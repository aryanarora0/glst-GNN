import os
from multiprocessing import Pool
from argparse import ArgumentParser

import numpy as np
import pickle as pkl
import networkx as nx

import uproot

from torch_geometric.utils.convert import from_networkx

class GraphBuilder:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        with uproot.open(self.input_path) as file:
            self.input_tree = file.get("tree")
        
    def load_event_data(self):
        for events in uproot.iterate(self.input_tree, step_size=1):
            for event in events:
                good_md1_idx = event["LS_MD_idx0"][event["LS_isFake"] == 0]
                good_md2_idx = event["LS_MD_idx1"][event["LS_isFake"] == 0]

                edges = np.transpose(np.vstack((good_md1_idx, good_md2_idx)))

                G = nx.Graph()
                for i in range(len(event["MD_0_x"])):
                    G.add_node(i, x1=event["MD_0_x"][i], y1=event["MD_0_y"][i], z1=event["MD_0_z"][i], x2=event["MD_1_x"][i], y2=event["MD_1_y"][i], z2=event["MD_1_z"][i], layer=event["MD_layer"][i])
                G.add_edges_from(edges)

                yield G

    def process_and_save_graph(self, idx_event):
        idx, event = idx_event
        graph = from_networkx(event, group_node_attrs=["x1", "y1", "z1", "x2", "y2", "z2", "layer"])
        with open(f"{self.output_path}/graph_{idx}.pkl", "wb") as f:
            pkl.dump(graph, f)

    def save_graphs(self):
        with Pool(4) as pool:
            results = pool.imap(self.process_and_save_graph, enumerate(self.load_event_data()))
            for _ in results:
                pass

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--input_path", type=str, required=True)
    argparser.add_argument("--output_path", type=str, default="./")

    args = argparser.parse_args()

    data = GraphBuilder(args.input_path, args.output_path)
    data.save_graphs()