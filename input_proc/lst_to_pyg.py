import os
import uproot
import numpy as np
import networkx as nx
from argparse import ArgumentParser
import pickle as pkl
from torch_geometric.utils import from_networkx
from concurrent.futures import ProcessPoolExecutor

class GraphBuilder:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.input_tree = uproot.open(self.input_path)["tree"]

    def process_event(self, event_data, idx):
        """
        Processes a single event and saves the corresponding graph.
        """
        good_md1_idx = event_data["LS_MD_idx0"][event_data["LS_isFake"] == 0]
        good_md2_idx = event_data["LS_MD_idx1"][event_data["LS_isFake"] == 0]

        edges = np.vstack((np.array(good_md1_idx), np.array(good_md2_idx))).T

        G = nx.Graph()
        for i in range(len(event_data["MD_0_x"])):
            G.add_node(i, 
                        x1=event_data["MD_0_x"][i],
                        y1=event_data["MD_0_y"][i],
                        z1=event_data["MD_0_z"][i],
                        x2=event_data["MD_1_x"][i],
                        y2=event_data["MD_1_y"][i],
                        z2=event_data["MD_1_z"][i],
                        eta=event_data["MD_eta"][i],
                        phi=event_data["MD_phi"][i],
                        dphichange=event_data["MD_dphichange"][i]
                    )
        G.add_edges_from(edges)

        graph = from_networkx(G, group_node_attrs=["x1", "y1", "z1", "x2", "y2", "z2", "eta", "phi", "dphichange"])
        output_file = os.path.join(self.output_path, f"graph_nolayer_{idx}.pkl")
        with open(output_file, "wb") as f:
            pkl.dump(graph, f)
            print(f"Saved graph {idx}")

    def process_events_in_parallel(self):
        with ProcessPoolExecutor() as executor:
            futures = []
            for idx, events in enumerate(uproot.iterate(self.input_tree, step_size=1)):
                for event in events:
                    futures.append(executor.submit(self.process_event, event, idx))

            for future in futures:
                future.result()

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--input_path", type=str, required=True)
    argparser.add_argument("--output_path", type=str, default="./")

    args = argparser.parse_args()

    data = GraphBuilder(args.input_path, args.output_path)
    data.process_events_in_parallel()