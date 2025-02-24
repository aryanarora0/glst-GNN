import os
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import pickle as pkl
import uproot

import torch
from torch_geometric.utils import from_networkx
import torch_geometric.transforms as T

class GraphBuilder:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.input_tree = uproot.open(self.input_path)["tree"]

    def process_event(self, event_data, idx, debug=False):
        if os.path.exists(os.path.join(self.output_path, f"graph_nolayer_{idx}.pkl")):
            print(f"Graph {idx} already exists, skipping...")
            return
        G = nx.DiGraph()
        for i in range(len(event_data["MD_0_x"])):
            G.add_node(i, 
                        x1=event_data["MD_0_x"][i],
                        y1=event_data["MD_0_y"][i],
                        z1=event_data["MD_0_z"][i],
                        r1=event_data["MD_0_r"][i],
                        x2=event_data["MD_1_x"][i],
                        y2=event_data["MD_1_y"][i],
                        z2=event_data["MD_1_z"][i],
                        r2=event_data["MD_1_r"][i],
                        eta=event_data["MD_eta"][i],
                        phi=event_data["MD_phi"][i],
                        dphichange=event_data["MD_dphichange"][i]
                    )
        
        for edge in range(len(event_data["LS_pt"])):
            G.add_edge(event_data["LS_MD_idx0"][edge], 
                       event_data["LS_MD_idx1"][edge],
                       pt=event_data["LS_pt"][edge])
    
        graph = from_networkx(G, group_node_attrs="all", group_edge_attrs="all")
        graph.y = torch.tensor((~event_data["LS_isFake"] + 2))

        graph = T.ToUndirected()(graph)

        if debug:
            print(graph)
        output_file = os.path.join(self.output_path, f"graph_nolayer_{idx}.pkl")
        with open(output_file, "wb") as f:
            pkl.dump(graph, f)
            print(f"Saved graph {idx}")

    def process_events_in_parallel(self, n_workers, debug=False):
        if not debug:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                for idx, events in enumerate(uproot.iterate(self.input_tree, step_size=1)):
                    for event in events:
                        futures.append(executor.submit(self.process_event, event, idx))

                for future in futures:
                    future.result()
        else:
            for idx, events in enumerate(uproot.iterate(self.input_tree, step_size=1)):
                for event in events:
                    self.process_event(event, idx, debug)
                    break
                break

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--input_path", type=str, required=True)
    argparser.add_argument("--output_path", type=str, default="./")
    argparser.add_argument("--n_workers", type=int, default=64)
    argparser.add_argument("--debug", action="store_true")

    args = argparser.parse_args()

    data = GraphBuilder(args.input_path, args.output_path)
    data.process_events_in_parallel(args.n_workers, args.debug)