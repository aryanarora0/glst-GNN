import os
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import uproot
import torch
from torch_geometric.data import Data
import awkward as ak

NODE_FEATURES = ["MD_0_x", "MD_0_y", "MD_0_z", "MD_0_r", "MD_1_x", "MD_1_y", "MD_1_z", "MD_1_r", "MD_eta", "MD_phi",
                 "MD_pt", "MD_dphichange"]
EDGE_FEATURES = ["LS_pt", "LS_eta", "LS_phi", "LS_dphichange"]
EDGE_INDEX = ["LS_MD_idx0", "LS_MD_idx1"]
Y_VAL = ["LS_isFake"]

class GraphBuilder:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.input_tree = uproot.open(self.input_path)["tree"]

    def process_event(self, event_data, idx, debug=False):
        if os.path.exists(os.path.join(self.output_path, f"graph_nolayer_{idx}.pt")):
            print(f"Graph {idx} already exists, skipping...")
            return

        node_features = torch.tensor(ak.to_dataframe(event_data.arrays(NODE_FEATURES)).values)
        edge_index = torch.tensor(ak.to_dataframe(event_data.arrays(EDGE_FEATURES)).values)
        edge_feature = torch.tensor(ak.to_dataframe(event_data.arrays(EDGE_INDEX)).values)
        y = torch.logical_not(torch.tensor(event_data.arrays(Y_VAL, library='np')['LS_isFake'][0])).int()

        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_feature, y=y)

        if debug:
            print(graph)
        torch.save(graph, os.path.join(self.output_path, f"graph_nolayer_{idx}.pt"))
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