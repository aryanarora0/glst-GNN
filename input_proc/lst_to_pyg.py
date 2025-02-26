import os
import torch
import awkward as ak
import uproot
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch_geometric.data import Data
from argparse import ArgumentParser
import time 

NODE_FEATURES = ["MD_0_x", "MD_0_y", "MD_0_z", "MD_0_r", "MD_1_x", "MD_1_y", "MD_1_z", "MD_1_r", "MD_eta", "MD_phi",
                 "MD_pt", "MD_dphichange"]
EDGE_FEATURES = ["LS_pt", "LS_eta", "LS_phi"]
EDGE_INDEX = ["LS_MD_idx0", "LS_MD_idx1"]
Y_VAL = ["LS_isFake"]


class GraphBuilder:
    def __init__(self, input_path, output_path):
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.input_tree = uproot.open(input_path)["tree"]

    def process_event(self, event_data, idx, debug=False):
        """Processes a single event and saves the graph."""
        output_file = os.path.join(self.output_path, f"graph_nolayer_{idx}.pt")

        if os.path.exists(output_file):
            print(f"Graph {idx} already exists, skipping...")
            return

        node_features = torch.tensor(event_data[NODE_FEATURES].values)
        edge_index = torch.tensor(event_data[EDGE_INDEX].values)  # Fixed variable
        edge_feature = torch.tensor(event_data[EDGE_FEATURES].values)  # Fixed variable
        y = torch.logical_not(torch.tensor(event_data[Y_VAL])).int()

        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_feature, y=y)

        if debug:
            print(graph)
        torch.save(graph, output_file)
        print(f"Saved graph {idx}")

    def process_events_in_parallel(self, n_workers, debug=False):
        num_events = self.input_tree.num_entries if not debug else 1
        t1 = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for idx in range(num_events):
                    executor.submit(
                    self.process_event,
                    ak.to_dataframe(self.input_tree.arrays(entry_start=idx, entry_stop=idx + 1)),
                    idx)

        t2 = time.time() 
        print("time", t2-t1)
if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--input_path", type=str, required=True)
    argparser.add_argument("--output_path", type=str, default="./")
    argparser.add_argument("--n_workers", type=int, default=16)
    argparser.add_argument("--debug", action="store_true")

    args = argparser.parse_args()

    data = GraphBuilder(args.input_path, args.output_path)
    data.process_events_in_parallel(args.n_workers, args.debug)
