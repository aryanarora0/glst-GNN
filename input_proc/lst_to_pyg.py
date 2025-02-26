import os
import torch
import awkward as ak
import uproot
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch_geometric.data import Data
from argparse import ArgumentParser

NODE_FEATURES = ["MD_0_x", "MD_0_y", "MD_0_z", "MD_0_r", "MD_1_x", "MD_1_y", "MD_1_z", "MD_1_r", "MD_eta", "MD_phi",
                 "MD_pt", "MD_dphichange"]
EDGE_FEATURES = ["LS_pt", "LS_eta", "LS_phi", "LS_dphichange"]
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
        """Processes events in parallel with a progress tracker."""
        num_events = self.input_tree.num_entries

        if debug:
            # Process first event for debugging
            event_data = ak.to_dataframe(self.input_tree.arrays(entry_start=0, entry_stop=1))
            self.process_event(event_data, 0, debug)
            return

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            with tqdm(total=num_events, desc="Processing Events", dynamic_ncols=True) as pbar:
                for idx in range(num_events):
                    event_data = ak.to_dataframe(self.input_tree.arrays(entry_start=idx, entry_stop=idx+1))
                    futures[executor.submit(self.process_event, event_data, idx)] = idx

                for future in as_completed(futures):
                    result = future.result()
                    print(result)  # Print progress/errors immediately
                    pbar.update(1)  # Update progress bar dynamically

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--input_path", type=str, required=True)
    argparser.add_argument("--output_path", type=str, default="./")
    argparser.add_argument("--n_workers", type=int, default=16)
    argparser.add_argument("--debug", action="store_true")

    args = argparser.parse_args()

    data = GraphBuilder(args.input_path, args.output_path)
    data.process_events_in_parallel(args.n_workers, args.debug)