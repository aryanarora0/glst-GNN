import os
from concurrent.futures import ProcessPoolExecutor
from argparse import ArgumentParser

import uproot
import awkward as ak
import numpy as np

import torch
from torch_geometric.data import Data

LS_VARS = ["LS_pt", "LS_eta", "LS_phi"]
MD_VARS = ["MD_0_x", "MD_0_y", "MD_0_z", "MD_0_r", "MD_dphichange", "MD_1_x", "MD_1_y", "MD_1_z", "MD_1_r", "MD_dphichange"]
MD_INDEX = ["LS_MD_idx0", "LS_MD_idx1"]
CUTS = ["LS_isFake"]
TARGET = ["LS_TCidx"]
ALL_COLUMNS = LS_VARS + MD_VARS + MD_INDEX + TARGET + CUTS

class GraphBuilder:
    def __init__(self, input_path, output_path):
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.input_tree = uproot.open(input_path)["tree"]

    def process_event(self, event_data, idx, debug=False, overwrite=False):
        """Processes a single event and saves the graph."""
        output_file = os.path.join(self.output_path, f"graph_{idx}.pt")
        if debug:
            ...
        elif ((not overwrite) and os.path.exists(output_file)):
            print(f"Graph {idx} already exists, skipping...")
            return
        
        mask = np.logical_not(ak.to_dataframe(event_data[CUTS]).values.flatten())
        ls_features = ak.to_dataframe(event_data[LS_VARS]).values[mask]
        md_idx = ak.to_dataframe(event_data[MD_INDEX]).values[mask]
        md_features = ak.to_dataframe(event_data[MD_VARS]).values[md_idx]

        md_features = md_features.reshape(-1, 2 * len(MD_VARS))

        node_features = torch.Tensor(ak.concatenate([ls_features, md_features], axis=1))
        target = torch.Tensor(ak.to_dataframe(event_data[TARGET]).values[mask])

        graph = Data(x=node_features, y=target)

        if debug:
            print(graph)
            return

        torch.save(graph, output_file)
        print(f"Processed graph {idx}")

    def process_events_in_parallel(self, n_workers, debug=False, overwrite=False):
        num_events = self.input_tree.num_entries if not debug else 1
        n_workers =  min(n_workers, os.cpu_count() // 2) if not debug else 1

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for idx in range(num_events):
                executor.submit(
                    self.process_event,
                    self.input_tree.arrays(ALL_COLUMNS, entry_start=idx, entry_stop=idx + 1),
                    idx, debug, overwrite
                )


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--input", type=str, required=True)
    argparser.add_argument("--output", type=str, default="./")
    argparser.add_argument("--n_workers", type=int, default=16)
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--overwrite", action="store_true", help="Overwrite existing graphs")

    args = argparser.parse_args()

    data = GraphBuilder(args.input, args.output)
    data.process_events_in_parallel(args.n_workers, args.debug, args.overwrite)