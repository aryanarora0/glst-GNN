import os
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser

import uproot
import awkward as ak

import torch
from torch_geometric.data import Data

NODE_FEATURES = ["MD_0_x", "MD_0_y", "MD_0_z", "MD_0_r", "MD_1_x", "MD_1_y", "MD_1_z", "MD_1_r", "MD_pt", "MD_eta", "MD_phi" ,"MD_dphichange"]
EDGE_FEATURES = ["LS_pt", "LS_eta", "LS_phi"]
EDGE_INDEX = ["LS_MD_idx0", "LS_MD_idx1"]
Y_VAL = ["LS_isFake"]
ALL_COLUMNS = NODE_FEATURES + EDGE_FEATURES + EDGE_INDEX + Y_VAL

class GraphBuilder:
    def __init__(self, input_path, output_path):
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.input_tree = uproot.open(input_path)["tree"]

    def _DeltaPhi(self, phi1: torch.Tensor, phi2: torch.Tensor):
        dphi = (phi1 - phi2) % (2 * torch.pi)
        dphi[dphi < -torch.pi] += 2 * torch.pi
        dphi[dphi > torch.pi] -= 2 * torch.pi
        return dphi

    def process_event(self, event_data, idx, debug=False, overwrite=False):
        """Processes a single event and saves the graph."""
        output_file = os.path.join(self.output_path, f"graph_{idx}.pt")
        if debug:
            ...
        elif ((not overwrite) and os.path.exists(output_file)):
            print(f"Graph {idx} already exists, skipping...")
            return
        
        node_features = torch.tensor(ak.to_dataframe(event_data[NODE_FEATURES]).values)
        edge_index = torch.tensor(ak.to_dataframe(event_data[EDGE_INDEX]).values.T).to(torch.int64)
        edge_feature = torch.tensor(ak.to_dataframe(event_data[EDGE_FEATURES]).values)
        
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        d_pt = abs(node_features[src_idx, 8] - node_features[dst_idx, 8])
        d_eta = abs(node_features[src_idx, 9] - node_features[dst_idx, 9])
        d_phi = self._DeltaPhi(node_features[src_idx, 10], node_features[dst_idx, 10])
        dr = torch.sqrt(d_eta ** 2 + d_phi ** 2)

        extra_edge_features = torch.stack([d_pt, d_eta, d_phi, dr], dim=1)
        edge_feature = torch.cat([edge_feature, extra_edge_features], dim=1)
        
        y = torch.logical_not(torch.tensor(ak.to_dataframe(event_data[Y_VAL]).values)).int().view(-1)

        node_features = torch.nan_to_num(node_features, nan=0, posinf=50000, neginf=-50000)
        edge_feature = torch.nan_to_num(edge_feature, nan=0, posinf=50000, neginf=-50000)

        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_feature, y=y)

        if debug:
            print(graph)
            return

        torch.save(graph, output_file)
        print(f"Processed graph {idx}")

    def process_events_in_parallel(self, n_workers, debug=False, overwrite=False):
        num_events = self.input_tree.num_entries if not debug else 1
        n_workers =  min(n_workers, os.cpu_count() // 2)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for idx in range(num_events):
                executor.submit(
                    self.process_event,
                    self.input_tree.arrays(ALL_COLUMNS, entry_start=idx, entry_stop=idx + 1),
                    idx, debug, overwrite
                )


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--input_path", type=str, required=True)
    argparser.add_argument("--output_path", type=str, default="./")
    argparser.add_argument("--n_workers", type=int, default=16)
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--overwrite", action="store_true", help="Overwrite existing graphs")

    args = argparser.parse_args()

    data = GraphBuilder(args.input_path, args.output_path)
    data.process_events_in_parallel(args.n_workers, args.debug, args.overwrite)