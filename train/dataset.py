from torch_geometric.data import Dataset
import pickle
from glob import glob

class PointCloudGraphDataset(Dataset):
    def __init__(self, input_path, regex):
        super().__init__()
        self.graphs = []
        for file in glob(input_path + regex):
            self.graphs.append(pickle.load(open(file, 'rb')))
        
        print(f"Loaded {len(self.graphs)} graphs")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]