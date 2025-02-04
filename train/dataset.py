from torch_geometric.data import Dataset
import pickle
from glob import glob

class PointCloudGraphDataset(Dataset):
    def __init__(self, input_path, regex, transform=None):
        super().__init__()
        self.graphs = []
        for file in glob(input_path + regex):
            self.graphs.append(pickle.load(open(file, 'rb')))
        print(f"Loaded {len(self.graphs)} graphs")
        
        self.transform = transform

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.graphs[idx])
        return self.graphs[idx]