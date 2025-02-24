from torch_geometric.data import Dataset
import pickle
from glob import glob

class GraphDataset(Dataset):
    def __init__(self, input_path, regex, subset=-1, transform=None):
        super().__init__()
        self.graphs = []
        files = glob(input_path + regex)
        for file in files[:subset]:
            with open(file, 'rb') as f:
                graph = pickle.load(f)
                if graph is not None:
                    self.graphs.append(graph)
        print(f"Loaded {len(self.graphs)} graphs")
        
        self.transform = transform

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.graphs[idx])
        return self.graphs[idx]