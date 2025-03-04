from torch_geometric.data import Dataset
import torch
from glob import glob
import numpy as np

class GraphDataset(Dataset):
    def __init__(self, input_path, regex, subset=None, transform=None):
        super().__init__()
        self.subset = subset
        self.files = np.random.permutation(glob(input_path + regex))
        self.transform = transform
        self.cache = {}

    def __len__(self):
        if self.subset is not None:
            return min(self.subset, len(self.files))
        return len(self.files)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        graph = torch.load(self.files[idx], weights_only=False)
        if self.transform:
            graph = self.transform(graph)
        self.cache[idx] = graph
        return graph