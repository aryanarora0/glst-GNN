from torch_geometric.data import Dataset
import torch
from glob import glob
import numpy as np

class GraphDataset(Dataset):
    def __init__(self, input_path, regex, subset=-1, transform=None):
        super().__init__()
        self.files = np.random.permutation(glob(input_path + regex))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        graph = torch.load(self.files[idx])
        if self.transform:
            return self.transform(graph)
        return graph