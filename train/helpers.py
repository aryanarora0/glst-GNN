import numpy as np
import torch

from torch_cluster import knn_graph

def sample_negative_edges(edge_index, num_nodes, num_samples):
    # Create adjacency set from existing edges
    existing_edges = set(tuple(edge_index[:, i].tolist()) for i in range(edge_index.shape[1]))
    # Sample random pairs
    neg_edges = []
    while len(neg_edges) < num_samples:
        u, v = np.random.randint(0, num_nodes, 2)
        if (u != v) and ((u, v) not in existing_edges) and ((v, u) not in existing_edges):
            neg_edges.append([u, v])
    
    return torch.tensor(neg_edges).T