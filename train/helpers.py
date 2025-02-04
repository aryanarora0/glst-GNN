import numpy as np
import torch

def compute_metrics(all_metrics, true_edge_index, knn_edge_index, edge_probabilities, threshold=0.2):
    candidate_edges = knn_edge_index.cpu().numpy()
    predicted_edges = [tuple(candidate_edges[:, i]) for i in range(candidate_edges.shape[1]) if edge_probabilities[i] >= threshold]

    true_edges_set = set(tuple(true_edge_index[:, i].tolist()) for i in range(true_edge_index.shape[1]))
    predicted_edges_set = set(predicted_edges)

    true_positive_edges = predicted_edges_set & true_edges_set
    false_positive_edges = predicted_edges_set - true_edges_set
    false_negative_edges = true_edges_set - predicted_edges_set

    tp = len(true_positive_edges)
    fp = len(false_positive_edges)
    fn = len(false_negative_edges)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    all_metrics.append((tp, fp, fn, precision, recall, f1))

def align_edge_indices(edge_index1, edge_index2):
    edge_set1 = set(map(tuple, edge_index1.t().tolist()))
    edge_set2 = set(map(tuple, edge_index2.t().tolist()))

    # print(f"Number of common edges: {len(edge_set1 & edge_set2)}")

    aligned_edges = list(edge_set1.union(edge_set2))
    aligned_labels1 = [1 if edge in edge_set1 else 0 for edge in aligned_edges]

    aligned_edge_index = torch.tensor(aligned_edges, dtype=torch.long).t()
    label1 = torch.tensor(aligned_labels1, dtype=torch.float32)

    return aligned_edge_index, label1

def sample_negative_edges(edge_index, num_nodes, num_neg_samples):
    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        src = torch.randint(0, num_nodes, (1,))
        dest = torch.randint(0, num_nodes, (1,))
        if src != dest and (src.item(), dest.item()) not in edge_index.t().tolist():
            neg_edges.append((src.item(), dest.item()))
    return torch.tensor(neg_edges, dtype=torch.long).t()

class MinMaxScalerColumns:
    def __call__(self, data):
        # Compute the min and max per column
        col_min = data.x.min(dim=0, keepdim=True).values
        col_max = data.x.max(dim=0, keepdim=True).values

        # Avoid division by zero by setting small epsilon where max == min
        col_range = col_max - col_min
        col_range[col_range == 0] = 1e-9

        # Rescale each column to [0, 1]
        data.x = (data.x - col_min) / col_range
        return data
