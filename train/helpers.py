import numpy as np
import torch

from torch_cluster import knn_graph

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
    """
    Align two edge index tensors by inserting zeros for missing edges.

    Parameters:
    edge_index1 (torch.Tensor): The first edge index of shape [2, num_edges1].
    edge_index2 (torch.Tensor): The second edge index of shape [2, num_edges2].

    Returns:
    torch.Tensor: Aligned edge indices for edge_index1 and edge_index2.
    torch.Tensor: Corresponding labels (1 for real edges, 0 for padded ones).
    """
    # Convert edge_index tensors to sets for easy comparison
    edge_set1 = set(map(tuple, edge_index1.t().tolist()))
    edge_set2 = set(map(tuple, edge_index2.t().tolist()))

    # Find unique edges in each set
    only_in_1 = edge_set1 - edge_set2
    only_in_2 = edge_set2 - edge_set1

    # Prepare aligned edge list and labels
    aligned_edges = list(edge_set1.union(edge_set2))
    aligned_labels1 = [1 if edge in edge_set1 else 0 for edge in aligned_edges]
    aligned_labels2 = [1 if edge in edge_set2 else 0 for edge in aligned_edges]

    # Convert back to torch tensors
    aligned_edge_index = torch.tensor(aligned_edges, dtype=torch.long).t()
    label1 = torch.tensor(aligned_labels1, dtype=torch.float32)
    label2 = torch.tensor(aligned_labels2, dtype=torch.float32)

    return aligned_edge_index, label1, label2