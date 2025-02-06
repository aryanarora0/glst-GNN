import numpy as np
import torch

class MinMaxScalerColumns:
    def __call__(self, data):
        col_min = data.x.min(dim=0, keepdim=True).values
        col_max = data.x.max(dim=0, keepdim=True).values

        col_range = col_max - col_min
        col_range[col_range == 0] = 1e-9

        data.x = (data.x - col_min) / col_range
        return data

def compute_metrics(all_metrics, true_edge_index, candidate_edges, edge_probabilities, threshold=0.9):
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

def align_edge_indices(true_edges, predicted_edges, probabilities):
    true_edges_set = set(map(tuple, true_edges.t().tolist()))
    predicted_edges_set = set(map(tuple, predicted_edges.t().tolist()))

    predicted_edge_to_prob = {tuple(edge): prob for edge, prob in zip(predicted_edges.t().tolist(), probabilities.squeeze().tolist())}

    aligned_edges_set = true_edges_set.union(predicted_edges_set)
    aligned_edges = list(aligned_edges_set)
    aligned_edge_index = torch.tensor(aligned_edges, dtype=torch.long).t()

    true_edges_labels = torch.tensor([1 if edge in true_edges_set else 0 for edge in aligned_edges], dtype=torch.float32, requires_grad=True)

    aligned_probabilities = torch.tensor([predicted_edge_to_prob.get(edge, 0.0) for edge in aligned_edges], dtype=torch.float32, requires_grad=True)

    return true_edges_labels, aligned_edge_index, aligned_probabilities