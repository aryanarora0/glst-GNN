from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F

class EdgePredictionGNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 20)
        self.conv2 = GCNConv(20, 16)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        src_nodes = x[edge_index[0]]
        dest_nodes = x[edge_index[1]]

        edge_features = torch.cat([src_nodes, dest_nodes], dim=-1)

        edge_logits = self.mlp(edge_features)
        return edge_logits