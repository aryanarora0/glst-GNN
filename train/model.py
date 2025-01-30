from torch_geometric.nn import GCNConv
import torch

class EdgePredictionGNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x, edge_index, edge_pairs):
        # Node embeddings from GNN layers
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        
        # Get embeddings for candidate edge pairs
        h_src, h_dst = x[edge_pairs[0]], x[edge_pairs[1]]
        edge_features = torch.cat([h_src, h_dst], dim=1)
        
        # Predict edge existence
        return torch.sigmoid(self.mlp(edge_features))   