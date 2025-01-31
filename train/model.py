from torch_geometric.nn import GCNConv
import torch

class EdgePredictionGNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64, add_self_loops=False)
        self.conv2 = GCNConv(64, 64)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x, edge_index):
        # Node embeddings from GNN layers
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        
        # Get embeddings for candidate edge pairs
        md0, md1 = x[edge_index[0]], x[edge_index[1]]
        edge_features = torch.cat([md0, md1], dim=1)
        
        # Predict edge existence
        return torch.sigmoid(self.mlp(edge_features))   