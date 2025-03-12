import torch
import torch.nn as nn
from torch_geometric.nn import SuperGATConv
    
class EdgePredictionGNN(torch.nn.Module):
    def __init__(self, in_channels, emb_channels, hidden_channels, edge_dim=1):
        super(EdgePredictionGNN, self).__init__()
        self.conv1 = SuperGATConv(in_channels, emb_channels, concat=False, is_undirected=True)
        self.conv2 = SuperGATConv(emb_channels, hidden_channels, concat=False, is_undirected=True)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels + edge_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr, neg_edge_index=None):
        x = self.conv1(x, edge_index, neg_edge_index).relu()
        x = self.conv2(x, edge_index, neg_edge_index).relu()

        src, dst = x[edge_index[0]], x[edge_index[1]]
        x = torch.cat([src, edge_attr, dst], dim=-1)

        return self.mlp(x)    