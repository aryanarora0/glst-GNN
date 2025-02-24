import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
    
class EdgePredictionGNN(torch.nn.Module):
    def __init__(self, in_channels, emb_channels, hidden_channels):
        super(EdgePredictionGNN, self).__init__()
        self.conv1 = GATConv(in_channels, emb_channels)
        self.conv2 = GATConv(emb_channels, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()

        src, dst = x[edge_index[0]], x[edge_index[1]]
        x = torch.cat([src, dst], dim=-1)

        x = self.mlp(x)
        return x