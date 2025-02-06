import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, knn_graph
    
class EdgePredictionGNN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 emb_channels,
                 hidden_channels,
                 k=16):
        super(EdgePredictionGNN, self).__init__()
        self.k = k
        self.conv1 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * in_channels, emb_channels),
                nn.ReLU(),
                nn.Linear(emb_channels, emb_channels)
            ),
            k=k
        )
        self.conv2 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * emb_channels, emb_channels),
                nn.ReLU(),
                nn.Linear(emb_channels, emb_channels)
            ),
            k=k
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, batch):
        x = self.conv1(x, batch=batch)
        x = self.conv2(x, batch=batch)
        
        edge_index = knn_graph(x, self.k, batch=batch, loop=False)

        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=1)
        link_probs = self.mlp(edge_features)

        return link_probs, edge_index