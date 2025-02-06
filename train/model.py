import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, knn_graph
    
class MLP(nn.Module):
    def __init__(self, n_in, n_out, n_hidden=20, c=False):
        super().__init__()
        layers = [
            nn.Linear(n_in, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(n_hidden, n_out)
        ]
        if c:
            layers.append(nn.Sigmoid())
        self.seq = nn.Sequential(*layers)

    def forward(self, *args, **kwargs):
        return self.seq(*args, **kwargs)
    

class EdgePredictionGNN(nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, out_channels=64, k=4):
        super(EdgePredictionGNN, self).__init__()
        self.k = k
        self.conv1 = DynamicEdgeConv(MLP(2 * num_node_features, hidden_channels // 2), k)
        self.conv2 = DynamicEdgeConv(MLP(hidden_channels, out_channels), k)
        self.link_predictor = MLP(2 * out_channels, 1, c=True)

    def predict_links(self, x, edge_index):
        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=1)
        link_probs = self.link_predictor(edge_features)
        return link_probs
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        edge_index = knn_graph(x, self.k)
        link_probs = self.predict_links(x, edge_index)
        return link_probs, edge_index