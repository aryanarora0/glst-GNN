import torch.nn as nn
from torch_geometric.nn import GravNetConv
    
class GravNetGNN(nn.Module):
    def __init__(self, in_dim: int = 23, k: int = 2):
        super().__init__()
        self._embedding1 = GravNetConv(
            in_channels=in_dim,
            out_channels=in_dim,
            space_dimensions=8,
            propagate_dimensions=8,
            k=k,
        )
        self._embedding2 = GravNetConv(
            in_channels=in_dim,
            out_channels=in_dim,
            space_dimensions=8,
            propagate_dimensions=8,
            k=k,
        )
        self._beta = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        latent = self._embedding1(data.x, batch=data.batch)
        latent = self._embedding2(latent, batch=data.batch)
        beta = self._beta(latent)
        eps = 1e-6
        beta = beta.clamp(eps, 1 - eps)
        return {
            "B": beta,
            "H": latent,
        }