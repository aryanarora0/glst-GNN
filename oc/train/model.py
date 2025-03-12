import torch.nn as nn
from torch_geometric.nn import GravNetConv
    
class GravNetGNN(nn.Module):
    def __init__(self, in_dim: int = 23, depth: int = 1, k: int = 2):
        super().__init__()
        layers = [
            GravNetConv(
                in_channels=in_dim,
                out_channels=in_dim,
                space_dimensions=3,
                propagate_dimensions=3,
                k=k,
            )
            for _ in range(depth)
        ]
        self._embedding = nn.Sequential(*layers)
        self._beta = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        latent = self._embedding(data.x)
        beta = self._beta(latent).squeeze()
        eps = 1e-6
        beta = beta.clamp(eps, 1 - eps)
        return {
            "B": beta,
            "H": latent,
        }