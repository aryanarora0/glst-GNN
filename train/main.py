import torch
torch.manual_seed(42)
import torch.optim as optim

from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch.utils.data import random_split

from model import EdgePredictionGNN
from dataset import PointCloudGraphDataset
from train import train, test

if __name__ == "__main__":
    transform = T.Compose([T.ToUndirected(), T.NormalizeFeatures()])
    dataset = PointCloudGraphDataset(input_path='../data/', regex='graph_*.pkl', transform=transform)

    train_size = 0.95
    test_size = 1 - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=5)    

    print("Train dataset length:", len(train_dataset))
    print("Test dataset length:", len(test_dataset))

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = EdgePredictionGNN(num_node_features=7)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    for epoch in range(10):
        print(f"Epoch {epoch}: ", end='')
        train(model, device, optimizer, criterion, train_loader)

    test(model, device, test_loader, threshold=0.5)