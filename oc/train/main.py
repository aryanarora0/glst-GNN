from argparse import ArgumentParser

import torch
torch.manual_seed(42)
import torch.optim as optim

from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN

from model import GravNetGNN
from dataset import GraphDataset
from train import train, test
from helpers import PerformanceEvaluator, plot_loss
from losses import CondensationLoss

from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    argparser.add_argument('--save_step', type=int, default=50, help='number of epochs between saves')
    argparser.add_argument('--debug', action='store_true', help='debug mode')
    args = argparser.parse_args()

    transform = None

    if args.debug:
        dataset = GraphDataset(input_path='../data/relval/', regex='graph_*.pt', subset=10, transform=transform)
    else:
        dataset = GraphDataset(input_path='../data/relval/', regex='graph_*.pt', transform=transform)

    test_size = 0.2
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size, random_state=42)

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)    

    print("Train dataset length:", len(train_dataset))
    print("Test dataset length:", len(test_dataset))

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    num_node_features = dataset[0].num_node_features

    model = GravNetGNN(
        in_dim=num_node_features,
        depth=1,
        k=4
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = CondensationLoss(q_min=0.1)

    for epoch in tqdm(range(1, args.epochs+1)):
        model_out, loss = train(model, device, optimizer, criterion, train_loader, epoch)
        if epoch > 0 and epoch % args.save_step == 0:
            torch.save(model.state_dict(), f"models/model_{epoch}.pt")
            print(f"Model saved at epoch {epoch}")

    # model.load_state_dict(torch.load("models/model_50.pt"))
    # model.eval()

    # for data in test_loader:
    #     data = data.to(device)
    #     out = model(data)
    #     X = out["H"].cpu().detach().numpy()
    #     cluster = DBSCAN(eps=1, min_samples=3).fit(X)
        
    #     data_labels = data.y.cpu().detach().numpy().flatten()
    #     uniq_data_labels = sorted(set(data_labels))
    #     for uniq_cl in uniq_data_labels:
    #         if uniq_cl == -1:
    #             continue
    #         num_elements_true = np.sum(data_labels == uniq_cl)
    #         first_idx = np.where(data_labels == uniq_cl)[0][0]
    #         if cluster.labels_[first_idx] == -1:
    #             continue
    #         num_elements_pred = np.sum(cluster.labels_ == cluster.labels_[first_idx])
    #         print(f"Cluster {uniq_cl}: {num_elements_true} true elements, {num_elements_pred} identified elements")
    #     print(f"Number of true clusters: {len(uniq_data_labels)}")
    #     print(f"Number of predicted clusters: {len(set(cluster.labels_)) - 1}")
    #     print(f"Number of noisy segments: {np.sum(cluster.labels_ == -1)}")
    #     break