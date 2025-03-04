from argparse import ArgumentParser

import torch
torch.manual_seed(42)
import torch.optim as optim

from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch.utils.data import random_split

from model import EdgePredictionGNN
from dataset import GraphDataset
from train import train, test
from helpers import PerformanceEvaluator, plot_loss, ColumnWiseNormalizeFeatures

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    argparser.add_argument('--test_step', type=int, default=5, help='number of epochs between tests')
    argparser.add_argument('--debug', action='store_true', help='debug mode')
    args = argparser.parse_args()

    #TODO: put this path in a config file
    transform = None
    if args.debug:
        dataset = GraphDataset(input_path='../data/relval/', regex='graph_*.pt', subset=10)
    else:
        dataset = GraphDataset(input_path='../data/relval/', regex='graph_*.pt', transform=transform, subset=3000)

    test_size = 0.2
    train_dataset, test_dataset = random_split(dataset, [1-test_size, test_size])

    print("Train dataset length:", len(train_dataset))
    print("Test dataset length:", len(test_dataset))

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_node_features = dataset[0].num_node_features
    num_edge_attrs = dataset[0].num_edge_features

    model = EdgePredictionGNN(
        in_channels=num_node_features,
        edge_dim=num_edge_attrs,
        emb_channels=64,
        hidden_channels=16
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([30.], device=device))
    lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5)

    train_loss, test_loss = [], []
    for epoch in range(args.epochs+1):
        print(f"Epoch {epoch}: ", end='')
        tr_loss = train(model, device, optimizer, lr_scheduler, criterion, train_loader)
        train_loss.append(tr_loss)

        if epoch % args.test_step == 0:
            te_loss = test(model, device, criterion, test_loader)
            test_loss.append(te_loss)
            print(f"Train Loss: {tr_loss:.4f}, Test loss: {te_loss:.4f}")
        else:
            print(f"Train Loss: {tr_loss:.4f}") 

    pe = PerformanceEvaluator(model, device, train_loader, test_loader)
    pe.plot_precision_recall_curve()
    pe.plot_roc_curve()
    
    plot_loss(train_loss, test_loss, args.test_step)