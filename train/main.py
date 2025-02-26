from argparse import ArgumentParser

import torch
torch.manual_seed(42)
import torch.optim as optim

from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from sklearn.model_selection import train_test_split

from model import EdgePredictionGNN
from dataset import GraphDataset
from train import train, test
from helpers import PerformanceEvaluator, plot_loss

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--epochs', type=int, default=31, help='number of epochs to train')
    argparser.add_argument('--debug', action='store_true', help='debug mode')
    args = argparser.parse_args()

    #transform = T.Compose([T.LocalDegreeProfile()])

    #TODO: put this path in a config file
    if args.debug:
        dataset = GraphDataset(input_path='/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/glst/lst_graphs/', regex='graph_nolayer_*.pt')
    else:
        dataset = GraphDataset(input_path='/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/glst/lst_graphs/', regex='graph_nolayer_*.pt')

    test_size = 0.2
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size, random_state=42)

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)    

    print("Train dataset length:", len(train_dataset))
    print("Test dataset length:", len(test_dataset))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EdgePredictionGNN(
        in_channels=dataset[0].num_node_features,
        emb_channels=64,
        hidden_channels=16,
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1000], device=device))
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    train_loss, test_loss = [], []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}: ", end='')
        tr_loss = train(model, device, optimizer, lr_scheduler, criterion, train_loader)
        train_loss.append(tr_loss)

        if epoch % 10 == 0:
            te_loss = test(model, device, criterion, test_loader)
            test_loss.append(te_loss)
            print(f"Train Loss: {tr_loss:.4f}, Test loss: {te_loss:.4f}")
        else:
            print(f"Train Loss: {tr_loss:.4f}") 

    pe = PerformanceEvaluator(model, device, train_loader, test_loader)
    pe.plot_precision_recall_curve()
    pe.plot_roc_curve()
    plot_loss(train_loss, test_loss)