from argparse import ArgumentParser

import torch
torch.manual_seed(42)
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN

from model import GravNetGNN
from dataset import GraphDataset
from train import train, validation
from helpers import plot_loss

import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.CMS)

from fastgraphcompute.object_condensation import ObjectCondensation

from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    argparser.add_argument('--save-step', type=int, default=50, help='number of epochs between saves')
    argparser.add_argument('--validation-only', action='store_true', help='only validation step')
    argparser.add_argument('--debug', action='store_true', help='debug mode')
    args = argparser.parse_args()

    transform = None

    if args.debug:
        dataset = GraphDataset(input_path='../data/relval/', regex='graph_*.pt', subset=100, transform=transform)
    else:
        dataset = GraphDataset(input_path='../data/relval/', regex='graph_*.pt', subset=10, transform=transform)

    test_size = 0.1
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size, random_state=42)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, drop_last=True, num_workers=4, pin_memory=True)    

    print("Train dataset length:", len(train_dataset))
    print("Test dataset length:", len(test_dataset))

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    num_node_features = dataset[0].num_node_features

    model = GravNetGNN(
        in_dim=num_node_features,
        k=12
    )
    model.to(device)

    if not args.validation_only:
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-4, threshold_mode='abs', cooldown=0, min_lr=1e-6)
        criterion = ObjectCondensation(q_min=0.1, s_B=1)

        tot_losses = {}
        for epoch in tqdm(range(1, args.epochs+1)):
            model_out, losses = train(model, device, optimizer, criterion, train_loader, epoch)
            for key, value in losses.items():
                if key not in tot_losses:
                    tot_losses[key] = []
                tot_losses[key].append(value)        
            if epoch > 0 and epoch % args.save_step == 0:
                torch.save(model.state_dict(), f"models/model_{epoch}.pt")
                print(f"Model saved at epoch {epoch}")

        plot_loss(tot_losses)

    else:
        model.load_state_dict(torch.load("models/model_450.pt", weights_only=False))
        model.eval()
        eps_range = np.arange(0.01, 0.21, 0.02)
        perfect_rates, lhc_rates, dm_rates = [], [], []
        
        for eps in tqdm(eps_range, desc="Testing epsilon values"):
            perfect_batch, lhc_batch, dm_batch = [], [], []
            
            for data in test_loader:
                data = data.to(device)
                perf, lhc, dm = validation(data, model, eps=eps)
                perfect_batch.append(perf)
                lhc_batch.append(lhc)
                dm_batch.append(dm)
            
            perfect_rates.append(np.mean(perfect_batch))
            lhc_rates.append(np.mean(lhc_batch))
            dm_rates.append(np.mean(dm_batch))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(eps_range, perfect_rates, 'o-', label='Perfect')
        ax.plot(eps_range, lhc_rates, 's-', label='LHC')
        ax.plot(eps_range, dm_rates, '^-', label='DM')
        
        ax.set_xlabel('Epsilon (Clustering Distance Threshold)')
        ax.set_ylabel('Success Rate')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig("plots/eps_test.png", dpi=300)
        plt.close()


    # # print explicitly for one event
    # perf_truth = []
    # perf_fakes = []

    # for data in test_loader:
    #     data = data.to(device)
    #     out = model(data)
    #     X = out["H"].cpu().detach().numpy()
    #     cluster = DBSCAN(eps=0.1, min_samples=2).fit(X)

    #     data_labels = data.y.cpu().detach().numpy().flatten()
    #     uniq_data_labels = sorted(set(data_labels))

    #     perf_truth_local = []
    #     perf_fakes_local = []

    #     for uniq_cl in uniq_data_labels:
    #         true_cluster_indices = np.where(data_labels == uniq_cl)[0]

    #          # Get all DBSCAN labels for points in this true cluster
    #         cluster_dbscan_labels = cluster.labels_[true_cluster_indices]
            
    #         # Skip if all points in the true cluster are noise
    #         if np.all(cluster_dbscan_labels == -1):
    #             continue

    #         # Find the most common non-noise DBSCAN label
    #         non_noise_labels = cluster_dbscan_labels[cluster_dbscan_labels != -1]
    #         if len(non_noise_labels) == 0:
    #             continue  # Skip if all points are noise
            
    #         # Get the most common non-noise label
    #         unique_labels, counts = np.unique(non_noise_labels, return_counts=True)
    #         dbscan_label = unique_labels[np.argmax(counts)]

    #         num_elements_true = len(true_cluster_indices)
    #         num_elements_pred = np.sum(cluster.labels_ == dbscan_label)
    #         num_elements_correct = np.sum(data_labels[cluster.labels_ == dbscan_label] == uniq_cl)
    #         num_elements_fake = num_elements_pred - num_elements_correct
            
    #         perf_truth_local.append(num_elements_correct / num_elements_true if num_elements_true > 0 else 0)
    #         perf_fakes_local.append(num_elements_fake / num_elements_pred if num_elements_pred > 0 else 0)
            
    #         print(f"Cluster {uniq_cl}: {num_elements_true} true elements, {num_elements_pred} identified elements, {num_elements_correct} correct, {num_elements_fake} fake")
        
    #     print(f"Number of true clusters: {len(uniq_data_labels)}")
    #     print(f"Number of predicted clusters: {len(set(cluster.labels_)) - 1}")
    #     print(f"Number of noisy segments: {np.sum(cluster.labels_ == -1)}")
    #     break