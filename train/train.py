import torch
from torch_geometric.nn import knn_graph

from helpers import align_edge_indices, compute_metrics

def train(model, device, optimizer, criterion, train_loader, k):
    model.train()
    total_loss = 0

    for batch in train_loader:
        x, edge_index = batch.x, batch.edge_index
        knn_edge_index = knn_graph(x, k=k, batch=batch.batch)

        aligned_edge_index, true_labels = align_edge_indices(edge_index, knn_edge_index)
        
        x = x.to(device)
        aligned_edge_index = aligned_edge_index.to(device)
        true_labels = true_labels.to(device)

        optimizer.zero_grad()
        predictions = model(x, aligned_edge_index)
        loss = criterion(predictions.squeeze(), true_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Train loss: {total_loss:.4f}")

@torch.no_grad()
def test(model, device, test_loader, k, threshold=0.5):
    model.eval()
    all_metrics = []
    
    for batch in test_loader:
        x, true_edge_index = batch.x, batch.edge_index
        knn_edge_index = knn_graph(x, k=k, batch=batch.batch)

        x = x.to(device)
        knn_edge_index = knn_edge_index.to(device)

        edge_probabilities = model(x, knn_edge_index).cpu().numpy().squeeze()
        
        compute_metrics(all_metrics, true_edge_index, knn_edge_index, edge_probabilities, threshold)
            
    metrics_tensor = torch.tensor(all_metrics, dtype=torch.float32)
    mean_metrics = metrics_tensor.mean(dim=0)

    print(f"Test metrics: TP={mean_metrics[0]:.4f}, FP={mean_metrics[1]:.4f}, FN={mean_metrics[2]:.4f}, Precision={mean_metrics[3]:.4f}, Recall={mean_metrics[4]:.4f}, F1={mean_metrics[5]:.4f}")