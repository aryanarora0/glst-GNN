import torch
from helpers import align_edge_indices, compute_metrics

def train(model, device, optimizer, criterion, train_loader):
    model.train()
    total_loss = 0

    for batch in train_loader:
        x, edge_index = batch.x, batch.edge_index
        x = x.to(device)

        optimizer.zero_grad()
        predictions, predicted_edge_index = model(x)

        true_edges_labels, _, aligned_probabilities = align_edge_indices(edge_index, predicted_edge_index, predictions)
        loss = criterion(aligned_probabilities, true_edges_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() / len(train_loader)

    print(f"Total Train loss: {total_loss:.4f}")
    return true_edges_labels, aligned_probabilities

@torch.no_grad()
def test(model, device, test_loader, threshold=0.5):
    model.eval()
    all_metrics = []
    
    for batch in test_loader:
        x, edge_index = batch.x, batch.edge_index
        x = x.to(device)

        predictions, predicted_edge_indices = model(x)
        
        predictions = predictions.cpu().numpy()
        predicted_edge_indices = predicted_edge_indices.cpu().numpy()
        
        compute_metrics(all_metrics, edge_index, predicted_edge_indices, predictions, threshold)
            
    metrics_tensor = torch.tensor(all_metrics, dtype=torch.float32)
    mean_metrics = metrics_tensor.mean(dim=0)

    print(f"Test metrics: TP={mean_metrics[0]:.4f}, FP={mean_metrics[1]:.4f}, FN={mean_metrics[2]:.4f}, Precision={mean_metrics[3]:.4f}, Recall={mean_metrics[4]:.4f}, F1={mean_metrics[5]:.4f}")