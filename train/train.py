from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch

torch.manual_seed(42)

from torch_cluster import knn_graph

from sklearn.model_selection import train_test_split

from helpers import sample_negative_edges
from model import EdgePredictionGNN
from dataset import PointCloudGraphDataset

def train(model, device, optimizer, train_loader, k=4):
    model.train()
    total_loss = 0

    for batch in train_loader:
        x, edge_index = batch.x, batch.edge_index

        num_nodes = x.size(0)
        
        pos_edges = edge_index
        neg_edges = sample_negative_edges(edge_index, num_nodes, pos_edges.size(1))
        
        all_edges = torch.cat([pos_edges, neg_edges], dim=1)

        x = x.to(device)
        edge_index = edge_index.to(device)
        all_edges = all_edges.to(device)

        labels = torch.cat([
            torch.ones(pos_edges.size(1)), 
            torch.zeros(neg_edges.size(1)) 
        ]).to(x.device)
        
        optimizer.zero_grad()
        predictions = model(x, edge_index, all_edges)
        loss = criterion(predictions.squeeze(), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Train loss: {total_loss:.4f}")

def test_edge_prediction_with_knn(model, device, test_loader, k=4, threshold=0.5):
    model.eval()
    metrics = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch.x
            num_nodes = x.size(0)

            knn_edge_index = knn_graph(x, k=k, batch=batch.batch)
            candidate_edges = sample_negative_edges(knn_edge_index, num_nodes, knn_edge_index.size(1))

            x = x.to(device)
            knn_edge_index = knn_edge_index.to(device)
            candidate_edges = candidate_edges.to(device)

            edge_probabilities = model(x, knn_edge_index, candidate_edges)
            edge_probabilities = edge_probabilities.squeeze()

            true_edges_set = set(tuple(batch.edge_index[:, i].tolist()) for i in range(batch.edge_index.shape[1]))
            candidate_edges_set = set(tuple(candidate_edges[:, i].tolist()) for i in range(candidate_edges.shape[1]))

            true_positive_edges = candidate_edges_set & true_edges_set
            false_positive_edges = candidate_edges_set - true_edges_set
            false_negative_edges = true_edges_set - candidate_edges_set

            tp = len(true_positive_edges) / len(true_edges_set)
            fp = len(false_positive_edges) / len(candidate_edges_set)
            fn = len(false_negative_edges) / len(true_edges_set)

            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

            labels = torch.tensor(
                [1 if edge in true_positive_edges else 0 for edge in candidate_edges_set]
            ).to(x.device)

            predictions = (edge_probabilities > threshold).float()

            metrics.append((tp, fp, fn, precision, recall, f1))

    metrics = torch.tensor(metrics).mean(dim=0)
    print(f"Test metrics: TP={metrics[0]:.4f}, FP={metrics[1]:.4f}, FN={metrics[2]:.4f}, Precision={metrics[3]:.4f}, Recall={metrics[4]:.4f}, F1={metrics[5]:.4f}")


if __name__ == "__main__":
    dataset = PointCloudGraphDataset(input_path='../data/', regex='graph_*.pkl')
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.03, random_state=42)

    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]

    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=5)    

    print("Train dataset length:", len(train_dataset))
    print("Test dataset length:", len(test_dataset))

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = EdgePredictionGNN(num_node_features=7)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    for epoch in range(20):
        print(f"Epoch {epoch}: ", end='')
        train(model, device, optimizer, train_loader)

    test_edge_prediction_with_knn(model, device, test_loader)