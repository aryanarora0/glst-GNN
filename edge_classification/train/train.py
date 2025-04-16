import torch

def train(model, device, optimizer, lr_scheduler, criterion, train_loader):
    model.train()
    total_loss = 0

    for data in train_loader:
        x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
        x, edge_index, edge_attr, y = [i.to(device) for i in (x, edge_index, edge_attr, y)]

        optimizer.zero_grad()
        predictions = model(x, edge_index, edge_attr, neg_edge_index=edge_index[:, (y==0)]).squeeze(-1)

        loss = criterion(predictions, y.float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item() / len(train_loader)
    
    lr_scheduler.step()
    
    return total_loss

@torch.no_grad()
def test(model, device, criterion, test_loader):
    model.eval()
    total_loss = 0

    for data in test_loader:
        x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        y = y.to(device)

        predictions = model(x, edge_index, edge_attr).squeeze()
        loss = criterion(predictions, y.float())

        total_loss += loss.item() / len(test_loader)

    return total_loss