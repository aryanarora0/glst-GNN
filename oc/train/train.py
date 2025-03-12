import torch
# import matplotlib.pyplot as plt

def train(model, device, optimizer, criterion, train_loader, epoch):
    model.train()
    # plot = True 
    for data in train_loader:
        data = data.to(device)

        optimizer.zero_grad()
        model_out = model(data)

        loss = criterion(
            beta = model_out["B"],
            x = model_out["H"],
            object_id = data.y,
        )
        
        loss = loss["attractive"] + loss["repulsive"] + loss["coward"] + loss["noise"]
        loss.backward()
        optimizer.step()
        
        # if plot:
        #     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        #     count = 0
        #     for x, y, col in zip(model_out["H"][:, 1].cpu().detach().numpy(), model_out["H"][:, 2].cpu().detach().numpy(), data.y.cpu().numpy()):
        #         ax.scatter(x, y, c = col % 20, cmap="tab20", vmin=0, vmax=20, s=40, edgecolor="none")
        #         count += 1
        #         if count > 1000:
        #             break

        #     ax.set_xlim(-100, 100)
        #     ax.set_ylim(-100, 100)
            
        #     plt.savefig(f"epoch_{epoch}.png")
        #     plt.close(fig)
        #     plot = False

    return model_out, loss

@torch.no_grad()
def test(model, device, criterion, test_loader):
    model.eval()

    for data in test_loader:
        data = data.to(device)

        model_out = model(data)

        loss = criterion(
            beta = model_out["B"],
            x = model_out["H"],
            object_id = data.y,
        )

    return model_out, loss