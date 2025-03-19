import torch
# import matplotlib.pyplot as plt

def train(model, device, optimizer, criterion, train_loader, epoch):
    # plot = True
    model.train()
    tot_attractive_loss = 0
    tot_repulsive_loss = 0
    tot_loss = 0

    for data in train_loader:
        data = data.to(device)

        optimizer.zero_grad()
        model_out = model(data)

        loss = criterion(
            beta = model_out["B"],
            x = model_out["H"],
            object_id = data.y,
        )

        # plot beta
        # if epoch > 9:
        #     fig, ax = plt.subplots()
        #     ax.hist(model_out["B"].cpu().detach().numpy(), bins=100)
        #     ax.set_title(f"Epoch {epoch}")
        #     ax.set_xlabel(r"$\beta$")
        #     ax.set_ylabel("Count")
        #     plt.savefig(f"epoch_{epoch}_beta.png")
        #     exit()
        
        tot_loss_batch = loss["attractive"] + loss["repulsive"]
        
        tot_loss_batch.backward()
        optimizer.step()

        tot_attractive_loss += loss["attractive"].item()
        tot_repulsive_loss += loss["repulsive"].item()
        tot_loss += tot_loss_batch.item()

    losses = {
        "attractive": tot_attractive_loss / len(train_loader),
        "repulsive": tot_repulsive_loss / len(train_loader),
        "loss": tot_loss / len(train_loader),
    }

    # if plot:
    #         fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    #         count = 0
    #         for x, y, col in zip(model_out["H"][:, 1].cpu().detach().numpy(), model_out["H"][:, 2].cpu().detach().numpy(), data.y.cpu().numpy()):
    #             ax.scatter(x, y, c = col % 20, cmap="tab20", vmin=0, vmax=20, s=40, edgecolor="none")
    #             count += 1
    #             if count > 1000:
    #                 break

    #         ax.set_xlim(-10, 100)
    #         ax.set_ylim(-10, 100)
            
    #         ax.set_title(f"Epoch {epoch}")

    #         plt.savefig(f"epoch_{epoch}.png")
    #         plt.close(fig)
    #         plot = False

    return model_out, losses

# @torch.no_grad()
# def test(model, device, criterion, test_loader):
#     model.eval()
#     for data in test_loader:
#         data = data.to(device)
#         model_out = model(data)

#     # run DBSCAN and print metrics