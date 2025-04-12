import torch
from helpers import batch_to_rowsplits
from sklearn.cluster import DBSCAN
import numpy as np

def train(model, device, optimizer, criterion, train_loader, epoch):
    model.train()
    tot_attractive_loss = 0
    tot_repulsive_loss = 0
    tot_loss = 0

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        with_stack=True
    ) as prof:

        for i, data in enumerate(train_loader):
            data = data.to(device)

            row_splits = batch_to_rowsplits(data.batch)

            scaler = torch.cuda.amp.GradScaler()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                model_out = model(data)

                L_att, L_rep, L_beta, _, _ = criterion(
                    beta = model_out["B"],
                    coords = model_out["H"],
                    asso_idx = data.y.to(torch.int32),
                    row_splits = row_splits
                )
            
                tot_loss_batch = L_att + L_rep + L_beta

            scaler.scale(tot_loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()        

            tot_attractive_loss += L_att.item()
            tot_repulsive_loss += L_rep.item()
            tot_noise_loss = L_beta.item()
            tot_loss += tot_loss_batch.item()

            del L_att, L_rep, L_beta, tot_loss_batch
            torch.cuda.empty_cache()

            prof.step()

    losses = {
        "attractive": tot_attractive_loss / len(train_loader),
        "repulsive": tot_repulsive_loss / len(train_loader),
        "noise": tot_noise_loss / len(train_loader),
        "loss": tot_loss / len(train_loader),
    }

    print("Memory used: ", torch.cuda.memory_allocated() / 1e9, "GB")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    return model_out, losses

def validation(data, model, eps=0.2):
    model.eval()
    out = model(data)
    X = out["H"].cpu().detach().numpy()
    cluster = DBSCAN(eps=eps, min_samples=2).fit(X)

    data_labels = data.y.cpu().detach().numpy().flatten()
    uniq_data_labels = sorted(set(data_labels))

    perfect = []
    lhc = []
    dm = []

    for uniq_cl in uniq_data_labels:
        true_cluster_indices = np.where(data_labels == uniq_cl)[0]

        cluster_dbscan_labels = cluster.labels_[true_cluster_indices]
        
        if np.all(cluster_dbscan_labels == -1):
            continue

        non_noise_labels = cluster_dbscan_labels[cluster_dbscan_labels != -1]
        if len(non_noise_labels) == 0:
            continue  # Skip if all points are noise
        
        unique_labels, counts = np.unique(non_noise_labels, return_counts=True)
        dbscan_label = unique_labels[np.argmax(counts)]

        num_elements_true = len(true_cluster_indices)
        num_elements_pred = np.sum(cluster.labels_ == dbscan_label)
        num_elements_correct = np.sum(data_labels[cluster.labels_ == dbscan_label] == uniq_cl)
        num_elements_fake = num_elements_pred - num_elements_correct
        
        if num_elements_true > 0 and num_elements_pred > 0:
            perfect.append(num_elements_correct == num_elements_true)
            lhc.append(1 if num_elements_correct / num_elements_true > 0.75 else 0)
            dm.append(1 if (num_elements_correct / num_elements_true >= 0.5 and 
                           num_elements_fake / num_elements_pred < 0.5) else 0)

    return np.mean(perfect), np.mean(lhc), np.mean(dm)