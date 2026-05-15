import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from model.multi_target_gnn import MultiTargetGINE
from data.gnn_dataset import GNNDataset


@torch.no_grad()
def eval_upstream_gnn(
    gnn: MultiTargetGINE, dataset: GNNDataset, batch_size: int, device: str = "cuda"
) -> dict[str, float]:

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    all_preds = []
    all_targets = []

    running_loss = 0.0
    total_samples = 0

    eval_pbar = tqdm(dataloader, desc="Evaluating", leave=False)

    for data in eval_pbar:
        target_embeddings = dataset.target_embeddings[data.target_id]

        preds = gnn(
            x=data.x.to(device),
            edge_index=data.edge_index.to(device),
            edge_attr=data.edge_attr.to(device),
            batch=data.batch.to(device),
            target_esm_embeddings=target_embeddings,
        )[:, 0]

        targets = data.y.to(device)

        loss = F.mse_loss(preds, targets, reduction="sum")

        running_loss += loss.item()
        total_samples += targets.shape[0]

        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    metrics = {
        "loss": running_loss / total_samples,
        "mse": mse,
        "rmse": rmse.item(),
        "mae": mae,
        "r2": r2,
    }

    return metrics
