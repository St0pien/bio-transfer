from collections import defaultdict

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

from data.gnn_dataset import GNNDataset
from model.multi_target_gnn import MultiTargetGINE


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


@torch.no_grad()
def eval_upstream_gnn_per_target(
    gnn: MultiTargetGINE,
    dataset: GNNDataset,
    batch_size: int,
    device: str = "cuda",
) -> dict[int, dict[str, float]]:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Store predictions/targets grouped by target_id
    per_target_preds = defaultdict(list)
    per_target_targets = defaultdict(list)

    eval_pbar = tqdm(dataloader, desc="Evaluating", leave=False)

    for data in eval_pbar:
        target_ids = data.target_id

        target_embeddings = dataset.target_embeddings[target_ids]

        preds = gnn(
            x=data.x.to(device),
            edge_index=data.edge_index.to(device),
            edge_attr=data.edge_attr.to(device),
            batch=data.batch.to(device),
            target_esm_embeddings=target_embeddings,
        )[:, 0]

        targets = data.y.to(device)

        preds_cpu = preds.detach().cpu()
        targets_cpu = targets.detach().cpu()
        target_ids_cpu = target_ids.detach().cpu()

        # Group by target_id
        for tid, pred, target in zip(
            target_ids_cpu,
            preds_cpu,
            targets_cpu,
        ):
            tid = int(tid.item())

            per_target_preds[tid].append(pred.item())
            per_target_targets[tid].append(target.item())

    # Compute metrics per target
    per_target_metrics = {}

    for tid in per_target_preds:
        preds = np.array(per_target_preds[tid])
        targets = np.array(per_target_targets[tid])

        mse = mean_squared_error(targets, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, preds)

        # R² is undefined for <2 samples
        if len(targets) > 1:
            r2 = r2_score(targets, preds)
        else:
            r2 = float("nan")

        per_target_metrics[tid] = {
            "num_samples": len(targets),
            "loss": mse,  # equivalent to mean MSE loss
            "mse": mse,
            "rmse": rmse.item(),
            "mae": mae,
            "r2": r2,
        }

    return per_target_metrics
