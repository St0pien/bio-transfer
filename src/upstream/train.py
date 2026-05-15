from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import wandb
from torch.optim.adamw import AdamW
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

from data.gnn_dataset import GNNDataset
from model.esm_target_embedder import ESMTargetEmbedder
from model.multi_target_gnn import MultiTargetGINE

from .eval import eval_upstream_gnn


@dataclass
class GNNConfig:
    node_dim: int = 13
    edge_dim: int = 4
    esm_dim: int = 640
    hidden_dim: int = 256
    target_dim: int = 256
    num_layers: int = 8
    dropout: float = 0.1
    out_dim: int = 1


@dataclass
class DatasetConfig:
    csv_path: str = ""
    esm_embedding_model: str = "facebook/esm2_t30_150M_UR50D"
    target_sequences_json: Optional[str] = None
    smiles_col: str = "canonical_smiles"
    target_id_col: str = "target_chembl_id"
    y_col: str = "pchembl_value"


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps_frac = 0.05
    clip_norm: float = 1.0


@dataclass
class WandbConfig:
    entity: str = ""
    project: str = "bio-transfer"
    experiment_name: str = ""
    log_period: int = 10


@dataclass
class GNNTrainConfig:
    checkpoint_path: str = "checkpoints/upstream/gnn.pt"
    seed: int = 42
    epochs: int = 10
    batch_size: int = 256
    device: str = "cuda"

    gnn: GNNConfig = field(default_factory=GNNConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    validation_dataset: Optional[DatasetConfig] = field(default=None)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    wandb: Optional[WandbConfig] = field(default=None)


def train_upstream_gnn(config: GNNTrainConfig):
    with wandb.init(
        entity=config.wandb.entity if config.wandb is not None else "",
        project=config.wandb.project if config.wandb is not None else "",
        name=config.wandb.experiment_name if config.wandb is not None else "",
        config=asdict(config),
        mode="disabled" if config.wandb is None else None,
    ) as run:
        esm_embedder = ESMTargetEmbedder(
            config.dataset.esm_embedding_model, device=config.device
        )

        dataset = GNNDataset.from_csv(
            csv_path=config.dataset.csv_path,
            embedder=esm_embedder,
            target_dict_json=config.dataset.target_sequences_json,
            smiles_col=config.dataset.smiles_col,
            target_id_col=config.dataset.target_id_col,
            y_col=config.dataset.y_col,
        )

        gnn = MultiTargetGINE(**asdict(config.gnn)).to(config.device)
        gnn.set_target_embedding_mean(dataset.target_embeddings.mean(dim=0))

        if config.validation_dataset is not None:
            validation_dataset = GNNDataset.from_csv(
                csv_path=config.validation_dataset.csv_path,
                embedder=esm_embedder,
                target_dict_json=config.validation_dataset.target_sequences_json,
                smiles_col=config.validation_dataset.smiles_col,
                target_id_col=config.validation_dataset.target_id_col,
                y_col=config.validation_dataset.y_col,
            )

        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        optimizer = AdamW(
            gnn.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )

        total_steps = config.epochs * len(dataloader)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * config.optimizer.warmup_steps_frac),
            num_training_steps=total_steps,
        )

        training_pbar = tqdm(
            range(config.epochs),
            desc="Training upstream GNN",
            position=0,
        )

        for epoch in training_pbar:
            epoch_pbar = tqdm(
                dataloader, desc=f"Epoch {epoch + 1}", position=1, leave=False
            )
            gnn.train()

            for i, data in enumerate(epoch_pbar):
                optimizer.zero_grad()

                target_embeddings = dataset.target_embeddings[data.target_id]

                model_preds = gnn(
                    x=data.x.to(config.device),
                    edge_index=data.edge_index.to(config.device),
                    edge_attr=data.edge_attr.to(config.device),
                    batch=data.batch.to(config.device),
                    target_esm_embeddings=target_embeddings,
                )[:, 0]

                loss = F.mse_loss(model_preds, data.y.to(config.device))

                loss.backward()

                if config.optimizer.clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        gnn.parameters(), config.optimizer.clip_norm
                    )

                optimizer.step()
                lr_scheduler.step()

                if config.wandb is not None:
                    if (i + 1) % config.wandb.log_period == 0:
                        run.log(
                            {"lr": lr_scheduler.get_last_lr()[0], "loss": loss.item()}
                        )

                epoch_pbar.set_postfix({"loss": loss.item()})

            if config.validation_dataset is not None:
                gnn.eval()

                metrics = eval_upstream_gnn(
                    gnn,
                    validation_dataset,
                    batch_size=config.batch_size,
                    device=config.device,
                )

                run.log(metrics)

    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(gnn.state_dict(), checkpoint_path)

    print(f"Checkpoint saved to: {checkpoint_path}!")
