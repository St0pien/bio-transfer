import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GraphNorm
from torch_geometric.nn import global_add_pool

from data.gnn_dataset import GNNData


class FiLM(nn.Module):
    def __init__(self, hidden_dim, target_dim):
        super().__init__()

        self.gamma = nn.Linear(target_dim, hidden_dim)
        self.beta = nn.Linear(target_dim, hidden_dim)

        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(
        self,
        x: torch.Tensor,
        target_embedding: torch.Tensor,
        batch: torch.Tensor = None,
    ):
        gamma = self.gamma(target_embedding) + 1
        beta = self.beta(target_embedding)

        if batch is not None:
            gamma = gamma[batch]
            beta = beta[batch]

        x = gamma * x + beta

        return x


class GINEFiLMBlock(nn.Module):
    def __init__(self, hidden_dim, edge_dim, target_dim, dropout=0.1):
        super().__init__()

        nn_edge = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINEConv(nn_edge, edge_dim=edge_dim)
        self.norm = GraphNorm(hidden_dim)
        self.film = FiLM(hidden_dim, target_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        target_embeddings: torch.Tensor,
    ):
        h = self.conv(x, edge_index, edge_attr)
        h = self.norm(h, batch)
        h = self.film(h, target_embeddings, batch)
        h = F.silu(h)
        h = self.dropout(h)

        return x + h


class MultiTargetGINE(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        esm_dim=640,
        hidden_dim=256,
        target_dim=256,
        num_layers=8,
        dropout=0.1,
        out_dim=1,
    ):
        super().__init__()

        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        self.target_mlp = nn.Sequential(
            nn.Linear(esm_dim, target_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(target_dim, target_dim),
        )

        self.layers = nn.ModuleList(
            [
                GINEFiLMBlock(
                    hidden_dim=hidden_dim,
                    edge_dim=hidden_dim,
                    target_dim=target_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.readout = global_add_pool

        self.head1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )

        self.head_film1 = FiLM(hidden_dim, target_dim)

        self.head2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        target_esm_embeddings: torch.Tensor,
    ):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        target_embeddings = self.target_mlp(target_esm_embeddings)

        for layer in self.layers:
            x = layer(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
                target_embeddings=target_embeddings,
            )

        graph_emb = self.readout(x, batch)

        y = self.head1(graph_emb)
        y = self.head_film1(y, target_embeddings)
        y = self.head2(y)

        return F.relu(y)
