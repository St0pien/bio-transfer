import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GraphNorm, global_add_pool


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

        self.register_buffer("target_embedding_mean", torch.zeros(esm_dim))

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

    def set_target_embedding_mean(self, mean: torch.Tensor):
        self.target_embedding_mean.copy_(mean)

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

        mean_centered_temb = target_esm_embeddings - self.target_embedding_mean
        norm_mean_centered_temb = F.normalize(mean_centered_temb)
        target_embeddings = self.target_mlp(norm_mean_centered_temb)

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
