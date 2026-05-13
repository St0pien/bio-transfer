import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    EsmModel,
)

from model.esm_target_embedder import ESMTargetEmbedder

from .config import ATOM_TYPES
from .downloading import fetch_protein_seqeuence, fetch_uniprot_from_chembl


class GNNData(Data):
    target_embedding: torch.Tensor
    target_id: str


def atom_features(atom, atom_types=ATOM_TYPES):
    atom_type = atom.GetSymbol()

    one_hot = [0] * (len(atom_types) + 1)

    if atom_type in atom_types:
        one_hot[atom_types.index(atom_type)] = 1
    else:
        one_hot[-1] = 1  # Unknown atom token

    return torch.tensor(
        one_hot
        + [
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
        ],
        dtype=torch.float,
    )


def bond_features(bond):

    bt = bond.GetBondType()

    return torch.tensor(
        [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
        ],
        dtype=torch.float,
    )


def smiles_to_graph(smiles: str, y: Optional[float] = None, atom_types=ATOM_TYPES):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = torch.stack([atom_features(atom, atom_types) for atom in mol.GetAtoms()])

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bf = bond_features(bond)

        edge_index.extend(
            [
                [i, j],
                [j, i],
            ]
        )

        edge_attr.extend(
            [
                bf,
                bf,
            ]
        )

    edge_index = torch.tensor(edge_index).t().contiguous()

    edge_attr = torch.stack(edge_attr)

    return GNNData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


class GNNDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        target_embedding_dict: dict[str, torch.Tensor],
        smiles_col="canonical_smiles",
        target_id_col="target_chembl_id",
        y_col="pchembl_value",
    ):
        self.df = df
        self.target_embedding_dict = target_embedding_dict
        self.smiles_col = smiles_col
        self.target_id_col = target_id_col
        self.y_col = y_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        smiles = row[self.smiles_col]
        target_id = row[self.target_id_col]
        y = float(row[self.y_col])

        graph = smiles_to_graph(smiles, y)
        graph.target_embedding = self.target_embedding_dict[target_id]
        graph.target_id = target_id

        return graph

    def from_csv(
        csv_path: str,
        embedder: ESMTargetEmbedder,
        target_dict_json: Optional[str] = None,
        smiles_col="canonical_smiles",
        target_id_col="target_chembl_id",
        y_col="pchembl_value",
    ) -> "GNNDataset":
        print(f"[+] Initializing dataset from: {csv_path}")
        df = pd.read_csv(csv_path)

        unique_targets = df[target_id_col].unique()
        print(
            f"[+] Dataset size: {len(df)}, detected {len(unique_targets)} unique targets"
        )

        if target_dict_json is not None:
            with open(target_dict_json) as f:
                target_sequences = json.load(f)
        else:
            target_sequences = {}
            for chembl_id in tqdm(
                unique_targets, desc="Downloading target protein sequences"
            ):
                target_sequences[chembl_id] = fetch_protein_seqeuence(
                    fetch_uniprot_from_chembl(chembl_id)
                )

        print("[+] Precomputing target embeddings")
        target_embmeddings = embedder.get_target_embeddings(target_sequences)

        return GNNDataset(df, target_embmeddings, smiles_col, target_id_col, y_col)
