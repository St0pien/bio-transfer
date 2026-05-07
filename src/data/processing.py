import random
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.SaltRemover import SaltRemover
from tqdm import tqdm

from .downloading import (
    fetch_data_from_chembl,
)


def get_largest_fragment(mol):
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    return max(frags, key=lambda m: m.GetNumAtoms())


remover = SaltRemover()


def clean_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Strip salts
    mol = remover.StripMol(mol, dontRemoveEverything=True)

    # Ensure we keep only the main fragment
    mol = get_largest_fragment(mol)
    if mol is None:
        return None

    # Final sanitization (optional but good practice)
    Chem.SanitizeMol(mol)

    # Canonical SMILES
    return Chem.MolToSmiles(mol, canonical=True)


def sanitize_bioactivity_data(
    df: pd.DataFrame, smiles_col="canonical_smiles", value_col="pchembl_value"
) -> pd.DataFrame:
    df = df.copy()

    df = df.dropna(subset=[smiles_col, value_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df[smiles_col] = df[smiles_col].apply(clean_smiles)
    df = df.dropna()

    return df.reset_index(drop=True)


def aggregate_bioactivity_duplicates(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    value_col: str = "pchembl_value",
) -> pd.DataFrame:
    df = df.copy()

    grouped = (
        df.groupby([smiles_col])[value_col]
        .agg(
            median_value="median",
            n_measurements="count",
        )
        .reset_index()
    )

    grouped = grouped.rename(columns={"median_value": value_col})

    return grouped


def generate_dataset(targets: dict[str, str], name: str, data_dir="data/sanitized"):
    pbar = tqdm(targets.items(), desc="Sanitizing dataset")

    upstream_parts = []
    for target_name, target_id in pbar:
        bioactivity_data = fetch_data_from_chembl(target_id)
        df = pd.DataFrame.from_records(bioactivity_data)
        sanitized_df = sanitize_bioactivity_data(df)
        deduplicated_df = aggregate_bioactivity_duplicates(sanitized_df)

        deduplicated_df["target_name"] = target_name
        deduplicated_df["target_chembl_id"] = target_id
        upstream_parts.append(deduplicated_df)

    save_path = Path(
        data_dir,
    )
    save_path.mkdir(exist_ok=True, parents=True)

    df_upstream_raw = pd.concat(upstream_parts, ignore_index=True)
    df_upstream_raw.to_csv(save_path / f"{name}.csv", index=False)


def compute_max_tanimoto(df1, df2, smiles_col="canonical_smiles"):
    gen = GetMorganGenerator(radius=2, fpSize=2048)

    fps2 = []
    for smi in df2[smiles_col]:
        fp = gen.GetFingerprint(Chem.MolFromSmiles(smi))
        if fp is not None:
            fps2.append(fp)

    fps2 = [fp for fp in fps2 if fp is not None]

    max_sims = []

    for smi in tqdm(df1[smiles_col], desc="Computing max similarity"):
        fp1 = gen.GetFingerprint(Chem.MolFromSmiles(smi))

        if fp1 is None or len(fps2) == 0:
            max_sims.append(None)
            continue

        sims = Chem.DataStructs.BulkTanimotoSimilarity(fp1, fps2)

        max_sims.append(max(sims))
    return max_sims


def filter_upstream_by_similarity_to_downstream(
    upstream: pd.DataFrame, downstream: pd.DataFrame, threshold=0.6
):
    max_sims = compute_max_tanimoto(upstream, downstream)
    mask = [sim < threshold for sim in max_sims]

    return upstream.copy()[mask]


def scaffold_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    seed: int = 42,
    smiles_col: str = "canonical_smiles",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    random.seed(seed)

    scaffold_map = defaultdict(list)

    # Build scaffold groups
    for idx, smi in tqdm(df[smiles_col].items(), total=len(df), desc="Scaffolding"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue  # skip invalid SMILES

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold is not None else ""

        scaffold_map[scaffold_smiles].append(idx)

    # Shuffle scaffolds
    scaffolds = list(scaffold_map.keys())
    random.shuffle(scaffolds)

    # Split scaffold groups
    n = len(scaffolds)
    train_cutoff = int(n * train_frac)
    valid_cutoff = int(n * (train_frac + valid_frac))

    train_scaffolds = scaffolds[:train_cutoff]
    valid_scaffolds = scaffolds[train_cutoff:valid_cutoff]
    test_scaffolds = scaffolds[valid_cutoff:]

    def collect(scaffold_list):
        idxs = [i for s in scaffold_list for i in scaffold_map[s]]
        return df.loc[idxs].reset_index(drop=True)

    train_df = collect(train_scaffolds)
    valid_df = collect(valid_scaffolds)
    test_df = collect(test_scaffolds)

    return train_df, valid_df, test_df
