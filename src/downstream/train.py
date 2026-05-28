import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data.gnn_dataset import smiles_to_graph
from downstream.eval import (
    eval_downstream_classification_model,
    eval_downstream_regression_model,
)
from model.multi_target_gnn import MultiTargetGINE


def parse_csv(
    csv_path: str,
    smiles_col="canonical_smiles",
    y_col="pchembl_value",
):
    df = pd.read_csv(csv_path)

    smiles = df[smiles_col].to_numpy()
    y = df[y_col].to_numpy()

    return smiles, y


def smiles_to_ecfp(smiles, radius=2, n_bits=2048):
    fp_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)

        assert mol is not None, "Smiles should represent valid molecule"

        fp = fp_gen.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)

    return np.array(fps)


def smiles_to_embeddings(
    smiles: list[str],
    target_embedding: torch.Tensor,
    gnn: MultiTargetGINE,
    batch_size=64,
):
    dataset = [smiles_to_graph(s) for s in smiles]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Precomputing compound embeddings"):
            batch = batch.to(target_embedding.device)
            emb = gnn.encode_graph(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                target_esm_embeddings=target_embedding.unsqueeze(0).repeat(
                    batch.batch.shape[0], 1
                ),
            )

            all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings).numpy()


def tune_hyperparams(
    model,
    csv_train: str,
    csv_val: str,
    param_grid,
    task="regression",
    threshold=None,
    gnn: MultiTargetGINE = None,
    target_embedding: torch.Tensor = None,
    gnn_batch_size=None,
):
    smiles_train, y_train = parse_csv(csv_train)
    smiles_val, y_val = parse_csv(csv_val)

    if task == "classification":
        if threshold is None:
            raise ValueError("Threshold must be provided for classification.")

        y_train = (y_train >= threshold).astype(int)
        y_val = (y_val >= threshold).astype(int)

    if gnn is None:
        X_train = smiles_to_ecfp(smiles_train)
        X_val = smiles_to_ecfp(smiles_val)
    else:
        X_train = smiles_to_embeddings(
            smiles_train,
            target_embedding=target_embedding,
            gnn=gnn,
            batch_size=gnn_batch_size,
        )
        X_val = smiles_to_embeddings(
            smiles_val,
            target_embedding=target_embedding,
            gnn=gnn,
            batch_size=gnn_batch_size,
        )

    X = np.vstack([X_train, X_val])
    y = np.concatenate([y_train, y_val])
    test_fold = np.concatenate([np.full(len(X_train), -1), np.zeros(len(X_val))])

    ps = PredefinedSplit(test_fold)

    scoring = "neg_root_mean_squared_error" if task == "regression" else "f1"

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=ps,
        verbose=2,
        n_jobs=-1,
    )

    grid.fit(X, y)

    best_params = grid.best_params_

    best_model = clone(model)
    best_model.set_params(**best_params)
    best_model.fit(X_train, y_train)

    eval_fn = eval_downstream_regression_model
    if task == "classification":
        eval_fn = eval_downstream_classification_model

    metrics = eval_fn(best_model, X_val, y_val)

    print("\n==== VALIDATION METRICS ====")
    for name, value in metrics.items():
        print(f"{name}:\t{value}")

    print("\n==== BEST PARAMETERS ====")
    print(best_params)

    return best_model, best_params
