import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from downstream.eval import (
    eval_downstream_classification_model,
    eval_downstream_regression_model,
)


def parse_csv(csv_path: str, smiles_col="canonical_smiles", y_col="pchembl_value"):
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


def tune_hyperparams(
    model, csv_train: str, csv_val: str, param_grid, task="regression", threshold=None
):
    smiles_train, y_train = parse_csv(csv_train)
    smiles_val, y_val = parse_csv(csv_val)

    if task == "classification":
        if threshold is None:
            raise ValueError("Threshold must be provided for classification.")

        y_train = (y_train >= threshold).astype(int)
        y_val = (y_val >= threshold).astype(int)

    X_train = smiles_to_ecfp(smiles_train)
    X_val = smiles_to_ecfp(smiles_val)

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
        n_jobs=1,
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
