import argparse
from statistics import LinearRegression

import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from data.downloading import fetch_protein_seqeuence, fetch_uniprot_from_chembl
from downstream.eval import (
    eval_downstream_classification_model,
    eval_downstream_regression_model,
)
from downstream.grid_params import LINEAR, MLP, RND_FOREST, SVM, XGB
from downstream.train import (
    parse_csv,
    smiles_to_ecfp,
    smiles_to_embeddings,
    tune_hyperparams,
)
from model.esm_target_embedder import ESMTargetEmbedder
from model.multi_target_gnn import MultiTargetGINE


def get_model_and_grid(model_name: str, task: str):
    model_name = model_name.lower()
    task = task.lower()

    if task == "regression":
        if model_name == "xgboost":
            model = XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
            )
            param_grid = XGB
        elif model_name == "randomforest":
            model = RandomForestRegressor(random_state=42)
            param_grid = RND_FOREST
        elif model_name == "svm":
            model = SVR()
            param_grid = SVM
        elif model_name == "linear":
            model = LinearRegression()
            param_grid = LINEAR
        elif model_name == "mlp":
            model = MLPRegressor(
                random_state=42,
                max_iter=500,
            )
            param_grid = MLP
        else:
            raise ValueError(f"Unknown model: {model_name}")

    elif task == "classification":
        if model_name == "xgboost":
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            )
            param_grid = XGB
        elif model_name == "randomforest":
            model = RandomForestClassifier(random_state=42)
            param_grid = RND_FOREST
        elif model_name == "svm":
            model = SVC(probability=True)
            param_grid = SVM
        elif model_name == "linear":
            model = LogisticRegression(max_iter=1000)
            param_grid = LINEAR
        elif model_name == "mlp":
            model = MLPClassifier(
                random_state=42,
                max_iter=500,
            )
            param_grid = MLP
        else:
            raise ValueError(f"Unknown model: {model_name}")

    else:
        raise ValueError(f"Unknown task: {task}")

    return model, param_grid


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "xgboost",
            "randomforest",
            "svm",
            "linear",
            "mlp",
        ],
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["regression", "classification"],
    )

    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--val-csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--test-csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for converting regression labels to binary.",
    )

    parser.add_argument(
        "--gnn",
        default=None,
        help="Path to GNN encoder checkpoint for transfer learning",
    )

    parser.add_argument(
        "-gnn-batch-size",
        type=int,
        default=64,
        help="Batch size for gnn encoding in transfer learning",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model, param_grid = get_model_and_grid(args.model, args.task)

    if args.gnn is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gnn = MultiTargetGINE.from_pretrained(args.gnn).to(device)
        gnn.eval()

        print(f"[+] Loaded GNN encoder from {args.gnn}")

        with open(args.train_csv) as f:
            f.readline()
            target_chembl_id = f.readline().split(",")[-1].strip()

        print(f"[+] Fetching protein sequence for {target_chembl_id}")
        target_sequence = fetch_protein_seqeuence(
            fetch_uniprot_from_chembl(target_chembl_id)
        )

        print("[+] Preparing target embedding")
        esm_embedder = ESMTargetEmbedder(device=device)
        target_embedding = esm_embedder.get_target_embeddings(
            {target_chembl_id: target_sequence}
        )[target_chembl_id]

    best_model, _ = tune_hyperparams(
        model=model,
        csv_train=args.train_csv,
        csv_val=args.val_csv,
        param_grid=param_grid,
        task=args.task,
        threshold=args.threshold,
        gnn=gnn if args.gnn is not None else None,
        target_embedding=target_embedding if args.gnn is not None else None,
        gnn_batch_size=args.gnn_batch_size,
    )

    smiles_test, y_test = parse_csv(args.test_csv)

    if args.task == "classification":
        y_test = (y_test >= args.threshold).astype(int)

    if args.gnn is None:
        X_test = smiles_to_ecfp(smiles_test)
    else:
        X_test = smiles_to_embeddings(
            smiles_test,
            target_embedding=target_embedding,
            gnn=gnn,
            batch_size=args.gnn_batch_size,
        )

    if args.task == "regression":
        metrics = eval_downstream_regression_model(
            best_model,
            X_test,
            y_test,
        )
    else:
        metrics = eval_downstream_classification_model(
            best_model,
            X_test,
            y_test,
        )

    print("\n==== TEST RESULTS ====")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
