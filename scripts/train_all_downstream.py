import argparse
import contextlib
import io
import sys
from pathlib import Path

import pandas as pd
import torch

# Ensure src/ is importable when running scripts from the project root.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
)
from sklearn.neural_network import (
    MLPClassifier,
    MLPRegressor,
)
from sklearn.svm import (
    SVC,
    SVR,
)
from xgboost import (
    XGBClassifier,
    XGBRegressor,
)

from data.downloading import (
    fetch_protein_seqeuence,
    fetch_uniprot_from_chembl,
)

from downstream.grid_params import (
    XGB,
    RND_FOREST,
    SVM,
    LINEAR,
    MLP,
)

from downstream.eval import (
    eval_downstream_classification_model,
    eval_downstream_regression_model,
)

from downstream.train import (
    parse_csv,
    smiles_to_ecfp,
    smiles_to_embeddings,
    tune_hyperparams,
)

from model.esm_target_embedder import ESMTargetEmbedder
from model.multi_target_gnn import MultiTargetGINE

# SEEDS = [42, 123, 2137]
SEEDS = [42]

SUBSETS = [
    "0.089799", # 500
    "0.01796", # 100
    "0.00898", # 50
    "0.003592", # 20
    "0.001796", # 10
]

TARGET = "BACE1"

def get_all_models(task):
    if task == "classification":
        return {
            "xgboost": (
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                ),
                XGB,
            ),
            "randomforest": (
                RandomForestClassifier(random_state=42),
                RND_FOREST,
            ),
            "svm": (
                SVC(probability=True),
                SVM,
            ),
            "linear": (
                LogisticRegression(max_iter=1000),
                LINEAR,
            ),
            "mlp": (
                MLPClassifier(
                    random_state=42,
                    max_iter=500,
                ),
                MLP,
            ),
        }

    elif task == "regression":
        return {
            "xgboost": (
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                ),
                XGB,
            ),
            "randomforest": (
                RandomForestRegressor(random_state=42),
                RND_FOREST,
            ),
            "svm": (
                SVR(),
                SVM,
            ),
            "linear": (
                LinearRegression(),
                LINEAR,
            ),
            "mlp": (
                MLPRegressor(
                    random_state=42,
                    max_iter=500,
                ),
                MLP,
            ),
        }

    else:
        raise ValueError(f"Unknown task: {task}")


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "regression"],
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Required for classification.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
    )

    parser.add_argument(
        "--gnn",
        default=None,
        help="Path to pretrained GNN checkpoint",
    )

    parser.add_argument(
        "--gnn-batch-size",
        type=int,
        default=64,
    )

    return parser.parse_args()


def prepare_transfer_learning(args):

    if args.gnn is None:
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[+] Using device: {device}")

    gnn = MultiTargetGINE.from_pretrained(args.gnn).to(device)
    gnn.eval()

    print(f"[+] Loaded GNN encoder from {args.gnn}")

    target_chembl_id = "CHEMBL4822"

    print(f"[+] Fetching protein sequence for {target_chembl_id}")

    target_sequence = fetch_protein_seqeuence(
        fetch_uniprot_from_chembl(target_chembl_id)
    )

    print("[+] Preparing target embedding")

    esm_embedder = ESMTargetEmbedder(device=device)

    target_embedding = esm_embedder.get_target_embeddings(
        {
            target_chembl_id: target_sequence
        }
    )[target_chembl_id]

    return gnn, target_embedding


def prepare_test_features(
    smiles_test,
    args,
    gnn,
    target_embedding,
):

    if args.gnn is None:

        return smiles_to_ecfp(smiles_test)

    return smiles_to_embeddings(
        smiles_test,
        target_embedding=target_embedding,
        gnn=gnn,
        batch_size=args.gnn_batch_size,
    )


def evaluate_model(
    task,
    model,
    X_test,
    y_test,
):

    if task == "classification":

        return eval_downstream_classification_model(
            model,
            X_test,
            y_test,
        )

    return eval_downstream_regression_model(
        model,
        X_test,
        y_test,
    )


def get_selection_metric(task, metrics):

    if task == "classification":
        return metrics["ROC_AUC"]

    return metrics["R2"]


def main():

    args = parse_args()

    if args.task == "classification" and args.threshold is None:
        raise ValueError(
            "--threshold is required for classification"
        )

    models = get_all_models(args.task)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = output_dir / f"downstream_{args.task}_filtered_gnn" / "BACE1"
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    gnn, target_embedding = prepare_transfer_learning(args)

    for seed in SEEDS:

        print("\n" + "#" * 100)
        print(f"RUNNING SEED: {seed}")
        print("#" * 100)

        base_dir = Path("data/splits/downstream") / str(seed)

        val_csv = base_dir / f"{TARGET}_val.csv"
        test_csv = base_dir / f"{TARGET}_test.csv"

        smiles_test, y_test = parse_csv(test_csv)

        if args.task == "classification":
            y_test = (y_test >= args.threshold).astype(int)

        X_test = prepare_test_features(
            smiles_test=smiles_test,
            args=args,
            gnn=gnn,
            target_embedding=target_embedding,
        )

        for subset in SUBSETS:

            print("\n" + "=" * 100)
            print(f"RUNNING SUBSET: {subset}")
            print("=" * 100)

            train_csv = (
                base_dir
                / subset
                / f"{TARGET}_train.csv"
            )

            for model_name, (model, param_grid) in models.items():

                print("\n" + "-" * 80)
                print(
                    f"TRAINING {model_name} | "
                    f"seed={seed} | "
                    f"subset={subset}"
                )
                print("-" * 80)

                transfer_tag = (
                    "transfer"
                    if args.gnn is not None
                    else "ecfp"
                )

                log_filename = (
                    f"{args.task}_"
                    f"{model_name}_"
                    f"seed-{seed}_"
                    f"size-{subset}.txt"
                )

                log_path = logs_dir / log_filename

                buffer = io.StringIO()

                with contextlib.redirect_stdout(buffer):

                    best_model, best_config = tune_hyperparams(
                        model=model,
                        csv_train=train_csv,
                        csv_val=val_csv,
                        param_grid=param_grid,
                        task=args.task,
                        threshold=args.threshold,
                        gnn=gnn,
                        target_embedding=target_embedding,
                        gnn_batch_size=args.gnn_batch_size,
                    )

                    metrics = evaluate_model(
                        task=args.task,
                        model=best_model,
                        X_test=X_test,
                        y_test=y_test,
                    )

                    print("\n==== TEST RESULTS ====")

                    for k, v in metrics.items():
                        print(f"{k}: {v:.4f}")

                log_text = buffer.getvalue()

                with open(log_path, "w") as f:
                    f.write(log_text)

                print(log_text)

                if args.task == "classification":
                    row = {
                        "target": TARGET,
                        "model": model_name,
                        "seed": seed,
                        "train_size": subset,
                        "accuracy": metrics["Accuracy"],
                        "precision": metrics["Precision"],
                        "recall": metrics["Recall"],
                        "f1": metrics["F1"],
                        "roc_auc": metrics["ROC_AUC"],
                        "best_params": str(best_config),
                        "file": log_filename,
                    }
                else:
                    row = {
                        "target": TARGET,
                        "model": model_name,
                        "seed": seed,
                        "train_size": subset,
                        "rmse": metrics["RMSE"],
                        "mae": metrics["MAE"],
                        "r2": metrics["R2"],
                        "best_params": str(best_config),
                        "file": log_filename,
                    }
                summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    summary_filename = (
        f"summary_{args.task}_"
        f"{'transfer' if args.gnn is not None else 'ecfp'}.csv"
    )

    summary_path = output_dir / summary_filename

    summary_df.to_csv(summary_path, index=False)

    print("\n" + "#" * 100)
    print("FINISHED ALL EXPERIMENTS")
    print("#" * 100)

    print(f"\nSaved summary:")
    print(summary_path)


if __name__ == "__main__":
    main()