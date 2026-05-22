import argparse
from statistics import LinearRegression

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from downstream.grid_params import XGB, RND_FOREST, SVM, LINEAR, MLP

from downstream.eval import (
    eval_downstream_classification_model,
    eval_downstream_regression_model,
)
from downstream.train import parse_csv, smiles_to_ecfp, tune_hyperparams


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

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model, param_grid = get_model_and_grid(args.model, args.task)

    best_model, _ = tune_hyperparams(
        model=model,
        csv_train=args.train_csv,
        csv_val=args.val_csv,
        param_grid=param_grid,
        task=args.task,
        threshold=args.threshold,
    )

    smiles_test, y_test = parse_csv(args.test_csv)

    if args.task == "classification":
        y_test = (y_test >= args.threshold).astype(int)

    X_test = smiles_to_ecfp(smiles_test)

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
