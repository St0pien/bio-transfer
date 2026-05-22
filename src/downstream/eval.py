from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    recall_score,
    roc_auc_score,
)
import numpy as np


def eval_downstream_regression_model(estimator, X, y):
    preds = estimator.predict(X)

    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)

    return {"R2": r2, "RMSE": rmse, "MAE": mae}


def eval_downstream_classification_model(estimator, X, y):
    preds = estimator.predict(X)

    metrics = {
        "Accuracy": accuracy_score(y, preds),
        "Precision": precision_score(y, preds),
        "Recall": recall_score(y, preds),
        "F1": f1_score(y, preds),
    }

    # Optional ROC-AUC if probabilities exist
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(X)[:, 1]
        metrics["ROC_AUC"] = roc_auc_score(y, probs)

    return metrics
