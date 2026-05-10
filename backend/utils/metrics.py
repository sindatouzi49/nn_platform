"""Classification and regression metrics."""
import numpy as np


def classification_report_dict(y_true, y_pred_classes, n_classes=2):
    y_t = y_true.ravel() if y_true.ndim > 1 else y_true
    acc = float(np.mean(y_t == y_pred_classes))
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_t, y_pred_classes):
        cm[int(t)][int(p)] += 1

    per_class = {}
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec   = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1     = 2 * prec * recall / (prec + recall + 1e-9)
        per_class[str(c)] = {
            "precision": float(prec),
            "recall":    float(recall),
            "f1":        float(f1),
            "support":   int(cm[c].sum()),
        }

    return {
        "accuracy": acc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def regression_metrics(y_true, y_pred):
    mse  = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2   = float(1 - ss_res / (ss_tot + 1e-9))
    return {"mse": mse, "rmse": rmse, "r2": r2}
