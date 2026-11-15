"""
Evaluation metrics, feature ranking, and reporting module.
CMPE 257 - Machine Learning
"""

import numpy as np
from typing import Dict, List, Tuple

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """Calculate core classification metrics."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["ROC-AUC"] = 0.5
    return metrics


def print_evaluation_summary(results: Dict[str, Dict[str, Tuple[float, float]]]):
    """Pretty-print cross-validation summary table."""
    print("=" * 80)
    print(f"{'Model':<22} | {'Accuracy':<11} | {'Precision':<11} | {'Recall':<11} | {'F1-Score':<11} | {'ROC-AUC':<11}")
    print("-" * 80)
    for model_name, metrics in results.items():
        acc_str = f"{metrics['Accuracy'][0]:.3f}±{metrics['Accuracy'][1]:.2f}"
        prec_str = f"{metrics['Precision'][0]:.3f}±{metrics['Precision'][1]:.2f}"
        rec_str = f"{metrics['Recall'][0]:.3f}±{metrics['Recall'][1]:.2f}"
        f1_str = f"{metrics['F1-Score'][0]:.3f}±{metrics['F1-Score'][1]:.2f}"
        auc_str = f"{metrics['ROC-AUC'][0]:.3f}±{metrics['ROC-AUC'][1]:.2f}"
        print(f"{model_name:<22} | {acc_str:<11} | {prec_str:<11} | {rec_str:<11} | {f1_str:<11} | {auc_str:<11}")
    print("=" * 80)


def get_feature_importances(model, feature_names: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
    """Extract top positive and negative features or tree Gini importances."""
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        sorted_pairs = sorted(zip(feature_names, imps), key=lambda x: x[1], reverse=True)
        return sorted_pairs[:top_n]
    elif hasattr(model, "coef_"):
        coefs = model.coef_[0]
        sorted_pairs = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
        return sorted_pairs[:top_n]
    return []
