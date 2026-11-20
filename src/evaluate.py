"""
Evaluation metrics, feature ranking, threshold tuning, and reporting module.
CMPE 257 - Machine Learning
Implements Step 5: Optimal Decision Threshold Tuning (ROC / F1 optimization).
"""

import numpy as np
from typing import Dict, List, Tuple

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        roc_curve, precision_recall_curve
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


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, method: str = "youden") -> Tuple[float, float]:
    """
    Step 5: Find optimal classification probability threshold.
    
    Methods:
        - 'youden': Maximizes Youden's J statistic (Sensitivity + Specificity - 1) on ROC curve.
        - 'f1': Maximizes the F1-Score across thresholds between 0.30 and 0.70.
        
    Returns:
        optimal_threshold: float between 0.0 and 1.0
        best_score: corresponding metric score at optimal threshold
    """
    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = float(np.clip(thresholds[best_idx], 0.30, 0.70))
        return best_threshold, float(j_scores[best_idx])
    else:
        # F1-score sweep
        best_f1 = -1.0
        best_threshold = 0.50
        for th in np.linspace(0.30, 0.70, 41):
            preds = (y_prob >= th).astype(int)
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(th)
        return best_threshold, best_f1


def evaluate_with_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict[str, float]:
    """Calculate classification metrics evaluated at a specific probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob)
    metrics["Threshold"] = threshold
    return metrics


def print_evaluation_summary(results: Dict[str, Dict[str, Tuple[float, float]]]):
    """Pretty-print cross-validation summary table."""
    print("=" * 82)
    print(f"{'Model / Ensemble Architecture':<28} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<8}")
    print("-" * 82)
    for model_name, metrics in results.items():
        acc_str = f"{metrics['Accuracy'][0]:.3f}±{metrics['Accuracy'][1]:.2f}"
        prec_str = f"{metrics['Precision'][0]:.3f}±{metrics['Precision'][1]:.2f}"
        rec_str = f"{metrics['Recall'][0]:.3f}±{metrics['Recall'][1]:.2f}"
        f1_str = f"{metrics['F1-Score'][0]:.3f}±{metrics['F1-Score'][1]:.2f}"
        auc_str = f"{metrics['ROC-AUC'][0]:.3f}±{metrics['ROC-AUC'][1]:.2f}"
        print(f"{model_name:<28} | {acc_str:<10} | {prec_str:<10} | {rec_str:<10} | {f1_str:<10} | {auc_str:<8}")
    print("=" * 82)


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
