"""
Machine learning models and match-level grouped cross-validation.
CMPE 257 - Machine Learning
"""

import numpy as np
from typing import Dict, Any

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Optional XGBoost import with graceful fallback
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def get_baseline_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiate the 4 core models defined in the CMPE 257 project proposal:
    1. Logistic Regression (Interpretable baseline)
    2. Decision Tree (Transparent white-box model for rule extraction)
    3. Random Forest (Bagging ensemble)
    4. Gradient Boosting / XGBoost (Boosting ensemble)
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required to instantiate baseline models.")
        
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
    }
    
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state
        )
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            random_state=random_state
        )
        
    return models


def train_and_evaluate_grouped_cv(models: Dict[str, Any], X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5):
    """
    Evaluate models using GroupKFold cross-validation grouped by match_id.
    This guarantees that rally sequences from the same match are never split across train and test folds.
    
    Returns:
        results: Dict containing metrics per model (mean and std across folds)
        oof_preds: Dict containing out-of-fold probability predictions per model
    """
    unique_groups = np.unique(groups)
    actual_splits = min(n_splits, len(unique_groups))
    gkf = GroupKFold(n_splits=actual_splits)
    
    results = {}
    oof_preds = {name: np.zeros(len(y)) for name in models.keys()}
    
    for name, model in models.items():
        fold_acc, fold_prec, fold_rec, fold_f1, fold_auc = [], [], [], [], []
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred.astype(float)
                
            oof_preds[name][test_idx] = y_prob
            
            # Record fold metrics
            fold_acc.append(accuracy_score(y_test, y_pred))
            fold_prec.append(precision_score(y_test, y_pred, zero_division=0))
            fold_rec.append(recall_score(y_test, y_pred, zero_division=0))
            fold_f1.append(f1_score(y_test, y_pred, zero_division=0))
            try:
                fold_auc.append(roc_auc_score(y_test, y_prob))
            except ValueError:
                fold_auc.append(0.5)
                
        results[name] = {
            "Accuracy": (np.mean(fold_acc), np.std(fold_acc)),
            "Precision": (np.mean(fold_prec), np.std(fold_prec)),
            "Recall": (np.mean(fold_rec), np.std(fold_rec)),
            "F1-Score": (np.mean(fold_f1), np.std(fold_f1)),
            "ROC-AUC": (np.mean(fold_auc), np.std(fold_auc)),
        }
        
    return results, oof_preds
