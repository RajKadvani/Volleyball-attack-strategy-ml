"""
Machine learning models, hyperparameter tuning, and ensemble architectures.
CMPE 257 - Machine Learning
Implements:
- Step 1: Soft-Voting & Stacking Ensembles
- Step 2: Hyperparameter-Tuned Models (XGBoost, Random Forest, Logistic Regression)
"""

import numpy as np
from typing import Dict, Any, Tuple

try:
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        VotingClassifier,
        StackingClassifier
    )
    from sklearn.model_selection import GroupKFold, RandomizedSearchCV
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
    Instantiate the 4 core baseline models:
    1. Logistic Regression (Linear baseline)
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


def get_tuned_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Step 2: Models with optimized hyperparameters discovered via cross-validated tuning.
    Includes regularization, deeper trees with minimum leaf constraints, and shrinkage.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required to instantiate tuned models.")
        
    tuned = {
        "Tuned Logistic Regression": LogisticRegression(
            C=0.5,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ),
        "Tuned Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=16,
            min_samples_leaf=12,
            class_weight="balanced",
            random_state=random_state
        ),
        "Tuned Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
    }
    
    if HAS_XGB:
        tuned["Tuned XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.5,
            min_child_weight=2,
            eval_metric="logloss",
            random_state=random_state
        )
    else:
        tuned["Tuned Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            min_samples_leaf=4,
            random_state=random_state
        )
        
    return tuned


def get_ensemble_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Step 1: Advanced Stacking and Soft-Voting Ensembles (Textbook ML Technique).
    Combines predictions from diverse model families (Linear, Bagging, Boosting).
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for ensemble models.")
        
    # Base estimators for ensembles
    lr = LogisticRegression(C=0.5, max_iter=1000, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=150, max_depth=7, min_samples_leaf=5, random_state=random_state, n_jobs=-1)
    
    if HAS_XGB:
        gb = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.06, subsample=0.85, random_state=random_state, eval_metric="logloss")
    else:
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.06, subsample=0.85, random_state=random_state)

    ensembles = {
        # Weighted Soft-Voting Classifier
        "Soft Voting Classifier": VotingClassifier(
            estimators=[("xgb", gb), ("rf", rf), ("lr", lr)],
            voting="soft",
            weights=[0.45, 0.40, 0.15]
        ),
        # Stacking Classifier with Logistic Regression Meta-Learner
        "Stacking Classifier": StackingClassifier(
            estimators=[("xgb", gb), ("rf", rf), ("lr", lr)],
            final_estimator=LogisticRegression(C=1.0, max_iter=500, random_state=random_state),
            cv=5,
            n_jobs=-1
        )
    }
    return ensembles


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
