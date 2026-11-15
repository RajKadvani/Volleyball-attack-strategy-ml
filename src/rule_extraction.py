"""
Rule extraction and tactical pattern mining module.
CMPE 257 - Machine Learning
Implements Phase 3 strategy extraction based on Van Haaren et al. (KDD 2016).
"""

from typing import List, Dict, Any
import numpy as np

try:
    from sklearn.tree import _tree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def extract_rules_from_tree(decision_tree, feature_names: List[str], min_support: float = 0.03, min_precision: float = 0.60) -> List[Dict[str, Any]]:
    """
    Mine interpretable decision paths from a trained Decision Tree Classifier.
    
    Args:
        decision_tree: Fitted DecisionTreeClassifier
        feature_names: List of feature names
        min_support: Minimum fraction of dataset supporting the rule
        min_precision: Minimum class-1 precision (confidence) required
        
    Returns:
        List of rule dictionaries ranked by precision and support.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for rule extraction.")
        
    tree_ = decision_tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    total_samples = tree_.n_node_samples[0]
    extracted_rules = []
    
    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Left branch: <= threshold
            left_cond = f"{name} <= {threshold:.2f}"
            recurse(tree_.children_left[node], conditions + [left_cond])
            
            # Right branch: > threshold
            right_cond = f"{name} > {threshold:.2f}"
            recurse(tree_.children_right[node], conditions + [right_cond])
        else:
            # Leaf node reached
            counts = tree_.value[node][0]
            node_samples = tree_.n_node_samples[node]
            support = node_samples / total_samples
            
            # Binary class assumption: class 0, class 1
            if len(counts) > 1:
                p1 = counts[1] / np.sum(counts)
            else:
                p1 = 0.0
                
            if support >= min_support and p1 >= min_precision:
                extracted_rules.append({
                    "conditions": conditions,
                    "readable_rule": " AND ".join(conditions),
                    "precision": float(p1),
                    "support": float(support),
                    "samples": int(node_samples),
                    "pred_class": 1
                })
                
    recurse(0, [])
    # Sort rules by highest precision then highest support
    extracted_rules.sort(key=lambda r: (r["precision"], r["support"]), reverse=True)
    return extracted_rules


def format_tactical_recommendations(rules: List[Dict[str, Any]]) -> List[str]:
    """Convert mathematical tree predicates into actionable volleyball coaching insights."""
    recommendations = []
    for idx, rule in enumerate(rules, 1):
        conds = rule["conditions"]
        prec = rule["precision"] * 100
        supp = rule["support"] * 100
        rec = f"Strategy Rule #{idx}: IF ({' AND '.join(conds)}) -> Predicted Kill Rate: {prec:.1f}% (Occurrence: {supp:.1f}% of rallies)"
        recommendations.append(rec)
    return recommendations
