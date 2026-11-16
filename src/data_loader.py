"""
Data loading and preprocessing module for Volleyball Rally Datasets.
CMPE 257 - Machine Learning
"""

import os
from typing import Tuple, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_CSV_NAME = "vren_dataset.csv"


def get_default_data_path() -> str:
    """Return path to primary volleyball rally dataset."""
    return os.path.join(DEFAULT_DATA_DIR, DATA_CSV_NAME)


def load_rally_data(file_path: Optional[str] = None):
    """
    Load volleyball rally dataset from CSV.
    
    Args:
        file_path: Optional custom path to CSV. If None, automatically checks data/ directory.
        
    Returns:
        pd.DataFrame or list of dicts: Processed rally dataset.
    """
    if file_path is None:
        file_path = get_default_data_path()

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. "
            "Please run 'python scripts/generate_sample_data.py' or 'python scripts/download_vren.py'."
        )

    if not HAS_PANDAS:
        import csv
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        print(f"Loaded {len(data)} records via standard CSV reader (Pandas not installed).")
        return data

    df = pd.read_csv(file_path)
    
    # Standardize column naming
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    # Target column validation
    target_candidates = ["attack_success", "rally_won", "win", "target", "point_won"]
    found_target = None
    for cand in target_candidates:
        if cand in df.columns:
            found_target = cand
            break
            
    if found_target and found_target != "attack_success":
        df["attack_success"] = df[found_target].astype(int)
        
    # Clean string columns
    str_cols = df.select_dtypes(include=["object"]).columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()
        
    # Remove leakage columns if present
    leakage_cols = [
        "win_reason", "lose_reason", "last_touch_error", "referee_call",
        "rally_winner", "winning_team", "result_type"
    ]
    to_drop = [c for c in leakage_cols if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        
    print(f"Successfully loaded dataset from '{file_path}' with shape: {df.shape}")
    return df


def split_features_target(df, target_col: str = "attack_success", group_col: str = "match_id") -> Tuple:
    """
    Split dataframe into features (X), target (y), and grouping labels (groups).
    """
    if not HAS_PANDAS or not isinstance(df, pd.DataFrame):
        raise ValueError("split_features_target requires pandas DataFrame input.")
        
    y = df[target_col].values
    groups = df[group_col].values if group_col in df.columns else None
    
    drop_cols = [target_col]
    if "rally_id" in df.columns:
        drop_cols.append("rally_id")
        
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y, groups
