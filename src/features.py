"""
Feature engineering module for volleyball possession and tactical attack analysis.
CMPE 257 - Machine Learning
"""

import numpy as np
try:
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def engineer_features(df):
    """
    Generate domain-specific volleyball features based on Van Haaren et al. (KDD 2016)
    and VREN notation specifications.
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        pd.DataFrame with engineered features added.
    """
    df_feat = df.copy()
    
    # 1. System state (In-system vs Out-of-system pass)
    if "pass_rating" in df_feat.columns:
        df_feat["is_in_system"] = (df_feat["pass_rating"].astype(float) == 3.0).astype(int)
        df_feat["is_out_of_system"] = (df_feat["pass_rating"].astype(float) == 1.0).astype(int)
    else:
        df_feat["is_in_system"] = 0
        df_feat["is_out_of_system"] = 0
        
    # 2. Attack tempo & quickness
    if "set_type" in df_feat.columns:
        df_feat["is_quick_attack"] = df_feat["set_type"].isin(["Quick", "Slide"]).astype(int)
        df_feat["is_high_ball"] = (df_feat["set_type"] == "High_Ball").astype(int)
    else:
        df_feat["is_quick_attack"] = 0
        df_feat["is_high_ball"] = 0
        
    # 3. Court geometry: Front-row attack (Zones 2, 3, 4) vs Back-row attack (Zones 1, 5, 6)
    if "attack_zone" in df_feat.columns:
        df_feat["is_back_row"] = df_feat["attack_zone"].isin(["Zone_1", "Zone_5", "Zone_6"]).astype(int)
    else:
        df_feat["is_back_row"] = 0
        
    # 4. Blocker differential & defensive pressure
    if "num_blockers" in df_feat.columns:
        df_feat["num_blockers"] = df_feat["num_blockers"].astype(float)
        df_feat["is_solo_or_no_block"] = (df_feat["num_blockers"] <= 1.0).astype(int)
        df_feat["is_triple_block"] = (df_feat["num_blockers"] >= 3.0).astype(int)
        # Net advantage: positive if fewer than 2 blockers, negative if more
        df_feat["block_advantage"] = 2.0 - df_feat["num_blockers"]
    else:
        df_feat["num_blockers"] = 2.0
        df_feat["is_solo_or_no_block"] = 0
        df_feat["is_triple_block"] = 0
        df_feat["block_advantage"] = 0.0
        
    # 5. Tempo x Blocker interaction: In-system quicks against weak block
    df_feat["tempo_blocker_edge"] = df_feat["is_quick_attack"] * df_feat["is_solo_or_no_block"]
    
    # 6. High ball against triple block: high risk situation
    df_feat["high_ball_triple_block"] = df_feat["is_high_ball"] * df_feat["is_triple_block"]
    
    # 7. Rally Phase: First attack (serve receive) vs Transition (counter-attack)
    if "round" in df_feat.columns:
        df_feat["is_transition_attack"] = (df_feat["round"].astype(int) > 1).astype(int)
    else:
        df_feat["is_transition_attack"] = 0
        
    return df_feat


def prepare_model_data(df, target_col="attack_success", group_col="match_id"):
    """
    Complete preprocessing pipeline transforming raw rally data into encoded numerical matrices.
    
    Returns:
        X_encoded: np.ndarray of encoded features
        y: np.ndarray target labels
        groups: np.ndarray match group identifiers
        feature_names: list of feature names corresponding to X columns
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for prepare_model_data.")
        
    # Feature engineering
    df_feat = engineer_features(df)
    
    # Separate target & groups
    y = df_feat[target_col].values.astype(int)
    groups = df_feat[group_col].values if group_col in df_feat.columns else np.zeros(len(df_feat))
    
    # Identify feature types
    drop_cols = [target_col, "rally_id", group_col, "team"]
    candidate_cols = [c for c in df_feat.columns if c not in drop_cols]
    
    cat_cols = [c for c in candidate_cols if df_feat[c].dtype == "object"]
    num_cols = [c for c in candidate_cols if c not in cat_cols]
    
    # Transformer pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )
    
    X_encoded = preprocessor.fit_transform(df_feat[candidate_cols])
    
    # Retrieve feature names
    cat_feature_names = []
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"]
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
        
    feature_names = num_cols + cat_feature_names
    return X_encoded, y, groups, feature_names
