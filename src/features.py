"""
Feature engineering module for volleyball possession and tactical attack analysis.
CMPE 257 - Machine Learning
Includes domain-specific features and polynomial tactical interaction terms (Step 3).
"""

import numpy as np

try:
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_selection import VarianceThreshold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def engineer_features(df, include_interactions: bool = True):
    """
    Generate domain-specific volleyball features based on Van Haaren et al. (KDD 2016)
    and VREN notation specifications, with optional polynomial tactical interactions (Step 3).
    
    Args:
        df: Input pandas DataFrame
        include_interactions: Whether to add 2nd-degree tactical interaction cross-products
        
    Returns:
        pd.DataFrame with engineered features added.
    """
    df_feat = df.copy()
    
    # 1. System state (In-system vs Out-of-system pass)
    if "pass_rating" in df_feat.columns:
        df_feat["pass_rating"] = df_feat["pass_rating"].astype(float)
        df_feat["is_in_system"] = (df_feat["pass_rating"] == 3.0).astype(int)
        df_feat["is_out_of_system"] = (df_feat["pass_rating"] == 1.0).astype(int)
    else:
        df_feat["pass_rating"] = 2.0
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
        df_feat["round"] = df_feat["round"].astype(float)
        df_feat["is_transition_attack"] = (df_feat["round"] > 1.0).astype(int)
    else:
        df_feat["round"] = 1.0
        df_feat["is_transition_attack"] = 0

    # 8. Step 3: Polynomial Interaction Features (Textbook ML Technique)
    # Generates cross-product interaction terms that reflect non-linear volleyball physics
    if include_interactions:
        # Pass quality x Blocker pressure
        df_feat["interact_pass_blockers"] = df_feat["pass_rating"] * (3.0 - df_feat["num_blockers"])
        # Pass quality x Tempo
        df_feat["interact_pass_tempo"] = df_feat["pass_rating"] * df_feat["is_quick_attack"]
        # Transition fatigue x Blocker count
        df_feat["interact_round_blockers"] = df_feat["round"] * df_feat["num_blockers"]
        # Block advantage x In-system pass
        df_feat["interact_advantage_insystem"] = df_feat["block_advantage"] * df_feat["is_in_system"]
        # Out of system breakdown index
        df_feat["interact_outofsystem_highball"] = df_feat["is_out_of_system"] * df_feat["is_high_ball"]
        # Back-row pipe attack vs formed block
        df_feat["interact_backrow_blockers"] = df_feat["is_back_row"] * df_feat["num_blockers"]
        
    return df_feat


def prepare_model_data(df, target_col="attack_success", group_col="match_id", include_interactions: bool = True):
    """
    Complete preprocessing pipeline transforming raw rally data into encoded numerical matrices.
    
    Args:
        df: Input pandas DataFrame
        target_col: Target binary column name
        group_col: Group column name for GroupKFold
        include_interactions: Whether to enable Step 3 polynomial interaction features
        
    Returns:
        X_encoded: np.ndarray of encoded features
        y: np.ndarray target labels
        groups: np.ndarray match group identifiers
        feature_names: list of feature names corresponding to X columns
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for prepare_model_data.")
        
    # Feature engineering with interactions
    df_feat = engineer_features(df, include_interactions=include_interactions)
    
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
