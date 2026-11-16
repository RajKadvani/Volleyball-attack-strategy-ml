# Volleyball Rally & Attack Strategy Datasets

This directory contains the datasets and schemas utilized for analyzing and discovering successful volleyball attack strategies using Machine Learning.

## Dataset Overview

The dataset (`data/vren_dataset.csv`) contains structured rally possession logs based on the **VREN (Volleyball Rally Dataset with Expression Notation Language)** format:
- **Repository Reference**: [https://github.com/haotianxia/VREN](https://github.com/haotianxia/VREN)
- **Academic Paper**: Xia et al., *"VREN: Volleyball Rally Dataset with Expression Notation Language"*, arXiv:2209.13846 (2022).
- **Scope**: 1,632 rallies across 15 collegiate and national matches, capturing sequence interactions from reception pass to attack execution against opposing blockers.

## Schema & Feature Dictionary

| Feature Name | Type | Description | Values / Range |
|---|---|---|---|
| `match_id` | Categorical / ID | Unique identifier for the match | `M01`, `M02`, ... |
| `rally_id` | Integer | Sequential rally number within match | `1, 2, ...` |
| `round` | Integer | Round sequence within the rally (1 = serve receive attack, 2+ = transition) | `1, 2, 3, ...` |
| `team` | Categorical | Attacking team identifier | `Team_A`, `Team_B` |
| `serve_type` | Categorical | Type of serve starting the rally | `Jump_Serve`, `Float_Serve`, `Topspin` |
| `pass_rating` | Ordinal | Reception / pass quality | `1` (Poor / Out-of-system), `2` (Medium), `3` (Perfect / In-system) |
| `set_type` | Categorical | Trajectory and tempo of set | `Quick`, `Outside`, `Slide`, `Pipe`, `High_Ball`, `Opposite` |
| `set_zone` | Categorical | Location along the net where set was delivered | `Zone_2`, `Zone_3`, `Zone_4` |
| `attack_type` | Categorical | Spiker offensive stroke execution | `Hard_Spike`, `Tip`, `Roll_Shot`, `Off_Speed`, `Tool_Block` |
| `attack_zone` | Categorical | Court zone from which the attack was launched | `Zone_1` to `Zone_6` |
| `num_blockers` | Integer | Number of opposing blockers contesting the attack | `0`, `1`, `2`, `3` |
| `block_touch` | Binary | Whether defensive block touched the ball | `0` (No), `1` (Yes) |
| `attack_success` | Binary (Target) | Whether the attacking possession won the point | `1` (Success / Point won), `0` (Defended / Lost point) |

## Data Preprocessing Guidelines
1. **Target Leakage Prevention**: Features encoding the terminal outcome (such as referee whistle codes, winning error reasons) are strictly excluded from input features.
2. **Match-Level Grouping**: Cross-validation uses grouped splits (`GroupKFold` on `match_id`) to ensure no rally sequences from the same match appear in both training and test sets.
