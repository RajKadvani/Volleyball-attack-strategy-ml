# Project Proposal Documentation

This folder preserves the original approved project proposal documents and academic literature for the CMPE 257 Machine Learning final project.

## Documents Included

1. **`Volleyball_Project_Proposal.pdf` / `Volleyball_Project_Proposal.docx`**:
   - Official project proposal submitted by **Raj Dharmendra Kadvani (Student ID: 018283798)** for CMPE 257 Machine Learning at San Jose State University.
2. **`KDD2016_Van_Haaren_Volleyball.pdf`**:
   - Academic reference paper: Jan Van Haaren, Horesh Ben Shitrit, Jesse Davis, and Pascal Fua. *"Analyzing Volleyball Match Data from the 2014 World Championships Using Machine Learning Techniques"*, Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2016). [https://doi.org/10.1145/2939672.2939725](https://doi.org/10.1145/2939672.2939725).

## Proposal Core Summary

- **Title**: Discovering Successful Volleyball Attack Strategies Using Machine Learning
- **Author**: Raj Dharmendra Kadvani
- **Course**: CMPE 257 Machine Learning, Fall 2026, San Jose State University
- **Problem Statement**: Discovering offensive tactical patterns and combinations of pass quality, setter choice, hitter attack type, and opponent blockers that correlate with winning rally outcomes.
- **Dataset**: VREN Volleyball Rally Dataset (haotianxia/VREN) containing 1,632 rallies and 12,112 action features from NCAA Division I and international games.
- **Methodology**: 
  - Phase 1: Data preparation, validation, and exploratory data analysis.
  - Phase 2: Domain feature engineering and baseline model evaluation (Logistic Regression, Decision Tree, Random Forest, XGBoost) using match-level grouped cross-validation.
  - Phase 3: Rule extraction for interpretable strategy discovery with support and validation precision constraints.
  - Deployment: Interactive Streamlit application (`app.py`).
