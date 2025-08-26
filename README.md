# Predicting House Prices

**Authors:** Sumana Chilakamarri, Yash Gupta, & Ayesha Uddin  
**Course:** Computational Data Analysis/Machine Learning
**Date:** April 23, 2025

---

## Problem Statement
Traditional home valuation methods often miss **non‑linear interactions** among features like school quality, walkability, renovations, and neighborhood effects. We apply **machine learning** to estimate residential property prices more **accurately and objectively**, using structured features & engineered components, and evaluate both **unsupervised** and **supervised** approaches.

---

## Data Source
Primary dataset: **Kaggle – Housing Prices** (tabular home attributes). We initially explored joining with an additional Kaggle dataset containing richer neighborhood attributes; due to heavy missingness after join, we proceeded with the second dataset alone.

Example features include: square footage, bedroom/bath counts, lot size, basement, AC, guest room, plus zoning/road access and within‑city location markers when available.

---

## Data Preprocessing
- Removed extreme sale price outliers using a **One‑Class SVM** (RBF γ=0.05 on `SalePrice`).
- Missing values: numerical → **median imputation**; categorical → **most frequent**.
- Standardized numeric features; **one‑hot encoded** categoricals.
- **PCA** retaining **95% variance** to reduce dimensionality and multicollinearity.
- Built two artifacts:  
  - `data/regression_data.csv` — PCA features + continuous `SalePrice` (for regression & clustering)  
  - `data/classification_data.csv` — PCA features + **binary** label (high/low vs. median) using **SMOTE** for balance.

---

## Methods

### Unsupervised: K‑Means Clustering
- Used the elbow method → **k = 3** clusters.
- Visualized clusters in 2D using the first two principal components.

### Supervised: Classification
- Benchmarked **Logistic Regression** and **SVM**; selected **Random Forest** for best performance & robustness.
- Train/test split **75/25** with hyperparameter tuning.
- **Results (classification):**
  - **Accuracy ~ 90.5%**
  - Precision (High‑price) **~ 92.9%**
  - Recall (High‑price) **~ 87.9%**
- Feature importance on PCA components showed **PC1** as dominant.

---

## Findings (from the report)
- **Cluster 0** (budget) ~ **$130.5k** average, **707** homes.
- **Cluster 1** (mid‑market) ~ **$217.1k** average, **314** homes.
- **Cluster 2** (premium) ~ **$228.0k** average, **302** homes.
- Confusion matrix: TN=124, TP=117, FP=9, FN=16.

---

## Repository
```
predicting-house-prices/
├─ notebooks/
│  └─ k_means.ipynb                 # PCA + elbow + k=3 clustering visuals
├─ scripts/
│  └─ random_forest.py              # trains & evaluates RF classifier
├─ data/
│  ├─ classification_data.csv       # PCA features + binary label
│  └─ regression_data.csv           # PCA features + continuous SalePrice
├─ Predicting_House_Prices.pdf      # final report (figures & discussion)
├─ requirements.txt
└─ README.md
```
---

## Running the files

### Setup
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1) Reproduce K‑Means (notebook)
```bash
jupyter notebook notebooks/k_means.ipynb
```
- Ensure `data/regression_data.csv` exists (PCA features + SalePrice).

### 2) Train / Evaluate Random Forest (script)
```bash
python scripts/random_forest.py     --train data/classification_data.csv     --test  data/classification_data.csv
```
Suggested script behavior (already supported by many templates):
- Reads a CSV of PCA features + binary label.
- Splits or uses provided train/test paths.
- Prints **accuracy, precision, recall**, and saves a **confusion matrix** to `reports/` if implemented.

---

## Report
Full details, figures, etc.: **[Predicting_House_Prices.pdf](./Predicting_House_Prices.pdf)**

---

## Tech Stack
- **Python**, **pandas**, **numpy**, **scikit‑learn**, **imbalanced‑learn**, **xgboost** (optional), **matplotlib**, **seaborn**, **Jupyter**

---
