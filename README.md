# World Cup 2026 Predictor ⚽

A machine learning project that predicts the outcomes of FIFA World Cup 2026 matches — and simulates the tournament bracket to crown a predicted winner — from player performance data.

## Overview

The dataset contains **per-player, per-match** performance statistics (~54,600 rows, 75 features). To predict match outcomes, individual player rows are aggregated up to the **team-vs-team match level**, then a classifier is trained to predict the result.

> **Note:** The 2026 tournament has not been played, so this dataset is **simulated/synthetic**. This project models that simulated data as a machine learning exercise — it is not a real-world forecast.

## Dataset

[FIFA World Cup 2026 Player Performance Dataset](https://www.kaggle.com/datasets/rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset) (Kaggle)

Downloaded automatically via KaggleHub — see the notebook. The raw CSV is not committed to this repo (see `.gitignore`).

## Approach

1. **Load** the data via KaggleHub
2. **Aggregate** player rows → one row per match (team-level features)
3. **EDA** — explore distributions, class balance, correlations
4. **Preprocess** — scale numeric features, encode categoricals (`ColumnTransformer` + `Pipeline`)
5. **Model** — Logistic Regression, Random Forest, Gradient Boosting
6. **Evaluate** — cross-validation, F1 / ROC-AUC, hyperparameter tuning (`GridSearchCV`)
7. **Interpret** — SHAP to see which stats drive winning
8. **Simulate** the bracket to predict the champion

## Tech

Python · pandas · scikit-learn · matplotlib / seaborn · SHAP · Jupyter

## Running it

```bash
pip install -r requirements.txt
jupyter lab
```

Open `world_cup_predictor.ipynb` and run the cells top to bottom.
