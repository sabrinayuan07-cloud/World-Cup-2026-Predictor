# World Cup 2026 Predictor

A machine learning project that predicts international football match outcomes (home win / away win / draw) and uses the final model to predict World Cup 2026 matchups.

## The data

The project uses a real dataset of international football results from 1872 to the present ([Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017), also on [GitHub](https://github.com/martj42/international_results)): about 50,000 matches, each with the two teams, the final score, the tournament, the city and whether it was played on neutral ground.

The raw data only has teams and scores, so the real work is feature engineering. For each match I compute each team's recent form (win rate, goals scored, goals conceded and points over its previous 10 games, using only matches played before that game) and take the difference between the two teams.

## Approach

Each row is one match, and the target is the result (home win / away win / draw).

- **Cleaning:** drop unplayed fixtures and duplicate rows, then build the target from the scores
- **Feature engineering:** rolling team form, joined back onto each match with a row-count check, plus a neutral-ground flag
- **Time-based split:** train on 1872–2015, test on 2016–2026, with `TimeSeriesSplit` cross-validation
- **EDA:** outcome balance, home advantage, the effect of neutral ground
- **Preprocessing:** a `ColumnTransformer` (impute and scale numeric features, one-hot encode teams)
- **Models:** `DummyClassifier` baseline, then logistic regression, decision tree, random forest and gradient boosting, tuned with `GridSearchCV`
- **Feature selection:** RFECV
- **Interpretation:** random forest feature importances and SHAP values for the final model
- **Predictions:** a helper that predicts any matchup from each team's current form

I evaluate with macro-averaged F1 as the main metric (plus accuracy), because the three outcomes are imbalanced and I want the model to handle draws and away wins, not just the common home win.

## Results

| Model | CV macro-F1 |
|---|---|
| Dummy (always home win) | 0.219 |
| **Logistic regression (final)** | **0.476** |
| Gradient boosting (tuned) | 0.420 |
| Random forest (tuned) | 0.401 |
| Decision tree (tuned) | 0.398 |

**Test set (2016–2026):** 0.505 macro-F1 and 54.7% accuracy, vs 0.216 and 47.8% for the baseline. Draws are the hardest outcome (F1 0.30).

## Tools

Python, pandas, scikit-learn, SHAP, matplotlib / seaborn, Jupyter.

## Running it

```bash
pip install -r requirements.txt
jupyter lab
```

Open `world_cup_predictor.ipynb` and run the cells top to bottom. The dataset downloads automatically through kagglehub, or from the dataset's GitHub repo if Kaggle isn't available, so it isn't stored in this repo.
