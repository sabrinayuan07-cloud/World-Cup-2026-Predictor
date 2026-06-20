# World Cup 2026 Predictor

A machine learning project that predicts international football match outcomes and simulates the World Cup 2026 bracket to pick a tournament winner.

## The data

The project uses a real dataset of international football results from 1872 to the present (from Kaggle) — around 45,000 actual matches, each with the two teams, the final score, the tournament, and the venue.

The raw data only has teams and scores, so the real work is feature engineering: for each match I compute each team's recent form (win rate, goals scored and conceded over their previous games) and compare the two teams. These engineered strength features are what let the model predict who wins a specific matchup.

## Approach

Each row is one match between two teams, and the target is the result (home win / away win / draw).

Workflow:
- Build the target from match scores and engineer team-form features
- Exploratory analysis — outcome balance, home advantage, goal distributions
- Preprocessing with a ColumnTransformer (impute and scale numeric features, one-hot encode teams)
- Baseline with DummyClassifier, then logistic regression, random forest, and gradient boosting
- Cross-validation and hyperparameter tuning with GridSearchCV
- SHAP to see which features drive the predictions
- Simulate the bracket to choose a predicted champion

I evaluate with accuracy and macro-averaged F1, since the three outcomes are imbalanced and I want the model to handle draws and away wins, not just the common home win.

## Tools

Python, pandas, scikit-learn, matplotlib / seaborn, SHAP, Jupyter.

## Running it

```bash
pip install -r requirements.txt
jupyter lab
```

Open `world_cup_predictor.ipynb` and run the cells top to bottom. The dataset downloads automatically through kagglehub, so it isn't stored in this repo.
