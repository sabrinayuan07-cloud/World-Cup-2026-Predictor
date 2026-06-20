# World Cup 2026 Predictor
A machine learning project that predicts FIFA World Cup 2026 match outcomes from player performance stats, then simulates the knockout bracket to pick a tournament winner.

## The data
The dataset comes from Kaggle and has per-player, per-match statistics — about 54,600 rows and 75 columns covering goals, expected goals (xG), passing, defensive actions, and physical metrics like distance covered and top speed.

Since the 2026 tournament hasn't actually been played, the data is simulated. I'm treating this as a modeling exercise to practice the full ML workflow end to end, not as a real forecast.

## Approach
Each row is one player in one match, but I want to predict the result of a match between two teams. So the main step is aggregating the player rows up to the team level for each match, then training a classifier on those team-vs-team features.

## Workflow:

Aggregate player stats to the match level
Exploratory analysis — distributions, class balance, feature correlations
Preprocessing with a ColumnTransformer (scale numeric features, one-hot encode categoricals)
Baseline with DummyClassifier, then logistic regression, random forest, and gradient boosting
Cross-validation and hyperparameter tuning with GridSearchCV
SHAP to see which stats drive the predictions
Simulate the bracket to choose a predicted champion
