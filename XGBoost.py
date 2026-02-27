import xgboost as xgb
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import joblib

STATS_DIR = "ProcessedData"
stats_path = os.path.join(STATS_DIR, f"CombinedData.csv")

stats_df = pd.read_csv(
    stats_path,
    encoding="cp1252"
    )

X = stats_df[["AdjOE_diff", "AdjDE_diff", "Adj T._diff"]]
y = stats_df["team1_win"]

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 42)

# define model
model = xgb.XGBClassifier(
    eta=0.05,
    max_depth=3,
    objective="binary:logistic",
    eval_metric="logloss"
)

# calibrated_model = CalibratedClassifierCV(
#     model,
#     method='isotonic', 
#     cv=5)

# train model
model.fit(X_train, y_train)

# prediction and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
win_probs = model.predict_proba(X_test)
print(f"Test Accuracy: {accuracy:.4f}")
print("Log Loss:", log_loss(y_test, win_probs))

results = stats_df.loc[X_test.index].copy()
results["predicted_team1_win_prob"] = win_probs[:,1]
print(results[["year", "team1", "team2", "team1_win", "predicted_team1_win_prob"]])

os.makedirs("Models", exist_ok=True)
joblib.dump(model, "Models/xgb_model.json")