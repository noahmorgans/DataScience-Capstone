import xgboost as xgb
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import joblib

STATS_DIR = "ProcessedData"
stats_path = os.path.join(STATS_DIR, f"CombinedData.csv")

stats_df = pd.read_csv(
    stats_path,
    encoding="cp1252"
    )

X = stats_df[["AdjOE_diff", "AdjDE_diff", "Adj T._diff", "Momentum_diff"]]
y = stats_df["team1_win"]

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

# define model
model = xgb.XGBClassifier(
    eta=0.35,
    max_depth=5,
    gamma=0.1,
    tree_method="exact",
    min_child_weight=2,
    max_delta_step=6,
    objective="binary:logistic",
    eval_metric="logloss"
)

# Best parameter values thus far:
# eta = 0.35
# max_depth = 5
# gamma = 0.1
# tree_method = "exact"
# min_child_weight = 2
# max_delta_step = 5 (minimal effect here)
# alpha = 0.1
#
# irrelevant parameters that were tested: 
# subsample, alpha, reg_lambda, colsample_bytree,
# ...

# train model
model.fit(X_train, y_train)

calibrated_model = CalibratedClassifierCV(
    model,
    method='sigmoid', 
    cv=10)

calibrated_model.fit(X_train, y_train)

# prediction and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
win_probs = model.predict_proba(X_test)[:,1]
print(f"Test Accuracy: {accuracy:.4f}")
print("Log Loss:", log_loss(y_test, win_probs))
print("Brier Score:", brier_score_loss(y_test, win_probs))

results = stats_df.loc[X_test.index].copy()
results["predicted_team1_win_prob"] = win_probs
print(results[["year", "team1", "team2", "team1_win", "predicted_team1_win_prob"]])

os.makedirs("Models", exist_ok=True)
joblib.dump(model, "Models/xgb_model.json")

# y_pred = calibrated_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# win_probs = calibrated_model.predict_proba(X_test)[:,1]
# print(f"Test Accuracy: {accuracy:.4f}")
# print("Log Loss:", log_loss(y_test, win_probs))
# print("Brier Score:", brier_score_loss(y_test, win_probs))

# results = stats_df.loc[X_test.index].copy()
# results["predicted_team1_win_prob"] = win_probs
# print(results[["year", "team1", "team2", "team1_win", "predicted_team1_win_prob"]])

# os.makedirs("Models", exist_ok=True)
# joblib.dump(calibrated_model, "Models/xgb_calibrated_model.json")

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt   # ← fix the typo too (matplotplib → matplotlib)

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt   # ← fix the typo too (matplotplib → matplotlib)

# After fitting the calibrated model
win_probs_raw = model.predict_proba(X_test)[:, 1]
win_probs_cal = calibrated_model.predict_proba(X_test)[:, 1]

# Raw model
prob_true_raw, prob_pred_raw = calibration_curve(y_test, win_probs_raw, n_bins=10)
# Calibrated model
prob_true_cal, prob_pred_cal = calibration_curve(y_test, win_probs_cal, n_bins=10)

plt.plot(prob_pred_raw, prob_true_raw, marker='o', label='Raw XGBoost')
plt.plot(prob_pred_cal, prob_true_cal, marker='s', label='Calibrated (sigmoid)')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect')
plt.xlabel('Predicted probability')
plt.ylabel('Actual win rate')
plt.title('Calibration Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()