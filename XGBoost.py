import xgboost as xgb
import pandas as pd

stats_df = pd.read_csv(
    "PreTournamentStats/2025.csv",
    encoding="cp1252"
)