import sys

import pandas as pd
import os
import re

START_YEAR = 2026
END_YEAR = 2026

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_DIR = "PreTournamentStats"
GAME_LOGS_DIR = "TournamentGameLogs"
MOMENTUM_DIR = "Momentum"
OUTPUT_DIR = os.path.join(BASE_DIR, "ProcessedData")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "CombinedData.csv")

STAT_COLUMNS = [
    "AdjOE", "AdjDE", "Adj T.", "Momentum"
]

def parse_game_line(line):
    parts = line.strip().split()

    if len(parts) < 6:
        return None
    
    date = parts[0]

    match = re.search(r"\d{2}/\d{2}/\d{4}\s+(.*?)\s+(\d+)\s+(.*?)\s+(\d+)", line)
    if not match:
        return None
    
    team1 = match.group(1).strip()
    score1 = int(match.group(2))
    team2 = match.group(3).strip()
    score2 = int(match.group(4))
    
    return team1, score1, team2, score2

all_games = []

for year in range(START_YEAR, END_YEAR + 1):

    print(f"Processing {year}...")

    stats_path = os.path.join(STATS_DIR, f"{year}.csv")
    games_path = os.path.join(GAME_LOGS_DIR, f"{year}.txt")
    momentum_path = os.path.join(MOMENTUM_DIR, f"{year}.csv")

    if not os.path.exists(stats_path) or not os.path.exists(games_path) or not os.path.exists(momentum_path):
        print(f"Missing data for {year}, skipping.")
        continue

    stats_df = pd.read_csv(
    stats_path,
    encoding="cp1252"
    )

    stats_df["Team"] = stats_df["Team"].str.strip()
    stats_df = stats_df.set_index("Team")
    stats_df['Momentum'] = pd.read_csv(momentum_path)['Momentum'].values

    with open(games_path, "r", encoding="utf-8") as f:
        for line in f:

            parsed = parse_game_line(line)
            if parsed is None:
                continue
            
            team1, score1, team2, score2 = parsed

            if team1 not in stats_df.index or team2 not in stats_df.index:
                print(f"Stats missing for {team1} or {team2} in {year}, skipping game.")
                continue

            team1_stats = stats_df.loc[team1]
            team2_stats = stats_df.loc[team2]

            row = {
                "year": year,
                "team1": team1,
                "team2": team2,
                "team1_score": score1,
                "team2_score": score2,
                "team1_win": int(score1 > score2)
            }

            for col in STAT_COLUMNS:
                row[f"{col}_1"] = team1_stats[col]
                row[f"{col}_2"] = team2_stats[col]
                row[f"{col}_diff"] = team1_stats[col] - team2_stats[col]

            all_games.append(row)

final_df = pd.DataFrame(all_games)

os.makedirs("ProcessedData", exist_ok=True)
final_df.to_csv(OUTPUT_PATH, index=False)

print(f"Combined data saved to {OUTPUT_PATH}")
print("Total games processed:", len(final_df))