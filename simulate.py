import xgboost as xgb
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import joblib

model = joblib.load("Models/xgb_model.json")

stats_df = pd.read_csv("PreTournamentStats/2023.csv", encoding="cp1252")
stats_df["Team"] = stats_df["Team"].str.strip()
momentum_df = pd.read_csv("Momentum/2023.csv")
stats_df["Momentum"] = momentum_df["Momentum"].values
stats = stats_df.set_index("Team")[["AdjOE", "AdjDE", "Adj T.", "Momentum"]].to_dict("index")

prob_cache = {}

def parse_bracket(filepath):
    games = []
    pattern = re.compile(r"(\d{2}/\d{2}/\d{4})\s+(.*?)\s+(\d+)\s+(.*?)\s+(\d+)\s+\d*NP")    
    first_four_dates = set()

    with open(filepath) as f:
        lines = f.readlines()

    parsed = []
    for line in lines:
        m = pattern.search(line)
        if m:
            date, t1, s1, t2, s2 = m.groups()
            parsed.append((date, t1.strip(), int(s1), t2.strip(), int(s2)))

    if not parsed:
        raise ValueError("No games found in bracket file.")
    
    from datetime import datetime
    parsed.sort(key=lambda x: datetime.strptime(x[0], "%m/%d/%Y"))
    first_four = {(t1, t2) for _, t1, _, t2, _ in parsed[:4]}

    for row in parsed[4:]:
        games.append(row)

    return games

def build_rounds(games):
    round_sizes = [32, 16, 8, 4, 2, 1]
    round_points = [10, 20, 40, 80, 160, 320]
    rounds = []
    idx = 0
    for size, pts in zip(round_sizes, round_points):
        rounds.append((games[idx:idx+size], pts))
        idx += size

    return rounds

from difflib import get_close_matches

def lookup_team(name, stats):
    if name in stats:
        return name
    matches = get_close_matches(name, stats.keys(), n=1, cutoff=0.6)
    if matches:
        return matches[0]
    raise ValueError(f"Team '{name}' not found in stats.")

def win_prob(team1, team2, stats, model):
    t1 = lookup_team(team1, stats)
    t2 = lookup_team(team2, stats)
    key = (team1, team2)
    if key not in prob_cache:
        diff = np.array([[
            stats[t1]["AdjOE"] - stats[t2]["AdjOE"],
            stats[t1]["AdjDE"] - stats[t2]["AdjDE"],
            stats[t1]["Adj T."] - stats[t2]["Adj T."],
            stats[t1]["Momentum"] - stats[t2]["Momentum"]
        ]])
        prob_cache[key] = model.predict_proba(diff)[0][1]
    return prob_cache[key]

def simulate_tournament(rounds, stats, model, actual_winners):
    total = 0
    current_teams = None

    for round_idx, (games, pts) in enumerate(rounds):
        if round_idx == 0:
            matchups = [(t1, t2) for _, t1, _, t2, _ in games]
        else:
            matchups = [(current_teams[i], current_teams[i+1]) for i in range(0, len(current_teams), 2)]

        round_winners = []
        for i, (t1, t2) in enumerate(matchups):
            try:
                p = win_prob(t1, t2, stats, model)
            except KeyError:
                p=0.5
            
            sim_winner = t1 if np.random.random() < p else t2

            if sim_winner in actual_winners[round_idx]:
                total += pts

            round_winners.append(sim_winner)

        current_teams = round_winners

    return total

def get_actual_winners(rounds):
    actual_winners = []
    for games, _ in rounds:
        winners = []
        for _, t1, s1, t2, s2, *_ in [(*g,) for g in games]:
            winners.append(t1 if s1 > s2 else t2)
        actual_winners.append(winners)
    return actual_winners

games = parse_bracket("TournamentGameLogs/2023.txt")
rounds = build_rounds(games)
actual_winners = get_actual_winners(rounds)

iteration_points = []

for i in range(25000):
    pts = simulate_tournament(rounds, stats, model, actual_winners)
    iteration_points.append(pts)

iteration_points = np.array(iteration_points)

print(f"Mean score:   {iteration_points.mean():.1f}")
print(f"Median score: {np.median(iteration_points):.1f}")
print(f"Std dev:      {iteration_points.std():.1f}")
print(f"Min:          {iteration_points.min()}")
print(f"Max:          {iteration_points.max()}")
print(f"Scores by percentile (25/50/75): "
      f"{np.percentile(iteration_points, 25):.0f} / "
      f"{np.percentile(iteration_points, 50):.0f} / "
      f"{np.percentile(iteration_points, 75):.0f}")