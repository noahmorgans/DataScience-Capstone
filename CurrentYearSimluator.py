import numpy as np
import pandas as pd
from difflib import get_close_matches
import joblib
import sys
import os

# ── Config ──────────────────────────────────────────────────────────────────
STATS_CSV      = "PreTournamentStats/2026.csv"
MOMENTUM_CSV   = "Momentum/2026.csv"
MODEL_PATH     = "Models/xgb_model.json"
BRACKET_TXT    = "CurrentYearBracket/bracket.txt"        # one team per line, adjacent pairs = matchups
OUTPUT_DIR     = "SimulationResults"
NUM_SIMS       = 25

ROUND_NAMES = [
    "Round of 64",
    "Round of 32",
    "Sweet 16",
    "Elite Eight",
    "Final Four",
    "Championship",
]
# ────────────────────────────────────────────────────────────────────────────


def load_stats(stats_csv, momentum_csv):
    stats_df = pd.read_csv(stats_csv, encoding="cp1252")
    stats_df["Team"] = stats_df["Team"].str.strip()
    momentum_df = pd.read_csv(momentum_csv)
    stats_df["Momentum"] = momentum_df["Momentum"].values
    return stats_df.set_index("Team")[["AdjOE", "AdjDE", "Adj T.", "Momentum"]].to_dict("index")


def load_bracket(filepath):
    """Read one team per line; return list of team names in order."""
    with open(filepath) as f:
        teams = [line.strip() for line in f if line.strip()]
    if len(teams) % 2 != 0:
        raise ValueError(f"Odd number of teams ({len(teams)}) — every team needs an opponent.")
    return teams


def lookup_team(name, stats):
    if name in stats:
        return name
    matches = get_close_matches(name, stats.keys(), n=1, cutoff=0.6)
    if matches:
        print(f"  [fuzzy match] '{name}' → '{matches[0]}'")
        return matches[0]
    raise ValueError(f"Team '{name}' not found in stats. Check spelling or add to CSV.")


def win_prob(team1, team2, stats, model):
    t1 = lookup_team(team1, stats)
    t2 = lookup_team(team2, stats)
    diff = np.array([[
        stats[t1]["AdjOE"] - stats[t2]["AdjOE"],
        stats[t1]["AdjDE"] - stats[t2]["AdjDE"],
        stats[t1]["Adj T."] - stats[t2]["Adj T."],
        stats[t1]["Momentum"] - stats[t2]["Momentum"],
    ]])
    return model.predict_proba(diff)[0][1]


def simulate_tournament(teams, stats, model, out_file):
    current_round = teams[:]
    round_index = 0
    lines = []

    lines.append("=" * 54)
    lines.append("       NCAA TOURNAMENT BRACKET SIMULATION")
    lines.append("=" * 54)

    while len(current_round) > 1:
        round_name = ROUND_NAMES[round_index] if round_index < len(ROUND_NAMES) else f"Round {round_index + 1}"
        lines.append(f"\n{'─' * 54}")
        lines.append(f"  {round_name}  ({len(current_round)} teams → {len(current_round) // 2})")
        lines.append(f"{'─' * 54}")
 
        next_round = []
        for i in range(0, len(current_round), 2):
            t1, t2 = current_round[i], current_round[i + 1]
            try:
                p = win_prob(t1, t2, stats, model)
            except ValueError as e:
                lines.append(f"  WARNING: {e}  — defaulting to 50/50")
                p = 0.5
 
            winner = t1 if np.random.random() < p else t2
            p_winner = p if winner == t1 else 1 - p
 
            lines.append(f"  {t1:30s} vs  {t2}")
            lines.append(f"    → {winner} wins  ({p_winner*100:.1f}%)\n")
            next_round.append(winner)
 
        current_round = next_round
        round_index += 1
 
    champion = current_round[0]
    lines.append("=" * 54)
    lines.append(f"  CHAMPION: {champion}")
    lines.append("=" * 54)
 
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
 
    return champion

    champion = current_round[0]
    print("=" * 54)
    print(f"  🏆  CHAMPION: {champion}")
    print("=" * 54 + "\n")
    return champion


def main():
    print("Loading model and stats...")
    model  = joblib.load(MODEL_PATH)
    stats  = load_stats(STATS_CSV, MOMENTUM_CSV)
    teams  = load_bracket(BRACKET_TXT)
    print(f"  {len(teams)} teams loaded from bracket.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in range(1, NUM_SIMS + 1):
        out_path = os.path.join(OUTPUT_DIR, f"Simulation{i:02d}.txt")
        champion = simulate_tournament(teams, stats, model, out_path)
        print(f"Sim {i:02d} -> {champion} (saved to {out_path})")

if __name__ == "__main__":
    main()