import pandas as pd
import re

def parse_game_line(line):
    # Extract date explicitly
    date_match = re.match(r"\s*(\d{1,2}/\d{1,2}/\d{4})", line)
    if not date_match:
        return None

    date = date_match.group(1)
    rest = line[date_match.end():]

    # Find remaining integers (scores)
    matches = list(re.finditer(r"\b\d+\b", rest))
    if len(matches) < 2:
        return None

    team1_score = int(matches[-2].group())
    team2_score = int(matches[-1].group())

    # Slice team names
    team1 = rest[:matches[-2].start()].strip()
    team2 = rest[matches[-2].end():matches[-1].start()].strip()

    if team1_score > team2_score:
        team1_result = "W"
        team2_result = "L"
    else:
        team1_result = "L"
        team2_result = "W"

    return {
        "date": date,
        "team1": team1,
        "team1_score": team1_score,
        "team1_result": team1_result,
        "team2": team2,
        "team2_score": team2_score,
        "team2_result": team2_result
    }

stats_df = pd.read_csv(
    "PreTournamentStats/2025.csv",
    encoding="cp1252"
)

print(stats_df)

gameDict = {}

records = []

with open("PreTournamentGameLogs/2025.txt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        record = parse_game_line(line)
        if record is not None:
            records.append(record)

games_df = pd.DataFrame(records)
stats_df.set_index("Team", inplace=True)

for i, row in games_df.iterrows():
    team1 = row["team1"]
    team2 = row["team2"]

    # Initialize empty dictionary to hold new features
    new_features = {}

    if team1 in stats_df.index:
        new_features["team1_rank"] = stats_df.loc[team1, "Rk"]

    if team2 in stats_df.index:
        new_features["team2_rank"] = stats_df.loc[team2, "Rk"]

    # Append these to DataFrame
    for key, value in new_features.items():
        games_df.at[i, key] = value

games_df = games_df[games_df["team1_rank"].notna() & games_df["team2_rank"].notna()]
    
stats_df["rating"] = 0.0
stats_df["AdjEM"] = stats_df["AdjOE"] - stats_df["AdjDE"]
stats_df["EFG%M"] = stats_df["EFG%"] - stats_df["EFGD%"]
stats_df["TORM"] = stats_df["TOR"] - stats_df["TORD"]
stats_df["FTRM"] = stats_df["FTR"] - stats_df["FTRD"]
stats_df["TRB"] = stats_df["ORB"] + stats_df["DRB"]
stats_df["FTRM"] = stats_df["FTR"] - stats_df["FTRD"]
stats_df["3P%M"] = stats_df["3P%"] - stats_df["3P%D"]
stats_df["2P%M"] = stats_df["2P%"] - stats_df["2P%D"]
stats_df["3PRM"] = stats_df["3PR"] - stats_df["3PRD"]

for i, row in games_df.iterrows():
    team1 = row["team1"]
    team2 = row["team2"]
    score1 = row["team1_score"]
    score2 = row["team2_score"]

    # Determine winner and loser
    if score1 > score2:
        winner = team1
        loser = team2
    elif score2 > score1:
        winner = team2
        loser = team1

    # Get ranks
    winner_rank = stats_df.loc[winner, "Rk"]
    loser_rank = stats_df.loc[loser, "Rk"]

    # Compute rating changes
    loser_change = (winner_rank - 1) / 364
    winner_change = (364 - loser_rank) / 364

    # Update running totals
    stats_df.loc[loser, "rating"] -= loser_change
    stats_df.loc[winner, "rating"] += winner_change

print(games_df)
print(stats_df)
