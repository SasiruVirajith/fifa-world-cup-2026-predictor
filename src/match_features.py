# Copyright (c) 2026 Sasiru Virajith Kankanamge
# SPDX-License-Identifier: MIT

"""
FIFA World Cup 2026 Predictor
Built by: K. Sasiru Virajith
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MATCH_MODEL_START_YEAR, PROCESSED_DIR, RAW_DIR
from src.historical_data import get_tournament_categories, load_and_normalize_results
from src.labels import normalize_team_name

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def calculate_elo_ratings(
    results_df: pd.DataFrame,
    k: int = 32,
    initial_elo: int = 1500,
) -> dict:
    elo = {}
    results_df = results_df.sort_values("date").copy()

    for _, row in results_df.iterrows():
        home = normalize_team_name(row["home_team"])
        away = normalize_team_name(row["away_team"])

        if home not in elo:
            elo[home] = initial_elo
        if away not in elo:
            elo[away] = initial_elo

        exp_home = 1 / (1 + 10 ** ((elo[away] - elo[home]) / 400))
        exp_away = 1 - exp_home

        home_goals = row["home_score"]
        away_goals = row["away_score"]

        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals == away_goals:
            actual_home, actual_away = 0.5, 0.5
        else:
            actual_home, actual_away = 0.0, 1.0

        elo[home] += k * (actual_home - exp_home)
        elo[away] += k * (actual_away - exp_away)

    return elo


def _team_result(home_score, away_score, perspective: str) -> float:
    if home_score == away_score:
        return 0.5
    if perspective == "home":
        return 1.0 if home_score > away_score else 0.0
    return 1.0 if away_score > home_score else 0.0


def _last_n_form(team: str, history: list, n: int = 5) -> float:
    if not history:
        return 0.5
    recent = history[-n:]
    return np.mean(recent) if recent else 0.5


def _h2h_stats(home: str, away: str, h2h: dict) -> dict:
    key = tuple(sorted([home, away]))
    stats = h2h.get(key, {"played": 0, "wins": {}, "draws": 0})
    played = stats["played"]
    home_wins = stats["wins"].get(home, 0)
    return {
        "h2h_matches_played": played,
        "h2h_home_win_rate": home_wins / max(played, 1),
        "h2h_draw_rate": stats["draws"] / max(played, 1),
    }


def build_match_features(start_year: int = MATCH_MODEL_START_YEAR) -> pd.DataFrame:
    print("Building match-level features (walk-forward)...")
    results = load_and_normalize_results()
    results = get_tournament_categories(results)

    fifa_path = RAW_DIR / "fifa_latest_ranking.csv"
    fifa_latest = {}
    if fifa_path.exists():
        fifa = pd.read_csv(fifa_path)
        fifa_latest = dict(zip(
            fifa["country"].apply(normalize_team_name),
            fifa["total_points"],
        ))

    pen_path = RAW_DIR / "shootouts.csv"
    penalty_win_rate = {}
    if pen_path.exists():
        shootouts = pd.read_csv(pen_path)
        shootouts["winner"] = shootouts["winner"].apply(normalize_team_name)
        shootouts["home_team"] = shootouts["home_team"].apply(normalize_team_name)
        shootouts["away_team"] = shootouts["away_team"].apply(normalize_team_name)
        wins = shootouts["winner"].value_counts()
        games = pd.concat([shootouts["home_team"], shootouts["away_team"]]).value_counts()
        penalty_win_rate = (wins / games).fillna(0.5).to_dict()

    team_history = defaultdict(list)
    h2h = {}
    elo_state = {}
    records = []

    all_results = results[results["year"] < MATCH_MODEL_START_YEAR]
    if not all_results.empty:
        elo_state = calculate_elo_ratings(all_results)

    train_matches = results[results["year"] >= start_year]

    # features use pre-match state only (no leakage from this row's result)
    for idx, row in train_matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs, aws = row["home_score"], row["away_score"]

        home_elo = elo_state.get(home, 1500)
        away_elo = elo_state.get(away, 1500)
        home_fifa = fifa_latest.get(home, home_elo)
        away_fifa = fifa_latest.get(away, away_elo)

        home_form = _last_n_form(home, team_history[home], 5)
        away_form = _last_n_form(away, team_history[away], 5)
        h2h_feat = _h2h_stats(home, away, h2h)

        fifa_diff = home_fifa - away_fifa
        elo_diff = home_elo - away_elo

        if hs > aws:
            outcome = 2
        elif hs < aws:
            outcome = 0
        else:
            outcome = 1

        records.append({
            "date": row["date"],
            "home_team": home,
            "away_team": away,
            "outcome": outcome,
            "home_score": hs,
            "away_score": aws,
            "match_type": row["match_type"],
            "tournament": row["tournament"],
            "neutral": row["neutral"],
            "fifa_diff": fifa_diff,
            "elo_diff": elo_diff,
            "home_last5_win_rate": home_form,
            "away_last5_win_rate": away_form,
            "h2h_home_win_rate": h2h_feat["h2h_home_win_rate"],
            "h2h_draw_rate": h2h_feat["h2h_draw_rate"],
            "h2h_matches_played": h2h_feat["h2h_matches_played"],
            "home_penalty_win_rate": penalty_win_rate.get(home, 0.5),
            "away_penalty_win_rate": penalty_win_rate.get(away, 0.5),
            "fifa_diff_x_home_form": fifa_diff * home_form,
            "fifa_diff_x_away_form": fifa_diff * away_form,
            "h2h_effective": h2h_feat["h2h_home_win_rate"] * min(1.0, h2h_feat["h2h_matches_played"] / 10),
            "is_friendly": int(row["match_type"] == "friendly"),
            "is_tournament": int(row["match_type"] in ("world_cup", "continental")),
        })

        home_res = _team_result(hs, aws, "home")
        away_res = _team_result(hs, aws, "away")
        team_history[home].append(home_res)
        team_history[away].append(away_res)

        key = tuple(sorted([home, away]))
        if key not in h2h:
            h2h[key] = {"played": 0, "wins": {}, "draws": 0}
        h2h[key]["played"] += 1
        if hs == aws:
            h2h[key]["draws"] += 1
        elif hs > aws:
            h2h[key]["wins"][home] = h2h[key]["wins"].get(home, 0) + 1
        else:
            h2h[key]["wins"][away] = h2h[key]["wins"].get(away, 0) + 1

        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        actual_home = home_res
        k = 32 if row["match_type"] != "friendly" else 16
        elo_state[home] = home_elo + k * (actual_home - exp_home)
        elo_state[away] = away_elo + k * ((1 - actual_home) - (1 - exp_home))

    df = pd.DataFrame(records)
    save_path = PROCESSED_DIR / "match_features.csv"
    df.to_csv(save_path, index=False)
    print(f"  [OK] Match features saved -> {save_path} ({len(df)} matches)")
    return df


def build_team_strength_table() -> pd.DataFrame:
    results = load_and_normalize_results()
    results = get_tournament_categories(results)

    elo = calculate_elo_ratings(results)

    fifa_path = RAW_DIR / "fifa_latest_ranking.csv"
    fifa_pts = {}
    if fifa_path.exists():
        fifa = pd.read_csv(fifa_path)
        fifa_pts = dict(zip(fifa["country"].apply(normalize_team_name), fifa["total_points"]))

    team_history = defaultdict(list)
    for _, row in results.iterrows():
        home, away = row["home_team"], row["away_team"]
        hs, aws = row["home_score"], row["away_score"]
        team_history[home].append(_team_result(hs, aws, "home"))
        team_history[away].append(_team_result(hs, aws, "away"))

    records = []
    all_teams = set(elo.keys()) | set(fifa_pts.keys())
    for team in all_teams:
        records.append({
            "team": team,
            "elo": elo.get(team, 1500),
            "fifa_points": fifa_pts.get(team, elo.get(team, 1500)),
            "last5_win_rate": _last_n_form(team, team_history[team], 5),
            "last20_win_rate": _last_n_form(team, team_history[team], 20),
        })

    df = pd.DataFrame(records).sort_values("fifa_points", ascending=False)
    save_path = PROCESSED_DIR / "team_strength_2026.csv"
    df.to_csv(save_path, index=False)
    print(f"  [OK] Team strength table -> {save_path}")
    return df


if __name__ == "__main__":
    build_match_features()
    build_team_strength_table()
