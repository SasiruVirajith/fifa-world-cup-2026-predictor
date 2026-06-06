"""
squad_depth.py
──────────────
Squad depth scoring and counterfactual win-probability impact.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR
from src.labels import normalize_team_name

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_squad_depth(year: int = 2022) -> pd.DataFrame:
    """
    Rate squad depth per team using player event volume by position.

    Uses StatsBomb events to identify top contributors per team.
    """
    path = RAW_DIR / f"statsbomb_events_wc{year}.csv"
    if not path.exists():
        return pd.DataFrame()

    events = pd.read_csv(path, low_memory=False)
    events["team"] = events["team"].apply(normalize_team_name)

    # Player contribution = event count (proxy for minutes/involvement)
    player_contrib = (
        events.groupby(["player", "team"])
        .size()
        .reset_index(name="events")
    )

    team_depth = player_contrib.groupby("team").agg(
        squad_size=("player", "nunique"),
        top3_events=("events", lambda x: x.nlargest(3).sum()),
        total_events=("events", "sum"),
    ).reset_index()

    team_depth["depth_score"] = team_depth["top3_events"] / team_depth["total_events"].replace(0, np.nan)
    team_depth["depth_score"] = team_depth["depth_score"].fillna(0.5)
    # Lower top3 concentration = better depth
    team_depth["depth_score"] = 1 - team_depth["depth_score"]
    team_depth["year"] = year

    save_path = OUTPUTS_DIR / f"squad_depth_{year}.csv"
    team_depth.to_csv(save_path, index=False)
    return team_depth


def counterfactual_win_impact(
    team: str,
    year: int = 2022,
    reduction_pct: float = 0.15,
) -> dict:
    """
    Estimate win probability drop if key player(s) are unavailable.

    Reduces team's ELO and form features by reduction_pct and re-scores.
    """
    team = normalize_team_name(team)
    features_path = PROCESSED_DIR / "team_features.csv"
    model_path = MODELS_DIR / "tournament_winner.pkl"

    if not features_path.exists() or not model_path.exists():
        return {"team": team, "baseline_prob": None, "counterfactual_prob": None, "drop_pct": None}

    df = pd.read_csv(features_path)
    team_row = df[(df["team"] == team) & (df["year"] == year)]
    if team_row.empty:
        team_row = df[df["team"] == team].tail(1)
    if team_row.empty:
        return {"team": team, "baseline_prob": None, "counterfactual_prob": None, "drop_pct": None}

    feature_cols = ["elo", "win_rate_last20", "goals_scored_avg", "goals_conceded_avg"]
    if "squad_value" in team_row.columns:
        feature_cols.append("squad_value")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_base = team_row[feature_cols].fillna(0)
    baseline_prob = float(model.predict_proba(X_base)[0][1])

    X_cf = X_base.copy()
    X_cf["elo"] = X_cf["elo"] * (1 - reduction_pct)
    X_cf["win_rate_last20"] = X_cf["win_rate_last20"] * (1 - reduction_pct)
    X_cf["goals_scored_avg"] = X_cf["goals_scored_avg"] * (1 - reduction_pct)
    if "squad_value" in X_cf.columns:
        X_cf["squad_value"] = X_cf["squad_value"] * (1 - reduction_pct)

    cf_prob = float(model.predict_proba(X_cf)[0][1])
    drop = baseline_prob - cf_prob

    return {
        "team": team,
        "year": year,
        "baseline_prob": round(baseline_prob, 4),
        "counterfactual_prob": round(cf_prob, 4),
        "drop_pct": round(drop * 100, 2),
        "reduction_pct": reduction_pct,
    }
