"""
predict.py
──────────
Loads saved models and returns predictions for new input.
Called by the Streamlit dashboard (app.py).

Usage:
    from src.predict import predict_winner, predict_golden_boot, get_playmaker_ranking
"""

import pickle
import numpy as np
import pandas as pd
import shap
from pathlib import Path

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")


def load_model(name: str):
    """Load a saved model by name from models/."""
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def models_available() -> bool:
    """Return True if trained models exist."""
    return (MODELS_DIR / "tournament_winner.pkl").exists()


def predict_winner(team_features: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Predict tournament winner probabilities for all teams.

    Args:
        team_features: DataFrame with columns [team, elo, win_rate_last20, ...]
        top_n: how many top teams to return

    Returns:
        DataFrame with [team, win_probability] sorted descending
    """
    model = load_model("tournament_winner")
    if model is None:
        # Return dummy data so the dashboard still renders during development
        return pd.DataFrame({
            "team": ["Brazil", "France", "England", "Spain", "Argentina"],
            "win_probability": [0.22, 0.18, 0.14, 0.12, 0.11],
        })

    feature_cols = ["elo", "win_rate_last20", "goals_scored_avg", "goals_conceded_avg"]
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    elif "squad_value" in team_features.columns:
        feature_cols.append("squad_value")
    available = [c for c in feature_cols if c in team_features.columns]
    X = team_features[available].fillna(team_features[available].mean())

    probs = model.predict_proba(X)[:, 1]
    result = team_features[["team"]].copy()
    result["win_probability"] = probs
    result = result.sort_values("win_probability", ascending=False).head(top_n)
    result["win_probability"] = result["win_probability"].round(4)

    return result.reset_index(drop=True)


def predict_golden_boot(player_features: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Predict top goal scorers for the tournament.

    Args:
        player_features: DataFrame with player stats
        top_n: number of players to return

    Returns:
        DataFrame with [player, team, predicted_goals]
    """
    if player_features is None or player_features.empty:
        return pd.DataFrame(columns=["player", "team", "predicted_goals"])

    result = player_features.copy()
    if "player" not in result.columns:
        return pd.DataFrame(columns=["player", "team", "predicted_goals"])

    # 2026-forward composite (martj42 + club layer)
    if "boot_score" in result.columns:
        result["predicted_goals"] = result.get(
            "predicted_goals",
            (result["boot_score"] * 7.5).round(2),
        )
        ranked = result.sort_values("boot_score", ascending=False).head(top_n)
        return ranked[["player", "team", "predicted_goals"]].reset_index(drop=True)

    model = load_model("golden_boot")
    if model is None:
        return pd.DataFrame(columns=["player", "team", "predicted_goals"])

    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        feature_cols = ["xg_per90", "npxg_per90", "shots_per90", "shots_on_target_pct"]
    available = [c for c in feature_cols if c in player_features.columns]
    if not available:
        return pd.DataFrame(columns=["player", "team", "predicted_goals"])

    X = player_features[available].fillna(0)
    predicted_goals = model.predict(X)
    result = player_features[["player", "team"]].copy()
    result["predicted_goals"] = predicted_goals.round(2)
    result = result.sort_values("predicted_goals", ascending=False).head(top_n)

    return result.reset_index(drop=True)


def predict_golden_glove(keeper_features: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Predict the Golden Glove (best goalkeeper) probabilities.

    Returns:
        DataFrame with [player, team, golden_glove_probability]
    """
    if keeper_features is None or keeper_features.empty:
        return pd.DataFrame(columns=["player", "team", "golden_glove_probability"])

    result = keeper_features.copy()
    if "player" not in result.columns:
        return pd.DataFrame(columns=["player", "team", "golden_glove_probability"])

    if "glove_score" in result.columns:
        if "golden_glove_probability" not in result.columns:
            total = result["glove_score"].sum()
            result["golden_glove_probability"] = (
                result["glove_score"] / total if total else 0
            ).round(4)
        ranked = result.sort_values("glove_score", ascending=False).head(top_n)
        return ranked[["player", "team", "golden_glove_probability"]].reset_index(drop=True)

    model = load_model("golden_glove")
    if model is None:
        return pd.DataFrame(columns=["player", "team", "golden_glove_probability"])

    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        feature_cols = ["save_pct", "clean_sheet_pct", "psxg_minus_ga", "ga90"]
    available = [c for c in feature_cols if c in keeper_features.columns]
    if not available:
        return pd.DataFrame(columns=["player", "team", "golden_glove_probability"])

    X = keeper_features[available].fillna(0)
    probs = model.predict_proba(X)[:, 1]
    result = keeper_features[["player", "team"]].copy()
    result["golden_glove_probability"] = probs.round(4)
    result = result.sort_values("golden_glove_probability", ascending=False).head(top_n)

    return result.reset_index(drop=True)


def get_playmaker_ranking(top_n: int = 10, year: int = 2026) -> pd.DataFrame:
    """
    Return the top-ranked playmakers from the pre-computed composite score.

    Returns:
        DataFrame with [player, team, playmaker_score]
    """
    for path in (
        Path("outputs") / f"playmaker_rankings_{year}.csv",
        Path("data/processed/playmaker_features.csv"),
        Path("outputs/playmaker_rankings.csv"),
    ):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "year" in df.columns:
            year_df = df[df["year"] == year]
            if not year_df.empty:
                df = year_df
        if "playmaker_score" not in df.columns:
            continue
        cols = [c for c in ["player", "team", "playmaker_score"] if c in df.columns]
        return df[cols].sort_values("playmaker_score", ascending=False).head(top_n).reset_index(drop=True)

    return pd.DataFrame(columns=["player", "team", "playmaker_score"])


def get_shap_explanation(model_name: str, team_features: pd.DataFrame, team: str) -> dict:
    """
    Return SHAP values for a single team - used in the dashboard explainability panel.

    Returns:
        dict with {feature: shap_value} for the given team
    """
    model = load_model(model_name)
    if model is None:
        return {}

    feature_cols = ["elo", "win_rate_last20", "goals_scored_avg", "goals_conceded_avg"]
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    elif "squad_value" in team_features.columns:
        feature_cols.append("squad_value")

    team_row = team_features[team_features["team"] == team]
    if team_row.empty:
        return {}

    available = [c for c in feature_cols if c in team_row.columns]
    X = team_row[available].fillna(0)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]

    return dict(zip(available, shap_values))
