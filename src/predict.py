# Copyright (c) 2026 Sasiru Virajith Kankanamge
# SPDX-License-Identifier: MIT

"""
FIFA World Cup 2026 Predictor
Built by: K. Sasiru Virajith
"""

import pickle
from pathlib import Path

import pandas as pd

MODELS_DIR = Path("models")


def load_model(name: str):
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_golden_boot(player_features: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if player_features is None or player_features.empty:
        return pd.DataFrame(columns=["player", "team", "predicted_goals"])

    result = player_features.copy()
    if "player" not in result.columns:
        return pd.DataFrame(columns=["player", "team", "predicted_goals"])

    if "boot_score" in result.columns:
        if "predicted_goals" not in result.columns:
            result["predicted_goals"] = (result["boot_score"] * 18).round(2)
        if "player_norm" in result.columns:
            result = (
                result.sort_values("boot_score", ascending=False)
                .drop_duplicates(subset=["player_norm", "team"], keep="first")
            )
        ranked = result.head(top_n)
        cols = ["player", "team", "boot_score", "predicted_goals"]
        for extra in ("p_qualify", "progression_factor", "league_difficulty", "group"):
            if extra in ranked.columns:
                cols.append(extra)
        return ranked[cols].reset_index(drop=True)

    return pd.DataFrame(columns=["player", "team", "predicted_goals"])


def predict_golden_glove(keeper_features: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if keeper_features is None or keeper_features.empty:
        return pd.DataFrame(columns=["player", "team", "golden_glove_probability"])

    result = keeper_features.copy()
    if "player" not in result.columns:
        return pd.DataFrame(columns=["player", "team", "golden_glove_probability"])

    if "glove_score" not in result.columns:
        return pd.DataFrame(columns=["player", "team", "golden_glove_probability"])

    if "golden_glove_probability" not in result.columns:
        total = result["glove_score"].sum()
        result["golden_glove_probability"] = (
            result["glove_score"] / total if total else 0
        ).round(4)
    ranked = result.sort_values("glove_score", ascending=False).head(top_n)
    out_cols = ["player", "team", "glove_score", "golden_glove_probability"]
    out_cols = [c for c in out_cols if c in ranked.columns]
    return ranked[out_cols].reset_index(drop=True)


def predict_golden_ball(top_n: int = 10) -> pd.DataFrame:
    path = Path("outputs/player_tournament_2026.csv")
    if not path.exists():
        return pd.DataFrame(columns=["player", "team", "pot_score"])
    df = pd.read_csv(path)
    if "player_norm" in df.columns:
        df = df.sort_values("pot_score", ascending=False).drop_duplicates(
            subset=["player_norm", "team"], keep="first",
        )
    cols = [c for c in ["player", "team", "pot_score", "p_qualify", "team_win_prob"] if c in df.columns]
    return df[cols].head(top_n).reset_index(drop=True)


def predict_team_surprise(top_n: int = 10) -> pd.DataFrame:
    from src.team_expectations import compute_underdog_surprises

    df = compute_underdog_surprises()
    cols = [
        c
        for c in ["team", "surprise_score", "fifa_rank", "p_qualify", "team_win_prob", "group", "sim_score"]
        if c in df.columns
    ]
    return df[cols].head(top_n).reset_index(drop=True)


def predict_team_upset(top_n: int = 10) -> pd.DataFrame:
    from src.team_expectations import compute_big_team_upsets

    df = compute_big_team_upsets()
    cols = [
        c
        for c in [
            "team",
            "upset_score",
            "rank_gap",
            "fifa_rank",
            "sim_win_rank",
            "p_qualify",
            "team_win_prob",
            "group",
        ]
        if c in df.columns
    ]
    return df[cols].head(top_n).reset_index(drop=True)
