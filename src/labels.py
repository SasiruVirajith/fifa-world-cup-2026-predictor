"""
labels.py
─────────
Target variable extraction for all ML models.
"""

import re
import unicodedata

import pandas as pd
from pathlib import Path

from src.config import (
    GOLDEN_GLOVE_WINNERS,
    PROCESSED_DIR,
    RAW_DIR,
    TEAM_ALIASES,
    WC_WINNERS,
)


def normalize_team_name(name: str) -> str:
    """Harmonize team names across data sources."""
    if pd.isna(name):
        return name
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def normalize_player_name(name: str) -> str:
    """Normalize player names for fuzzy matching."""
    if pd.isna(name):
        return ""
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"[^a-zA-Z0-9\s]", "", name).lower().strip()
    return name


def add_winner_labels(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Add won_tournament column (1 = winner, 0 = other)."""
    df = df.copy()
    winner = WC_WINNERS.get(year)
    df["year"] = year
    df["won_tournament"] = (df["team"].apply(normalize_team_name) == winner).astype(int)
    return df


def add_glove_labels(gk_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Add won_golden_glove column for goalkeeper features."""
    gk_df = gk_df.copy()
    winner = GOLDEN_GLOVE_WINNERS.get(year, "")
    winner_norm = normalize_player_name(winner)

    player_col = _find_column(gk_df, ["player", "Player"])
    if player_col:
        gk_df["won_golden_glove"] = (
            gk_df[player_col].apply(normalize_player_name) == winner_norm
        ).astype(int)
    else:
        gk_df["won_golden_glove"] = 0

    gk_df["year"] = year
    return gk_df


def extract_tournament_goals(events_df: pd.DataFrame, year: int = None) -> pd.DataFrame:
    """
    Count goals per player from StatsBomb event data.

    Returns DataFrame with [player, team, tournament_goals, year].
    """
    df = events_df.copy()

    # StatsBomb type column can be dict-like string or nested
    if "type" in df.columns:
        if df["type"].dtype == object:
            df["event_type"] = df["type"].apply(
                lambda x: x.get("name") if isinstance(x, dict) else str(x)
            )
        else:
            df["event_type"] = df["type"]
    else:
        df["event_type"] = ""

    shots = df[df["event_type"].str.contains("Shot", case=False, na=False)].copy()

    if "shot_outcome" in shots.columns:
        shots["outcome"] = shots["shot_outcome"].apply(
            lambda x: x.get("name") if isinstance(x, dict) else str(x)
        )
    else:
        shots["outcome"] = ""

    goals = shots[shots["outcome"].str.contains("Goal", case=False, na=False)]

    if goals.empty:
        return pd.DataFrame(columns=["player", "team", "tournament_goals", "year"])

    player_goals = (
        goals.groupby(["player", "team"])
        .size()
        .reset_index(name="tournament_goals")
        .sort_values("tournament_goals", ascending=False)
    )

    if year is not None:
        player_goals["year"] = year

    return player_goals


def merge_tournament_goals(
    player_df: pd.DataFrame,
    goals_df: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Merge StatsBomb goal counts onto FBref player feature rows."""
    df = player_df.copy()
    goals = goals_df.copy()

    player_col = _find_column(df, ["player", "Player"])
    team_col = _find_column(df, ["team", "Team", "Squad", "squad"])

    if not player_col:
        df["tournament_goals"] = 0
        df["year"] = year
        return df

    df["_player_key"] = df[player_col].apply(normalize_player_name)
    if team_col:
        df["_team_key"] = df[team_col].apply(normalize_team_name)
    else:
        df["_team_key"] = ""

    goals["_player_key"] = goals["player"].apply(normalize_player_name)
    goals["_team_key"] = goals["team"].apply(normalize_team_name)

    merged = df.merge(
        goals[["_player_key", "_team_key", "tournament_goals"]],
        on=["_player_key", "_team_key"],
        how="left",
    )
    merged["tournament_goals"] = merged["tournament_goals"].fillna(0).astype(int)
    merged["year"] = year
    merged = merged.drop(columns=["_player_key", "_team_key"], errors="ignore")

    return merged


def load_events_for_year(year: int) -> pd.DataFrame:
    """Load combined StatsBomb events CSV for a WC year."""
    path = RAW_DIR / f"statsbomb_events_wc{year}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _find_column(df: pd.DataFrame, candidates: list) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        for cand in candidates:
            if cand.lower() in str(col).lower():
                return col
    return None
