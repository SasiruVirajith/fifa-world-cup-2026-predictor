"""
player_tournament.py
────────────────────
Player of the Tournament composite ranking from StatsBomb events.
"""

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, RAW_DIR
from src.labels import normalize_team_name
from src.statsbomb_features import _get_event_name, _parse_json_col

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _normalise(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def build_player_tournament_scores(year: int = 2022) -> pd.DataFrame:
    """
    Composite Player of the Tournament score.

    Weights:
        40% goals + assists
        25% defensive actions (tackles, interceptions, pressures)
        20% progressive carries + key passes
        15% duel win rate
    """
    path = RAW_DIR / f"statsbomb_events_wc{year}.csv"
    if not path.exists():
        return pd.DataFrame()

    events = pd.read_csv(path, low_memory=False)
    events["event_type"] = events["type"].apply(_get_event_name)
    events["team"] = events["team"].apply(normalize_team_name)

    player_mins = events.groupby(["player", "team"]).size().reset_index(name="events")
    player_mins["minutes"] = (player_mins["events"] * 0.5).clip(lower=90)

    # Goals
    shots = events[events["event_type"] == "Shot"].copy()
    shots["outcome"] = shots["shot_outcome"].apply(_get_event_name)
    goals = shots[shots["outcome"] == "Goal"].groupby(["player", "team"]).size().reset_index(name="goals")

    # Assists from goal-assist passes
    passes = events[events["event_type"] == "Pass"].copy()
    def is_assist(row):
        ga = _parse_json_col(row.get("pass_goal_assist", {}))
        return 1 if ga else 0
    passes["is_assist"] = passes.apply(is_assist, axis=1)
    assists = passes[passes["is_assist"] == 1].groupby(["player", "team"]).size().reset_index(name="assists")

    # Defensive actions
    def_actions = events[events["event_type"].isin(["Pressure", "Interception", "Duel"])].copy()
    def_agg = def_actions.groupby(["player", "team"]).size().reset_index(name="def_actions")

    # Key passes
    def is_key(row):
        sa = _parse_json_col(row.get("pass_shot_assist", {}))
        return 1 if sa else 0
    passes["is_key"] = passes.apply(is_key, axis=1)
    key_passes = passes[passes["is_key"] == 1].groupby(["player", "team"]).size().reset_index(name="key_passes")

    # Carries
    carries = events[events["event_type"] == "Carry"].copy()
    carry_agg = carries.groupby(["player", "team"]).size().reset_index(name="carries")

    # Duel win rate
    duels = events[events["event_type"] == "Duel"].copy()
    def duel_won(row):
        outcome = _parse_json_col(row.get("duel_outcome", {}))
        return 1 if outcome.get("name") == "Won" else 0
    duels["won"] = duels.apply(duel_won, axis=1)
    duel_agg = duels.groupby(["player", "team"]).agg(
        duels_total=("won", "count"),
        duels_won=("won", "sum"),
    ).reset_index()
    duel_agg["duel_win_rate"] = duel_agg["duels_won"] / duel_agg["duels_total"].replace(0, np.nan)

    # Merge all
    df = player_mins[["player", "team", "minutes"]].copy()
    for sub in [goals, assists, def_agg, key_passes, carry_agg, duel_agg]:
        df = df.merge(sub, on=["player", "team"], how="left")

    for col in ["goals", "assists", "def_actions", "key_passes", "carries"]:
        df[col] = df[col].fillna(0)

    df["duel_win_rate"] = df["duel_win_rate"].fillna(0.5)
    mins90 = df["minutes"] / 90

    df["goals_assists_per90"] = (df["goals"] + df["assists"]) / mins90
    df["def_per90"] = df["def_actions"] / mins90
    df["creation_per90"] = (df["key_passes"] + df["carries"] * 0.1) / mins90

    df["pot_score"] = (
        0.40 * _normalise(df["goals_assists_per90"]) +
        0.25 * _normalise(df["def_per90"]) +
        0.20 * _normalise(df["creation_per90"]) +
        0.15 * _normalise(df["duel_win_rate"])
    )

    df["year"] = year
    ranked = df.sort_values("pot_score", ascending=False).reset_index(drop=True)

    save_path = OUTPUTS_DIR / f"player_tournament_{year}.csv"
    ranked.to_csv(save_path, index=False)
    return ranked
