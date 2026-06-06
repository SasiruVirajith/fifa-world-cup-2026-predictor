"""
statsbomb_features.py
─────────────────────
Build player features from StatsBomb event data when FBref is unavailable.
"""

import ast
import json

import numpy as np
import pandas as pd

from src.config import RAW_DIR
from src.labels import normalize_player_name, normalize_team_name


def _parse_json_col(val):
    """Parse dict-like column values from CSV."""
    if pd.isna(val):
        return {}
    if isinstance(val, dict):
        return val
    try:
        return json.loads(val.replace("'", '"'))
    except Exception:
        try:
            return ast.literal_eval(val)
        except Exception:
            return {}


def _get_event_name(val) -> str:
    d = _parse_json_col(val)
    return d.get("name", str(val)) if d else str(val)


def load_events(year: int) -> pd.DataFrame:
    path = RAW_DIR / f"statsbomb_events_wc{year}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def build_striker_features_from_events(year: int) -> pd.DataFrame:
    """Extract striker/attacker features from StatsBomb shot events."""
    events = load_events(year)
    if events.empty:
        return pd.DataFrame()

    events["event_type"] = events["type"].apply(_get_event_name)
    shots = events[events["event_type"] == "Shot"].copy()

    if shots.empty:
        return pd.DataFrame()

    shots["outcome"] = shots["shot_outcome"].apply(_get_event_name)
    shots["is_goal"] = (shots["outcome"] == "Goal").astype(int)
    shots["xg"] = pd.to_numeric(shots.get("shot_statsbomb_xg", 0), errors="coerce").fillna(0)
    shots["team"] = shots["team"].apply(normalize_team_name)

    # Minutes played per player (from all events)
    player_mins = (
        events.groupby(["player", "team"])
        .size()
        .reset_index(name="event_count")
    )
    # Approximate minutes: events / 2 ~ minutes (rough proxy)
    player_mins["minutes"] = player_mins["event_count"] * 0.5
    player_mins["minutes"] = player_mins["minutes"].clip(lower=90)

    agg = shots.groupby(["player", "team"]).agg(
        shots_total=("is_goal", "count"),
        goals_total=("is_goal", "sum"),
        xg_total=("xg", "sum"),
        shots_on_target=("outcome", lambda x: (x == "Goal").sum() + (x == "Saved").sum()),
    ).reset_index()

    agg = agg.merge(player_mins[["player", "team", "minutes"]], on=["player", "team"], how="left")
    agg["minutes"] = agg["minutes"].fillna(90)
    mins90 = agg["minutes"] / 90

    agg["xg_per90"] = agg["xg_total"] / mins90
    agg["npxg_per90"] = agg["xg_per90"]  # StatsBomb xG is mostly non-penalty in open play
    agg["shots_per90"] = agg["shots_total"] / mins90
    agg["goals_per90"] = agg["goals_total"] / mins90
    agg["shots_on_target_pct"] = (agg["shots_on_target"] / agg["shots_total"].replace(0, np.nan)).fillna(0)
    agg["tournament_goals"] = agg["goals_total"].astype(int)
    agg["year"] = year

    return agg


def build_goalkeeper_features_from_events(year: int) -> pd.DataFrame:
    """Extract goalkeeper features from StatsBomb events."""
    events = load_events(year)
    if events.empty:
        return pd.DataFrame()

    events["event_type"] = events["type"].apply(_get_event_name)
    events["team"] = events["team"].apply(normalize_team_name)

    # Identify goalkeepers from position
    gk_players = set()
    if "position" in events.columns:
        for _, row in events.drop_duplicates(["player", "team"]).iterrows():
            pos = _get_event_name(row.get("position", {}))
            if "Goalkeeper" in pos:
                gk_players.add((row["player"], row["team"]))

    shots = events[events["event_type"] == "Shot"].copy()
    shots["outcome"] = shots["shot_outcome"].apply(_get_event_name)
    shots["is_goal"] = (shots["outcome"] == "Goal").astype(int)
    shots["xg"] = pd.to_numeric(shots.get("shot_statsbomb_xg", 0), errors="coerce").fillna(0)

    records = []
    for player, team in gk_players:
        faced = shots[shots["team"] != team]  # shots against this GK's team
        # Better: shots where opposition scored against team
        team_shots_against = shots[
            shots["team"].apply(normalize_team_name) != team
        ]
        # Filter to matches this GK played - use all for simplicity
        ga = team_shots_against[team_shots_against["is_goal"] == 1]
        total_shots = len(team_shots_against)

        player_events = events[(events["player"] == player) & (events["team"] == team)]
        minutes = max(len(player_events) * 0.5, 90)

        goals_allowed = shots[
            (shots["team"] != team) & (shots["is_goal"] == 1)
        ].groupby("team").size().sum() if not shots.empty else 0

        # Simpler per-GK: count goals in their matches
        match_ids = player_events["match_id"].unique()
        match_shots = shots[shots["match_id"].isin(match_ids)]
        goals_in_matches = match_shots[
            (match_shots["is_goal"] == 1) &
            (match_shots["team"].apply(normalize_team_name) != team)
        ]
        ga_count = len(goals_in_matches)
        psxg = match_shots[match_shots["team"].apply(normalize_team_name) != team]["xg"].sum()
        saves = len(match_shots[
            (match_shots["team"].apply(normalize_team_name) != team) &
            (match_shots["outcome"] == "Saved")
        ])

        total_faced = len(match_shots[match_shots["team"].apply(normalize_team_name) != team])
        save_pct = saves / total_faced if total_faced > 0 else 0.5
        clean_sheets = 1.0 if ga_count == 0 else 0.0

        records.append({
            "player": player,
            "team": team,
            "save_pct": save_pct,
            "clean_sheet_pct": clean_sheets,
            "psxg_minus_ga": psxg - ga_count,
            "ga90": ga_count / (minutes / 90),
            "year": year,
        })

    return pd.DataFrame(records)


def build_playmaker_features_from_events(year: int) -> pd.DataFrame:
    """Extract playmaker features from StatsBomb pass events."""
    events = load_events(year)
    if events.empty:
        return pd.DataFrame()

    events["event_type"] = events["type"].apply(_get_event_name)
    events["team"] = events["team"].apply(normalize_team_name)
    passes = events[events["event_type"] == "Pass"].copy()

    if passes.empty:
        return pd.DataFrame()

    def is_key_pass(row):
        ret = _parse_json_col(row.get("pass_goal_assist", {}))
        shot = _parse_json_col(row.get("pass_shot_assist", {}))
        return 1 if ret or shot else 0

    passes["is_key_pass"] = passes.apply(is_key_pass, axis=1)
    passes["pass_length"] = pd.to_numeric(passes.get("pass_length", 0), errors="coerce").fillna(0)
    passes["is_progressive"] = (passes["pass_length"] >= 10).astype(int)

    player_mins = (
        events.groupby(["player", "team"])
        .size()
        .reset_index(name="event_count")
    )
    player_mins["minutes"] = (player_mins["event_count"] * 0.5).clip(lower=90)

    agg = passes.groupby(["player", "team"]).agg(
        key_passes_total=("is_key_pass", "sum"),
        progressive_passes_total=("is_progressive", "sum"),
        passes_total=("is_key_pass", "count"),
    ).reset_index()

    # xA from shot assists
    if "shot_statsbomb_xg" in passes.columns:
        passes["xa_val"] = pd.to_numeric(passes["shot_statsbomb_xg"], errors="coerce").fillna(0)
        xa_agg = passes.groupby(["player", "team"])["xa_val"].sum().reset_index(name="xa_total")
        agg = agg.merge(xa_agg, on=["player", "team"], how="left")
    else:
        agg["xa_total"] = agg["key_passes_total"] * 0.05

    agg = agg.merge(player_mins[["player", "team", "minutes"]], on=["player", "team"], how="left")
    agg["minutes"] = agg["minutes"].fillna(90)
    mins90 = agg["minutes"] / 90

    agg["key_passes_per90"] = agg["key_passes_total"] / mins90
    agg["progressive_passes_per90"] = agg["progressive_passes_total"] / mins90
    agg["xa_per90"] = agg["xa_total"] / mins90
    agg["pass_completion_pct"] = 0.75  # default when not available
    agg["year"] = year

    return agg
