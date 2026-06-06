"""
player_international.py
───────────────────────
Weighted international player stats from martj42 results + goalscorers.
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    PLAYER_INTL_START,
    PLAYER_INTL_TRAIN_START,
    RAW_DIR,
    TOURNAMENT_TIER_WEIGHTS,
    WC2026_ALL_TEAMS,
    WC_START_DATES,
)
from src.labels import normalize_player_name, normalize_team_name


def classify_tournament(tournament: str) -> str:
    """Map martj42 tournament label to a weight tier."""
    t = str(tournament).lower()
    if "world cup" in t and "qualif" not in t:
        return "world_cup"
    if "qualif" in t:
        return "qualifier"
    if any(
        kw in t
        for kw in (
            "euro",
            "copa am",
            "africa cup",
            "asian cup",
            "gold cup",
            "nations league",
        )
    ):
        if "qualif" in t:
            return "qualifier"
        if "nations league" in t:
            return "nations_league"
        return "major_continental"
    if "friendly" in t:
        return "friendly"
    return "other"


def tournament_weight(tournament: str) -> float:
    tier = classify_tournament(tournament)
    return TOURNAMENT_TIER_WEIGHTS.get(tier, 0.5)


def build_international_player_stats(
    teams: list[str] | None = None,
    start_date: str = PLAYER_INTL_START,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Aggregate scorer stats per (player, national team) with tournament weights.

    Returns columns used by striker / playmaker / POT feature builders.
    """
    teams = set(teams or WC2026_ALL_TEAMS)
    results_path = RAW_DIR / "results.csv"
    scorers_path = RAW_DIR / "goalscorers.csv"
    if not results_path.exists() or not scorers_path.exists():
        return pd.DataFrame()

    results = pd.read_csv(results_path)
    scorers = pd.read_csv(scorers_path)
    results["date"] = pd.to_datetime(results["date"])
    scorers["date"] = pd.to_datetime(scorers["date"])

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else results["date"].max()
    results = results[(results["date"] >= start) & (results["date"] <= end)].copy()
    scorers = scorers[(scorers["date"] >= start) & (scorers["date"] <= end)].copy()

    results["home_team"] = results["home_team"].apply(normalize_team_name)
    results["away_team"] = results["away_team"].apply(normalize_team_name)
    scorers["home_team"] = scorers["home_team"].apply(normalize_team_name)
    scorers["away_team"] = scorers["away_team"].apply(normalize_team_name)
    scorers["team"] = scorers["team"].apply(normalize_team_name)
    scorers["scorer"] = scorers["scorer"].astype(str).str.strip()

    merged = scorers.merge(
        results[["date", "home_team", "away_team", "tournament"]],
        on=["date", "home_team", "away_team"],
        how="left",
    )
    merged = merged[merged["team"].isin(teams)].copy()
    if merged.empty:
        return pd.DataFrame()

    merged["player"] = merged["scorer"]
    merged["own_goal"] = merged["own_goal"].astype(str).str.upper().isin(["TRUE", "1", "T", "YES"])
    merged = merged[~merged["own_goal"]].copy()
    merged["tier"] = merged["tournament"].apply(classify_tournament)
    merged["weight"] = merged["tournament"].apply(tournament_weight)
    merged["weighted_goal"] = merged["weight"]
    merged["is_penalty"] = merged["penalty"].astype(str).str.upper().isin(["TRUE", "1", "T", "YES"])
    merged["is_major"] = merged["tier"].isin(["world_cup", "major_continental"]).astype(int)

    agg = merged.groupby(["player", "team"], as_index=False).agg(
        intl_goals=("player", "count"),
        intl_weighted_goals=("weighted_goal", "sum"),
        intl_penalty_goals=("is_penalty", "sum"),
        intl_major_goals=("is_major", "sum"),
        intl_scoring_matches=("date", "nunique"),
        intl_last_goal_date=("date", "max"),
    )

    # Matches played per nation in window (team-level, attached to scorers + GKs later)
    team_matches = pd.concat(
        [
            results[["date", "home_team"]].rename(columns={"home_team": "team"}),
            results[["date", "away_team"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    )
    team_matches = team_matches[team_matches["team"].isin(teams)]
    team_match_counts = team_matches.groupby("team")["date"].nunique().to_dict()

    agg["intl_goal_rate"] = agg["intl_goals"] / agg["intl_scoring_matches"].clip(lower=1)
    agg["team_intl_matches"] = agg["team"].map(team_match_counts).fillna(0)
    agg["intl_goals_per_team_match"] = agg["intl_goals"] / agg["team_intl_matches"].replace(0, pd.NA)
    agg["intl_goals_per_team_match"] = agg["intl_goals_per_team_match"].fillna(0)
    agg["player_norm"] = agg["player"].apply(normalize_player_name)
    return agg.sort_values("intl_weighted_goals", ascending=False).reset_index(drop=True)


def build_international_team_defense(
    teams: list[str] | None = None,
    start_date: str = PLAYER_INTL_START,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Team defensive form from martj42 for Golden Glove context."""
    teams = set(teams or WC2026_ALL_TEAMS)
    results_path = RAW_DIR / "results.csv"
    if not results_path.exists():
        return pd.DataFrame()

    results = pd.read_csv(results_path)
    results["date"] = pd.to_datetime(results["date"])
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else results["date"].max()
    results = results[(results["date"] >= start) & (results["date"] <= end)].copy()
    results["home_team"] = results["home_team"].apply(normalize_team_name)
    results["away_team"] = results["away_team"].apply(normalize_team_name)

    records = []
    for team in teams:
        home = results[results["home_team"] == team]
        away = results[results["away_team"] == team]
        goals_against = pd.concat(
            [home["away_score"], away["home_score"]], ignore_index=True
        )
        clean_sheets = int((goals_against == 0).sum())
        matches = len(goals_against)
        records.append({
            "team": team,
            "intl_matches": matches,
            "intl_goals_conceded": float(goals_against.sum()),
            "intl_ga90": float(goals_against.mean()) if matches else 2.0,
            "intl_clean_sheet_pct": clean_sheets / matches if matches else 0.0,
        })
    return pd.DataFrame(records)


def international_stats_for_year(year: int, teams: list[str] | None = None) -> pd.DataFrame:
    """International window ending at WC kickoff for backtest enrichment."""
    start = PLAYER_INTL_TRAIN_START if year <= 2022 else PLAYER_INTL_START
    end = WC_START_DATES.get(year, f"{year}-06-11")
    return build_international_player_stats(teams=teams, start_date=start, end_date=end)
