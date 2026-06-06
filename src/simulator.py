"""
simulator.py
────────────
Monte Carlo group stage simulator using ELO-based match outcomes.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import RAW_DIR, OUTPUTS_DIR
from src.features import calculate_elo_ratings
from src.labels import normalize_team_name

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

POINTS_WIN = 3
POINTS_DRAW = 1


def elo_win_probability(elo_a: float, elo_b: float) -> float:
    """Probability team A wins (excluding draw adjustment)."""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def simulate_match(elo_home: float, elo_away: float, rng: np.random.Generator) -> tuple:
    """
    Simulate a single match outcome.
    Returns (home_points, away_points).
    """
    p_home = elo_win_probability(elo_home, elo_away)
    p_away = elo_win_probability(elo_away, elo_home)
    p_draw = max(0.15, 1 - p_home - p_away + 0.25)  # base draw rate
    total = p_home + p_draw + p_away
    p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

    outcome = rng.choice(["home", "draw", "away"], p=[p_home, p_draw, p_away])
    if outcome == "home":
        return POINTS_WIN, 0
    if outcome == "draw":
        return POINTS_DRAW, POINTS_DRAW
    return 0, POINTS_WIN


def get_group_fixtures(year: int = 2022) -> pd.DataFrame:
    """
    Build group-stage fixtures from StatsBomb match data.
    Returns DataFrame with [group, home_team, away_team].
    """
    matches_path = RAW_DIR / f"statsbomb_matches_wc{year}.csv"
    if not matches_path.exists():
        return pd.DataFrame()

    matches = pd.read_csv(matches_path)
    stage_col = "competition_stage" if "competition_stage" in matches.columns else "stage"

    if stage_col in matches.columns:
        stage_mask = matches[stage_col].astype(str).str.contains("Group", case=False, na=False)
        group_matches = matches[stage_mask].copy()
    else:
        group_matches = pd.DataFrame()

    if group_matches.empty:
        if stage_col in matches.columns:
            knockout_mask = matches[stage_col].astype(str).str.contains(
                "Round of|Quarter|Semi|Final", case=False, na=False
            )
            group_matches = matches[~knockout_mask].copy()
        else:
            group_matches = matches.copy()

    records = []
    for _, m in group_matches.iterrows():
        # Derive pseudo-group from match_week when group letter unavailable
        group = m.get("group", f"W{m.get('match_week', 0)}")
        records.append({
            "group": str(group),
            "home_team": normalize_team_name(m["home_team"]),
            "away_team": normalize_team_name(m["away_team"]),
        })

    return pd.DataFrame(records)


_ELO_CACHE = None


def get_elo_ratings() -> dict:
    """Load current ELO ratings from international results (cached)."""
    global _ELO_CACHE
    if _ELO_CACHE is not None:
        return _ELO_CACHE
    results_path = RAW_DIR / "international_results.csv"
    if not results_path.exists():
        return {}
    results = pd.read_csv(results_path)
    results["date"] = pd.to_datetime(results["date"])
    _ELO_CACHE = calculate_elo_ratings(results)
    return _ELO_CACHE


def run_group_simulation(
    year: int = 2022,
    n_simulations: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation of group stage.

    Returns DataFrame with [team, group, p_qualify, expected_points, p_top_group].
    """
    fixtures = get_group_fixtures(year)
    if fixtures.empty:
        return pd.DataFrame()

    elo = get_elo_ratings()
    rng = np.random.default_rng(seed)

    # Teams per group
    groups = fixtures.groupby("group")
    team_groups = {}
    for grp, grp_fixtures in groups:
        teams = set(grp_fixtures["home_team"]) | set(grp_fixtures["away_team"])
        for t in teams:
            team_groups[t] = grp

    all_teams = list(team_groups.keys())
    qualify_counts = {t: 0 for t in all_teams}
    top_group_counts = {t: 0 for t in all_teams}
    points_total = {t: 0 for t in all_teams}

    for _ in range(n_simulations):
        group_points = {t: 0 for t in all_teams}

        for _, fix in fixtures.iterrows():
            home, away = fix["home_team"], fix["away_team"]
            elo_h = elo.get(home, 1500)
            elo_a = elo.get(away, 1500)
            pts_h, pts_a = simulate_match(elo_h, elo_a, rng)
            group_points[home] += pts_h
            group_points[away] += pts_a

        for grp, grp_fixtures in groups:
            teams = list(set(grp_fixtures["home_team"]) | set(grp_fixtures["away_team"]))
            ranked = sorted(teams, key=lambda t: group_points[t], reverse=True)
            # Top 2 qualify (2022 format)
            for t in ranked[:2]:
                qualify_counts[t] += 1
            top_group_counts[ranked[0]] += 1

        for t in all_teams:
            points_total[t] += group_points[t]

    results = []
    for team in all_teams:
        results.append({
            "team": team,
            "group": team_groups[team],
            "p_qualify": qualify_counts[team] / n_simulations,
            "p_top_group": top_group_counts[team] / n_simulations,
            "expected_points": points_total[team] / n_simulations,
            "year": year,
        })

    df = pd.DataFrame(results).sort_values("p_qualify", ascending=False)
    save_path = OUTPUTS_DIR / f"group_simulation_{year}.csv"
    df.to_csv(save_path, index=False)
    return df
