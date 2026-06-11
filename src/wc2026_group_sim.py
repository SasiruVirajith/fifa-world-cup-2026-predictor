# Copyright (c) 2026 Sasiru Virajith Kankanamge
# SPDX-License-Identifier: MIT

"""
FIFA World Cup 2026 Predictor
Built by: K. Sasiru Virajith
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, WC2026_GROUPS
from src.match_model import FEATURE_COLS, load_match_model
from src.wc2026_simulator import TournamentState, _make_sim_context, simulate_match

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def get_wc2026_group_fixtures() -> pd.DataFrame:
    records = []
    for group, teams in WC2026_GROUPS.items():
        for home, away in itertools.combinations(teams, 2):
            records.append({"group": group, "home_team": home, "away_team": away})
    return pd.DataFrame(records)


def _team_strength(team: str, state: TournamentState) -> float:
    return state.fifa_points.get(team, 1350) + state.modern_strength(team) * 8


def run_wc2026_group_simulation(n_simulations: int = 2000, seed: int = 42) -> pd.DataFrame:
    fixtures = get_wc2026_group_fixtures()
    if fixtures.empty:
        return pd.DataFrame()

    model, feature_cols = load_match_model()
    feature_cols = feature_cols or FEATURE_COLS
    ctx = _make_sim_context(model, feature_cols)
    state = ctx.state
    rng = np.random.default_rng(seed)

    team_groups = {team: grp for grp, teams in WC2026_GROUPS.items() for team in teams}
    all_teams = list(team_groups.keys())
    qualify_counts = {t: 0 for t in all_teams}
    top_group_counts = {t: 0 for t in all_teams}
    points_total = {t: 0.0 for t in all_teams}

    for _ in range(n_simulations):
        group_points = {t: 0.0 for t in all_teams}
        for _, fix in fixtures.iterrows():
            home, away = fix["home_team"], fix["away_team"]
            if model is not None:
                result = simulate_match(home, away, ctx, rng, round_name="group")
                if result == "A":
                    group_points[home] += 3
                elif result == "D":
                    group_points[home] += 1
                    group_points[away] += 1
                else:
                    group_points[away] += 3
            else:
                diff = _team_strength(home, state) - _team_strength(away, state)
                p_home = 1 / (1 + 10 ** (-diff / 400))
                p_draw = 0.26
                p_away = 1 - p_home - p_draw
                total = p_home + p_draw + p_away
                outcome = rng.choice(["H", "D", "A"], p=[p_home / total, p_draw / total, p_away / total])
                if outcome == "H":
                    group_points[home] += 3
                elif outcome == "D":
                    group_points[home] += 1
                    group_points[away] += 1
                else:
                    group_points[away] += 3

        for group in WC2026_GROUPS:
            teams = WC2026_GROUPS[group]
            ranked = sorted(teams, key=lambda t: group_points[t], reverse=True)
            for t in ranked[:2]:
                qualify_counts[t] += 1
            top_group_counts[ranked[0]] += 1

        for t in all_teams:
            points_total[t] += group_points[t]

    rows = []
    for team in all_teams:
        rows.append({
            "team": team,
            "group": team_groups[team],
            "p_qualify": qualify_counts[team] / n_simulations,
            "p_top_group": top_group_counts[team] / n_simulations,
            "expected_points": points_total[team] / n_simulations,
            "year": 2026,
        })

    df = pd.DataFrame(rows).sort_values("p_qualify", ascending=False)
    save_path = OUTPUTS_DIR / "group_simulation_2026.csv"
    df.to_csv(save_path, index=False)
    print(f"  [OK] WC 2026 group simulation -> {save_path}")
    return df
