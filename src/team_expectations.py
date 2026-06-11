# Copyright (c) 2026 Sasiru Virajith Kankanamge
# SPDX-License-Identifier: MIT

"""
FIFA World Cup 2026 Predictor
Built by: K. Sasiru Virajith
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR, WC2026_ALL_TEAMS, WC2026_GROUPS
from src.labels import normalize_team_name


def _group_for_team(team: str) -> str | None:
    for letter, teams in WC2026_GROUPS.items():
        if team in teams:
            return letter
    return None


def _load_fifa_ranks() -> pd.DataFrame:
    path = RAW_DIR / "fifa_latest_ranking.csv"
    if not path.exists():
        return pd.DataFrame({"team": WC2026_ALL_TEAMS, "fifa_rank": 50, "fifa_points": 1500.0})
    df = pd.read_csv(path)
    out = df[["country", "rank", "total_points"]].rename(
        columns={"country": "team", "rank": "fifa_rank", "total_points": "fifa_points"}
    )
    wc = pd.DataFrame({"team": WC2026_ALL_TEAMS})
    return wc.merge(out, on="team", how="left").fillna({"fifa_rank": 80, "fifa_points": 1400.0})


def _load_group_sim() -> pd.DataFrame:
    path = OUTPUTS_DIR / "group_simulation_2026.csv"
    if not path.exists():
        base = pd.DataFrame({"team": WC2026_ALL_TEAMS})
        base["group"] = base["team"].map(_group_for_team)
        base["p_qualify"] = 0.25
        base["p_top_group"] = 0.05
        base["expected_points"] = 4.0
        return base
    return pd.read_csv(path)


def _load_champion_probs() -> pd.DataFrame:
    path = OUTPUTS_DIR / "wc2026_champion_probabilities.csv"
    if not path.exists():
        return pd.DataFrame({"team": WC2026_ALL_TEAMS, "team_win_prob": 0.02})
    df = pd.read_csv(path)
    prob_col = next(
        (c for c in ("win_probability", "champion_probability", "probability") if c in df.columns),
        None,
    )
    if prob_col is None:
        cols = [c for c in df.columns if c != "team"]
        prob_col = cols[0] if cols else None
    if prob_col is None:
        return pd.DataFrame({"team": WC2026_ALL_TEAMS, "team_win_prob": 0.02})
    return df[["team", prob_col]].rename(columns={prob_col: "team_win_prob"})


def _load_team_strength() -> pd.DataFrame:
    path = PROCESSED_DIR / "team_strength_2026.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    wc = set(WC2026_ALL_TEAMS)
    return df[df["team"].isin(wc)][["team", "last5_win_rate", "last20_win_rate"]].copy()


def _group_opponent_fifa(team: str, fifa: pd.DataFrame) -> float:
    group = _group_for_team(team)
    if not group:
        return 1600.0
    opponents = [t for t in WC2026_GROUPS[group] if t != team]
    pts = fifa.set_index("team")["fifa_points"]
    vals = [float(pts.get(o, 1500.0)) for o in opponents]
    return float(np.mean(vals)) if vals else 1600.0


def _safe_max(series: pd.Series, floor: float = 1.0) -> float:
    val = float(series.max()) if len(series) else floor
    return max(val, floor)


def load_team_tournament_context() -> pd.DataFrame:
    fifa = _load_fifa_ranks()
    group_sim = _load_group_sim()
    champs = _load_champion_probs()
    strength = _load_team_strength()

    df = pd.DataFrame({"team": WC2026_ALL_TEAMS})
    df["group"] = df["team"].map(_group_for_team)
    df = df.merge(group_sim[["team", "p_qualify", "p_top_group", "expected_points"]], on="team", how="left")
    df = df.merge(champs, on="team", how="left")
    df = df.merge(fifa[["team", "fifa_rank", "fifa_points"]], on="team", how="left")
    if not strength.empty:
        df = df.merge(strength, on="team", how="left")

    df["p_qualify"] = df["p_qualify"].fillna(0.20)
    df["p_top_group"] = df["p_top_group"].fillna(0.05)
    df["expected_points"] = df["expected_points"].fillna(4.0)
    df["team_win_prob"] = df["team_win_prob"].fillna(0.001)
    df["fifa_rank"] = df["fifa_rank"].fillna(80)
    df["last5_win_rate"] = df.get("last5_win_rate", pd.Series(0.5, index=df.index)).fillna(0.5)
    df["last20_win_rate"] = df.get("last20_win_rate", pd.Series(0.5, index=df.index)).fillna(0.5)

    df["group_opponent_fifa"] = df["team"].apply(lambda t: _group_opponent_fifa(t, fifa))
    df["group_difficulty"] = df["group_opponent_fifa"] / df["fifa_points"].clip(lower=800)

    # Expected knockout matches beyond group stage (3 group games always played)
    df["expected_knockout_matches"] = (
        1.2 * df["p_qualify"]
        + 2.5 * df["team_win_prob"].clip(upper=0.30)
        + 0.8 * df["p_top_group"]
    ).clip(0, 7)

    # Multiplier for player awards: team must qualify and progress for goals / POT
    qualify = df["p_qualify"].clip(0.02, 1.0)
    win = df["team_win_prob"].clip(0, 1.0)
    form = df["last5_win_rate"].clip(0, 1.0)
    knockout = df["expected_knockout_matches"] / _safe_max(df["expected_knockout_matches"], 1.0)

    df["progression_factor"] = (
        0.40 * qualify
        + 0.30 * (win / _safe_max(win, 0.001))
        + 0.20 * knockout
        + 0.10 * form
    ).clip(0.03, 1.0)

    df["expected_tournament_matches"] = 3.0 + df["expected_knockout_matches"]
    return df.sort_values("progression_factor", ascending=False).reset_index(drop=True)


def _sim_performance_score(ctx: pd.DataFrame) -> pd.Series:
    q = ctx["p_qualify"]
    w = ctx["team_win_prob"]
    ep = ctx["expected_points"] / _safe_max(ctx["expected_points"], 1.0)
    return (0.45 * q + 0.35 * (w / _safe_max(w, 0.001)) + 0.20 * ep).clip(0, 1)


def _fifa_expected_score(ctx: pd.DataFrame) -> pd.Series:
    rank = ctx["fifa_rank"].astype(float)
    inv = 1.0 / rank
    mn, mx = inv.min(), inv.max()
    if mx == mn:
        return pd.Series(0.5, index=ctx.index)
    return (inv - mn) / (mx - mn)


def compute_team_surprise_disappointment() -> pd.DataFrame:
    ctx = load_team_tournament_context()
    ctx["sim_score"] = _sim_performance_score(ctx)
    ctx["fifa_expected"] = _fifa_expected_score(ctx)
    ctx["surprise_delta"] = (ctx["sim_score"] - ctx["fifa_expected"]).round(4)
    ctx["year"] = 2026
    return ctx.sort_values("surprise_delta", ascending=False).reset_index(drop=True)


# FIFA rank cutoffs among WC 2026 participants
BIG_TEAM_MAX_RANK = 20
UNDERDOG_MIN_RANK = 28


def compute_big_team_upsets() -> pd.DataFrame:
    ctx = compute_team_surprise_disappointment()
    big = ctx[ctx["fifa_rank"] <= BIG_TEAM_MAX_RANK].copy()

    # Win-probability rank among all 48 WC teams (1 = highest sim win %)
    all_ctx = ctx.copy()
    all_ctx["sim_win_rank"] = all_ctx["team_win_prob"].rank(ascending=False, method="min")
    big = big.merge(all_ctx[["team", "sim_win_rank"]], on="team", how="left")

    big["rank_gap"] = (big["sim_win_rank"] - big["fifa_rank"]).round(1)
    # Composite upset: rank gap + weak knockout path for a big nation
    rank_gap_norm = big["rank_gap"].clip(lower=0) / max(big["rank_gap"].clip(lower=0).max(), 1)
    qualify_shortfall = (1.0 - big["p_qualify"]).clip(lower=0) * (1.0 / big["fifa_rank"].clip(lower=1))
    win_shortfall = (
        (big["fifa_expected"] - big["sim_score"]).clip(lower=0)
        * (1.0 / big["fifa_rank"].clip(lower=1))
    )
    big["upset_score"] = (
        0.50 * rank_gap_norm
        + 0.30 * qualify_shortfall / max(qualify_shortfall.max(), 1e-9)
        + 0.20 * win_shortfall / max(win_shortfall.max(), 1e-9)
    ).round(4)

    # Still projected to go deep  -  not an "upset" in the football sense
    big = big[~((big["p_qualify"] >= 0.95) & (big["team_win_prob"] >= 0.10))]

    # Must be rated below FIFA rank in sims, or at real risk of missing knockouts
    big = big[(big["rank_gap"] > 0) | (big["p_qualify"] < 0.85)]
    return big.sort_values(["upset_score", "rank_gap"], ascending=False).reset_index(drop=True)


def compute_underdog_surprises() -> pd.DataFrame:
    ctx = compute_team_surprise_disappointment()
    dogs = ctx[ctx["fifa_rank"] >= UNDERDOG_MIN_RANK].copy()
    dogs["surprise_score"] = (dogs["sim_score"] - dogs["fifa_expected"]).round(4)
    return dogs.sort_values("surprise_score", ascending=False).reset_index(drop=True)


def save_team_expectations() -> pd.DataFrame:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df = compute_team_surprise_disappointment()
    ctx_path = OUTPUTS_DIR / "team_tournament_context_2026.csv"
    df.to_csv(ctx_path, index=False)
    return df
