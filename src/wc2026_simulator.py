# Copyright (c) 2026 Sasiru Virajith Kankanamge
# SPDX-License-Identifier: MIT

"""
FIFA World Cup 2026 Predictor
Built by: K. Sasiru Virajith
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DRAW_THRESHOLD,
    MODERN_TEAMS,
    N_SIMULATIONS_DEFAULT,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    ROUND_VARIANCE,
    WC2026_ALL_TEAMS,
    WC2026_GROUPS,
    WC2026_PLAYOFF_CANDIDATES,
)
from src.match_model import FEATURE_COLS, load_match_model
from src.labels import normalize_team_name

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn.base",
)

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CURRENT_YEAR = 2026

# Map any alias back to the official WC 2026 draw name
_WC_TEAM_LOOKUP = {
    normalize_team_name(team): team for team in WC2026_ALL_TEAMS
}

_STATIC_STATE = None


def to_wc_team(name: str) -> str:
    return _WC_TEAM_LOOKUP.get(normalize_team_name(name), name)


class TournamentState:

    def __init__(self):
        self.fifa_points: dict[str, float] = {}
        self.last5_form: dict[str, float] = {}
        self.penalty_win_rate: dict[str, float] = {}
        self.h2h: dict = {}
        self.achievements = None
        self._load_data()

    def _load_data(self):
        strength_path = PROCESSED_DIR / "team_strength_2026.csv"
        if strength_path.exists():
            strength = pd.read_csv(strength_path)
            teams = strength["team"].map(normalize_team_name)
            pts_col = "fifa_points" if "fifa_points" in strength.columns else "elo"
            self.fifa_points.update(dict(zip(teams, strength[pts_col].fillna(1500))))
            if "last5_win_rate" in strength.columns:
                self.last5_form.update(dict(zip(teams, strength["last5_win_rate"].fillna(0.5))))

        fifa_path = RAW_DIR / "fifa_latest_ranking.csv"
        if fifa_path.exists():
            fifa = pd.read_csv(fifa_path)
            for country, points in zip(fifa["country"], fifa["total_points"]):
                t = to_wc_team(country)
                self.fifa_points[t] = points
                if t not in self.last5_form:
                    self.last5_form[t] = 0.5

        ach_path = RAW_DIR / "team_achievements.csv"
        if ach_path.exists():
            ach = pd.read_csv(ach_path)
            ach["team"] = ach["team"].apply(to_wc_team)
            self.achievements = ach.set_index("team")

        pen_path = RAW_DIR / "shootouts.csv"
        if pen_path.exists():
            so = pd.read_csv(pen_path)
            so["winner"] = so["winner"].apply(normalize_team_name)
            so["home_team"] = so["home_team"].apply(normalize_team_name)
            so["away_team"] = so["away_team"].apply(normalize_team_name)
            wins = so["winner"].value_counts()
            games = pd.concat([so["home_team"], so["away_team"]]).value_counts()
            self.penalty_win_rate = (wins / games).fillna(0.5).to_dict()

        for playoff, candidates in WC2026_PLAYOFF_CANDIDATES.items():
            pts = [self.fifa_points.get(normalize_team_name(c), 1400) for c in candidates]
            forms = [self.last5_form.get(normalize_team_name(c), 0.5) for c in candidates]
            self.fifa_points[playoff] = float(np.mean(pts))
            self.last5_form[playoff] = float(np.mean(forms))

        for team in WC2026_ALL_TEAMS:
            t = normalize_team_name(team)
            if t not in self.fifa_points:
                self.fifa_points[t] = 1350
            if t not in self.last5_form:
                self.last5_form[t] = 0.35

        self._smooth_form_cache = {
            team: self.smooth_form(team) for team in self.fifa_points
        }

    def smooth_form(self, team: str, alpha: float = 0.65) -> float:
        cached = getattr(self, "_smooth_form_cache", None)
        if cached is not None and team in cached and alpha == 0.65:
            return cached[team]
        wr = self.last5_form.get(team, 0.5)
        return alpha * wr + (1 - alpha) * 0.5

    def modern_strength(self, team: str) -> float:
        score = (self.smooth_form(team) - 0.5) * 110
        score += MODERN_TEAMS.get(team, 0)

        if self.achievements is not None and team in self.achievements.index:
            row = self.achievements.loc[team]
            for col, weight, decay in [
                ("wc_last_semi_year", 22, 6), ("wc_last_final_year", 28, 6),
                ("wc_last_win_year", 32, 8), ("cont_last_semi_year", 14, 5),
                ("cont_last_final_year", 18, 5), ("cont_last_win_year", 22, 6),
            ]:
                val = row.get(col)
                if pd.notna(val):
                    score += weight * np.exp(-(CURRENT_YEAR - val) / decay)
        return score

    def h2h_get(self, a: str, b: str) -> dict:
        key = tuple(sorted([a, b]))
        return self.h2h.get(key, {"home_win_rate": 0.5, "draw_rate": 0.25, "matches_played": 0})


def get_tournament_state() -> TournamentState:
    global _STATIC_STATE
    if _STATIC_STATE is None:
        _STATIC_STATE = TournamentState()
    return _STATIC_STATE


@dataclass
class _SimContext:
    model: object | None
    feature_cols: list[str]
    col_idx: dict[str, int]
    state: TournamentState
    feature_buf: np.ndarray
    group_buf: np.ndarray


def _make_sim_context(model, feature_cols: list[str]) -> _SimContext:
    cols = feature_cols or FEATURE_COLS
    col_idx = {name: i for i, name in enumerate(cols)}
    n = len(cols)
    return _SimContext(
        model=model,
        feature_cols=cols,
        col_idx=col_idx,
        state=get_tournament_state(),
        feature_buf=np.zeros((1, n), dtype=np.float64),
        group_buf=np.zeros((6, n), dtype=np.float64),
    )


def _write_feature_row(
    buf: np.ndarray,
    row: int,
    col_idx: dict[str, int],
    features: dict[str, float],
) -> None:
    for name, value in features.items():
        idx = col_idx.get(name)
        if idx is not None:
            buf[row, idx] = value


def _match_features(
    state: TournamentState,
    team_a: str,
    team_b: str,
    fifa_diff: float,
) -> dict[str, float]:
    h2h = state.h2h_get(team_a, team_b)
    form_a = state.smooth_form(team_a)
    form_b = state.smooth_form(team_b)
    h2h_weight = min(1.0, h2h["matches_played"] / 10)
    return {
        "fifa_diff": fifa_diff,
        "elo_diff": fifa_diff * 0.95,
        "home_last5_win_rate": form_a,
        "away_last5_win_rate": form_b,
        "h2h_home_win_rate": h2h["home_win_rate"],
        "h2h_draw_rate": h2h["draw_rate"],
        "h2h_matches_played": h2h["matches_played"],
        "home_penalty_win_rate": state.penalty_win_rate.get(team_a, 0.5),
        "away_penalty_win_rate": state.penalty_win_rate.get(team_b, 0.5),
        "fifa_diff_x_home_form": fifa_diff * form_a,
        "fifa_diff_x_away_form": fifa_diff * form_b,
        "h2h_effective": h2h["home_win_rate"] * h2h_weight,
        "is_friendly": 0,
        "is_tournament": 1,
    }


def _base_fifa_diff(state: TournamentState, team_a: str, team_b: str) -> float:
    fifa_diff = state.fifa_points.get(team_a, 1500) - state.fifa_points.get(team_b, 1500)
    tactical_diff = MODERN_TEAMS.get(team_a, 0) - MODERN_TEAMS.get(team_b, 0)
    return fifa_diff + tactical_diff * 12


def _outcome_from_probs(
    p_away: float,
    p_draw: float,
    p_home: float,
    rng: np.random.Generator,
) -> str:
    if p_draw > DRAW_THRESHOLD:
        return "D"
    p_nodraw_home = p_home / (p_home + p_away)
    return "A" if rng.random() < p_nodraw_home else "B"


def _apply_group_result(table: dict, t1: str, t2: str, result: str) -> None:
    if result == "A":
        table[t1]["points"] += 3
        table[t1]["gd"] += 1
        table[t1]["gf"] += 1
        table[t2]["gd"] -= 1
    elif result == "B":
        table[t2]["points"] += 3
        table[t2]["gd"] += 1
        table[t2]["gf"] += 1
        table[t1]["gd"] -= 1
    else:
        table[t1]["points"] += 1
        table[t2]["points"] += 1


def _predict_match_probs(ctx: _SimContext, features: dict) -> np.ndarray:
    _write_feature_row(ctx.feature_buf, 0, ctx.col_idx, features)
    return ctx.model.predict_proba(ctx.feature_buf)[0]


def simulate_match(
    team_a: str,
    team_b: str,
    ctx: _SimContext,
    rng: np.random.Generator,
    round_name: str = "group",
) -> str:
    state = ctx.state
    fifa_diff = _base_fifa_diff(state, team_a, team_b)
    fifa_diff += rng.normal(0, ROUND_VARIANCE.get(round_name, 20))
    features = _match_features(state, team_a, team_b, fifa_diff)

    if ctx.model is not None:
        probs = _predict_match_probs(ctx, features)
        p_away, p_draw, p_home = probs[0], probs[1], probs[2]
    else:
        p_home = 1 / (1 + 10 ** (-fifa_diff / 400))
        p_away = 1 - p_home
        p_draw = 0.25
        total = p_home + p_away + p_draw
        p_away, p_draw, p_home = p_away / total, p_draw / total, p_home / total

    return _outcome_from_probs(p_away, p_draw, p_home, rng)


def simulate_penalty(team_a: str, team_b: str, state: TournamentState, rng) -> str:
    pa = state.penalty_win_rate.get(team_a, 0.5)
    pb = state.penalty_win_rate.get(team_b, 0.5)
    return team_a if rng.random() < pa / (pa + pb) else team_b


def _sort_group_table(table: dict) -> list[tuple[str, dict]]:
    return sorted(
        table.items(),
        key=lambda item: (item[1]["points"], item[1]["gd"], item[1]["gf"]),
        reverse=True,
    )


def simulate_group(
    teams: list,
    ctx: _SimContext,
    rng: np.random.Generator,
) -> list[tuple[str, dict]]:
    table = {t: {"points": 0, "gd": 0, "gf": 0} for t in teams}
    pairs: list[tuple[str, str]] = []

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            t1, t2 = teams[i], teams[j]
            if rng.random() < 0.5:
                t1, t2 = t2, t1
            pairs.append((t1, t2))

    if ctx.model is None:
        for t1, t2 in pairs:
            result = simulate_match(t1, t2, ctx, rng, "group")
            _apply_group_result(table, t1, t2, result)
        return _sort_group_table(table)

    state = ctx.state
    group_variance = ROUND_VARIANCE.get("group", 20)
    n_matches = len(pairs)
    buf = ctx.group_buf[:n_matches]

    for row, (t1, t2) in enumerate(pairs):
        fifa_diff = _base_fifa_diff(state, t1, t2)
        fifa_diff += rng.normal(0, group_variance)
        _write_feature_row(
            buf, row, ctx.col_idx,
            _match_features(state, t1, t2, fifa_diff),
        )

    all_probs = ctx.model.predict_proba(buf)

    for row, (t1, t2) in enumerate(pairs):
        p_away, p_draw, p_home = all_probs[row]
        result = _outcome_from_probs(p_away, p_draw, p_home, rng)
        _apply_group_result(table, t1, t2, result)

    return _sort_group_table(table)


def rank_third_places(third_places: list) -> list:
    return sorted(
        third_places,
        key=lambda x: (x["points"], x["gd"], x["gf"]),
        reverse=True,
    )


def simulate_knockout_match(
    team_a: str,
    team_b: str,
    ctx: _SimContext,
    rng: np.random.Generator,
    round_name: str,
) -> str:
    result = simulate_match(team_a, team_b, ctx, rng, round_name)
    if result == "D":
        return simulate_penalty(team_a, team_b, ctx.state, rng)
    return team_a if result == "A" else team_b


def simulate_one_tournament(
    ctx: _SimContext,
    rng: np.random.Generator,
) -> tuple[str, dict[str, dict]]:
    state = ctx.state
    third_places = []
    qualified = []
    team_groups = {
        team: grp for grp, teams in WC2026_GROUPS.items() for team in teams
    }
    group_stats: dict[str, dict] = {
        team: {
            "points": 0.0,
            "qualified": False,
            "top_group": False,
            "group": team_groups.get(team),
        }
        for team in WC2026_ALL_TEAMS
    }

    for group_name, teams in WC2026_GROUPS.items():
        ranked = simulate_group(teams, ctx, rng)
        group_winner = ranked[0][0]
        group_stats[group_winner]["top_group"] = True
        for team, stats in ranked:
            group_stats[team]["points"] = float(stats["points"])
            group_stats[team]["group"] = group_name

        qualified.append(ranked[0][0])
        qualified.append(ranked[1][0])
        third = ranked[2][1]
        third_places.append({
            "team": ranked[2][0],
            "points": third["points"],
            "gd": third["gd"],
            "gf": third["gf"],
            "group": group_name,
        })

    ranked_third = rank_third_places(third_places)
    qualified += [r["team"] for r in ranked_third[:8]]
    qualified = list(dict.fromkeys(qualified))
    for team in qualified:
        group_stats[team]["qualified"] = True

    qualified.sort(key=lambda t: state.fifa_points.get(t, 0), reverse=True)

    rounds = ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final"]
    current = qualified[:32] if len(qualified) >= 32 else qualified

    for round_name in rounds:
        if len(current) <= 1:
            break
        rng.shuffle(current)
        next_round = []
        for i in range(0, len(current) - 1, 2):
            w = simulate_knockout_match(
                current[i], current[i + 1], ctx, rng, round_name
            )
            next_round.append(w)
        if len(current) % 2 == 1:
            next_round.append(current[-1])
        current = next_round

    champion = current[0] if current else qualified[0]
    return champion, group_stats


def _empty_team_counters() -> tuple[dict, dict, dict, dict]:
    return (
        {team: 0 for team in WC2026_ALL_TEAMS},
        {team: 0 for team in WC2026_ALL_TEAMS},
        {team: 0 for team in WC2026_ALL_TEAMS},
        {team: 0.0 for team in WC2026_ALL_TEAMS},
    )


def _accumulate_tournament(
    win_counts: dict,
    qualify_counts: dict,
    top_group_counts: dict,
    points_total: dict,
    champion: str,
    group_stats: dict[str, dict],
) -> None:
    champion = to_wc_team(champion)
    win_counts[champion] = win_counts.get(champion, 0) + 1
    for team, stats in group_stats.items():
        points_total[team] += stats["points"]
        if stats["qualified"]:
            qualify_counts[team] += 1
        if stats["top_group"]:
            top_group_counts[team] += 1


def _run_simulation_batch(n_sims: int, seed: int) -> tuple[dict, dict, dict, dict]:
    model, feature_cols = load_match_model()
    ctx = _make_sim_context(model, feature_cols)
    rng = np.random.default_rng(seed)
    win_counts, qualify_counts, top_group_counts, points_total = _empty_team_counters()

    for _ in range(n_sims):
        champion, group_stats = simulate_one_tournament(ctx, rng)
        _accumulate_tournament(
            win_counts, qualify_counts, top_group_counts, points_total,
            champion, group_stats,
        )

    return win_counts, qualify_counts, top_group_counts, points_total


def _merge_counter_dicts(target: dict, source: dict) -> None:
    for key, value in source.items():
        target[key] += value


def _split_simulation_batches(
    n_simulations: int,
    n_workers: int,
    seed: int,
) -> list[tuple[int, int]]:
    base, remainder = divmod(n_simulations, n_workers)
    batches = []
    for worker_id in range(n_workers):
        count = base + (1 if worker_id < remainder else 0)
        if count > 0:
            batches.append((count, seed + worker_id * 1_000_003))
    return batches


def _write_group_simulation_csv(
    qualify_counts: dict[str, int],
    top_group_counts: dict[str, int],
    points_total: dict[str, float],
    n_simulations: int,
) -> pd.DataFrame:
    team_groups = {
        team: grp for grp, teams in WC2026_GROUPS.items() for team in teams
    }
    rows = []
    for team in WC2026_ALL_TEAMS:
        rows.append({
            "team": team,
            "group": team_groups.get(team),
            "p_qualify": qualify_counts[team] / n_simulations,
            "p_top_group": top_group_counts[team] / n_simulations,
            "expected_points": points_total[team] / n_simulations,
            "year": 2026,
        })
    df = pd.DataFrame(rows).sort_values("p_qualify", ascending=False)
    save_path = OUTPUTS_DIR / "group_simulation_2026.csv"
    df.to_csv(save_path, index=False)
    print(f"  [OK] Group-stage simulation -> {save_path}")
    return df


def _default_worker_count() -> int:
    # Cap at 4 by default — safer on laptops with limited RAM; override with --workers
    return max(1, min(os.cpu_count() or 1, 4))


def run_wc2026_simulation(
    n_simulations: int = N_SIMULATIONS_DEFAULT,
    seed: int = 42,
    n_workers: int | None = None,
) -> pd.DataFrame:
    workers = n_workers if n_workers is not None else _default_worker_count()
    workers = max(1, min(workers, n_simulations))
    mode = f"{workers} worker{'s' if workers > 1 else ''}"
    print(f"Running {n_simulations} WC 2026 tournament simulations ({mode})...")
    t0 = time.perf_counter()

    model, feature_cols = load_match_model()
    if model is None:
        print("  [WARN] No match model found - using ELO fallback")

    win_counts, qualify_counts, top_group_counts, points_total = _empty_team_counters()

    if workers == 1:
        ctx = _make_sim_context(model, feature_cols)
        rng = np.random.default_rng(seed)
        for i in range(n_simulations):
            if (i + 1) % 500 == 0:
                print(f"  ... {i + 1}/{n_simulations}")
            champion, group_stats = simulate_one_tournament(ctx, rng)
            _accumulate_tournament(
                win_counts, qualify_counts, top_group_counts, points_total,
                champion, group_stats,
            )
    else:
        batches = _split_simulation_batches(n_simulations, workers, seed)
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_simulation_batch, count, batch_seed): count
                for count, batch_seed in batches
            }
            for future in as_completed(futures):
                batch_wins, batch_qualify, batch_top, batch_points = future.result()
                _merge_counter_dicts(win_counts, batch_wins)
                _merge_counter_dicts(qualify_counts, batch_qualify)
                _merge_counter_dicts(top_group_counts, batch_top)
                _merge_counter_dicts(points_total, batch_points)
                completed += futures[future]
                print(f"  ... {completed}/{n_simulations}")

    elapsed = time.perf_counter() - t0
    print(f"  Simulations finished in {elapsed:.1f}s ({n_simulations / elapsed:.0f} sims/s)")

    _write_group_simulation_csv(
        qualify_counts, top_group_counts, points_total, n_simulations,
    )

    results = pd.DataFrame([
        {
            "team": team,
            "win_probability": count / n_simulations,
            "probability_percent": round(count / n_simulations * 100, 2),
            "simulations": count,
        }
        for team, count in win_counts.items()
    ]).sort_values("win_probability", ascending=False).reset_index(drop=True)

    save_path = OUTPUTS_DIR / "wc2026_champion_probabilities.csv"
    results.to_csv(save_path, index=False)
    print(f"  [OK] Champion probabilities -> {save_path}")
    print(f"  Top 5: {results.head(5)[['team', 'probability_percent']].to_dict('records')}")
    return results


if __name__ == "__main__":
    run_wc2026_simulation(n_simulations=N_SIMULATIONS_DEFAULT)
