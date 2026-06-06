"""
wc2026_simulator.py
───────────────────
Full FIFA World Cup 2026 Monte Carlo tournament simulation.

Format (official 2026):
  - 12 groups of 4 (round robin, neutral venues)
  - Top 2 per group (24 teams) + 8 best third-place teams -> Round of 32
  - Knockout bracket through the final
"""

import sys
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

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CURRENT_YEAR = 2026

# Map any alias back to the official WC 2026 draw name
_WC_TEAM_LOOKUP = {
    normalize_team_name(team): team for team in WC2026_ALL_TEAMS
}


def to_wc_team(name: str) -> str:
    """Resolve harmonized data names to official WC 2026 team labels."""
    return _WC_TEAM_LOOKUP.get(normalize_team_name(name), name)


class TournamentState:
    """Holds team ratings and H2H for one simulation run."""

    def __init__(self):
        self.fifa_points = {}
        self.last5_form = {}
        self.penalty_win_rate = {}
        self.h2h = {}
        self.achievements = None
        self._load_data()

    def _load_data(self):
        strength_path = PROCESSED_DIR / "team_strength_2026.csv"
        if strength_path.exists():
            strength = pd.read_csv(strength_path)
            for _, row in strength.iterrows():
                t = normalize_team_name(row["team"])
                self.fifa_points[t] = row.get("fifa_points", row.get("elo", 1500))
                self.last5_form[t] = row.get("last5_win_rate", 0.5)

        fifa_path = RAW_DIR / "fifa_latest_ranking.csv"
        if fifa_path.exists():
            fifa = pd.read_csv(fifa_path)
            for _, row in fifa.iterrows():
                t = to_wc_team(row["country"])
                self.fifa_points[t] = row["total_points"]
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

        # Playoff placeholder strengths (only if placeholders remain in groups)
        for playoff, candidates in WC2026_PLAYOFF_CANDIDATES.items():
            pts = [self.fifa_points.get(normalize_team_name(c), 1400) for c in candidates]
            forms = [self.last5_form.get(normalize_team_name(c), 0.5) for c in candidates]
            self.fifa_points[playoff] = np.mean(pts)
            self.last5_form[playoff] = np.mean(forms)

        # Ensure every confirmed WC team has FIFA/form defaults
        for team in WC2026_ALL_TEAMS:
            t = normalize_team_name(team)
            if t not in self.fifa_points:
                self.fifa_points[t] = 1350
            if t not in self.last5_form:
                self.last5_form[t] = 0.35

    def smooth_form(self, team: str, alpha: float = 0.65) -> float:
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


def simulate_match(
    team_a: str,
    team_b: str,
    model,
    feature_cols: list,
    state: TournamentState,
    rng: np.random.Generator,
    round_name: str = "group",
) -> str:
    """
    Simulate one match. Returns 'A' (team_a wins), 'B' (team_b wins), or 'D' (draw).
    """
    h2h = state.h2h_get(team_a, team_b)
    fifa_diff = state.fifa_points.get(team_a, 1500) - state.fifa_points.get(team_b, 1500)
    form_a = state.smooth_form(team_a)
    form_b = state.smooth_form(team_b)

    # Tactical/squad-quality edge (achievements already baked into FIFA points)
    tactical_diff = MODERN_TEAMS.get(team_a, 0) - MODERN_TEAMS.get(team_b, 0)
    fifa_diff += tactical_diff * 12

    # Tournament variance injection
    fifa_diff += rng.normal(0, ROUND_VARIANCE.get(round_name, 20))

    h2h_weight = min(1.0, h2h["matches_played"] / 10)
    features = {
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

    if model is not None:
        X = pd.DataFrame([{c: features.get(c, 0) for c in feature_cols}])
        probs = model.predict_proba(X)[0]
        # sklearn order: classes 0=away, 1=draw, 2=home -> map to B, D, A
        p_away, p_draw, p_home = probs[0], probs[1], probs[2]
    else:
        # ELO fallback
        p_home = 1 / (1 + 10 ** (-fifa_diff / 400))
        p_away = 1 - p_home
        p_draw = 0.25
        total = p_home + p_away + p_draw
        p_home, p_away, p_draw = p_home / total, p_away / total, p_draw / total

    if p_draw > DRAW_THRESHOLD:
        return "D"

    p_nodraw_home = p_home / (p_home + p_away)
    return "A" if rng.random() < p_nodraw_home else "B"


def simulate_penalty(team_a: str, team_b: str, state: TournamentState, rng) -> str:
    pa = state.penalty_win_rate.get(team_a, 0.5)
    pb = state.penalty_win_rate.get(team_b, 0.5)
    return team_a if rng.random() < pa / (pa + pb) else team_b


def simulate_group(
    teams: list,
    model,
    feature_cols: list,
    state: TournamentState,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Round-robin group stage (6 matches)."""
    table = {t: {"points": 0, "gd": 0, "gf": 0} for t in teams}

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            # Neutral-site WC matches — randomise designated home side per fixture
            t1, t2 = teams[i], teams[j]
            if rng.random() < 0.5:
                t1, t2 = t2, t1
            result = simulate_match(t1, t2, model, feature_cols, state, rng, "group")
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

    return pd.DataFrame(table).T.sort_values(
        ["points", "gd", "gf"], ascending=False
    )


def rank_third_places(third_places: list) -> list:
    """Rank third-place teams for best-8 qualification (points, GD, GF)."""
    return sorted(
        third_places,
        key=lambda x: (x["points"], x["gd"], x["gf"]),
        reverse=True,
    )


def simulate_knockout_match(
    team_a: str,
    team_b: str,
    model,
    feature_cols: list,
    state: TournamentState,
    rng: np.random.Generator,
    round_name: str,
) -> str:
    result = simulate_match(team_a, team_b, model, feature_cols, state, rng, round_name)
    if result == "D":
        return simulate_penalty(team_a, team_b, state, rng)
    return team_a if result == "A" else team_b


def simulate_one_tournament(
    model,
    feature_cols: list,
    rng: np.random.Generator,
) -> str:
    """Run one full WC 2026 simulation. Returns champion team name."""
    state = TournamentState()

    third_places = []
    qualified = []

    for group_name, teams in WC2026_GROUPS.items():
        table = simulate_group(teams, model, feature_cols, state, rng)
        ranked = table.reset_index().rename(columns={"index": "team"})
        qualified.append(ranked.iloc[0]["team"])
        qualified.append(ranked.iloc[1]["team"])
        third_places.append({
            "team": ranked.iloc[2]["team"],
            "points": ranked.iloc[2]["points"],
            "gd": ranked.iloc[2]["gd"],
            "gf": ranked.iloc[2]["gf"],
            "group": group_name,
        })

    # 8 best third-place teams join the 24 automatic qualifiers -> Round of 32
    ranked_third = rank_third_places(third_places)
    qualified += [r["team"] for r in ranked_third[:8]]
    qualified = list(dict.fromkeys(qualified))

    # Seed knockout bracket by FIFA points (higher seed = slight structural edge via ordering)
    qualified.sort(key=lambda t: state.fifa_points.get(t, 0), reverse=True)

    # Standard knockout rounds (32 -> 16 -> 8 -> 4 -> 2 -> 1)
    rounds = ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final"]
    current = qualified[:32] if len(qualified) >= 32 else qualified

    for round_name in rounds:
        if len(current) <= 1:
            break
        rng.shuffle(current)
        next_round = []
        for i in range(0, len(current) - 1, 2):
            w = simulate_knockout_match(
                current[i], current[i + 1], model, feature_cols, state, rng, round_name
            )
            next_round.append(w)
        if len(current) % 2 == 1:
            next_round.append(current[-1])
        current = next_round

    return current[0] if current else qualified[0]


def run_wc2026_simulation(
    n_simulations: int = N_SIMULATIONS_DEFAULT,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run N full tournament simulations.

    Returns DataFrame with [team, win_probability, simulations].
    """
    print(f"Running {n_simulations} WC 2026 tournament simulations...")
    model, feature_cols = load_match_model()
    if model is None:
        print("  [WARN] No match model found — using ELO fallback")

    rng = np.random.default_rng(seed)
    win_counts = {team: 0 for team in WC2026_ALL_TEAMS}

    for i in range(n_simulations):
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{n_simulations}")
        champion = to_wc_team(simulate_one_tournament(model, feature_cols, rng))
        win_counts[champion] = win_counts.get(champion, 0) + 1

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
