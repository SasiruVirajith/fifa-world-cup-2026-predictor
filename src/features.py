"""
features.py
───────────
Builds feature matrices for each prediction model from cleaned data.
Takes raw CSVs from data/raw/ and outputs processed feature files to data/processed/.

Usage:
    python src/features.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    PROCESSED_DIR,
    RAW_DIR,
    WC_START_DATES,
    WC_YEARS,
    WC_WINNERS,
)
from src.labels import (
    add_glove_labels,
    add_winner_labels,
    extract_tournament_goals,
    load_events_for_year,
    merge_tournament_goals,
    normalize_team_name,
)

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# FBref flattened column name mappings (update if soccerdata column names change)
STRIKER_COLUMN_MAP = {
    "Per 90 Minutes_xG": "xg_per90",
    "Per 90 Minutes_npxG": "npxg_per90",
    "Per 90 Minutes_Gls": "goals_per90",
    "Per 90 Minutes_Sh": "shots_per90",
    "Standard_Sh": "shots_total",
    "Standard_SoT": "shots_on_target",
    "Standard_SoT%": "shots_on_target_pct",
    "Standard_Gls": "goals_total",
    "Standard_xG": "xg_total",
}

GK_COLUMN_MAP = {
    "Shot Stopping_Save%": "save_pct",
    "Shot Stopping_CS%": "clean_sheet_pct",
    "Goals Against_GA": "ga_total",
    "Goals Against_GA90": "ga90",
    "Expected_PSxG": "psxg_total",
    "Expected_PSxG+/-": "psxg_minus_ga",
    "Expected_PSxG-GA": "psxg_minus_ga",
}

PLAYMAKER_COLUMN_MAP = {
    "Total_KP": "key_passes_total",
    "Total_PrgP": "progressive_passes_total",
    "Total_xAG": "xa_total",
    "Total_Ast": "assists_total",
    "Total_Cmp%": "pass_completion_pct",
    "Per 90 Minutes_KP": "key_passes_per90",
    "Per 90 Minutes_PrgP": "progressive_passes_per90",
    "Per 90 Minutes_xAG": "xa_per90",
}


def calculate_elo_ratings(
    results_df: pd.DataFrame,
    k: int = 32,
    initial_elo: int = 1500,
) -> dict:
    """
    Calculate ELO ratings for all teams from international match results.
    Returns dict {team: elo} at end of provided results.
    """
    elo = {}
    results_df = results_df.sort_values("date").copy()

    for _, row in results_df.iterrows():
        home = normalize_team_name(row["home_team"])
        away = normalize_team_name(row["away_team"])

        if home not in elo:
            elo[home] = initial_elo
        if away not in elo:
            elo[away] = initial_elo

        exp_home = 1 / (1 + 10 ** ((elo[away] - elo[home]) / 400))
        exp_away = 1 - exp_home

        home_goals = row["home_score"]
        away_goals = row["away_score"]

        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals == away_goals:
            actual_home, actual_away = 0.5, 0.5
        else:
            actual_home, actual_away = 0.0, 1.0

        elo[home] += k * (actual_home - exp_home)
        elo[away] += k * (actual_away - exp_away)

    return elo


def _compute_form(results_df: pd.DataFrame, team: str, n: int = 20) -> dict:
    """Compute recent form stats for a team from filtered results."""
    home = results_df[results_df["home_team"] == team].copy()
    away = results_df[results_df["away_team"] == team].copy()

    home["goals_for"] = home["home_score"]
    home["goals_against"] = home["away_score"]
    home["win"] = (home["home_score"] > home["away_score"]).astype(int)

    away["goals_for"] = away["away_score"]
    away["goals_against"] = away["home_score"]
    away["win"] = (away["away_score"] > away["home_score"]).astype(int)

    all_matches = pd.concat([
        home[["date", "goals_for", "goals_against", "win"]],
        away[["date", "goals_for", "goals_against", "win"]],
    ]).sort_values("date").tail(n)

    if all_matches.empty:
        return {
            "win_rate_last20": 0.5,
            "goals_scored_avg": 1.0,
            "goals_conceded_avg": 1.0,
            "matches_played": 0,
        }

    return {
        "win_rate_last20": all_matches["win"].mean(),
        "goals_scored_avg": all_matches["goals_for"].mean(),
        "goals_conceded_avg": all_matches["goals_against"].mean(),
        "matches_played": len(all_matches),
    }


def _get_wc_participants(year: int) -> list:
    """Get teams that participated in a given World Cup from StatsBomb or matches data."""
    matches_path = RAW_DIR / f"statsbomb_matches_wc{year}.csv"
    if matches_path.exists():
        matches = pd.read_csv(matches_path)
        home = matches["home_team"].apply(normalize_team_name).tolist()
        away = matches["away_team"].apply(normalize_team_name).tolist()
        return list(set(home + away))

    # Fallback: all teams from winners dict era
    return list(WC_WINNERS.values())


def build_team_features_for_year(
    results_df: pd.DataFrame,
    year: int,
    squad_values: pd.DataFrame = None,
    wc_history: pd.DataFrame = None,
) -> pd.DataFrame:
    """Build team features for one WC year, computed at tournament start."""
    cutoff = pd.Timestamp(WC_START_DATES.get(year, f"{year}-06-01"))
    historical = results_df[results_df["date"] < cutoff].copy()
    historical["home_team"] = historical["home_team"].apply(normalize_team_name)
    historical["away_team"] = historical["away_team"].apply(normalize_team_name)

    elo = calculate_elo_ratings(historical)
    participants = _get_wc_participants(year)

    records = []
    for team in participants:
        form = _compute_form(historical, team)
        record = {
            "team": team,
            "year": year,
            "elo": elo.get(team, 1500),
            **form,
        }

        # WC historical win rate
        if wc_history is not None and not wc_history.empty:
            wc_team = wc_history[
                (wc_history["home_team"].apply(normalize_team_name) == team)
                | (wc_history["away_team"].apply(normalize_team_name) == team)
            ]
            if not wc_team.empty:
                wins = 0
                total = 0
                for _, m in wc_team.iterrows():
                    h = normalize_team_name(m["home_team"])
                    a = normalize_team_name(m["away_team"])
                    if h == team:
                        total += 1
                        if m["home_score"] > m["away_score"]:
                            wins += 1
                    elif a == team:
                        total += 1
                        if m["away_score"] > m["home_score"]:
                            wins += 1
                record["wc_win_rate"] = wins / total if total > 0 else 0.5
                record["wc_appearances"] = wc_team["date"].dt.year.nunique() if "date" in wc_team.columns else 1
            else:
                record["wc_win_rate"] = 0.5
                record["wc_appearances"] = 0
        else:
            record["wc_win_rate"] = 0.5
            record["wc_appearances"] = 0

        # Squad value
        if squad_values is not None and not squad_values.empty:
            sv = squad_values.copy()
            team_col = _find_column(sv, ["team", "Team", "squad"])
            val_col = _find_column(sv, ["squad_value", "value", "market_value"])
            if team_col and val_col:
                sv["team_norm"] = sv[team_col].apply(normalize_team_name)
                match = sv[sv["team_norm"] == team]
                record["squad_value"] = match[val_col].iloc[0] if not match.empty else np.nan
            else:
                record["squad_value"] = np.nan
        else:
            record["squad_value"] = np.nan

        records.append(record)

    df = pd.DataFrame(records)
    return add_winner_labels(df, year)


def build_team_features(
    results_path: str = None,
    years: list = None,
) -> pd.DataFrame:
    """Build multi-year team feature matrix for tournament winner model."""
    print("Building team features (multi-year)...")

    results_path = results_path or str(RAW_DIR / "results.csv")
    if not Path(results_path).exists():
        results_path = str(RAW_DIR / "international_results.csv")
    if not Path(results_path).exists():
        print(f"  [ERR] Results not found at {results_path}")
        return pd.DataFrame()

    results = pd.read_csv(results_path)
    results["date"] = pd.to_datetime(results["date"])

    wc_history_path = RAW_DIR / "wc_historical_matches.csv"
    wc_history = None
    if wc_history_path.exists():
        wc_history = pd.read_csv(wc_history_path)
        wc_history["date"] = pd.to_datetime(wc_history["date"], errors="coerce")

    years = years or [y for y in WC_YEARS if (RAW_DIR / f"statsbomb_matches_wc{y}.csv").exists()]
    if not years:
        years = [2022]

    all_features = []
    for year in years:
        print(f"  Processing WC {year}...")
        tm_path = RAW_DIR / f"transfermarkt_squad_values_{year}.csv"
        squad_values = pd.read_csv(tm_path) if tm_path.exists() else None
        year_df = build_team_features_for_year(results, year, squad_values, wc_history)
        all_features.append(year_df)

    team_features = pd.concat(all_features, ignore_index=True)
    save_path = PROCESSED_DIR / "team_features.csv"
    team_features.to_csv(save_path, index=False)
    print(f"  [OK] Team features saved -> {save_path} ({len(team_features)} rows)")
    return team_features


def _load_fbref_csv(path: Path) -> pd.DataFrame:
    """Load FBref CSV, handling multi-level headers."""
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, header=[0, 1])
        df.columns = ["_".join(str(c) for c in col).strip("_") for col in df.columns]
    except Exception:
        df = pd.read_csv(path)

    # Reset index to get player/team columns
    df = df.reset_index()
    return df


def _rename_fbref_columns(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """Map FBref flattened columns to standard feature names."""
    df = df.copy()
    rename = {}
    for col in df.columns:
        col_str = str(col)
        for pattern, new_name in column_map.items():
            if pattern in col_str or col_str.endswith(pattern.replace(" ", "_")):
                rename[col] = new_name
                break
    df = df.rename(columns=rename)

    # Fuzzy fallback: find columns containing key substrings
    fuzzy_map = {
        "xg_per90": ["per", "90", "xg"],
        "npxg_per90": ["per", "90", "npxg"],
        "shots_per90": ["per", "90", "sh"],
        "shots_on_target_pct": ["sot%"],
        "save_pct": ["save%"],
        "clean_sheet_pct": ["cs%"],
        "ga90": ["ga90"],
        "psxg_minus_ga": ["psxg+/-", "psxg-ga"],
        "xa_per90": ["per", "90", "xag"],
        "key_passes_per90": ["per", "90", "kp"],
        "progressive_passes_per90": ["per", "90", "prgp"],
        "pass_completion_pct": ["cmp%"],
    }

    for target, keywords in fuzzy_map.items():
        if target not in df.columns:
            for col in df.columns:
                col_lower = str(col).lower()
                if all(kw.lower() in col_lower for kw in keywords):
                    df[target] = pd.to_numeric(df[col], errors="coerce")
                    break

    # Ensure player/team columns
    if "player" not in df.columns:
        for col in df.columns:
            if "player" in str(col).lower():
                df["player"] = df[col]
                break
    if "team" not in df.columns:
        for col in df.columns:
            if any(k in str(col).lower() for k in ["squad", "team"]):
                df["team"] = df[col].apply(normalize_team_name)
                break

    return df


def build_striker_features(year: int = 2022) -> pd.DataFrame:
    """Build striker features for one WC year with tournament_goals labels."""
    print(f"Building striker features for WC {year}...")

    events_path = RAW_DIR / f"statsbomb_events_wc{year}.csv"
    if events_path.exists():
        from src.statsbomb_features import build_striker_features_from_events
        return build_striker_features_from_events(year)

    path = RAW_DIR / f"fbref_shooting_{year}.csv"
    if not path.exists():
        print(f"  [ERR] No shooting data for WC {year}")
        return pd.DataFrame()

    shooting = _load_fbref_csv(path)
    shooting = _rename_fbref_columns(shooting, STRIKER_COLUMN_MAP)

    mins_col = _find_column(shooting, ["min", "minutes", "90s"])
    if mins_col:
        mins = pd.to_numeric(shooting[mins_col], errors="coerce").replace(0, np.nan)
        for total, per90 in [("xg_total", "xg_per90"), ("goals_total", "goals_per90")]:
            if total in shooting.columns and per90 not in shooting.columns:
                shooting[per90] = pd.to_numeric(shooting[total], errors="coerce") / (mins / 90)

    events = load_events_for_year(year)
    if not events.empty:
        goals_df = extract_tournament_goals(events, year)
        shooting = merge_tournament_goals(shooting, goals_df, year)
    else:
        shooting["tournament_goals"] = 0
        shooting["year"] = year

    return shooting


def build_goalkeeper_features(year: int = 2022) -> pd.DataFrame:
    """Build goalkeeper features for one WC year with golden glove labels."""
    print(f"Building goalkeeper features for WC {year}...")

    events_path = RAW_DIR / f"statsbomb_events_wc{year}.csv"
    if events_path.exists():
        from src.statsbomb_features import build_goalkeeper_features_from_events
        gk = build_goalkeeper_features_from_events(year)
        return add_glove_labels(gk, year) if not gk.empty else gk

    path = RAW_DIR / f"fbref_goalkeeper_{year}.csv"
    if not path.exists():
        print(f"  [ERR] Goalkeeper data not found at {path}")
        return pd.DataFrame()

    gk = _load_fbref_csv(path)
    gk = _rename_fbref_columns(gk, GK_COLUMN_MAP)
    gk = add_glove_labels(gk, year)
    return gk


def build_playmaker_features(year: int = 2022) -> pd.DataFrame:
    """Build playmaker features for one WC year."""
    print(f"Building playmaker features for WC {year}...")

    events_path = RAW_DIR / f"statsbomb_events_wc{year}.csv"
    if events_path.exists():
        from src.statsbomb_features import build_playmaker_features_from_events
        return build_playmaker_features_from_events(year)

    path = RAW_DIR / f"fbref_passing_{year}.csv"
    if not path.exists():
        print(f"  [ERR] Passing data not found at {path}")
        return pd.DataFrame()

    passing = _load_fbref_csv(path)
    passing = _rename_fbref_columns(passing, PLAYMAKER_COLUMN_MAP)

    mins_col = _find_column(passing, ["min", "minutes", "90s"])
    if mins_col:
        mins = pd.to_numeric(passing[mins_col], errors="coerce").replace(0, np.nan)
        for total, per90 in [
            ("xa_total", "xa_per90"),
            ("key_passes_total", "key_passes_per90"),
            ("progressive_passes_total", "progressive_passes_per90"),
        ]:
            if total in passing.columns and per90 not in passing.columns:
                passing[per90] = pd.to_numeric(passing[total], errors="coerce") / (mins / 90)

    passing["year"] = year
    return passing


def build_all_player_features(years: list = None) -> tuple:
    """Build and save all player feature CSVs across multiple years."""
    years = years or [
        y for y in [2018, 2022]
        if (RAW_DIR / f"statsbomb_events_wc{y}.csv").exists()
        or (RAW_DIR / f"fbref_shooting_{y}.csv").exists()
    ]
    if not years:
        years = [2022]

    strikers, gks, playmakers = [], [], []
    for year in years:
        s = build_striker_features(year)
        if not s.empty:
            strikers.append(s)
        g = build_goalkeeper_features(year)
        if not g.empty:
            gks.append(g)
        p = build_playmaker_features(year)
        if not p.empty:
            playmakers.append(p)

    striker_df = pd.concat(strikers, ignore_index=True) if strikers else pd.DataFrame()
    gk_df = pd.concat(gks, ignore_index=True) if gks else pd.DataFrame()
    playmaker_df = pd.concat(playmakers, ignore_index=True) if playmakers else pd.DataFrame()

    if not striker_df.empty:
        striker_df.to_csv(PROCESSED_DIR / "striker_features.csv", index=False)
        print(f"  [OK] Striker features saved ({len(striker_df)} rows)")
    if not gk_df.empty:
        gk_df.to_csv(PROCESSED_DIR / "goalkeeper_features.csv", index=False)
        print(f"  [OK] Goalkeeper features saved ({len(gk_df)} rows)")
    if not playmaker_df.empty:
        playmaker_df.to_csv(PROCESSED_DIR / "playmaker_features.csv", index=False)
        print(f"  [OK] Playmaker features saved ({len(playmaker_df)} rows)")

    return striker_df, gk_df, playmaker_df


def _find_column(df: pd.DataFrame, candidates: list) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        for cand in candidates:
            if cand.lower() in str(col).lower():
                return col
    return None


if __name__ == "__main__":
    print("=" * 50)
    print("  World Cup Predictor - Feature Engineering")
    print("=" * 50)

    print("\n[1/2] Building team features...")
    build_team_features()

    print("\n[2/2] Building player features...")
    build_all_player_features()

    print("\n[DONE] Feature engineering complete. Check data/processed/")
    print("   Next step: python src/models.py")
