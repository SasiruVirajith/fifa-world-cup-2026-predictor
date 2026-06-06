"""
player_features_2026.py
───────────────────────
Build 2026-forward player feature tables and award rankings.

Data layers:
  - martj42 international (2023+)
  - FBref club CSVs in data/raw/club/ (optional)
  - WC 2026 team sim for knockout context
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    OUTPUTS_DIR,
    PRIMARY_GK_2026,
    PROCESSED_DIR,
    WC2026_ALL_TEAMS,
)
from src.labels import normalize_player_name
from src.player_club import fetch_club_stats_via_soccerdata, load_club_player_stats
from src.player_international import (
    build_international_player_stats,
    build_international_team_defense,
    international_stats_for_year,
)


def _col_numeric(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _normalize(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def _load_team_win_probs() -> pd.DataFrame:
    path = OUTPUTS_DIR / "wc2026_champion_probabilities.csv"
    if not path.exists():
        return pd.DataFrame({"team": WC2026_ALL_TEAMS, "team_win_prob": 0.02})
    df = pd.read_csv(path)
    prob_col = "champion_probability" if "champion_probability" in df.columns else "probability"
    if prob_col not in df.columns:
        cols = [c for c in df.columns if c != "team"]
        prob_col = cols[0] if cols else None
    if prob_col is None:
        return pd.DataFrame({"team": WC2026_ALL_TEAMS, "team_win_prob": 0.02})
    out = df[["team", prob_col]].rename(columns={prob_col: "team_win_prob"})
    return out


def _merge_club_shooting(intl: pd.DataFrame, shooting: pd.DataFrame) -> pd.DataFrame:
    df = intl.copy()
    if shooting.empty:
        return df
    club_cols = [
        c
        for c in shooting.columns
        if c not in {"player", "team", "player_norm", "nation_raw"}
    ]
    df = df.merge(
        shooting[["player_norm", "team"] + club_cols],
        on=["player_norm", "team"],
        how="left",
    )
    if "player_club" not in df.columns and "player" in shooting.columns:
        names = shooting.groupby(["player_norm", "team"])["player"].first().reset_index()
        names = names.rename(columns={"player": "player_club_name"})
        df = df.merge(names, on=["player_norm", "team"], how="left")
    return df


def _fill_striker_proxies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["xg_per90"] = pd.to_numeric(out.get("xg_per90"), errors="coerce")
    out["npxg_per90"] = pd.to_numeric(out.get("npxg_per90"), errors="coerce")
    out["shots_per90"] = pd.to_numeric(out.get("shots_per90"), errors="coerce")
    out["goals_per90"] = pd.to_numeric(out.get("goals_per90"), errors="coerce")
    out["shots_on_target_pct"] = pd.to_numeric(out.get("shots_on_target_pct"), errors="coerce")

    rate = out["intl_goal_rate"].fillna(0)
    out["xg_per90"] = out["xg_per90"].fillna(rate * 0.9)
    out["npxg_per90"] = out["npxg_per90"].fillna(out["xg_per90"])
    out["shots_per90"] = out["shots_per90"].fillna(rate * 3.2 + out["intl_goals_per_team_match"].fillna(0) * 0.5)
    out["goals_per90"] = out["goals_per90"].fillna(out["intl_goals_per_team_match"].fillna(0) * 0.85)
    out["shots_on_target_pct"] = out["shots_on_target_pct"].fillna(0.33)
    return out


def build_striker_features_2026() -> pd.DataFrame:
    intl = build_international_player_stats()
    shooting, _, _ = load_club_player_stats()
    if intl.empty:
        return pd.DataFrame()

    df = _merge_club_shooting(intl, shooting)
    df = _fill_striker_proxies(df)
    df = df.merge(_load_team_win_probs(), on="team", how="left")
    df["team_win_prob"] = df["team_win_prob"].fillna(0.01)

    df["boot_score"] = (
        0.32 * _normalize(df["intl_weighted_goals"])
        + 0.22 * _normalize(df["xg_per90"].fillna(0))
        + 0.18 * _normalize(df["goals_per90"].fillna(0))
        + 0.13 * _normalize(df["shots_per90"].fillna(0))
        + 0.10 * _normalize(df["intl_major_goals"].fillna(0))
        + 0.05 * _normalize(df["team_win_prob"].fillna(0))
    )
    df["predicted_goals"] = (df["boot_score"] * 7.5).round(2)
    df["year"] = 2026
    df["tournament_goals"] = np.nan
    return df.sort_values("boot_score", ascending=False).reset_index(drop=True)


def build_goalkeeper_features_2026() -> pd.DataFrame:
    team_def = build_international_team_defense()
    _, _, gk_club = load_club_player_stats()

    rows = []
    for team, player in PRIMARY_GK_2026.items():
        row = {"player": player, "team": team, "player_norm": normalize_player_name(player), "year": 2026}
        rows.append(row)
    df = pd.DataFrame(rows)

    if not gk_club.empty:
        gk_cols = [c for c in gk_club.columns if c not in {"player", "team", "player_norm"}]
        df = df.merge(gk_club[["player_norm", "team"] + gk_cols], on=["player_norm", "team"], how="left")

    if not team_def.empty:
        df = df.merge(team_def, on="team", how="left")

    df["save_pct"] = _col_numeric(df, "save_pct", 68.0)
    cs_fallback = (
        df["intl_clean_sheet_pct"] * 100
        if "intl_clean_sheet_pct" in df.columns
        else pd.Series(25.0, index=df.index)
    )
    df["clean_sheet_pct"] = (
        _col_numeric(df, "clean_sheet_pct", 25.0)
        if "clean_sheet_pct" in df.columns
        else cs_fallback.fillna(25.0)
    )
    ga_fallback = df["intl_ga90"] if "intl_ga90" in df.columns else pd.Series(1.0, index=df.index)
    df["ga90"] = (
        _col_numeric(df, "ga90", 1.0) if "ga90" in df.columns else ga_fallback.fillna(1.0)
    )
    df["psxg_minus_ga"] = _col_numeric(df, "psxg_minus_ga", 0.0)

    df = df.merge(_load_team_win_probs(), on="team", how="left")
    df["team_win_prob"] = df["team_win_prob"].fillna(0.01)

    df["glove_score"] = (
        0.30 * _normalize(df["save_pct"])
        + 0.25 * _normalize(-df["ga90"])
        + 0.20 * _normalize(df["clean_sheet_pct"])
        + 0.15 * _normalize(df["psxg_minus_ga"])
        + 0.10 * _normalize(df["team_win_prob"])
    )
    df["golden_glove_probability"] = (df["glove_score"] / df["glove_score"].sum()).round(4)
    df["won_golden_glove"] = 0
    return df.sort_values("glove_score", ascending=False).reset_index(drop=True)


def build_playmaker_features_2026() -> pd.DataFrame:
    intl = build_international_player_stats()
    _, passing, _ = load_club_player_stats()
    if intl.empty:
        return pd.DataFrame()

    df = intl.copy()
    if not passing.empty:
        pass_cols = [c for c in passing.columns if c not in {"player", "team", "player_norm", "nation_raw"}]
        df = df.merge(passing[["player_norm", "team"] + pass_cols], on=["player_norm", "team"], how="left")

    df["xa_per90"] = pd.to_numeric(df.get("xa_per90"), errors="coerce")
    df["key_passes_per90"] = pd.to_numeric(df.get("key_passes_per90"), errors="coerce")
    df["progressive_passes_per90"] = pd.to_numeric(df.get("progressive_passes_per90"), errors="coerce")
    df["pass_completion_pct"] = pd.to_numeric(df.get("pass_completion_pct"), errors="coerce")

    proxy = df["intl_weighted_goals"].fillna(0) * 0.15
    df["xa_per90"] = df["xa_per90"].fillna(proxy)
    df["key_passes_per90"] = df["key_passes_per90"].fillna(proxy * 2.5)
    df["progressive_passes_per90"] = df["progressive_passes_per90"].fillna(proxy * 2.0)
    df["pass_completion_pct"] = df["pass_completion_pct"].fillna(78.0)

    df["playmaker_score"] = (
        0.40 * _normalize(df["xa_per90"])
        + 0.30 * _normalize(df["key_passes_per90"])
        + 0.20 * _normalize(df["progressive_passes_per90"])
        + 0.10 * _normalize(df["pass_completion_pct"])
    )
    df["year"] = 2026
    return df.sort_values("playmaker_score", ascending=False).reset_index(drop=True)


def build_player_of_tournament_2026(
    strikers: pd.DataFrame,
    playmakers: pd.DataFrame,
) -> pd.DataFrame:
    if strikers.empty:
        return pd.DataFrame()

    pot = strikers[["player", "team", "intl_goals", "boot_score"]].copy()
    pot = pot.rename(columns={"boot_score": "attack_score"})
    if not playmakers.empty:
        pm = playmakers[["player_norm", "team", "playmaker_score"]].copy()
        pot["player_norm"] = pot["player"].apply(normalize_player_name)
        pot = pot.merge(pm, on=["player_norm", "team"], how="left")
    else:
        pot["playmaker_score"] = 0.0

    pot = pot.merge(_load_team_win_probs(), on="team", how="left")
    pot["team_win_prob"] = pot["team_win_prob"].fillna(0.01)
    pot["pot_score"] = (
        0.45 * _normalize(pot["attack_score"].fillna(0))
        + 0.30 * _normalize(pot["playmaker_score"].fillna(0))
        + 0.15 * _normalize(pot["intl_goals"].fillna(0))
        + 0.10 * _normalize(pot["team_win_prob"].fillna(0))
    )
    pot["goals"] = pot["intl_goals"]
    pot["assists"] = 0
    pot["year"] = 2026
    return pot.sort_values("pot_score", ascending=False).reset_index(drop=True)


def _append_year_rows(existing_path, new_df: pd.DataFrame, year: int = 2026) -> pd.DataFrame:
    if new_df.empty:
        return pd.DataFrame()
    if existing_path.exists():
        old = pd.read_csv(existing_path)
        if "year" in old.columns:
            old = old[old["year"] != year]
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(existing_path, index=False)
    return combined


def enrich_backtest_with_international(features: pd.DataFrame, year: int) -> pd.DataFrame:
    """Add international columns to 2018/2022 StatsBomb rows for richer training."""
    if features.empty or "year" not in features.columns:
        return features
    subset = features[features["year"] == year].copy()
    if subset.empty:
        return features

    intl = international_stats_for_year(year, teams=subset["team"].unique().tolist())
    if intl.empty:
        return features

    intl_cols = [c for c in intl.columns if c not in {"player", "player_norm", "team"}]
    subset = subset.drop(columns=[c for c in intl_cols if c in subset.columns], errors="ignore")
    subset["player_norm"] = subset["player"].apply(normalize_player_name)
    merged = subset.merge(
        intl.drop(columns=["player"], errors="ignore"),
        on=["player_norm", "team"],
        how="left",
    )
    merged = merged.drop(columns=["player_norm"], errors="ignore")
    other = features[features["year"] != year]
    return pd.concat([other, merged], ignore_index=True)


def build_all_2026_player_features(
    try_club_scrape: bool = False,
    enrich_backtest: bool = False,
) -> dict:
    """
    Full 2026 player pipeline. Returns dict of output DataFrames.
    """
    print("=" * 60)
    print("  2026 Player Feature Pipeline")
    print("=" * 60)

    if try_club_scrape:
        print("\n[1/6] Attempting FBref club scrape...")
        fetch_club_stats_via_soccerdata()
    else:
        print("\n[1/6] Club layer: using data/raw/club/ CSVs if present")

    shooting, passing, gk = load_club_player_stats()
    club_status = "loaded" if not shooting.empty or not passing.empty or not gk.empty else "intl-only (drop FBref CSVs into data/raw/club/)"
    print(f"       Club status: {club_status}")

    print("\n[2/6] Building international player stats (martj42 2023+)...")
    intl = build_international_player_stats()
    intl_path = PROCESSED_DIR / "player_intl_2026.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    intl.to_csv(intl_path, index=False)
    print(f"       [OK] {len(intl)} scorers -> {intl_path}")

    if not shooting.empty:
        club_path = PROCESSED_DIR / "player_club_2026.csv"
        shooting.to_csv(club_path, index=False)
        print(f"       [OK] {len(shooting)} club rows -> {club_path}")

    print("\n[3/6] Striker / Golden Boot features...")
    strikers = build_striker_features_2026()
    striker_path = PROCESSED_DIR / "striker_features.csv"
    strikers_all = _append_year_rows(striker_path, strikers, year=2026)
    print(f"       [OK] {len(strikers)} rows for 2026 ({len(strikers_all)} total)")

    print("\n[4/6] Goalkeeper / Golden Glove features...")
    gk_df = build_goalkeeper_features_2026()
    gk_path = PROCESSED_DIR / "goalkeeper_features.csv"
    gk_all = _append_year_rows(gk_path, gk_df, year=2026)
    print(f"       [OK] {len(gk_df)} rows for 2026 ({len(gk_all)} total)")

    print("\n[5/6] Playmaker features...")
    playmakers = build_playmaker_features_2026()
    pm_path = PROCESSED_DIR / "playmaker_features.csv"
    pm_all = _append_year_rows(pm_path, playmakers, year=2026)
    playmakers.to_csv(OUTPUTS_DIR / "playmaker_rankings_2026.csv", index=False)
    # Back-compat: full rankings file prefers 2026 when present
    playmakers.head(500).to_csv(OUTPUTS_DIR / "playmaker_rankings.csv", index=False)
    print(f"       [OK] {len(playmakers)} rows for 2026 ({len(pm_all)} total)")

    print("\n[6/6] Player of the Tournament composite...")
    pot = build_player_of_tournament_2026(strikers, playmakers)
    pot_path = OUTPUTS_DIR / "player_tournament_2026.csv"
    pot.to_csv(pot_path, index=False)
    print(f"       [OK] {len(pot)} rows -> {pot_path}")

    if enrich_backtest and striker_path.exists():
        print("\n[Extra] Enriching 2018/2022 rows with international features...")
        enriched = pd.read_csv(striker_path)
        for y in [2018, 2022]:
            enriched = enrich_backtest_with_international(enriched, y)
        enriched.to_csv(striker_path, index=False)

    return {
        "international": intl,
        "strikers": strikers,
        "goalkeepers": gk_df,
        "playmakers": playmakers,
        "pot": pot,
    }


if __name__ == "__main__":
    outputs = build_all_2026_player_features()
    print("\nTop 10 Golden Boot candidates (2026):")
    print(outputs["strikers"][["player", "team", "intl_weighted_goals", "boot_score", "predicted_goals"]].head(10).to_string(index=False))
