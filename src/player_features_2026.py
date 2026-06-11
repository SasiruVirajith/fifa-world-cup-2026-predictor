# Copyright (c) 2026 Sasiru Virajith Kankanamge
# SPDX-License-Identifier: MIT

"""
FIFA World Cup 2026 Predictor
Built by: K. Sasiru Virajith
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    GOLDEN_BOOT_GOALS_CALIBRATION,
    OUTPUTS_DIR,
    PRIMARY_GK_2026,
    PROCESSED_DIR,
)
from src.labels import normalize_player_name
from src.player_club import (
    club_data_status,
    fetch_and_cache_club_stats,
    load_club_from_cache,
    load_club_player_stats,
)
from src.player_international import (
    build_international_player_stats,
    build_international_team_defense,
)
from src.team_expectations import load_team_tournament_context, save_team_expectations


def _col_numeric(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _normalize(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def _load_team_context() -> pd.DataFrame:
    return load_team_tournament_context()


def _merge_team_context(df: pd.DataFrame) -> pd.DataFrame:
    ctx = _load_team_context()
    keep = [
        "team",
        "group",
        "p_qualify",
        "team_win_prob",
        "progression_factor",
        "expected_tournament_matches",
        "group_difficulty",
        "last5_win_rate",
    ]
    return df.merge(ctx[keep], on="team", how="left")


def _club_shooting_table(intl: pd.DataFrame, shooting: pd.DataFrame) -> pd.DataFrame:
    # prefer committed CSV when in-memory fetch is sparse
    path = PROCESSED_DIR / "player_club_2026.csv"
    if path.exists():
        snap = pd.read_csv(path)
        if not snap.empty and (shooting.empty or len(snap) >= len(shooting)):
            return snap
    return shooting


def _apply_club_merge(
    df: pd.DataFrame,
    shooting: pd.DataFrame,
    club_cols: list[str],
    on: list[str],
    suffix: str,
) -> pd.DataFrame:
    if shooting.empty or not on:
        return df
    cols = on + club_cols
    cols = [c for c in cols if c in shooting.columns]
    if len(cols) <= len(on):
        return df
    extra = shooting[cols].drop_duplicates(subset=on)
    return df.merge(extra, on=on, how="left", suffixes=("", suffix))


def _merge_club_shooting(intl: pd.DataFrame, shooting: pd.DataFrame) -> pd.DataFrame:
    df = intl.copy()
    shooting = _club_shooting_table(intl, shooting)
    if shooting.empty:
        df["has_club_stats"] = False
        return df

    club_cols = [
        c
        for c in shooting.columns
        if c not in {"player", "team", "player_norm", "nation_raw", "source"}
    ]

    df = _apply_club_merge(df, shooting, club_cols, ["player_norm", "team"], "_m1")

    # martj42 uses full names; club API often uses initials (Harry Kane vs h kane)
    shoot = shooting.copy()
    shoot["_last"] = shoot["player_norm"].str.split().str[-1]
    df["_last"] = df["player_norm"].str.split().str[-1]
    df = _apply_club_merge(df, shoot, club_cols, ["_last", "team"], "_m2")
    for col in club_cols:
        fb = f"{col}_m2"
        if col in df.columns and fb in df.columns:
            df[col] = df[col].fillna(df[fb])
            df = df.drop(columns=[fb], errors="ignore")
    df = df.drop(columns=["_last"], errors="ignore")

    has_minutes = df["minutes"].notna() if "minutes" in df.columns else pd.Series(False, index=df.index)
    has_goals = df["goals_total"].notna() if "goals_total" in df.columns else pd.Series(False, index=df.index)
    has_per90 = df["goals_per90"].notna() if "goals_per90" in df.columns else pd.Series(False, index=df.index)
    df["has_club_stats"] = has_minutes | has_goals | has_per90

    if "player" in shooting.columns:
        names = shooting.groupby(["player_norm", "team"])["player"].first().reset_index()
        names = names.rename(columns={"player": "player_club_name"})
        df = df.merge(names, on=["player_norm", "team"], how="left")
        if df["player_club_name"].isna().any() and "_last" not in df.columns:
            df["_last"] = df["player_norm"].str.split().str[-1]
            shoot_names = shoot.groupby(["_last", "team"])["player"].first().reset_index()
            shoot_names = shoot_names.rename(columns={"player": "player_club_name_fb"})
            df = df.merge(shoot_names, on=["_last", "team"], how="left")
            df["player_club_name"] = df["player_club_name"].fillna(df["player_club_name_fb"])
            df = df.drop(columns=["player_club_name_fb", "_last"], errors="ignore")
    return df


def _fill_striker_proxies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["xg_per90"] = pd.to_numeric(out.get("xg_per90"), errors="coerce")
    out["npxg_per90"] = pd.to_numeric(out.get("npxg_per90"), errors="coerce")
    out["shots_per90"] = pd.to_numeric(out.get("shots_per90"), errors="coerce")
    out["goals_per90"] = pd.to_numeric(out.get("goals_per90"), errors="coerce")
    out["shots_on_target_pct"] = pd.to_numeric(out.get("shots_on_target_pct"), errors="coerce")
    if "league_difficulty" in out.columns:
        out["league_difficulty"] = pd.to_numeric(out["league_difficulty"], errors="coerce").fillna(0.5)
    else:
        out["league_difficulty"] = 0.5

    has_club = out["has_club_stats"] if "has_club_stats" in out.columns else pd.Series(False, index=out.index)
    gpg = out["intl_goals_per_team_match"].fillna(0)
    # no club row: impute from intl rate per team match (not per scoring appearance)
    out.loc[~has_club, "xg_per90"] = out.loc[~has_club, "xg_per90"].fillna(gpg * 0.85)
    out["npxg_per90"] = out["npxg_per90"].fillna(out["xg_per90"])
    out.loc[~has_club, "shots_per90"] = out.loc[~has_club, "shots_per90"].fillna(gpg * 2.8)
    out.loc[~has_club, "goals_per90"] = out.loc[~has_club, "goals_per90"].fillna(gpg * 0.75)
    out["shots_on_target_pct"] = out["shots_on_target_pct"].fillna(0.33)
    return out


def build_striker_features_2026(intl: pd.DataFrame | None = None) -> pd.DataFrame:
    intl = intl if intl is not None else build_international_player_stats()
    shooting, _, _ = load_club_player_stats(intl_lookup=intl)
    if intl.empty:
        return pd.DataFrame()

    df = _merge_club_shooting(intl, shooting)
    df = _fill_striker_proxies(df)
    df = _merge_team_context(df)
    df["progression_factor"] = df["progression_factor"].fillna(0.05)
    df["expected_tournament_matches"] = df["expected_tournament_matches"].fillna(3.0)
    df["p_qualify"] = df["p_qualify"].fillna(0.10)
    df["team_win_prob"] = df["team_win_prob"].fillna(0.001)

    df["adj_xg_per90"] = df["xg_per90"].fillna(0)
    df["adj_goals_per90"] = df["goals_per90"].fillna(0)
    df["adj_shots_per90"] = df["shots_per90"].fillna(0)

    has_club = df["has_club_stats"] if "has_club_stats" in df.columns else pd.Series(False, index=df.index)
    cred = df["league_difficulty"].fillna(0.5)
    cred = np.where(has_club, cred, 0.85)

    df["player_skill"] = (
        0.38 * _normalize(df["intl_weighted_goals"]) * cred
        + 0.24 * _normalize(df["adj_goals_per90"])
        + 0.18 * _normalize(df["adj_xg_per90"])
        + 0.10 * _normalize(df["adj_shots_per90"])
        + 0.10 * _normalize(df["intl_major_goals"].fillna(0)) * cred
    )

    df["boot_score"] = (df["player_skill"] * df["progression_factor"]).clip(lower=0)

    goals_per_match = (
        0.50 * df["intl_goals_per_team_match"].fillna(0)
        + 0.50 * df["adj_goals_per90"].fillna(0)
    )
    df["predicted_goals"] = (
        goals_per_match
        * df["expected_tournament_matches"]
        * GOLDEN_BOOT_GOALS_CALIBRATION
    ).clip(0, 12).round(2)
    df["year"] = 2026
    df["tournament_goals"] = np.nan
    return df.sort_values("boot_score", ascending=False).reset_index(drop=True)


def build_goalkeeper_features_2026(intl: pd.DataFrame | None = None) -> pd.DataFrame:
    team_def = build_international_team_defense()
    intl = intl if intl is not None else build_international_player_stats()
    _, _, gk_club = load_club_player_stats(intl_lookup=intl)

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

    df = _merge_team_context(df)
    df["progression_factor"] = df["progression_factor"].fillna(0.05)

    gk_skill = (
        0.35 * _normalize(df["save_pct"])
        + 0.30 * _normalize(-df["ga90"])
        + 0.20 * _normalize(df["clean_sheet_pct"])
        + 0.15 * _normalize(df["psxg_minus_ga"])
    )
    df["glove_score"] = (gk_skill * df["progression_factor"]).clip(lower=0)
    df["golden_glove_probability"] = (df["glove_score"] / df["glove_score"].sum()).round(4)
    df["won_golden_glove"] = 0
    return df.sort_values("glove_score", ascending=False).reset_index(drop=True)


def build_playmaker_features_2026(intl: pd.DataFrame | None = None) -> pd.DataFrame:
    intl = intl if intl is not None else build_international_player_stats()
    _, passing, _ = load_club_player_stats(intl_lookup=intl)
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

    pot = _merge_team_context(pot)
    pot["progression_factor"] = pot["progression_factor"].fillna(0.05)

    pot_skill = (
        0.40 * _normalize(pot["attack_score"].fillna(0))
        + 0.35 * _normalize(pot["playmaker_score"].fillna(0))
        + 0.25 * _normalize(pot["intl_goals"].fillna(0))
    )
    pot["pot_score"] = (pot_skill * pot["progression_factor"]).clip(lower=0)
    pot["team_win_prob"] = pot["team_win_prob"].fillna(0.001)
    pot["p_qualify"] = pot["p_qualify"].fillna(0.10)
    pot["goals"] = pot["intl_goals"]
    pot["assists"] = 0
    pot["year"] = 2026
    return pot.sort_values("pot_score", ascending=False).reset_index(drop=True)


def _write_year_features(path: Path, new_df: pd.DataFrame, year: int = 2026) -> pd.DataFrame:
    if new_df.empty:
        return pd.DataFrame()
    out = new_df.copy()
    out["year"] = year
    out.to_csv(path, index=False)
    return out


def build_all_2026_player_features(
    fetch_club: bool = True,
    fetch_club_force: bool = False,
) -> dict:
    if fetch_club:
        print("\n[1/7] Club stats...")
        fetch_and_cache_club_stats(force=fetch_club_force)
    else:
        print("\n[1/7] Club stats (cache only)...")
        load_club_from_cache()

    status = club_data_status()
    if status["loaded"]:
        src = status.get("source") or "raw_cache"
        club_status = (
            f"loaded ({status['player_rows']} players; source={src}; "
            f"understat={status['understat_files']}, api={status['api_files']} cache files)"
        )
    else:
        club_status = "intl-only (no club cache; set APIFOOTBALL_KEY or run with fetch)"
    print(f"       Club status: {club_status}")

    print("\n[2/7] International player stats...")
    intl = build_international_player_stats()
    intl_path = PROCESSED_DIR / "player_intl_2026.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    intl.to_csv(intl_path, index=False)
    print(f"       [OK] {len(intl)} scorers -> {intl_path}")

    shooting, _, _ = load_club_player_stats(intl_lookup=intl)
    club_path = PROCESSED_DIR / "player_club_2026.csv"
    if not shooting.empty:
        if club_path.exists():
            existing = pd.read_csv(club_path)
            # keep committed snapshot if the new fetch is much smaller
            if len(shooting) < max(50, int(len(existing) * 0.75)):
                print(
                    f"       [WARN] Keeping existing club snapshot ({len(existing)} rows); "
                    f"new fetch only {len(shooting)} rows"
                )
            else:
                shooting.to_csv(club_path, index=False)
                print(f"       [OK] {len(shooting)} club rows -> {club_path}")
        else:
            shooting.to_csv(club_path, index=False)
            print(f"       [OK] {len(shooting)} club rows -> {club_path}")

    print("\n[3/7] Striker / Golden Boot features...")
    strikers = build_striker_features_2026(intl=intl)
    striker_path = PROCESSED_DIR / "striker_features.csv"
    _write_year_features(striker_path, strikers, year=2026)
    print(f"       [OK] {len(strikers)} rows for 2026")

    print("\n[4/7] Goalkeeper / Golden Glove features...")
    gk_df = build_goalkeeper_features_2026(intl=intl)
    gk_path = PROCESSED_DIR / "goalkeeper_features.csv"
    _write_year_features(gk_path, gk_df, year=2026)
    print(f"       [OK] {len(gk_df)} rows for 2026")

    print("\n[5/7] Playmaker features (feeds Golden Ball)...")
    playmakers = build_playmaker_features_2026(intl=intl)
    pm_path = PROCESSED_DIR / "playmaker_features.csv"
    _write_year_features(pm_path, playmakers, year=2026)
    print(f"       [OK] {len(playmakers)} rows for 2026")

    print("\n[6/7] Golden Ball composite...")
    pot = build_player_of_tournament_2026(strikers, playmakers)
    pot_path = OUTPUTS_DIR / "player_tournament_2026.csv"
    pot.to_csv(pot_path, index=False)
    print(f"       [OK] {len(pot)} rows -> {pot_path}")

    print("\n[7/7] Team surprise / disappointment vs FIFA ranking...")
    team_exp = save_team_expectations()
    print(f"       [OK] {len(team_exp)} teams -> outputs/team_tournament_context_2026.csv")

    return {
        "international": intl,
        "strikers": strikers,
        "goalkeepers": gk_df,
        "playmakers": playmakers,
        "pot": pot,
        "team_expectations": team_exp,
    }


if __name__ == "__main__":
    outputs = build_all_2026_player_features()
    print("\nTop 10 Golden Boot candidates (2026):")
    print(outputs["strikers"][["player", "team", "intl_weighted_goals", "boot_score", "predicted_goals"]].head(10).to_string(index=False))
