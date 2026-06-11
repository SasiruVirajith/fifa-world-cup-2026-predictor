# Copyright (c) 2026 Sasiru Virajith Kankanamge
# SPDX-License-Identifier: MIT

"""
FIFA World Cup 2026 Predictor
Built by: K. Sasiru Virajith
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.club_fetcher import fetch_club_stats
from src.api_football_scraper import load_cached_api_football_stats
from src.league_difficulty import attach_league_difficulty
from src.understat_scraper import load_cached_understat_stats
from src.config import (
    CLUB_RAW_DIR,
    NATION_ALIASES,
    PROCESSED_DIR,
    WC2026_ALL_TEAMS,
)
from src.labels import normalize_player_name, normalize_team_name

CLUB_RAW_DIR.mkdir(parents=True, exist_ok=True)

_CLUB_UNIFIED: pd.DataFrame | None = None


def _nation_to_team(nation: str) -> str | None:
    if pd.isna(nation) or not str(nation).strip():
        return None
    raw = str(nation).strip()
    raw = re.sub(r"^[a-z]{3}\s+", "", raw, flags=re.I)
    team = NATION_ALIASES.get(raw, raw)
    team = normalize_team_name(team)
    return team if team in WC2026_ALL_TEAMS else None


def _normalize_unified_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["player_norm"] = out["player"].apply(normalize_player_name)
    for col in (
        "minutes", "goals_total", "xg_total", "shots_total", "assists_total",
        "xa_total", "key_passes_total", "progressive_passes_total",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out


def _consolidate_unified(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = attach_league_difficulty(df)
    sum_cols = [
        "goals_total", "xg_total", "shots_total", "assists_total",
        "xa_total", "key_passes_total", "progressive_passes_total", "ga_total",
    ]
    mean_cols = ["shots_on_target_pct", "pass_completion_pct", "save_pct", "clean_sheet_pct", "ga90", "psxg_minus_ga"]

    rows: list[dict] = []
    for player_norm, grp in df.groupby("player_norm"):
        mins = grp["minutes"].fillna(0).clip(lower=0)
        diff = grp["league_difficulty"].fillna(0.5)
        weights = mins * diff
        wsum = float(weights.sum())
        total_mins = float(mins.sum())
        if wsum <= 0:
            weights = mins.clip(lower=1)
            wsum = float(weights.sum())

        best = grp.sort_values("league_difficulty", ascending=False).iloc[0]
        rec: dict = {
            "player": best["player"],
            "player_norm": player_norm,
            "nation_raw": best.get("nation_raw", ""),
            "source": best.get("source", ""),
            "minutes": total_mins,
            "league_difficulty": float((mins * diff).sum() / max(total_mins, 1)),
        }

        for col in sum_cols:
            if col not in grp.columns:
                continue
            vals = pd.to_numeric(grp[col], errors="coerce").fillna(0)
            per90 = vals / (mins.replace(0, np.nan) / 90)
            adj_per90 = (per90.fillna(0) * weights).sum() / wsum
            rec[col] = adj_per90 * total_mins / 90 if total_mins > 0 else 0.0

        for col in mean_cols:
            if col not in grp.columns:
                continue
            vals = pd.to_numeric(grp[col], errors="coerce")
            rec[col] = float((vals.fillna(0) * weights).sum() / wsum)

        rows.append(rec)
    return pd.DataFrame(rows)


def _dedupe_unified(df: pd.DataFrame) -> pd.DataFrame:
    return _consolidate_unified(df)


def _attach_wc_team(unified: pd.DataFrame, intl_lookup: pd.DataFrame | None) -> pd.DataFrame:
    df = unified.copy()
    df["team"] = df["nation_raw"].apply(_nation_to_team)

    if intl_lookup is not None and not intl_lookup.empty:
        weight_col = "intl_weighted_goals" if "intl_weighted_goals" in intl_lookup.columns else "intl_goals"
        intl_teams = (
            intl_lookup.groupby("player_norm")
            .apply(lambda g: g.sort_values(weight_col, ascending=False).iloc[0]["team"])
            .reset_index(name="team_intl")
        )
        df = df.merge(intl_teams, on="player_norm", how="left")
        df["team"] = df["team"].fillna(df["team_intl"])
        df = df.drop(columns=["team_intl"], errors="ignore")

    return df[df["team"].notna()].copy()


def _aggregate_stat_frames(frames: list[pd.DataFrame], sum_cols: list[str], mean_cols: list[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    agg = {c: "sum" for c in sum_cols if c in combined.columns}
    agg.update({c: "mean" for c in mean_cols if c in combined.columns})
    if "minutes" in combined.columns:
        agg["minutes"] = "sum"
    grouped = combined.groupby(["player", "player_norm", "team"], as_index=False).agg(agg)
    if "minutes" not in grouped.columns:
        return grouped
    mins90 = grouped["minutes"].replace(0, np.nan) / 90
    if "goals_total" in grouped.columns:
        grouped["goals_per90"] = grouped["goals_total"] / mins90
    if "xg_total" in grouped.columns:
        grouped["xg_per90"] = grouped["xg_total"] / mins90
        grouped["npxg_per90"] = grouped["xg_per90"]
    if "shots_total" in grouped.columns:
        grouped["shots_per90"] = grouped["shots_total"] / mins90
    if "xa_total" in grouped.columns:
        grouped["xa_per90"] = grouped["xa_total"] / mins90
    if "key_passes_total" in grouped.columns:
        grouped["key_passes_per90"] = grouped["key_passes_total"] / mins90
    if "progressive_passes_total" in grouped.columns:
        grouped["progressive_passes_per90"] = grouped["progressive_passes_total"] / mins90
    return grouped


def _unified_to_stat_frames(unified: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if unified.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = unified.copy()
    if "league_difficulty" not in df.columns:
        df = attach_league_difficulty(df)
    mins90 = df["minutes"].replace(0, np.nan) / 90

    shooting = df[
        [
            "player", "player_norm", "team", "minutes", "league_difficulty",
            "goals_total", "xg_total", "shots_total", "shots_on_target_pct",
        ]
    ].copy()
    shooting["goals_per90"] = shooting["goals_total"] / mins90
    shooting["xg_per90"] = shooting["xg_total"] / mins90
    shooting["npxg_per90"] = shooting["xg_per90"]
    shooting["shots_per90"] = shooting["shots_total"] / mins90

    passing = df[
        [
            "player", "player_norm", "team", "minutes",
            "assists_total", "xa_total", "key_passes_total",
            "progressive_passes_total", "pass_completion_pct",
        ]
    ].copy()
    passing["xa_per90"] = passing["xa_total"] / mins90
    passing["key_passes_per90"] = passing["key_passes_total"] / mins90
    passing["progressive_passes_per90"] = passing["progressive_passes_total"] / mins90

    save_col = df["save_pct"] if "save_pct" in df.columns else pd.Series(np.nan, index=df.index)
    ga90_col = df["ga90"] if "ga90" in df.columns else pd.Series(np.nan, index=df.index)
    gk_cols = ["save_pct", "clean_sheet_pct", "ga_total", "ga90", "psxg_minus_ga"]
    gk_cols = [c for c in gk_cols if c in df.columns]
    gk_mask = save_col.notna() | ga90_col.notna()
    gk = df.loc[gk_mask, ["player", "player_norm", "team"] + gk_cols].copy() if gk_cols else pd.DataFrame()

    shooting_agg = _aggregate_stat_frames(
        [shooting], ["goals_total", "xg_total", "shots_total"], ["shots_on_target_pct", "league_difficulty"],
    )
    passing_agg = _aggregate_stat_frames(
        [passing],
        ["assists_total", "xa_total", "key_passes_total", "progressive_passes_total"],
        ["pass_completion_pct"],
    )
    gk_agg = _aggregate_stat_frames(
        [gk] if not gk.empty else [],
        ["ga_total"],
        ["save_pct", "clean_sheet_pct", "ga90", "psxg_minus_ga"],
    )
    return shooting_agg, passing_agg, gk_agg


def fetch_and_cache_club_stats(force: bool = False) -> None:
    global _CLUB_UNIFIED
    unified, _ = fetch_club_stats(force=force)
    unified = _normalize_unified_rows(unified)
    _CLUB_UNIFIED = _dedupe_unified(unified)


def _load_club_from_processed_snapshot() -> pd.DataFrame:
    # used on fresh clones when data/raw/club/ is empty
    path = PROCESSED_DIR / "player_club_2026.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "player_norm" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    if "nation_raw" not in out.columns:
        out["nation_raw"] = ""
    if "source" not in out.columns:
        out["source"] = "processed_snapshot"
    for col in (
        "assists_total", "xa_total", "key_passes_total", "progressive_passes_total",
        "ga_total", "save_pct", "clean_sheet_pct", "ga90", "psxg_minus_ga",
        "pass_completion_pct",
    ):
        if col not in out.columns:
            out[col] = np.nan
    return out


def load_club_from_cache() -> None:
    global _CLUB_UNIFIED
    frames = [load_cached_understat_stats(), load_cached_api_football_stats()]
    frames = [f for f in frames if not f.empty]
    if frames:
        unified = pd.concat(frames, ignore_index=True)
        unified = _normalize_unified_rows(unified)
        _CLUB_UNIFIED = _dedupe_unified(unified)
        return

    snapshot = _load_club_from_processed_snapshot()
    _CLUB_UNIFIED = snapshot if not snapshot.empty else pd.DataFrame()


def club_data_status() -> dict:
    status = {
        "loaded": False,
        "understat_files": 0,
        "api_files": 0,
        "player_rows": 0,
        "source": None,
    }
    understat_dir = CLUB_RAW_DIR / "understat"
    api_dir = CLUB_RAW_DIR / "api_football"
    if understat_dir.exists():
        status["understat_files"] = len(list(understat_dir.glob("*.json")))
    if api_dir.exists():
        status["api_files"] = len(list(api_dir.glob("*.json")))
    if _CLUB_UNIFIED is not None and not _CLUB_UNIFIED.empty:
        status["loaded"] = True
        status["player_rows"] = len(_CLUB_UNIFIED)
        src = _CLUB_UNIFIED["source"] if "source" in _CLUB_UNIFIED.columns else pd.Series(dtype=str)
        if not src.empty and (src == "processed_snapshot").any():
            status["source"] = "processed_snapshot"
        else:
            status["source"] = "raw_cache"
    elif status["understat_files"] or status["api_files"]:
        status["loaded"] = True
    elif (PROCESSED_DIR / "player_club_2026.csv").exists():
        status["source"] = "processed_snapshot_available"
    return status


def load_club_player_stats(
    intl_lookup: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global _CLUB_UNIFIED

    shooting_frames: list[pd.DataFrame] = []
    passing_frames: list[pd.DataFrame] = []
    gk_frames: list[pd.DataFrame] = []

    if _CLUB_UNIFIED is not None and not _CLUB_UNIFIED.empty:
        attached = _attach_wc_team(_CLUB_UNIFIED, intl_lookup)
        s, p, g = _unified_to_stat_frames(attached)
        if not s.empty:
            shooting_frames.append(s)
        if not p.empty:
            passing_frames.append(p)
        if not g.empty:
            gk_frames.append(g)

    shooting = _aggregate_stat_frames(
        shooting_frames, ["goals_total", "xg_total", "shots_total"], ["shots_on_target_pct", "league_difficulty"],
    )
    passing = _aggregate_stat_frames(
        passing_frames,
        ["assists_total", "xa_total", "key_passes_total", "progressive_passes_total"],
        ["pass_completion_pct"],
    )
    gk = _aggregate_stat_frames(
        gk_frames, ["ga_total"], ["save_pct", "clean_sheet_pct", "ga90", "psxg_minus_ga"],
    )
    return shooting, passing, gk


__all__ = [
    "club_data_status",
    "fetch_and_cache_club_stats",
    "load_club_from_cache",
    "load_club_player_stats",
]
