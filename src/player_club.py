"""
player_club.py
──────────────
Club-season player stats for 2024–25 / 2025–26.

Primary: CSV drops in data/raw/club/ (manual FBref export when scraping is blocked).
Fallback: soccerdata FBref scrape (often 403).
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLUB_RAW_DIR,
    FBREF_CLUB_LEAGUES,
    FBREF_CLUB_SEASONS,
    NATION_ALIASES,
    RAW_DIR,
    WC2026_ALL_TEAMS,
)
from src.labels import normalize_player_name, normalize_team_name

CLUB_RAW_DIR.mkdir(parents=True, exist_ok=True)

STAT_FILES = {
    "shooting": ["shooting", "standard"],
    "passing": ["passing"],
    "goalkeeper": ["keeper", "goalkeeper", "keepers"],
}


def _nation_to_team(nation: str) -> str | None:
    if pd.isna(nation):
        return None
    raw = str(nation).strip()
    # FBref often uses "eng England" style prefixes
    raw = re.sub(r"^[a-z]{3}\s+", "", raw, flags=re.I)
    team = NATION_ALIASES.get(raw, raw)
    team = normalize_team_name(team)
    return team if team in WC2026_ALL_TEAMS else None


def _load_fbref_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, header=[0, 1])
        df.columns = ["_".join(str(c) for c in col).strip("_") for col in df.columns]
    except Exception:
        df = pd.read_csv(path)
    return df.reset_index(drop=True)


def _pick_column(df: pd.DataFrame, patterns: list[str]) -> str | None:
    for col in df.columns:
        cl = str(col).lower()
        if all(p.lower() in cl for p in patterns):
            return col
    return None


def _numeric(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _parse_club_frame(df: pd.DataFrame, stat_type: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = pd.DataFrame(index=df.index)
    player_col = _pick_column(df, ["player"]) or _pick_column(df, ["unnamed"])
    nation_col = _pick_column(df, ["nation"]) or _pick_column(df, ["nat"])
    mins_col = _pick_column(df, ["min"]) or _pick_column(df, ["90s"])

    out["player"] = df[player_col] if player_col else df.iloc[:, 0]
    out["nation_raw"] = df[nation_col] if nation_col else ""
    out["team"] = out["nation_raw"].apply(_nation_to_team)
    out["minutes"] = _numeric(df, mins_col)
    if out["minutes"].max() and out["minutes"].max() < 50:
        out["minutes"] = out["minutes"] * 90

    if stat_type == "shooting":
        out["goals_total"] = _numeric(df, _pick_column(df, ["standard", "gls"]) or _pick_column(df, ["gls"]))
        out["xg_total"] = _numeric(df, _pick_column(df, ["expected", "xg"]) or _pick_column(df, ["xg"]))
        out["shots_total"] = _numeric(df, _pick_column(df, ["standard", "sh"]) or _pick_column(df, ["sh"]))
        sot_col = _pick_column(df, ["sot%"]) or _pick_column(df, ["sot"])
        out["shots_on_target_pct"] = _numeric(df, sot_col)
        mins90 = out["minutes"].replace(0, np.nan) / 90
        out["goals_per90"] = out["goals_total"] / mins90
        out["xg_per90"] = out["xg_total"] / mins90
        out["npxg_per90"] = out["xg_per90"]
        out["shots_per90"] = out["shots_total"] / mins90
    elif stat_type == "passing":
        out["assists_total"] = _numeric(df, _pick_column(df, ["total", "ast"]) or _pick_column(df, ["ast"]))
        out["xa_total"] = _numeric(df, _pick_column(df, ["total", "xag"]) or _pick_column(df, ["xag"]))
        out["key_passes_total"] = _numeric(df, _pick_column(df, ["total", "kp"]) or _pick_column(df, ["kp"]))
        out["progressive_passes_total"] = _numeric(df, _pick_column(df, ["total", "prgp"]) or _pick_column(df, ["prgp"]))
        cmp_col = _pick_column(df, ["cmp%"]) or _pick_column(df, ["cmp"])
        out["pass_completion_pct"] = _numeric(df, cmp_col)
        mins90 = out["minutes"].replace(0, np.nan) / 90
        out["xa_per90"] = out["xa_total"] / mins90
        out["key_passes_per90"] = out["key_passes_total"] / mins90
        out["progressive_passes_per90"] = out["progressive_passes_total"] / mins90
    elif stat_type == "goalkeeper":
        out["save_pct"] = _numeric(df, _pick_column(df, ["save%"]))
        out["clean_sheet_pct"] = _numeric(df, _pick_column(df, ["cs%"]))
        out["ga_total"] = _numeric(df, _pick_column(df, ["ga90"]) or _pick_column(df, ["ga"]))
        out["ga90"] = _numeric(df, _pick_column(df, ["ga90"]))
        out["psxg_minus_ga"] = _numeric(
            df,
            _pick_column(df, ["psxg+/-"]) or _pick_column(df, ["psxg-ga"]),
        )

    out["player_norm"] = out["player"].apply(normalize_player_name)
    out = out[out["team"].notna()].copy()
    return out


def _manual_club_files() -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {k: [] for k in STAT_FILES}
    for season in FBREF_CLUB_SEASONS:
        season_tag = season.replace("-", "")
        for stat_type, keywords in STAT_FILES.items():
            for kw in keywords:
                for path in CLUB_RAW_DIR.glob(f"*{kw}*{season_tag}*.csv"):
                    found[stat_type].append(path)
                for path in CLUB_RAW_DIR.glob(f"fbref_{stat_type}_{season}.csv"):
                    found[stat_type].append(path)
    return found


def fetch_club_stats_via_soccerdata(force: bool = False) -> bool:
    """Attempt automated FBref club fetch; returns True if any file saved."""
    try:
        import soccerdata as sd
    except ImportError:
        return False

    saved = False
    for league in FBREF_CLUB_LEAGUES:
        for season in FBREF_CLUB_SEASONS:
            tag = league.split("-")[0].lower()
            for stat_type, sd_name in [
                ("shooting", "shooting"),
                ("passing", "passing"),
                ("goalkeeper", "keeper"),
            ]:
                out = CLUB_RAW_DIR / f"fbref_{stat_type}_{tag}_{season}.csv"
                if out.exists() and not force:
                    saved = True
                    continue
                try:
                    fbref = sd.FBref(leagues=league, seasons=season)
                    time.sleep(3)
                    df = fbref.read_player_season_stats(stat_type=sd_name)
                    df.to_csv(out)
                    print(f"  [OK] Club {stat_type} {league} {season} -> {out.name}")
                    saved = True
                except Exception as e:
                    print(f"  [WARN] FBref {league} {season} {stat_type}: {e}")
    return saved


def load_club_player_stats() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and aggregate club stats across manual/scraped CSV files.

    Returns (shooting, passing, goalkeeper) DataFrames keyed by player_norm + team.
    """
    files = _manual_club_files()
    shooting_frames, passing_frames, gk_frames = [], [], []

    for path in files["shooting"]:
        parsed = _parse_club_frame(_load_fbref_table(path), "shooting")
        if not parsed.empty:
            shooting_frames.append(parsed)
    for path in files["passing"]:
        parsed = _parse_club_frame(_load_fbref_table(path), "passing")
        if not parsed.empty:
            passing_frames.append(parsed)
    for path in files["goalkeeper"]:
        parsed = _parse_club_frame(_load_fbref_table(path), "goalkeeper")
        if not parsed.empty:
            gk_frames.append(parsed)

    def _aggregate(frames: list[pd.DataFrame], sum_cols: list[str], mean_cols: list[str]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        agg = {c: "sum" for c in sum_cols if c in df.columns}
        agg.update({c: "mean" for c in mean_cols if c in df.columns})
        if "minutes" in df.columns:
            agg["minutes"] = "sum"
        grouped = df.groupby(["player", "player_norm", "team"], as_index=False).agg(agg)
        if "minutes" in grouped.columns:
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

    shooting = _aggregate(
        shooting_frames,
        ["goals_total", "xg_total", "shots_total"],
        ["shots_on_target_pct", "goals_per90", "xg_per90", "shots_per90"],
    )
    passing = _aggregate(
        passing_frames,
        ["assists_total", "xa_total", "key_passes_total", "progressive_passes_total"],
        ["pass_completion_pct", "xa_per90", "key_passes_per90", "progressive_passes_per90"],
    )
    gk = _aggregate(
        gk_frames,
        ["ga_total"],
        ["save_pct", "clean_sheet_pct", "ga90", "psxg_minus_ga"],
    )
    return shooting, passing, gk
