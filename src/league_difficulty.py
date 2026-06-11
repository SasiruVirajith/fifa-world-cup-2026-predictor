"""
league_difficulty.py
────────────────────
Difficulty multipliers for domestic leagues (0–1).

Low values down-weight stat-padded leagues (e.g. Saudi Pro League, MLS)
so Golden Boot / Golden Ball favor production in top competitions.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.config import API_FOOTBALL_CONFIG_PATH

# league_id → difficulty (1.0 = elite, ~0.35 = heavily padded)
_DEFAULT_DIFFICULTY: dict[int, float] = {
    # Big 5
    39: 1.00,
    140: 1.00,
    78: 1.00,
    135: 1.00,
    61: 1.00,
    # Tier B  -  proper leagues
    94: 0.82,
    88: 0.78,
    144: 0.76,
    203: 0.72,
    40: 0.68,
    71: 0.74,
    128: 0.70,
    # Lower-tier / padded
    253: 0.52,
    262: 0.54,
    307: 0.35,
}

# API target-team squad fetches (league_id missing in row)
_TARGET_TEAM_LEAGUE: dict[int, int] = {
    9568: 253,
    2939: 307,
    2932: 307,
    1616: 253,
    1597: 253,
}

DEFAULT_DIFFICULTY = 0.50
UNDERSTAT_DIFFICULTY = 1.00


def _load_config_overrides() -> dict:
    if not API_FOOTBALL_CONFIG_PATH.exists():
        return {}
    try:
        cfg = json.loads(API_FOOTBALL_CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return cfg.get("league_difficulty") or {}


def league_difficulty_map() -> dict[int, float]:
    merged = dict(_DEFAULT_DIFFICULTY)
    for key, val in _load_config_overrides().items():
        try:
            merged[int(key)] = float(val)
        except (TypeError, ValueError):
            continue
    return merged


def difficulty_for_league_id(league_id: int | None, source: str = "") -> float:
    if source and str(source).startswith("understat"):
        return UNDERSTAT_DIFFICULTY
    if not league_id:
        return DEFAULT_DIFFICULTY
    lid = int(league_id)
    if lid == 0:
        return DEFAULT_DIFFICULTY
    return league_difficulty_map().get(lid, DEFAULT_DIFFICULTY)


def difficulty_for_row(row: pd.Series | dict[str, Any]) -> float:
    if isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = row
    source = str(data.get("source") or "")
    league_id = data.get("league_id")
    try:
        league_id = int(league_id) if league_id is not None and league_id != "" else 0
    except (TypeError, ValueError):
        league_id = 0

    if league_id == 0 and source.startswith("api_team_"):
        try:
            team_id = int(str(source).replace("api_team_", ""))
            league_id = _TARGET_TEAM_LEAGUE.get(team_id, 0)
        except ValueError:
            pass

    return difficulty_for_league_id(league_id, source)


def attach_league_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["league_difficulty"] = pd.Series(dtype=float)
        return out
    out = df.copy()
    out["league_difficulty"] = out.apply(difficulty_for_row, axis=1)
    return out
