"""
api_football_scraper.py
───────────────────────
Layer 3: domestic league player stats via API-Football (api-sports.io).

Strategy (fits free 100 req/day tier):
  - /players/topscorers + /players/topassists per league (~2 calls each)
  - /players?team=&season= for high-impact clubs (Messi/MLS, Saudi, etc.)

Responses are cached under data/raw/club/api_football/ (long TTL  -  frozen snapshot).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.cache_utils import is_cache_fresh
from src.config import (
    API_FOOTBALL_BASE_URL,
    API_FOOTBALL_CONFIG_PATH,
    API_FOOTBALL_KEY_ENV,
    API_FOOTBALL_RATE_LIMIT_SEC,
    CLUB_CACHE_TTL_DAYS,
    CLUB_RAW_API_FOOTBALL,
    club_seasons_from_config,
)


@dataclass
class ApiFootballReport:
    api_calls: int = 0
    cached: int = 0
    ok: int = 0
    failed: int = 0
    skipped: bool = False
    errors: list[str] = field(default_factory=list)


def _load_config() -> dict:
    if not API_FOOTBALL_CONFIG_PATH.exists():
        return {"seasons": [2024, 2025], "big5_leagues": {}, "tier_b_leagues": {}, "target_teams": []}
    return json.loads(API_FOOTBALL_CONFIG_PATH.read_text(encoding="utf-8"))


def get_api_key() -> str | None:
    key = os.environ.get(API_FOOTBALL_KEY_ENV) or os.environ.get("API_FOOTBALL_KEY")
    if key:
        return key.strip()
    return None


def _cache_file(endpoint: str, league_id: int | None, team_id: int | None, season: int) -> Path:
    CLUB_RAW_API_FOOTBALL.mkdir(parents=True, exist_ok=True)
    if team_id is not None:
        return CLUB_RAW_API_FOOTBALL / f"api_{endpoint}_team{team_id}_s{season}.json"
    return CLUB_RAW_API_FOOTBALL / f"api_{endpoint}_league{league_id}_s{season}.json"


def _has_any_cache(seasons: list[int], ttl_days: int) -> bool:
    if not CLUB_RAW_API_FOOTBALL.exists():
        return False
    for season in seasons:
        for path in CLUB_RAW_API_FOOTBALL.glob(f"*_s{season}.json"):
            if is_cache_fresh(path, ttl_days):
                return True
    return False


def _season_from_cache_path(path: Path) -> int | None:
    match = re.search(r"_s(\d+)\.json$", path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _api_get(
    path: str,
    params: dict[str, Any],
    cache_path: Path,
    force: bool,
    ttl_days: int,
    report: ApiFootballReport,
) -> dict | None:
    if not force and is_cache_fresh(cache_path, ttl_days):
        report.cached += 1
        return json.loads(cache_path.read_text(encoding="utf-8"))

    api_key = get_api_key()
    if not api_key:
        report.skipped = True
        return None

    url = f"{API_FOOTBALL_BASE_URL}/{path.lstrip('/')}"
    headers = {"x-apisports-key": api_key}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        report.api_calls += 1
        time.sleep(API_FOOTBALL_RATE_LIMIT_SEC)
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            report.failed += 1
            report.errors.append(f"{path} {params}: {payload['errors']}")
            return None
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        report.ok += 1
        return payload
    except Exception as exc:
        report.failed += 1
        report.errors.append(f"{path} {params}: {exc}")
        return None


def _stat_block(entry: dict) -> dict | None:
    stats = entry.get("statistics") or []
    if not stats:
        return None
    return stats[0]


def _safe_float(val: Any) -> float | None:
    try:
        if val is None or val == "":
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> int | None:
    try:
        if val is None or val == "":
            return None
        return int(val)
    except (TypeError, ValueError):
        return None


def _record_from_top_entry(
    entry: dict,
    *,
    source_tag: str,
    league_id: int,
    season: int,
) -> dict | None:
    player = entry.get("player") or {}
    stat = _stat_block(entry)
    if not stat:
        return None

    games = stat.get("games") or {}
    goals = stat.get("goals") or {}
    shots = stat.get("shots") or {}
    passes = stat.get("passes") or {}
    team = stat.get("team") or {}
    league = stat.get("league") or {}
    resolved_league = league_id or _safe_int(league.get("id")) or 0
    gk = stat.get("goalkeeper") or {}

    minutes = _safe_int(games.get("minutes")) or 0
    gls = _safe_int(goals.get("total")) or 0
    ast = _safe_int(goals.get("assists")) or 0
    xg = _safe_float(goals.get("expected"))
    xa = _safe_float(goals.get("expected_assists") or goals.get("expectedAssists"))

    position = str(games.get("position") or "").lower()
    save_pct = _safe_float(gk.get("saves_percentage") or gk.get("save_percentage"))
    clean_sheets = _safe_int(gk.get("clean_sheets"))
    appearances = _safe_int(games.get("appearences") or games.get("appearances")) or 0
    cs_pct = (clean_sheets / appearances * 100) if clean_sheets and appearances else None
    ga_total = _safe_int(goals.get("conceded"))
    ga90 = (ga_total / (minutes / 90)) if ga_total is not None and minutes > 0 else None

    return {
        "source": source_tag,
        "league_id": resolved_league,
        "season": season,
        "player": player.get("name") or "",
        "club_team": team.get("name") or "",
        "nation_raw": player.get("nationality") or "",
        "minutes": minutes,
        "goals_total": gls,
        "xg_total": xg if xg is not None else 0.0,
        "shots_total": _safe_int(shots.get("total")) or 0,
        "assists_total": ast,
        "xa_total": xa if xa is not None else 0.0,
        "key_passes_total": _safe_int(passes.get("key")) or 0,
        "progressive_passes_total": 0,
        "pass_completion_pct": _safe_float(passes.get("accuracy")),
        "shots_on_target_pct": None,
        "save_pct": save_pct if "goalkeeper" in position or save_pct else None,
        "clean_sheet_pct": cs_pct,
        "ga_total": ga_total,
        "ga90": ga90,
        "psxg_minus_ga": None,
    }


def _records_from_team_squad(payload: dict, *, team_id: int, season: int) -> list[dict]:
    rows: list[dict] = []
    for entry in payload.get("response") or []:
        rec = _record_from_top_entry(
            entry,
            source_tag=f"api_team_{team_id}",
            league_id=0,
            season=season,
        )
        if rec:
            rows.append(rec)
    return rows


def _fetch_league_endpoint(
    endpoint: str,
    league_id: int,
    season: int,
    force: bool,
    ttl_days: int,
    report: ApiFootballReport,
) -> list[dict]:
    cache = _cache_file(endpoint, league_id, None, season)
    payload = _api_get(
        f"players/{endpoint}",
        {"league": league_id, "season": season},
        cache,
        force,
        ttl_days,
        report,
    )
    if not payload:
        return []

    rows: list[dict] = []
    for entry in payload.get("response") or []:
        rec = _record_from_top_entry(
            entry,
            source_tag=f"api_{endpoint}",
            league_id=league_id,
            season=season,
        )
        if rec:
            rows.append(rec)
    return rows


def _fetch_team_squad(
    team_id: int,
    season: int,
    force: bool,
    ttl_days: int,
    report: ApiFootballReport,
) -> list[dict]:
    cache = _cache_file("team_squad", None, team_id, season)
    payload = _api_get(
        "players",
        {"team": team_id, "season": season},
        cache,
        force,
        ttl_days,
        report,
    )
    if not payload:
        return []
    return _records_from_team_squad(payload, team_id=team_id, season=season)


def fetch_api_football_player_stats(
    *,
    include_big5: bool = False,
    include_tier_b: bool = True,
    include_target_teams: bool = True,
    force: bool = False,
    ttl_days: int = CLUB_CACHE_TTL_DAYS,
) -> tuple[pd.DataFrame, ApiFootballReport]:
    """Fetch league top scorers/assists (+ optional targeted club squads) for all configured seasons."""
    cfg = _load_config()
    seasons = club_seasons_from_config(cfg)
    report = ApiFootballReport()

    if not get_api_key() and not _has_any_cache(seasons, ttl_days) and not force:
        report.skipped = True
        report.errors.append("No APIFOOTBALL_KEY in .env and no cached API-Football files")
        return pd.DataFrame(), report

    league_map: dict[int, str] = {}
    if include_big5:
        league_map.update({int(k): v for k, v in cfg.get("big5_leagues", {}).items()})
    if include_tier_b:
        league_map.update({int(k): v for k, v in cfg.get("tier_b_leagues", {}).items()})

    all_rows: list[dict] = []
    for season in seasons:
        for league_id in league_map:
            for endpoint in ("topscorers", "topassists"):
                all_rows.extend(
                    _fetch_league_endpoint(endpoint, league_id, season, force, ttl_days, report),
                )

        if include_target_teams:
            seen_teams: set[int] = set()
            for team in cfg.get("target_teams", []):
                tid = int(team["id"])
                if tid in seen_teams:
                    continue
                seen_teams.add(tid)
                all_rows.extend(_fetch_team_squad(tid, season, force, ttl_days, report))

    if not all_rows:
        return pd.DataFrame(), report
    return pd.DataFrame(all_rows), report


def load_cached_api_football_stats() -> pd.DataFrame:
    """Load player rows from cached API-Football JSON only (no HTTP)."""
    cfg = _load_config()
    seasons = set(club_seasons_from_config(cfg))
    all_rows: list[dict] = []
    if not CLUB_RAW_API_FOOTBALL.exists():
        return pd.DataFrame()

    for path in CLUB_RAW_API_FOOTBALL.glob("api_topscorers_league*_s*.json"):
        season = _season_from_cache_path(path)
        if season is None or season not in seasons:
            continue
        try:
            league_id = int(path.stem.split("league")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload.get("response") or []:
            rec = _record_from_top_entry(
                entry, source_tag="api_topscorers", league_id=league_id, season=season,
            )
            if rec:
                all_rows.append(rec)

    for path in CLUB_RAW_API_FOOTBALL.glob("api_topassists_league*_s*.json"):
        season = _season_from_cache_path(path)
        if season is None or season not in seasons:
            continue
        try:
            league_id = int(path.stem.split("league")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload.get("response") or []:
            rec = _record_from_top_entry(
                entry, source_tag="api_topassists", league_id=league_id, season=season,
            )
            if rec:
                all_rows.append(rec)

    for path in CLUB_RAW_API_FOOTBALL.glob("api_team_squad_team*_s*.json"):
        season = _season_from_cache_path(path)
        if season is None or season not in seasons:
            continue
        try:
            team_id = int(path.stem.split("team")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        all_rows.extend(_records_from_team_squad(payload, team_id=team_id, season=season))

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


__all__ = [
    "fetch_api_football_player_stats",
    "ApiFootballReport",
    "get_api_key",
    "load_cached_api_football_stats",
]
