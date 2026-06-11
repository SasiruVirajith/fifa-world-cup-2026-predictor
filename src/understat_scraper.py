"""
understat_scraper.py
────────────────────
Layer 2: Big 5 player season stats from Understat (xG, xA, goals).

Uses cloudscraper + JSON.parse extraction (same approach as soccerdata).
Understat occasionally changes page layout; failures are non-fatal  -  API-Football
covers Big 5 when this layer returns empty.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path

import pandas as pd

from src.cache_utils import is_cache_fresh
from src.config import (
    CLUB_CACHE_TTL_DAYS,
    CLUB_RAW_UNDERSTAT,
    UNDERSTAT_LEAGUES_PATH,
    club_seasons_from_config,
)

UNDERSTAT_URL = "https://understat.com"


@dataclass
class UnderstatFetchReport:
    ok: int = 0
    cached: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def _load_config() -> dict:
    if not UNDERSTAT_LEAGUES_PATH.exists():
        return {"leagues": {}, "seasons": [2024, 2025]}
    return json.loads(UNDERSTAT_LEAGUES_PATH.read_text(encoding="utf-8"))


def _cache_path(league_key: str, season: int) -> Path:
    tag = league_key.lower().replace(" ", "_").replace("-", "_")
    return CLUB_RAW_UNDERSTAT / f"understat_{tag}_{season}.json"


def _extract_json_vars(html: bytes, var_names: list[str]) -> dict:
    """Extract Understat JSON.parse('...') blobs from league page HTML."""
    found: dict = {}
    for var in var_names:
        pattern = var.encode("utf-8") + br"[\s\t]*=[\s\t]*JSON\.parse\('(.*?)'\)"
        match = re.search(pattern, html, re.S)
        if match:
            raw = match.group(1).decode("utf-8")
            found[var] = json.loads(raw.encode().decode("unicode_escape"))
    return found


def _fetch_league_season(slug: str, season: int, league_key: str, force: bool) -> dict | None:
    CLUB_RAW_UNDERSTAT.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(league_key, season)
    if not force and is_cache_fresh(cache, CLUB_CACHE_TTL_DAYS):
        return json.loads(cache.read_text(encoding="utf-8"))

    try:
        import cloudscraper
    except ImportError:
        return None

    url = f"{UNDERSTAT_URL}/league/{slug}/{season}"
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False},
    )
    try:
        response = scraper.get(url, timeout=45)
        response.raise_for_status()
        data = _extract_json_vars(response.content, ["playersData", "teamsData"])
        if "playersData" not in data:
            return None
        payload = {"playersData": data["playersData"], "teamsData": data.get("teamsData", {})}
        cache.write_text(json.dumps(payload), encoding="utf-8")
        time.sleep(2)
        return payload
    except Exception:
        return None


def _players_to_records(league_key: str, season: int, payload: dict) -> list[dict]:
    players = payload.get("playersData") or []
    if isinstance(players, dict):
        players = list(players.values())

    rows: list[dict] = []
    for p in players:
        if not isinstance(p, dict):
            continue
        minutes = int(p.get("time") or 0)
        rows.append(
            {
                "source": "understat",
                "league_key": league_key,
                "season": season,
                "player": unescape(str(p.get("player_name", ""))),
                "club_team": unescape(str(p.get("team_title", "")).split(",")[0]),
                "nation_raw": "",
                "minutes": minutes,
                "goals_total": int(p.get("goals") or 0),
                "xg_total": float(p.get("xG") or 0),
                "shots_total": int(p.get("shots") or 0),
                "assists_total": int(p.get("assists") or 0),
                "xa_total": float(p.get("xA") or 0),
                "key_passes_total": int(p.get("key_passes") or 0),
                "progressive_passes_total": 0,
                "pass_completion_pct": None,
                "shots_on_target_pct": None,
                "save_pct": None,
                "clean_sheet_pct": None,
                "ga_total": None,
                "ga90": None,
                "psxg_minus_ga": None,
            },
        )
    return rows


def fetch_understat_player_stats(
    force: bool = False,
    ttl_days: int = CLUB_CACHE_TTL_DAYS,
) -> tuple[pd.DataFrame, UnderstatFetchReport]:
    """Download Big 5 player season aggregates from Understat."""
    cfg = _load_config()
    leagues = cfg.get("leagues", {})
    seasons = club_seasons_from_config(cfg)
    report = UnderstatFetchReport()
    all_rows: list[dict] = []

    for league_key, meta in leagues.items():
        slug = meta["slug"]
        for season in seasons:
            cache = _cache_path(league_key, season)
            label = f"{league_key} {season}"
            if not force and is_cache_fresh(cache, ttl_days):
                report.cached += 1
                payload = json.loads(cache.read_text(encoding="utf-8"))
            else:
                payload = _fetch_league_season(slug, season, league_key, force=force)
                if payload is None:
                    report.failed += 1
                    report.errors.append(f"{label}: fetch failed (layout/block)")
                    continue
                report.ok += 1
            all_rows.extend(_players_to_records(league_key, season, payload))

    if not all_rows:
        return pd.DataFrame(), report
    return pd.DataFrame(all_rows), report


def load_cached_understat_stats() -> pd.DataFrame:
    """Load player rows from cached Understat JSON only (no HTTP)."""
    cfg = _load_config()
    leagues = cfg.get("leagues", {})
    seasons = club_seasons_from_config(cfg)
    all_rows: list[dict] = []
    if not CLUB_RAW_UNDERSTAT.exists():
        return pd.DataFrame()
    for league_key in leagues:
        for season in seasons:
            cache = _cache_path(league_key, season)
            if cache.exists():
                payload = json.loads(cache.read_text(encoding="utf-8"))
                all_rows.extend(_players_to_records(league_key, season, payload))
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


__all__ = ["fetch_understat_player_stats", "UnderstatFetchReport", "load_cached_understat_stats"]
