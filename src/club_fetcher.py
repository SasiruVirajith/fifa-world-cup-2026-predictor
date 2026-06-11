"""
club_fetcher.py
───────────────
Orchestrate club stat layers for the 2026 player pipeline:

  Layer 2  -  Understat (Big 5 xG/xA)
  Layer 3  -  API-Football (Tier B + Big 5 fallback)
  Layer 4  -  intl proxies applied downstream in player_features_2026
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.api_football_scraper import ApiFootballReport, fetch_api_football_player_stats
from src.config import CLUB_CACHE_TTL_DAYS
from src.understat_scraper import UnderstatFetchReport, fetch_understat_player_stats


@dataclass
class ClubFetchReport:
    understat: UnderstatFetchReport | None = None
    api_football: ApiFootballReport | None = None
    understat_rows: int = 0
    api_rows: int = 0
    messages: list[str] = field(default_factory=list)


def fetch_club_stats(
    force: bool = False,
    ttl_days: int = CLUB_CACHE_TTL_DAYS,
) -> tuple[pd.DataFrame, ClubFetchReport]:
    """
    Fetch all club player rows and return a unified normalized DataFrame.

    API-Football Big 5 is only requested when Understat returns no rows.
    """
    report = ClubFetchReport()

    print("  [Layer 2] Understat  -  Big 5...")
    understat_df, u_report = fetch_understat_player_stats(force=force, ttl_days=ttl_days)
    report.understat = u_report
    report.understat_rows = len(understat_df)

    if report.understat_rows:
        print(
            f"       Understat: {report.understat_rows} player rows "
            f"({u_report.ok} fetched, {u_report.cached} cached)",
        )
    elif u_report.failed:
        print("       Understat: unavailable  -  will use API-Football for Big 5")
        for err in u_report.errors[:3]:
            print(f"         {err}")

    need_big5_api = understat_df.empty
    suffix = " + Big 5 fallback" if need_big5_api else ""
    print(f"  [Layer 3] API-Football  -  Tier B{suffix}...")
    api_df, a_report = fetch_api_football_player_stats(
        include_big5=need_big5_api,
        include_tier_b=True,
        include_target_teams=True,
        force=force,
        ttl_days=ttl_days,
    )
    report.api_football = a_report
    report.api_rows = len(api_df)

    if a_report.skipped and api_df.empty:
        print("       API-Football: skipped (no APIFOOTBALL_KEY in .env and no cache)")
        report.messages.append("api_skipped_no_key")
    elif not api_df.empty or a_report.api_calls or a_report.cached:
        print(
            f"       API-Football: {report.api_rows} rows "
            f"({a_report.api_calls} API calls, {a_report.cached} cached, {a_report.failed} failed)",
        )

    frames = [df for df in (understat_df, api_df) if not df.empty]
    if not frames:
        return pd.DataFrame(), report

    return pd.concat(frames, ignore_index=True), report


__all__ = ["fetch_club_stats", "ClubFetchReport"]
