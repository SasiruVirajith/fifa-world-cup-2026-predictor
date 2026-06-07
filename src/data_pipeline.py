"""
data_pipeline.py
────────────────
Fetches raw data from StatsBomb, FBref, and ELO ratings.
Run this first to populate data/raw/ before any modelling.

Usage:
    python src/data_pipeline.py
    python src/data_pipeline.py --refresh
    python src/data_pipeline.py --skip-events
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    CACHE_TTL_DAYS,
    INTERNATIONAL_RESULTS_URL,
    RAW_DIR,
    STATSBOMB_COMPETITION_ID,
    STATSBOMB_SEASONS,
)

RAW_DIR.mkdir(parents=True, exist_ok=True)


def _is_cache_fresh(path: Path, ttl_days: int = CACHE_TTL_DAYS) -> bool:
    """Return True if file exists and is younger than ttl_days."""
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime < timedelta(days=ttl_days)


def refresh_data(force: bool = False, skip_events: bool = False):
    """
    Run the full data pipeline with optional cache skipping.

    Args:
        force: Re-download even if cached files are fresh.
        skip_events: Skip StatsBomb event fetch (large/slow).
    """
    print("=" * 50)
    print("  World Cup Predictor - Data Pipeline")
    print("=" * 50)

    for year, season_id in STATSBOMB_SEASONS.items():
        match_path = RAW_DIR / f"statsbomb_matches_wc{year}.csv"
        if force or not _is_cache_fresh(match_path):
            print(f"\n[StatsBomb] Fetching WC {year} match list...")
            fetch_world_cup_matches(season_id=season_id)
        else:
            print(f"\n[StatsBomb] WC {year} matches cached - skipping")

        events_path = RAW_DIR / f"statsbomb_events_wc{year}.csv"
        if not skip_events:
            if force or not _is_cache_fresh(events_path):
                print(f"\n[StatsBomb] Fetching WC {year} events (this takes several minutes)...")
                fetch_all_wc_events(season_id=season_id, force=force)
            else:
                print(f"[StatsBomb] WC {year} events cached - skipping")
        else:
            print(f"[StatsBomb] Skipping WC {year} events (--skip-events)")

    print("\n[Historical] Downloading full martj42 dataset (1872-present)...")
    from src.historical_data import (
        build_team_achievements,
        download_martj42_dataset,
        fetch_fifa_rankings,
    )
    download_martj42_dataset(force=force)
    fetch_fifa_rankings(force=force)
    build_team_achievements()

    results_path = RAW_DIR / "results.csv"
    if force or not _is_cache_fresh(results_path):
        print("\n[ELO] Syncing international results alias...")
        fetch_elo_ratings()
    else:
        print("\n[ELO] International results cached - skipping")

    wc_hist_path = RAW_DIR / "wc_historical_matches.csv"
    if force or not _is_cache_fresh(wc_hist_path):
        print("\n[WC History] Fetching historical WC match results...")
        fetch_wc_historical_matches()
    else:
        print("\n[WC History] Historical matches cached - skipping")

    for year in [2018, 2022]:
        shooting_path = RAW_DIR / f"fbref_shooting_{year}.csv"
        if force or not _is_cache_fresh(shooting_path):
            print(f"\n[FBref] Fetching {year} WC player stats...")
            fetch_fbref_player_stats(season=year)
            time.sleep(2)
        else:
            print(f"\n[FBref] {year} stats cached - skipping")

        tm_path = RAW_DIR / f"transfermarkt_squad_values_{year}.csv"
        if force or not _is_cache_fresh(tm_path):
            print(f"\n[Transfermarkt] Fetching {year} squad values...")
            fetch_transfermarkt_squad_values(season=year)
            time.sleep(2)
        else:
            print(f"\n[Transfermarkt] {year} squad values cached - skipping")

    print("\n[DONE] Data pipeline complete. Check data/raw/ for your files.")
    print("   Next step: python src/features.py")


# ── 1. StatsBomb open data ─────────────────────────────────────────────────

def fetch_statsbomb_competitions():
    """Return all available StatsBomb competitions as a DataFrame."""
    try:
        from statsbombpy import sb
        comps = sb.competitions()
        print(f"  [OK] Found {len(comps)} StatsBomb competitions")
        return comps
    except ImportError:
        print("  [ERR] statsbombpy not installed. Run: pip install statsbombpy")
        return None


def fetch_world_cup_matches(
    season_id: int,
    competition_id: int = STATSBOMB_COMPETITION_ID,
):
    """Fetch all matches for a World Cup season."""
    year = {v: k for k, v in STATSBOMB_SEASONS.items()}.get(season_id, season_id)
    try:
        from statsbombpy import sb
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        save_path = RAW_DIR / f"statsbomb_matches_wc{year}.csv"
        matches.to_csv(save_path, index=False)
        print(f"  [OK] Saved {len(matches)} matches -> {save_path}")
        return matches
    except Exception as e:
        print(f"  [ERR] Error fetching matches: {e}")
        return None


def fetch_match_events(match_id: int):
    """Fetch all event data for a single match."""
    try:
        from statsbombpy import sb
        return sb.events(match_id=match_id)
    except Exception as e:
        print(f"  [ERR] Error fetching events for match {match_id}: {e}")
        return None


def fetch_all_wc_events(season_id: int, force: bool = False):
    """Fetch and save event data for every match in a World Cup."""
    year = {v: k for k, v in STATSBOMB_SEASONS.items()}.get(season_id, season_id)
    save_path = RAW_DIR / f"statsbomb_events_wc{year}.csv"

    if not force and _is_cache_fresh(save_path):
        print(f"  [OK] Events already cached at {save_path}")
        return pd.read_csv(save_path, low_memory=False)

    matches_path = RAW_DIR / f"statsbomb_matches_wc{year}.csv"
    if matches_path.exists():
        matches = pd.read_csv(matches_path)
    else:
        matches = fetch_world_cup_matches(season_id)

    if matches is None or matches.empty:
        return None

    print(f"\n  Fetching events for all {len(matches)} WC {year} matches...")
    all_events = []

    for _, match in tqdm(matches.iterrows(), total=len(matches)):
        match_id = match["match_id"]
        events = fetch_match_events(match_id)
        if events is not None:
            events["match_id"] = match_id
            all_events.append(events)

    if all_events:
        combined = pd.concat(all_events, ignore_index=True)
        combined.to_csv(save_path, index=False)
        print(f"  [OK] Saved {len(combined)} events -> {save_path}")
        return combined

    return None


# ── 2. FBref via soccerdata ────────────────────────────────────────────────

def fetch_fbref_player_stats(league: str = "INT-World Cup", season: int = 2022):
    """Fetch player stats from FBref for a given competition."""
    try:
        import soccerdata as sd
        fbref = sd.FBref(leagues=league, seasons=season)

        print(f"  Fetching FBref shooting stats ({league} {season})...")
        time.sleep(2)
        shooting = fbref.read_player_season_stats("shooting")
        shooting.to_csv(RAW_DIR / f"fbref_shooting_{season}.csv")
        print(f"  [OK] Saved shooting stats ({len(shooting)} rows)")

        print("  Fetching FBref passing stats...")
        time.sleep(2)
        passing = fbref.read_player_season_stats("passing")
        passing.to_csv(RAW_DIR / f"fbref_passing_{season}.csv")
        print(f"  [OK] Saved passing stats ({len(passing)} rows)")

        print("  Fetching FBref goalkeeper stats...")
        time.sleep(2)
        gk = fbref.read_player_season_stats("keeper")
        gk.to_csv(RAW_DIR / f"fbref_goalkeeper_{season}.csv")
        print(f"  [OK] Saved goalkeeper stats ({len(gk)} rows)")

        return shooting, passing, gk

    except ImportError:
        print("  [ERR] soccerdata not installed. Run: pip install soccerdata")
    except Exception as e:
        print(f"  [ERR] FBref error: {e}")
        _create_fbref_fallback(season)


def _create_fbref_fallback(season: int):
    """Create minimal fallback CSVs when FBref scraping fails."""
    print(f"  -> Creating fallback FBref data for {season} (scraping unavailable)")
    # Minimal structure so pipeline can continue
    fallback_players = pd.DataFrame({
        "player": ["Kylian Mbappé", "Lionel Messi", "Harry Kane", "Alisson"],
        "team": ["France", "Argentina", "England", "Brazil"],
    })
    for stat_type in ["shooting", "passing", "goalkeeper"]:
        path = RAW_DIR / f"fbref_{stat_type}_{season}.csv"
        if not path.exists():
            fallback_players.to_csv(path, index=False)


# ── 3. ELO ratings ────────────────────────────────────────────────────────

def fetch_elo_ratings():
    """Download international match results used for ELO calculation."""
    try:
        print("  Fetching international match results (ELO source)...")
        df = pd.read_csv(INTERNATIONAL_RESULTS_URL)
        save_path = RAW_DIR / "results.csv"
        df.to_csv(save_path, index=False)
        # Backward-compatible alias
        df.to_csv(RAW_DIR / "international_results.csv", index=False)
        print(f"  [OK] Saved {len(df)} international results -> {save_path}")
        return df
    except Exception as e:
        print(f"  [ERR] Could not fetch ELO data: {e}")
        return None


def fetch_wc_historical_matches():
    """
    Download World Cup match results for wc_win_rate and group fixtures.
    Uses international results filtered to World Cup tournaments.
    """
    try:
        url = (
            "https://raw.githubusercontent.com/"
            "martj42/international_results/master/results.csv"
        )
        df = pd.read_csv(url)
        # Filter to World Cup matches (tournament column if present, else by tournament name)
        if "tournament" in df.columns:
            wc = df[df["tournament"].str.contains("World Cup", case=False, na=False)]
        else:
            wc = df[df.get("competition", pd.Series()).str.contains("World Cup", case=False, na=False)]

        if wc.empty:
            # Fallback: use all matches - wc_win_rate will be approximate
            wc = df

        save_path = RAW_DIR / "wc_historical_matches.csv"
        wc.to_csv(save_path, index=False)
        print(f"  [OK] Saved {len(wc)} WC historical matches -> {save_path}")
        return wc
    except Exception as e:
        print(f"  [ERR] Could not fetch WC historical data: {e}")
        return None


# ── 4. Squad market values from Transfermarkt ─────────────────────────────

def fetch_transfermarkt_squad_values(season: int = 2022):
    """Build squad value estimates from StatsBomb match participants."""
    try:
        matches_path = RAW_DIR / f"statsbomb_matches_wc{season}.csv"
        if not matches_path.exists():
            _create_transfermarkt_fallback(season)
            return None

        matches = pd.read_csv(matches_path)
        teams = set()
        for col in ["home_team", "away_team"]:
            if col in matches.columns:
                teams.update(matches[col].unique())

        # ELO-based squad value proxy when market data unavailable
        results_path = RAW_DIR / "international_results.csv"
        if results_path.exists():
            from src.features import calculate_elo_ratings
            from src.labels import normalize_team_name
            results = pd.read_csv(results_path)
            results["date"] = pd.to_datetime(results["date"])
            elo = calculate_elo_ratings(results)
            squad_values = pd.DataFrame({
                "team": [normalize_team_name(t) for t in teams],
            })
            elo_df = pd.DataFrame(list(elo.items()), columns=["team", "elo"])
            squad_values = squad_values.merge(elo_df, on="team", how="left")
            squad_values["squad_value"] = squad_values["elo"] * 1e6
            squad_values = squad_values.drop(columns=["elo"])
        else:
            squad_values = pd.DataFrame({
                "team": list(teams),
                "squad_value": [1e9] * len(teams),
            })

        save_path = RAW_DIR / f"transfermarkt_squad_values_{season}.csv"
        squad_values.to_csv(save_path, index=False)
        print(f"  [OK] Saved squad values (ELO proxy) -> {save_path}")
        return squad_values
    except Exception as e:
        print(f"  [ERR] Squad values error: {e}")
        _create_transfermarkt_fallback(season)
        return None


def _create_transfermarkt_fallback(season: int):
    """Create fallback squad values when Transfermarkt scraping fails."""
    path = RAW_DIR / f"transfermarkt_squad_values_{season}.csv"
    if path.exists():
        return
    fallback = pd.DataFrame({
        "team": ["Brazil", "France", "England", "Argentina", "Germany"],
        "squad_value": [1.1e9, 1.0e9, 1.4e9, 0.7e9, 0.9e9],
    })
    fallback.to_csv(path, index=False)
    print(f"  -> Created fallback squad values -> {path}")


# ── Main runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World Cup Predictor data pipeline")
    parser.add_argument(
        "--refresh", action="store_true",
        help="Force re-download even if cached files are fresh",
    )
    parser.add_argument(
        "--skip-events", action="store_true",
        help="Skip StatsBomb event fetch (large/slow)",
    )
    args = parser.parse_args()
    refresh_data(force=args.refresh, skip_events=args.skip_events)
