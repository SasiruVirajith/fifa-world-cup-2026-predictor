"""Fetch domestic club stats (Understat + API-Football)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.player_club import club_data_status, fetch_and_cache_club_stats, load_club_from_cache


def main(force: bool = False, cache_only: bool = False) -> None:
    load_dotenv()
    print("=" * 60)
    print("  Club stats fetch (Understat + API-Football)")
    print("=" * 60)

    if cache_only:
        load_club_from_cache()
    else:
        fetch_and_cache_club_stats(force=force)

    status = club_data_status()
    print(f"\n  Status: {status}")
    if not cache_only and not force and status["api_files"] == 0:
        print("\n  Tip: set APIFOOTBALL_KEY in .env for Tier B / MLS / Saudi coverage.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch club player stats")
    parser.add_argument("--force", action="store_true", help="Ignore cache TTL")
    parser.add_argument("--cache-only", action="store_true", help="Load from cache only")
    args = parser.parse_args()
    main(force=args.force, cache_only=args.cache_only)
