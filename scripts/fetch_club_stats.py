# Copyright (c) 2026 Sasiru Virajith Kankanamge
# SPDX-License-Identifier: MIT

"""
FIFA World Cup 2026 Predictor
Built by: K. Sasiru Virajith
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.player_club import club_data_status, fetch_and_cache_club_stats, load_club_from_cache


def main(force: bool = False, cache_only: bool = False) -> None:
    load_dotenv()

    if cache_only:
        load_club_from_cache()
    else:
        fetch_and_cache_club_stats(force=force)

    status = club_data_status()
    print(f"Club stats: {status}")
    if not cache_only and not force and status["api_files"] == 0:
        print("Tip: set APIFOOTBALL_KEY in .env for Tier B / MLS / Saudi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch club player stats")
    parser.add_argument("--force", action="store_true", help="Ignore cache TTL")
    parser.add_argument("--cache-only", action="store_true", help="Load from cache only")
    args = parser.parse_args()
    main(force=args.force, cache_only=args.cache_only)
