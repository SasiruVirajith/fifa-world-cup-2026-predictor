# Copyright (c) 2026 Sasiru Virajith Kankanamge
# SPDX-License-Identifier: MIT

"""
FIFA World Cup 2026 Predictor
Built by: K. Sasiru Virajith
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.build_metadata import write_build_metadata
from src.historical_data import (
    build_team_achievements,
    download_martj42_dataset,
    fetch_fifa_rankings,
)
from src.match_features import build_match_features, build_team_strength_table
from src.match_model import train_match_model
from src.wc2026_simulator import run_wc2026_simulation


def run_wc2026_pipeline(
    n_simulations: int = 5000,
    *,
    refresh_data: bool = True,
    n_workers: int | None = None,
):
    print("\n[1/5] Historical data...")
    if refresh_data:
        print("  fetching martj42 + FIFA")
    else:
        print("  using cached raw files")
    download_martj42_dataset(force=refresh_data)
    fetch_fifa_rankings(force=refresh_data)
    build_team_achievements()

    print("\n[2/5] Building match features...")
    build_match_features()
    build_team_strength_table()

    print("\n[3/5] Training match outcome model...")
    train_match_model()

    print("\n[4/5] Running tournament simulations...")
    results = run_wc2026_simulation(n_simulations=n_simulations, n_workers=n_workers)

    meta_path = write_build_metadata(
        n_simulations=n_simulations,
        refresh_data=refresh_data,
    )
    print(f"\n[5/5] Build metadata -> {meta_path}")
    print(results.head(15).to_string(index=False))
    return results


def main(
    n_simulations: int = 5000,
    refresh_data: bool = True,
    n_workers: int | None = None,
):
    return run_wc2026_pipeline(
        n_simulations=n_simulations,
        refresh_data=refresh_data,
        n_workers=n_workers,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WC 2026 champion pipeline")
    parser.add_argument("--simulations", type=int, default=5000)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel simulation workers (default: min(CPU cores, 4))",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Skip forced re-download; refresh martj42/FIFA only when older than TTL",
    )
    args = parser.parse_args()
    main(
        n_simulations=args.simulations,
        refresh_data=not args.use_cache,
        n_workers=args.workers,
    )
