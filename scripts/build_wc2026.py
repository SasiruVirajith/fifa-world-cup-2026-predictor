"""
Full WC 2026 pipeline:
  1. Download historical data (martj42 + FIFA)
  2. Build match features
  3. Train match model
  4. Run 5000 tournament simulations
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.historical_data import (
    build_team_achievements,
    download_martj42_dataset,
    fetch_fifa_rankings,
)
from src.match_features import build_match_features, build_team_strength_table
from src.match_model import train_match_model
from src.wc2026_simulator import run_wc2026_simulation


def main(n_simulations: int = 5000):
    print("=" * 60)
    print("  WC 2026 Full Build Pipeline")
    print("=" * 60)

    print("\n[1/5] Downloading historical data...")
    download_martj42_dataset()
    fetch_fifa_rankings(force=True)
    build_team_achievements()

    print("\n[2/5] Building match features...")
    build_match_features()
    build_team_strength_table()

    print("\n[3/5] Training match outcome model...")
    train_match_model()

    print("\n[4/5] Running tournament simulations...")
    results = run_wc2026_simulation(n_simulations=n_simulations)

    print("\n[5/5] Done!")
    print(results.head(15).to_string(index=False))
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations", type=int, default=5000)
    args = parser.parse_args()
    main(n_simulations=args.simulations)
