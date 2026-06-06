"""
Build 2026 player award features and supporting outputs.

Usage:
    python scripts/build_player_2026.py
    python scripts/build_player_2026.py --scrape-club
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.historical_data import download_martj42_dataset
from src.player_features_2026 import build_all_2026_player_features
from src.upset_detector import predict_upsets
from src.wc2026_group_sim import run_wc2026_group_simulation


def main(try_club_scrape: bool = False, skip_sim: bool = False):
    print("Downloading martj42 data (if needed)...")
    download_martj42_dataset()

    build_all_2026_player_features(try_club_scrape=try_club_scrape)

    if not skip_sim:
        print("\n[Extra] WC 2026 group simulation...")
        run_wc2026_group_simulation(n_simulations=500)

    print("\n[Extra] WC 2026 upset detector...")
    predict_upsets(year=2026)

    print("\n[DONE] Player 2026 rebuild complete.")
    print("  Next: streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scrape-club", action="store_true", help="Try soccerdata FBref club scrape")
    parser.add_argument("--skip-sim", action="store_true", help="Skip group-stage Monte Carlo (faster)")
    args = parser.parse_args()
    main(try_club_scrape=args.scrape_club, skip_sim=args.skip_sim)
