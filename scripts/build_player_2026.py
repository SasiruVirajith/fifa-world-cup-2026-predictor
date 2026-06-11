"""
Build 2026 player award features and supporting outputs.

Group simulation must exist before player features are built (for progression_factor).
Normally written by build_wc2026.py from the same full-tournament sim run.
If missing, this script runs a fallback group-only sim first (unless --skip-sim).

Usage:
    python scripts/build_player_2026.py
    python scripts/build_player_2026.py --no-fetch-club --use-cache
    python scripts/build_player_2026.py --run-group-sim   REM player-only refresh
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.config import OUTPUTS_DIR
from src.historical_data import download_martj42_dataset
from src.player_features_2026 import build_all_2026_player_features
from src.wc2026_group_sim import run_wc2026_group_simulation

GROUP_SIM_PATH = OUTPUTS_DIR / "group_simulation_2026.csv"


def ensure_group_simulation(
    *,
    skip_sim: bool,
    run_group_sim: bool,
    n_simulations: int = 500,
) -> None:
    """
    Player features read group_simulation_2026.csv during build  -  run sim first if needed.
    """
    if GROUP_SIM_PATH.exists() and not run_group_sim:
        print(f"Using group simulation -> {GROUP_SIM_PATH}")
        return

    if skip_sim:
        if not GROUP_SIM_PATH.exists():
            print(
                "[WARN] group_simulation_2026.csv missing. Run scripts/build_wc2026.py first "
                "or omit --skip-sim / pass --run-group-sim.",
            )
        return

    print("\n[Pre] WC 2026 group simulation (before player features)...")
    run_wc2026_group_simulation(n_simulations=n_simulations)


def main(
    fetch_club: bool = True,
    fetch_force: bool = False,
    skip_sim: bool = False,
    run_group_sim: bool = False,
    group_simulations: int = 500,
    refresh_data: bool = True,
):
    load_dotenv()

    if refresh_data:
        print("Downloading latest martj42 data...")
        download_martj42_dataset(force=True)
        from src.historical_data import fetch_fifa_rankings
        fetch_fifa_rankings(force=True)
    else:
        print("Downloading martj42 data (TTL cache)...")
        download_martj42_dataset(force=False)

    ensure_group_simulation(
        skip_sim=skip_sim,
        run_group_sim=run_group_sim,
        n_simulations=group_simulations,
    )

    build_all_2026_player_features(
        fetch_club=fetch_club,
        fetch_club_force=fetch_force,
    )

    print("\n[DONE] Player 2026 rebuild complete.")
    print("  Next: streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 2026 player award features")
    parser.add_argument(
        "--no-fetch-club",
        action="store_true",
        help="Skip club HTTP fetch (use cached data/raw/club/ JSON only)",
    )
    parser.add_argument(
        "--fetch-club-force",
        action="store_true",
        help="Force re-download club stats (ignore long club cache TTL)",
    )
    parser.add_argument(
        "--no-scrape",
        action="store_true",
        help="Alias for --no-fetch-club (legacy)",
    )
    parser.add_argument(
        "--skip-sim",
        action="store_true",
        help="Do not run fallback group sim if CSV missing (requires prior build_wc2026.py)",
    )
    parser.add_argument(
        "--run-group-sim",
        action="store_true",
        help="Force group-only sim before features (player-only path; overwrites group CSV)",
    )
    parser.add_argument(
        "--group-simulations",
        type=int,
        default=500,
        help="Sim count for fallback/--run-group-sim group-only run (default 500)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Skip forced re-download of martj42/FIFA",
    )
    args = parser.parse_args()
    no_fetch = args.no_fetch_club or args.no_scrape
    main(
        fetch_club=not no_fetch,
        fetch_force=args.fetch_club_force,
        skip_sim=args.skip_sim,
        run_group_sim=args.run_group_sim,
        group_simulations=args.group_simulations,
        refresh_data=not args.use_cache,
    )
