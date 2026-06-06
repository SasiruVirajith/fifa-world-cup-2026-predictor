"""Regenerate Tier 2 output files (simulations, upsets, POT, depth)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.player_tournament import build_player_tournament_scores
from src.simulator import run_group_simulation
from src.squad_depth import compute_squad_depth
from src.upset_detector import predict_upsets

if __name__ == "__main__":
    for year in [2018, 2022]:
        print(f"Generating outputs for WC {year}...")
        run_group_simulation(year, n_simulations=2000)
        predict_upsets(year)
        build_player_tournament_scores(year)
        compute_squad_depth(year)
    print("Done.")
