# Copyright (c) 2026 Sasiru Virajith Kankanamge
# SPDX-License-Identifier: MIT

"""
FIFA World Cup 2026 Predictor
Built by: K. Sasiru Virajith
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR

METADATA_PATH = OUTPUTS_DIR / "build_metadata.json"


def _martj42_max_date() -> str | None:
    results_path = RAW_DIR / "results.csv"
    if not results_path.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(results_path, usecols=["date"])
        return str(pd.to_datetime(df["date"]).max().date())
    except Exception:
        return None


def write_build_metadata(
    *,
    n_simulations: int,
    refresh_data: bool,
    pipeline: str = "wc2026",
) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    champion_path = OUTPUTS_DIR / "wc2026_champion_probabilities.csv"
    match_features_path = PROCESSED_DIR / "match_features.csv"
    model_path = Path("models") / "match_outcome.pkl"

    meta = {
        "pipeline": pipeline,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_simulations": n_simulations,
        "refresh_data": refresh_data,
        "martj42_results_max_date": _martj42_max_date(),
        "artifacts": {
            "champion_probabilities": champion_path.exists(),
            "group_simulation": (OUTPUTS_DIR / "group_simulation_2026.csv").exists(),
            "match_features": match_features_path.exists(),
            "match_model": model_path.exists(),
        },
    }

    METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return METADATA_PATH


def load_build_metadata() -> dict | None:
    if not METADATA_PATH.exists():
        return None
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
