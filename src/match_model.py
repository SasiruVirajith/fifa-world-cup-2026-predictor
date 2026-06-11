# Copyright (c) 2026 Sasiru Virajith Kankanamge
# SPDX-License-Identifier: MIT

"""
FIFA World Cup 2026 Predictor
Built by: K. Sasiru Virajith
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODELS_DIR, PROCESSED_DIR

MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "fifa_diff", "elo_diff",
    "home_last5_win_rate", "away_last5_win_rate",
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_matches_played",
    "home_penalty_win_rate", "away_penalty_win_rate",
    "fifa_diff_x_home_form", "fifa_diff_x_away_form",
    "h2h_effective", "is_friendly", "is_tournament",
]


def train_match_model(
    features_path: str = None,
    sample_weight_tournaments: bool = True,
) -> GradientBoostingClassifier:
    print("Training match outcome model (Gradient Boosting)...")

    path = Path(features_path or PROCESSED_DIR / "match_features.csv")
    if not path.exists():
        print(f"  [ERR] {path} not found. Run match_features.py first.")
        return None

    df = pd.read_csv(path)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0)
    y = df["outcome"]

    # Weight competitive matches higher than friendlies
    weights = np.ones(len(df))
    if sample_weight_tournaments and "is_tournament" in df.columns:
        weights += df["is_tournament"] * 2
        weights += (1 - df.get("is_friendly", 0)) * 0.5

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, weights, test_size=0.15, random_state=42, stratify=y,
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["away", "draw", "home"]))

    save_path = MODELS_DIR / "match_outcome.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": available}, f)
    print(f"  [OK] Model saved -> {save_path}")
    return model


def load_match_model() -> tuple:
    path = MODELS_DIR / "match_outcome.pkl"
    if not path.exists():
        return None, FEATURE_COLS
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data.get("feature_cols", FEATURE_COLS)


if __name__ == "__main__":
    train_match_model()
