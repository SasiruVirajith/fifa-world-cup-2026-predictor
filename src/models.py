"""
models.py
─────────
Trains and saves all prediction models.

Models:
    1. tournament_winner    - XGBoost classifier (team features -> win probability)
    2. golden_boot          - XGBoost regressor (player features -> predicted goals)
    3. golden_glove         - XGBoost classifier (keeper features -> best keeper score)
    4. best_playmaker       - Multi-metric ranking (no training needed - composite score)

Usage:
    python src/models.py

    from src.models import train_winner_model, train_golden_boot_model
"""

import pickle
import sys
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────

def save_model(model, name: str):
    """Save a trained model to the models/ directory."""
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  [OK] Model saved -> {path}")


def load_model(name: str):
    """Load a saved model from the models/ directory."""
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Train it first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_shap_summary(model, X: pd.DataFrame, model_name: str):
    """Generate and save a SHAP feature importance plot."""
    print(f"  Generating SHAP plot for {model_name}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"Feature importance - {model_name}")
    plt.tight_layout()

    save_path = OUTPUTS_DIR / f"shap_{model_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] SHAP plot saved -> {save_path}")


# ── 1. Tournament winner model ─────────────────────────────────────────────

def train_winner_model(features_path: str = "data/processed/team_features.csv"):
    """
    Train an XGBoost classifier to predict World Cup winner probability.

    Target variable: 'won_tournament' (1 if team won WC that year, 0 otherwise)
    This requires labelled training data - you need to add a 'won_tournament'
    column to your team_features.csv based on historical WC results.

    TODO: Once you have labelled data, this function will:
        1. Split into train/test sets
        2. Train XGBoost with cross-validation
        3. Evaluate on holdout set
        4. Generate SHAP feature importance
        5. Save model to models/tournament_winner.pkl
    """
    print("Training tournament winner model...")

    if not Path(features_path).exists():
        print(f"  [ERR] Features not found at {features_path}")
        print("    Run src/features.py first")
        return None

    df = pd.read_csv(features_path)

    # ── Check target column exists
    if "won_tournament" not in df.columns:
        print("  [ERR] 'won_tournament' column missing from team_features.csv")
        print("    You need to add historical WC winner labels first.")
        print("    Add labels via src/labels.py (won_tournament from WC_WINNERS).")
        return None

    feature_cols = ["elo", "win_rate_last20", "goals_scored_avg", "goals_conceded_avg"]
    if "squad_value" in df.columns:
        feature_cols.append("squad_value")

    df = df.dropna(subset=feature_cols + ["won_tournament"])
    X = df[feature_cols].fillna(0)
    y = df["won_tournament"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scale_pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  [OK] Test accuracy: {acc:.3f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"  [OK] 5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    plot_shap_summary(model, X_test, "tournament_winner")
    save_model(model, "tournament_winner")

    return model


# ── 2. Golden Boot model ───────────────────────────────────────────────────

def train_golden_boot_model(features_path: str = "data/processed/striker_features.csv"):
    """
    Train an XGBoost regressor to predict tournament goals scored.

    Target variable: 'tournament_goals' - goals scored in that tournament.
    Feature set: xG per 90, shots per 90, club season goals, etc.

    TODO: Add 'tournament_goals' column using StatsBomb event data.
    """
    print("Training Golden Boot model...")

    if not Path(features_path).exists():
        print(f"  [ERR] Features not found at {features_path}")
        print("    Run src/features.py first")
        return None

    df = pd.read_csv(features_path)

    if "tournament_goals" not in df.columns:
        print("  [ERR] 'tournament_goals' column missing.")
        print("    Derive this from StatsBomb event data via src/labels.py")
        return None

    feature_cols = ["xg_per90", "npxg_per90", "shots_per90", "shots_on_target_pct"]
    df = df.dropna(subset=feature_cols + ["tournament_goals"])

    X = df[feature_cols]
    y = df["tournament_goals"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"  [OK] Test MAE: {mae:.3f} goals")

    plot_shap_summary(model, X_test, "golden_boot")
    save_model(model, "golden_boot")

    return model


# ── 3. Golden Glove model ──────────────────────────────────────────────────

def train_golden_glove_model(features_path: str = "data/processed/goalkeeper_features.csv"):
    """
    Train an XGBoost classifier to predict the Golden Glove winner.

    Target variable: 'won_golden_glove' (1 = award winner, 0 = other).
    Key features: PSxG, save %, clean sheets, goals allowed vs expected.
    """
    print("Training Golden Glove model...")

    if not Path(features_path).exists():
        print(f"  [ERR] Features not found at {features_path}")
        return None

    df = pd.read_csv(features_path)

    if "won_golden_glove" not in df.columns:
        print("  [ERR] 'won_golden_glove' label missing.")
        print("    Add this column manually from historical WC data.")
        return None

    feature_cols = ["save_pct", "clean_sheet_pct", "psxg_minus_ga", "ga90"]
    df = df.dropna(subset=feature_cols + ["won_golden_glove"])

    X = df[feature_cols]
    y = df["won_golden_glove"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scale_pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  [OK] Test accuracy: {acc:.3f}")

    plot_shap_summary(model, X_test, "golden_glove")
    save_model(model, "golden_glove")

    return model


# ── 4. Best Playmaker - composite ranking ──────────────────────────────────

def rank_playmakers(features_path: str = "data/processed/playmaker_features.csv") -> pd.DataFrame:
    """
    Rank players by a weighted composite playmaking score.
    No ML training needed - this is a weighted multi-metric ranking.

    Weights:
        40% xA per 90
        30% key passes per 90
        20% progressive passes per 90
        10% pass completion %

    Returns:
        DataFrame sorted by composite score, descending
    """
    print("Ranking playmakers...")

    if not Path(features_path).exists():
        print(f"  [ERR] Features not found at {features_path}")
        return pd.DataFrame()

    df = pd.read_csv(features_path)

    # Normalise each metric to 0–1 range so weights are meaningful
    def normalise(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.0] * len(series), index=series.index)
        return (series - mn) / (mx - mn)

    required = ["xa_per90", "key_passes_per90", "progressive_passes_per90", "pass_completion_pct"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"  [ERR] Missing columns: {missing}")
        print("    Check your FBref data column names in src/features.py")
        return df

    df["playmaker_score"] = (
        0.40 * normalise(df["xa_per90"]) +
        0.30 * normalise(df["key_passes_per90"]) +
        0.20 * normalise(df["progressive_passes_per90"]) +
        0.10 * normalise(df["pass_completion_pct"])
    )

    ranked = df.sort_values("playmaker_score", ascending=False).reset_index(drop=True)
    save_path = OUTPUTS_DIR / "playmaker_rankings.csv"
    ranked.to_csv(save_path, index=False)
    print(f"  [OK] Playmaker rankings saved -> {save_path}")

    return ranked


# ── Main runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  World Cup Predictor - Model Training")
    print("=" * 50)

    print("\n[1/4] Tournament winner model...")
    train_winner_model()

    print("\n[2/4] Golden Boot model...")
    train_golden_boot_model()

    print("\n[3/4] Golden Glove model...")
    train_golden_glove_model()

    print("\n[4/4] Playmaker rankings...")
    rank_playmakers()

    print("\n[DONE] All models trained. Check models/ and outputs/")
    print("   Next step: streamlit run app.py")
