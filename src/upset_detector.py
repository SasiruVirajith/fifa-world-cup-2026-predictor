"""
upset_detector.py
─────────────────
Detect matches most likely to produce giant-killings based on ELO gaps.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR
from src.features import calculate_elo_ratings, _compute_form
from src.labels import normalize_team_name
from src.wc2026_group_sim import get_wc2026_group_fixtures

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def build_historical_upsets() -> pd.DataFrame:
    """Build training data from historical WC matches with upset labels."""
    records = []
    for year in [2018, 2022]:
        matches_path = RAW_DIR / f"statsbomb_matches_wc{year}.csv"
        if not matches_path.exists():
            continue
        matches = pd.read_csv(matches_path)

        results_path = RAW_DIR / "international_results.csv"
        if not results_path.exists():
            continue
        results = pd.read_csv(results_path)
        results["date"] = pd.to_datetime(results["date"])

        for _, m in matches.iterrows():
            home = normalize_team_name(m["home_team"])
            away = normalize_team_name(m["away_team"])
            home_score = m.get("home_score", 0)
            away_score = m.get("away_score", 0)

            if pd.isna(home_score) or pd.isna(away_score):
                continue

            cutoff = pd.Timestamp(str(m.get("match_date", f"{year}-06-01"))[:10])
            historical = results[results["date"] < cutoff].copy()
            elo = calculate_elo_ratings(historical)
            elo_home = elo.get(home, 1500)
            elo_away = elo.get(away, 1500)
            elo_gap = elo_home - elo_away

            home_form = _compute_form(historical, home)
            away_form = _compute_form(historical, away)
            form_volatility = abs(home_form["win_rate_last20"] - away_form["win_rate_last20"])

            # Upset: lower-rated team wins
            if elo_gap > 0:
                is_upset = int(away_score > home_score)
                underdog = away
                favorite = home
            elif elo_gap < 0:
                is_upset = int(home_score > away_score)
                underdog = home
                favorite = away
            else:
                is_upset = 0
                underdog = away
                favorite = home

            records.append({
                "year": year,
                "home_team": home,
                "away_team": away,
                "favorite": favorite,
                "underdog": underdog,
                "elo_gap": abs(elo_gap),
                "form_volatility": form_volatility,
                "is_upset": is_upset,
            })

    return pd.DataFrame(records)


def predict_upsets(year: int = 2022, top_n: int = 10) -> pd.DataFrame:
    """
    Rank group-stage matches by upset probability.

    Uses a logistic model on ELO gap and form volatility.
    """
    from sklearn.linear_model import LogisticRegression

    history = build_historical_upsets()
    if history.empty:
        return pd.DataFrame()

    feature_cols = ["elo_gap", "form_volatility"]
    X = history[feature_cols].fillna(0)
    y = history["is_upset"]

    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X, y)

    results_path = RAW_DIR / "international_results.csv"
    if not results_path.exists():
        results_path = RAW_DIR / "results.csv"
    results = pd.read_csv(results_path)
    results["date"] = pd.to_datetime(results["date"])
    elo = calculate_elo_ratings(results)

    if year == 2026:
        group_matches = get_wc2026_group_fixtures()
        strength_path = PROCESSED_DIR / "team_strength_2026.csv"
        if strength_path.exists():
            strength = pd.read_csv(strength_path)
            rating = dict(zip(strength["team"], strength.get("fifa_points", strength.get("elo", 1500))))
            elo = {normalize_team_name(k): v for k, v in rating.items()}
    else:
        matches_path = RAW_DIR / f"statsbomb_matches_wc{year}.csv"
        if not matches_path.exists():
            return pd.DataFrame()
        matches = pd.read_csv(matches_path)
        stage_col = "competition_stage" if "competition_stage" in matches.columns else "stage"
        if stage_col in matches.columns:
            group_matches = matches[
                matches[stage_col].astype(str).str.contains("Group", case=False, na=False)
            ]
        else:
            group_matches = matches

    if group_matches.empty:
        return pd.DataFrame()

    predictions = []
    for _, m in group_matches.iterrows():
        home = normalize_team_name(m["home_team"])
        away = normalize_team_name(m["away_team"])
        elo_home = float(elo.get(home, 1500))
        elo_away = float(elo.get(away, 1500))
        elo_gap = abs(elo_home - elo_away)

        if elo_home >= elo_away:
            favorite, underdog = home, away
        else:
            favorite, underdog = away, home

        home_form = _compute_form(results, home)
        away_form = _compute_form(results, away)
        form_volatility = abs(home_form["win_rate_last20"] - away_form["win_rate_last20"])

        X_pred = pd.DataFrame([{"elo_gap": elo_gap, "form_volatility": form_volatility}])
        upset_prob = model.predict_proba(X_pred)[0][1]

        predictions.append({
            "home_team": home,
            "away_team": away,
            "favorite": favorite,
            "underdog": underdog,
            "elo_gap": round(elo_gap, 1),
            "upset_probability": round(upset_prob, 4),
            "year": year,
        })

    df = pd.DataFrame(predictions).sort_values("upset_probability", ascending=False).head(top_n)
    save_path = OUTPUTS_DIR / f"upset_predictions_{year}.csv"
    df.to_csv(save_path, index=False)
    return df
