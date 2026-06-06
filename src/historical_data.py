"""
historical_data.py
──────────────────
Download and prepare international football data.

Sources:
  - martj42 GitHub (results, goalscorers, shootouts)
  - FIFA api.fifa.com (men's world rankings)
  - Team achievements: canonical record-book honours + martj42 participation recency
"""

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.achievements_canonical import build_canonical_achievements, canon_team
from src.config import (
    FIFA_RANKINGS_API,
    FIFA_RANKINGS_LIMIT,
    MARTJ42_BASE,
    MARTJ42_FILES,
    RAW_DIR,
    WC2026_ALL_TEAMS,
    WC_YEARS,
)
from src.labels import normalize_team_name

RAW_DIR.mkdir(parents=True, exist_ok=True)

CONTINENTAL_KEYWORDS = {
    "euro": ["european championship", "uefa euro", "euro "],
    "copa": ["copa am"],
    "afcon": ["africa cup", "african cup", "afcon"],
    "asian": ["asian cup", "afc asian"],
    "gold": ["gold cup", "concacaf championship"],
}


def download_martj42_dataset(force: bool = False) -> dict:
    """Download all martj42 international results files from GitHub."""
    paths = {}
    for name, filename in MARTJ42_FILES.items():
        path = RAW_DIR / filename
        if path.exists() and not force:
            print(f"  [cached] {filename}")
            paths[name] = path
            continue
        url = f"{MARTJ42_BASE}/{filename}"
        print(f"  Downloading {filename}...")
        try:
            df = pd.read_csv(url)
            df.to_csv(path, index=False)
            print(f"  [OK] {filename} ({len(df)} rows)")
            paths[name] = path
        except Exception as e:
            print(f"  [ERR] {filename}: {e}")
    return paths


def _fifa_team_name(raw: str) -> str:
    """Map FIFA API country label to project team name."""
    return normalize_team_name(raw.strip())


def fetch_fifa_rankings(force: bool = False) -> pd.DataFrame:
    """Fetch latest men's FIFA world rankings from api.fifa.com."""
    path = RAW_DIR / "fifa_latest_ranking.csv"
    if path.exists() and not force:
        return pd.read_csv(path)

    print("  Fetching FIFA men's rankings from api.fifa.com...")
    try:
        resp = requests.get(
            FIFA_RANKINGS_API,
            params={"gender": 1, "limit": FIFA_RANKINGS_LIMIT},
            timeout=30,
            headers={"User-Agent": "football-predictor/1.0"},
        )
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("Results", [])
        if not results:
            raise ValueError("FIFA API returned no ranking rows")

        rows = []
        ranking_date = results[0].get("PubDate") or results[0].get("PrePubDate")
        for entry in results:
            name = entry["TeamName"][0]["Description"]
            rows.append({
                "date": pd.to_datetime(ranking_date).strftime("%Y-%m-%d")
                if ranking_date
                else pd.Timestamp.today().strftime("%Y-%m-%d"),
                "country": _fifa_team_name(name),
                "rank": entry.get("Rank"),
                "previous_rank": entry.get("PrevRank"),
                "total_points": entry.get("DecimalTotalPoints") or entry.get("TotalPoints"),
                "previous_points": entry.get("DecimalPrevPoints") or entry.get("PrevPoints"),
                "conf": entry.get("ConfederationName"),
            })

        df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
        df["country"] = df["country"].apply(normalize_team_name)
        df.to_csv(path, index=False)
        print(f"  [OK] FIFA rankings ({len(df)} teams, date {df['date'].iloc[0]})")
        return df
    except Exception as e:
        print(f"  [ERR] FIFA rankings: {e}")
        if path.exists():
            print("  [WARN] Using cached fifa_latest_ranking.csv")
            return pd.read_csv(path)
        return pd.DataFrame()


def _continental_bucket(tournament: str) -> str | None:
    t = str(tournament).lower()
    for bucket, keywords in CONTINENTAL_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return bucket
    return None


def _max_year(current, year: int | None) -> int | None:
    if year is None:
        return current
    if current is None or year > current:
        return year
    return current


def _participation_from_martj42(
    results: pd.DataFrame,
    stats: dict[str, dict],
) -> None:
    """
    Overlay participation recency from martj42 only.

    Honour counts and last achievement years stay canonical; martj42 never
    overwrites those fields.
    """
    wc_valid_years = set(WC_YEARS)
    max_cont_year = 2025

    wc = results[
        results["tournament"].str.contains("FIFA World Cup", case=False, na=False)
        & results["year"].isin(wc_valid_years)
    ]
    for year, edition in wc.groupby("year"):
        participants = set(edition["home_team"]) | set(edition["away_team"])
        for raw_team in participants:
            team = canon_team(raw_team)
            if team not in stats:
                continue
            stats[team]["last_world_cup_participation"] = _max_year(
                stats[team]["last_world_cup_participation"], int(year)
            )

    cont = results[
        results["tournament"].apply(_continental_bucket).notna()
        & (results["year"] <= max_cont_year)
    ]
    for year, edition in cont.groupby("year"):
        participants = set(edition["home_team"]) | set(edition["away_team"])
        for raw_team in participants:
            team = canon_team(raw_team)
            if team not in stats:
                continue
            stats[team]["last_continental_participation"] = _max_year(
                stats[team]["last_continental_participation"], int(year)
            )


def build_team_achievements() -> pd.DataFrame:
    """
    Hybrid team achievements for WC 2026 participants.

    - Canonical record-book honours (counts + last achievement years)
    - martj42 for last World Cup / continental participation only
    - All 48 confirmed WC 2026 teams get last_world_cup_participation = 2026
    """
    stats = build_canonical_achievements(WC2026_ALL_TEAMS)

    results_path = RAW_DIR / "results.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
        results["date"] = pd.to_datetime(results["date"])
        results["home_team"] = results["home_team"].apply(normalize_team_name)
        results["away_team"] = results["away_team"].apply(normalize_team_name)
        results["year"] = results["date"].dt.year
        _participation_from_martj42(results, stats)
    else:
        print("  [WARN] results.csv missing — participation years left blank")

    for team in WC2026_ALL_TEAMS:
        stats[team]["last_world_cup_participation"] = 2026

    df = pd.DataFrame(stats.values()).sort_values("team").reset_index(drop=True)
    path = RAW_DIR / "team_achievements.csv"
    df.to_csv(path, index=False)
    print(f"  [OK] Team achievements (hybrid) -> {path} ({len(df)} teams)")
    return df


def download_fifa_rankings(force: bool = False) -> pd.DataFrame:
    """Backward-compatible alias for FIFA rankings fetch."""
    return fetch_fifa_rankings(force=force)


def download_team_achievements(force: bool = False) -> pd.DataFrame:
    """Build hybrid team achievements locally (canonical + martj42 participation)."""
    path = RAW_DIR / "team_achievements.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    return build_team_achievements()


def load_and_normalize_results() -> pd.DataFrame:
    """Load results.csv with normalized team names and parsed dates."""
    path = RAW_DIR / "results.csv"
    if not path.exists():
        path = RAW_DIR / "international_results.csv"
    if not path.exists():
        raise FileNotFoundError("Run historical data download first")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["home_team"] = df["home_team"].apply(normalize_team_name)
    df["away_team"] = df["away_team"].apply(normalize_team_name)
    df["neutral"] = df["neutral"].astype(str).str.upper().isin(["TRUE", "1", "T", "YES"])
    df["year"] = df["date"].dt.year
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_tournament_categories(results: pd.DataFrame) -> pd.DataFrame:
    """Tag matches by type: World Cup, continental, friendly, qualifier, other."""
    df = results.copy()
    t = df["tournament"].fillna("").str.lower()

    df["match_type"] = "other"
    df.loc[t.str.contains("world cup"), "match_type"] = "world_cup"
    df.loc[t.str.contains("friendly"), "match_type"] = "friendly"
    df.loc[t.str.contains("qualif"), "match_type"] = "qualifier"
    continental_kw = [
        "euro", "copa america", "africa cup", "asian cup", "gold cup",
        "nations league", "confederations", "continental",
    ]
    for kw in continental_kw:
        df.loc[t.str.contains(kw), "match_type"] = "continental"

    return df


if __name__ == "__main__":
    print("Downloading martj42 historical dataset...")
    download_martj42_dataset()
    fetch_fifa_rankings(force=True)
    build_team_achievements()
    df = load_and_normalize_results()
    print(f"Loaded {len(df)} matches from {df.date.min().date()} to {df.date.max().date()}")
    cats = get_tournament_categories(df)
    print(cats.match_type.value_counts())
