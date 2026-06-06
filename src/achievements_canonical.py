"""
achievements_canonical.py
─────────────────────────
Record-book team honours for WC 2026 participants.

Counts and last-achievement years come from canonical football history
(edition tables below). Martj42 is used only for participation recency
(see historical_data.build_team_achievements).
"""

from __future__ import annotations

from src.config import WC2026_ALL_TEAMS
from src.labels import normalize_team_name

# Map dissolved / renamed nations to current WC 2026 labels
HISTORICAL_TEAM_NAMES = {
    "West Germany": "Germany",
    "Germany FR": "Germany",
    "Turkey": "Türkiye",
    "Czechoslovakia": "Czechia",
    "Zaire": "Congo DR",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Cabo Verde": "Cape Verde",
    "DR Congo": "Congo DR",
    "Democratic Republic of the Congo": "Congo DR",
    "Czech Republic": "Czechia",
}

# (year, winner, runner_up, semi_finalists)
WC_EDITIONS: list[tuple[int, str, str, tuple[str, str]]] = [
    (1930, "Uruguay", "Argentina", ("USA", "Yugoslavia")),
    (1934, "Italy", "Czechoslovakia", ("Germany", "Austria")),
    (1938, "Italy", "Hungary", ("Brazil", "Sweden")),
    (1950, "Uruguay", "Brazil", ("Sweden", "Spain")),
    (1954, "West Germany", "Hungary", ("Austria", "Uruguay")),
    (1958, "Brazil", "Sweden", ("France", "West Germany")),
    (1962, "Brazil", "Czechoslovakia", ("Chile", "Yugoslavia")),
    (1966, "England", "West Germany", ("Portugal", "Soviet Union")),
    (1970, "Brazil", "Italy", ("West Germany", "Uruguay")),
    (1974, "West Germany", "Netherlands", ("Brazil", "Poland")),
    (1978, "Argentina", "Netherlands", ("Brazil", "Italy")),
    (1982, "Italy", "West Germany", ("France", "Poland")),
    (1986, "Argentina", "West Germany", ("France", "Belgium")),
    (1990, "Germany", "Argentina", ("Italy", "England")),
    (1994, "Brazil", "Italy", ("Sweden", "Bulgaria")),
    (1998, "France", "Brazil", ("Netherlands", "Croatia")),
    (2002, "Brazil", "Germany", ("South Korea", "Turkey")),
    (2006, "Italy", "France", ("Germany", "Portugal")),
    (2010, "Spain", "Netherlands", ("Germany", "Uruguay")),
    (2014, "Germany", "Argentina", ("Brazil", "Netherlands")),
    (2018, "France", "Croatia", ("Belgium", "England")),
    (2022, "Argentina", "France", ("Croatia", "Morocco")),
]

# Continental honours curated for WC 2026 teams (record-book totals + recency through 2024/25 cycle)
CONTINENTAL_BY_TEAM: dict[str, dict] = {
    "Algeria": {
        "cont_titles": 2, "cont_finals": 3, "cont_semis": 6,
        "cont_last_win_year": 2019, "cont_last_final_year": 2019, "cont_last_semi_year": 2023,
    },
    "Argentina": {
        "cont_titles": 16, "cont_finals": 24, "cont_semis": 30,
        "cont_last_win_year": 2024, "cont_last_final_year": 2024, "cont_last_semi_year": 2024,
    },
    "Australia": {
        "cont_titles": 5, "cont_finals": 6, "cont_semis": 8,
        "cont_last_win_year": 2015, "cont_last_final_year": 2015, "cont_last_semi_year": 2019,
    },
    "Austria": {
        "cont_titles": 0, "cont_finals": 1, "cont_semis": 2,
        "cont_last_win_year": None, "cont_last_final_year": 1959, "cont_last_semi_year": 2008,
    },
    "Belgium": {
        "cont_titles": 0, "cont_finals": 1, "cont_semis": 2,
        "cont_last_win_year": None, "cont_last_final_year": 1980, "cont_last_semi_year": 1980,
    },
    "Bosnia and Herzegovina": {
        "cont_titles": 0, "cont_finals": 0, "cont_semis": 0,
        "cont_last_win_year": None, "cont_last_final_year": None, "cont_last_semi_year": None,
    },
    "Brazil": {
        "cont_titles": 9, "cont_finals": 12, "cont_semis": 15,
        "cont_last_win_year": 2019, "cont_last_final_year": 2021, "cont_last_semi_year": 2021,
    },
    "Canada": {
        "cont_titles": 2, "cont_finals": 3, "cont_semis": 4,
        "cont_last_win_year": 2000, "cont_last_final_year": 2000, "cont_last_semi_year": 2023,
    },
    "Cape Verde": {
        "cont_titles": 0, "cont_finals": 0, "cont_semis": 1,
        "cont_last_win_year": None, "cont_last_final_year": None, "cont_last_semi_year": 2013,
    },
    "Colombia": {
        "cont_titles": 1, "cont_finals": 2, "cont_semis": 6,
        "cont_last_win_year": 2001, "cont_last_final_year": 2001, "cont_last_semi_year": 2024,
    },
    "Congo DR": {
        "cont_titles": 2, "cont_finals": 2, "cont_semis": 4,
        "cont_last_win_year": 1974, "cont_last_final_year": 1974, "cont_last_semi_year": 2015,
    },
    "Croatia": {
        "cont_titles": 0, "cont_finals": 0, "cont_semis": 1,
        "cont_last_win_year": None, "cont_last_final_year": None, "cont_last_semi_year": 2024,
    },
    "Curaçao": {
        "cont_titles": 0, "cont_finals": 0, "cont_semis": 0,
        "cont_last_win_year": None, "cont_last_final_year": None, "cont_last_semi_year": None,
    },
    "Czechia": {
        "cont_titles": 1, "cont_finals": 1, "cont_semis": 2,
        "cont_last_win_year": 1976, "cont_last_final_year": 1976, "cont_last_semi_year": 1996,
    },
    "Ecuador": {
        "cont_titles": 0, "cont_finals": 0, "cont_semis": 1,
        "cont_last_win_year": None, "cont_last_final_year": None, "cont_last_semi_year": 2024,
    },
    "Egypt": {
        "cont_titles": 7, "cont_finals": 9, "cont_semis": 12,
        "cont_last_win_year": 2010, "cont_last_final_year": 2010, "cont_last_semi_year": 2021,
    },
    "England": {
        "cont_titles": 0, "cont_finals": 2, "cont_semis": 3,
        "cont_last_win_year": None, "cont_last_final_year": 2021, "cont_last_semi_year": 2024,
    },
    "France": {
        "cont_titles": 2, "cont_finals": 3, "cont_semis": 5,
        "cont_last_win_year": 2000, "cont_last_final_year": 2016, "cont_last_semi_year": 2016,
    },
    "Germany": {
        "cont_titles": 3, "cont_finals": 4, "cont_semis": 8,
        "cont_last_win_year": 1996, "cont_last_final_year": 1996, "cont_last_semi_year": 2016,
    },
    "Ghana": {
        "cont_titles": 4, "cont_finals": 5, "cont_semis": 8,
        "cont_last_win_year": 1982, "cont_last_final_year": 1982, "cont_last_semi_year": 2015,
    },
    "Haiti": {
        "cont_titles": 1, "cont_finals": 1, "cont_semis": 2,
        "cont_last_win_year": 1973, "cont_last_final_year": 1973, "cont_last_semi_year": 1973,
    },
    "Iran": {
        "cont_titles": 0, "cont_finals": 3, "cont_semis": 5,
        "cont_last_win_year": None, "cont_last_final_year": 2007, "cont_last_semi_year": 2024,
    },
    "Iraq": {
        "cont_titles": 1, "cont_finals": 1, "cont_semis": 2,
        "cont_last_win_year": 2007, "cont_last_final_year": 2007, "cont_last_semi_year": 2007,
    },
    "Ivory Coast": {
        "cont_titles": 2, "cont_finals": 3, "cont_semis": 6,
        "cont_last_win_year": 2015, "cont_last_final_year": 2015, "cont_last_semi_year": 2024,
    },
    "Japan": {
        "cont_titles": 4, "cont_finals": 5, "cont_semis": 7,
        "cont_last_win_year": 2011, "cont_last_final_year": 2011, "cont_last_semi_year": 2023,
    },
    "Jordan": {
        "cont_titles": 0, "cont_finals": 0, "cont_semis": 1,
        "cont_last_win_year": None, "cont_last_final_year": None, "cont_last_semi_year": 2024,
    },
    "Mexico": {
        "cont_titles": 9, "cont_finals": 12, "cont_semis": 14,
        "cont_last_win_year": 2023, "cont_last_final_year": 2023, "cont_last_semi_year": 2023,
    },
    "Morocco": {
        "cont_titles": 1, "cont_finals": 3, "cont_semis": 5,
        "cont_last_win_year": 1976, "cont_last_final_year": 2004, "cont_last_semi_year": 2022,
    },
    "Netherlands": {
        "cont_titles": 1, "cont_finals": 1, "cont_semis": 4,
        "cont_last_win_year": 1988, "cont_last_final_year": 1988, "cont_last_semi_year": 2014,
    },
    "New Zealand": {
        "cont_titles": 5, "cont_finals": 6, "cont_semis": 7,
        "cont_last_win_year": 2016, "cont_last_final_year": 2016, "cont_last_semi_year": 2024,
    },
    "Norway": {
        "cont_titles": 0, "cont_finals": 0, "cont_semis": 0,
        "cont_last_win_year": None, "cont_last_final_year": None, "cont_last_semi_year": None,
    },
    "Panama": {
        "cont_titles": 0, "cont_finals": 0, "cont_semis": 1,
        "cont_last_win_year": None, "cont_last_final_year": None, "cont_last_semi_year": 2023,
    },
    "Paraguay": {
        "cont_titles": 2, "cont_finals": 4, "cont_semis": 6,
        "cont_last_win_year": 1979, "cont_last_final_year": 1979, "cont_last_semi_year": 2015,
    },
    "Portugal": {
        "cont_titles": 1, "cont_finals": 1, "cont_semis": 3,
        "cont_last_win_year": 2016, "cont_last_final_year": 2016, "cont_last_semi_year": 2024,
    },
    "Qatar": {
        "cont_titles": 1, "cont_finals": 1, "cont_semis": 2,
        "cont_last_win_year": 2019, "cont_last_final_year": 2019, "cont_last_semi_year": 2019,
    },
    "Saudi Arabia": {
        "cont_titles": 0, "cont_finals": 1, "cont_semis": 3,
        "cont_last_win_year": None, "cont_last_final_year": 2007, "cont_last_semi_year": 2024,
    },
    "Scotland": {
        "cont_titles": 0, "cont_finals": 0, "cont_semis": 0,
        "cont_last_win_year": None, "cont_last_final_year": None, "cont_last_semi_year": None,
    },
    "Senegal": {
        "cont_titles": 1, "cont_finals": 2, "cont_semis": 4,
        "cont_last_win_year": 2021, "cont_last_final_year": 2021, "cont_last_semi_year": 2023,
    },
    "South Africa": {
        "cont_titles": 1, "cont_finals": 1, "cont_semis": 3,
        "cont_last_win_year": 1996, "cont_last_final_year": 1996, "cont_last_semi_year": 2013,
    },
    "South Korea": {
        "cont_titles": 2, "cont_finals": 4, "cont_semis": 6,
        "cont_last_win_year": 1960, "cont_last_final_year": 2015, "cont_last_semi_year": 2023,
    },
    "Spain": {
        "cont_titles": 4, "cont_finals": 4, "cont_semis": 5,
        "cont_last_win_year": 2024, "cont_last_final_year": 2024, "cont_last_semi_year": 2024,
    },
    "Sweden": {
        "cont_titles": 0, "cont_finals": 0, "cont_semis": 1,
        "cont_last_win_year": None, "cont_last_final_year": None, "cont_last_semi_year": 1992,
    },
    "Switzerland": {
        "cont_titles": 0, "cont_finals": 0, "cont_semis": 0,
        "cont_last_win_year": None, "cont_last_final_year": None, "cont_last_semi_year": None,
    },
    "Tunisia": {
        "cont_titles": 1, "cont_finals": 2, "cont_semis": 4,
        "cont_last_win_year": 2004, "cont_last_final_year": 2004, "cont_last_semi_year": 2021,
    },
    "Türkiye": {
        "cont_titles": 0, "cont_finals": 0, "cont_semis": 1,
        "cont_last_win_year": None, "cont_last_final_year": None, "cont_last_semi_year": 2008,
    },
    "United States": {
        "cont_titles": 7, "cont_finals": 9, "cont_semis": 11,
        "cont_last_win_year": 2021, "cont_last_final_year": 2021, "cont_last_semi_year": 2023,
    },
    "Uruguay": {
        "cont_titles": 15, "cont_finals": 18, "cont_semis": 20,
        "cont_last_win_year": 2011, "cont_last_final_year": 2011, "cont_last_semi_year": 2024,
    },
    "Uzbekistan": {
        "cont_titles": 0, "cont_finals": 0, "cont_semis": 1,
        "cont_last_win_year": None, "cont_last_final_year": None, "cont_last_semi_year": 2015,
    },
}


def canon_team(name: str) -> str:
    """Resolve historical labels to WC 2026 team names."""
    return normalize_team_name(HISTORICAL_TEAM_NAMES.get(name, name))


def _empty_row(team: str) -> dict:
    return {
        "team": team,
        "wc_titles": 0,
        "wc_finals": 0,
        "wc_semis": 0,
        "wc_last_win_year": None,
        "wc_last_final_year": None,
        "wc_last_semi_year": None,
        "cont_titles": 0,
        "cont_finals": 0,
        "cont_semis": 0,
        "cont_last_win_year": None,
        "cont_last_final_year": None,
        "cont_last_semi_year": None,
        "last_world_cup_participation": None,
        "last_continental_participation": None,
    }


def _bump_year(current, year: int | None) -> int | None:
    if year is None:
        return current
    if current is None or year > current:
        return year
    return current


def compute_wc_stats(team: str) -> dict:
    """Derive World Cup honour counts and recency from edition table."""
    row = {
        "wc_titles": 0,
        "wc_finals": 0,
        "wc_semis": 0,
        "wc_last_win_year": None,
        "wc_last_final_year": None,
        "wc_last_semi_year": None,
    }
    team = canon_team(team)

    for year, winner, runner_up, semis in WC_EDITIONS:
        winner = canon_team(winner)
        runner_up = canon_team(runner_up)
        semi_teams = {canon_team(s) for s in semis}
        finalists = {winner, runner_up}

        if team == winner:
            row["wc_titles"] += 1
            row["wc_last_win_year"] = _bump_year(row["wc_last_win_year"], year)
        if team in finalists:
            row["wc_finals"] += 1
            row["wc_last_final_year"] = _bump_year(row["wc_last_final_year"], year)
        if team in finalists | semi_teams:
            row["wc_semis"] += 1
            row["wc_last_semi_year"] = _bump_year(row["wc_last_semi_year"], year)

    return row


def get_canonical_achievements(team: str) -> dict:
    """Full canonical honour row for one team (no participation fields)."""
    team = normalize_team_name(team)
    row = _empty_row(team)
    row.update(compute_wc_stats(team))

    cont = CONTINENTAL_BY_TEAM.get(team)
    if cont:
        row.update(cont)
    return row


def build_canonical_achievements(teams: list[str] | None = None) -> dict[str, dict]:
    """Build canonical honour rows keyed by team name."""
    target = teams or WC2026_ALL_TEAMS
    return {team: get_canonical_achievements(team) for team in target}
