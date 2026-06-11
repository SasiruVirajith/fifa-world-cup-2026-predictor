"""
config.py
─────────
Central constants for the World Cup Predictor project.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# ── Tournament years ─────────────────────────────────────────────────────────
WC_YEARS = [1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]

WC_START_DATES = {
    1990: "1990-06-08", 1994: "1994-06-17", 1998: "1998-06-10",
    2002: "2002-05-31", 2006: "2006-06-09", 2010: "2010-06-11",
    2014: "2014-06-12", 2018: "2018-06-14", 2022: "2022-11-20",
    2026: "2026-06-11",
}

# ── martj42 international results (Kaggle mirror on GitHub) ─────────────────
MARTJ42_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
MARTJ42_FILES = {
    "results": "results.csv",
    "goalscorers": "goalscorers.csv",
    "shootouts": "shootouts.csv",
    "former_names": "former_names.csv",
}

INTERNATIONAL_RESULTS_URL = f"{MARTJ42_BASE}/results.csv"

# Official FIFA rankings API (men's)
FIFA_RANKINGS_API = "https://api.fifa.com/api/v3/rankings"
FIFA_RANKINGS_LIMIT = 300

# Match model training window
MATCH_MODEL_START_YEAR = 2000
RECENT_FORM_YEARS = 4  # friendlies + tournaments in last N years weighted heavily

# ── Team name harmonization ───────────────────────────────────────────────────
TEAM_ALIASES = {
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Korea, South": "South Korea",
    "USA": "United States",
    "United States": "United States",
    "United States of America": "United States",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Czech Republic": "Czechia",
    "Turkey": "Türkiye",
    "DR Congo": "Congo DR",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Cabo Verde": "Cape Verde",
    "Democratic Republic of the Congo": "Congo DR",
    "Chinese Taipei": "Taiwan",
    "FYR Macedonia": "North Macedonia",
    "Macedonia": "North Macedonia",
    "Switzerland": "Switzerland",
    "SUI": "Switzerland",
    "KSA": "Saudi Arabia",
    "CIV": "Ivory Coast",
    "MAR": "Morocco",
    "RSA": "South Africa",
    "IRL": "Republic of Ireland",
    "COD": "DR Congo",
    "CPV": "Cape Verde",
    "CUW": "Curaçao",
    "NCL": "New Caledonia",
    "SUR": "Suriname",
}

# ── WC 2026 format ───────────────────────────────────────────────────────────
# Official Final Draw (all 48 teams confirmed, playoffs resolved)
# Knockout: top 2 per group (24) + 8 best third-place teams -> Round of 32
WC2026_GROUPS = {
    "A": ["Mexico", "South Korea", "South Africa", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "Congo DR"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

WC2026_ALL_TEAMS = sorted({
    team for teams in WC2026_GROUPS.values() for team in teams
})

# Legacy - no playoff placeholders in the confirmed draw
WC2026_PLAYOFF_CANDIDATES = {}

ROUND_VARIANCE = {
    "group": 22,
    "play-in": 35,
    "round_of_32": 50,
    "round_of_16": 55,
    "quarterfinal": 65,
    "semifinal": 75,
    "final": 90,
}

# Recency-weighted tactical/squad quality (2022-2026 cycle)
# Belgium golden generation faded; France/Spain/Argentina remain elite
MODERN_TEAMS = {
    "France": 6, "Argentina": 6, "Spain": 6, "England": 5, "Brazil": 5,
    "Portugal": 5, "Germany": 5, "Morocco": 4, "Netherlands": 4, "Croatia": 4,
    "Colombia": 3, "Japan": 3, "Senegal": 3, "Uruguay": 3, "United States": 3,
    "Belgium": 1, "Switzerland": 1, "Mexico": 1, "Italy": 4,
}

DRAW_THRESHOLD = 0.30
N_SIMULATIONS_DEFAULT = 5000
CACHE_TTL_DAYS = 7  # martj42 / FIFA refresh window
CLUB_CACHE_TTL_DAYS = 3650  # club snapshot  -  frozen post-season
# API season years: 2024 = 2024/25, 2025 = 2025/26 (both blended in player_club)
CLUB_SEASONS = [2024, 2025]


def club_seasons_from_config(cfg: dict) -> list[int]:
    """Resolve club stat seasons from config (supports legacy single ``season`` key)."""
    if cfg.get("seasons"):
        return [int(s) for s in cfg["seasons"]]
    if cfg.get("season") is not None:
        return [int(cfg["season"])]
    return list(CLUB_SEASONS)

# ── 2026 player award pipeline ───────────────────────────────────────────────
PLAYER_INTL_START = "2023-01-01"
# Scale raw rate×matches estimate to historical WC Golden Boot band (~6–8 for favorites)
GOLDEN_BOOT_GOALS_CALIBRATION = 1.70
CLUB_RAW_DIR = RAW_DIR / "club"
CLUB_RAW_UNDERSTAT = CLUB_RAW_DIR / "understat"
CLUB_RAW_API_FOOTBALL = CLUB_RAW_DIR / "api_football"

UNDERSTAT_LEAGUES_PATH = ROOT_DIR / "config" / "understat_leagues.json"
API_FOOTBALL_CONFIG_PATH = ROOT_DIR / "config" / "api_football.json"
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
API_FOOTBALL_KEY_ENV = "APIFOOTBALL_KEY"
API_FOOTBALL_RATE_LIMIT_SEC = 6  # free tier: 10 req/min

# Big 5  -  Understat (primary) with API-Football fallback
UNDERSTAT_LEAGUE_KEYS = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

# Tier B  -  API-Football only (MLS, Saudi, etc.)
API_FOOTBALL_TIER_B_KEYS = [
    "POR-Primeira Liga",
    "NED-Eredivisie",
    "BEL-Belgian Pro League",
    "TUR-Super Lig",
    "ENG-Championship",
    "USA-MLS",
    "MEX-Liga MX",
    "KSA-Saudi Pro League",
    "BRA-Serie A",
    "ARG-Primera Division",
]

# Tournament weighting for international goal involvement
TOURNAMENT_TIER_WEIGHTS = {
    "world_cup": 1.0,
    "major_continental": 1.0,
    "qualifier": 0.85,
    "nations_league": 0.6,
    "friendly": 0.35,
    "other": 0.5,
}

# Likely #1 GK per confirmed WC 2026 nation (updated when squads are named)
PRIMARY_GK_2026 = {
    "Algeria": "Anthony Mandrea",
    "Argentina": "Emiliano Martínez",
    "Australia": "Mathew Ryan",
    "Austria": "Daniel Bachmann",
    "Belgium": "Koen Casteels",
    "Bosnia and Herzegovina": "Asmir Begović",
    "Brazil": "Alisson",
    "Canada": "Dayne St. Clair",
    "Cape Verde": "Vozinha",
    "Colombia": "Camilo Vargas",
    "Congo DR": "Joël Kiassumbua",
    "Croatia": "Dominik Livaković",
    "Curaçao": "Eloy Room",
    "Czechia": "Matěj Kovář",
    "Ecuador": "Hernán Galíndez",
    "Egypt": "Mohamed El-Shenawy",
    "England": "Jordan Pickford",
    "France": "Mike Maignan",
    "Germany": "Marc-André ter Stegen",
    "Ghana": "Lawrence Ati-Zigi",
    "Haiti": "Johny Placide",
    "Iran": "Alireza Beiranvand",
    "Iraq": "Jalal Hassan",
    "Ivory Coast": "Badra Ali Sangaré",
    "Japan": "Zion Suzuki",
    "Jordan": "Yazeed Abulaila",
    "Mexico": "Luis Malagón",
    "Morocco": "Yassine Bounou",
    "Netherlands": "Bart Verbruggen",
    "New Zealand": "Max Crocombe",
    "Norway": "Ørjan Nyland",
    "Panama": "Orlando Mosquera",
    "Paraguay": "Anthony Silva",
    "Portugal": "Diogo Costa",
    "Qatar": "Saad Al-Sheeb",
    "Saudi Arabia": "Mohammed Al-Owais",
    "Scotland": "Angus Gunn",
    "Senegal": "Édouard Mendy",
    "South Africa": "Ronwen Williams",
    "South Korea": "Jo Hyeon-woo",
    "Spain": "Unai Simón",
    "Sweden": "Robin Olsen",
    "Switzerland": "Yann Sommer",
    "Tunisia": "Aymen Dahmen",
    "Türkiye": "Uğurcan Çakır",
    "United States": "Matt Turner",
    "Uruguay": "Sergio Rochet",
    "Uzbekistan": "Utkir Boymurodov",
}

NATION_ALIASES = {
    **TEAM_ALIASES,
    "Korea, South": "South Korea",
    "USA": "United States",
    "Cote d Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "Curaçao": "Curaçao",
    "Türkiye": "Türkiye",
    "Turkiye": "Türkiye",
}
