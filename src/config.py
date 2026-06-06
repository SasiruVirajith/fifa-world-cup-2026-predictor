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

# ── StatsBomb IDs ────────────────────────────────────────────────────────────
STATSBOMB_COMPETITION_ID = 43
STATSBOMB_SEASONS = {2018: 3, 2022: 106}

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

# ── WC winners & awards (labels) ─────────────────────────────────────────────
WC_WINNERS = {
    2022: "Argentina", 2018: "France", 2014: "Germany", 2010: "Spain",
    2006: "Italy", 2002: "Brazil", 1998: "France", 1994: "Brazil",
    1990: "Germany", 1986: "Argentina",
}

GOLDEN_GLOVE_WINNERS = {
    2022: "Emiliano Martínez", 2018: "Thibaut Courtois", 2014: "Manuel Neuer",
    2010: "Iker Casillas", 2006: "Gianluigi Buffon", 2002: "Oliver Kahn",
    1998: "Fabien Barthez", 1994: "Michel Preud'homme", 1990: "Luis Goycochea",
}

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

# Legacy — no playoff placeholders in the confirmed draw
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
CACHE_TTL_DAYS = 7

# ── 2026 player award pipeline ───────────────────────────────────────────────
PLAYER_INTL_START = "2023-01-01"
PLAYER_INTL_TRAIN_START = "2019-01-01"
CLUB_RAW_DIR = RAW_DIR / "club"

FBREF_CLUB_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]
FBREF_CLUB_SEASONS = ["2024-2025", "2025-2026"]

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
