# Copyright (c) 2026 Sasiru Virajith Kankanamge
# SPDX-License-Identifier: MIT

"""
FIFA World Cup 2026 Predictor
Built by: K. Sasiru Virajith
"""

import re
import unicodedata

import pandas as pd

from src.config import TEAM_ALIASES


def normalize_team_name(name: str) -> str:
    if pd.isna(name):
        return name
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def normalize_player_name(name: str) -> str:
    if pd.isna(name):
        return ""
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"[^a-zA-Z0-9\s]", "", name).lower().strip()
    return name
