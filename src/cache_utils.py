# Copyright (c) 2026 Sasiru Virajith Kankanamge
# SPDX-License-Identifier: MIT

"""
FIFA World Cup 2026 Predictor
Built by: K. Sasiru Virajith
"""

from datetime import datetime, timedelta
from pathlib import Path

from src.config import CACHE_TTL_DAYS


def is_cache_fresh(path: Path, ttl_days: int = CACHE_TTL_DAYS) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime < timedelta(days=ttl_days)


def cache_age_days(path: Path) -> float | None:
    if not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).total_seconds() / 86400
