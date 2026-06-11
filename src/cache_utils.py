"""
cache_utils.py
──────────────
Shared file-cache helpers for raw data downloads.
"""

from datetime import datetime, timedelta
from pathlib import Path

from src.config import CACHE_TTL_DAYS


def is_cache_fresh(path: Path, ttl_days: int = CACHE_TTL_DAYS) -> bool:
    """Return True if file exists and is younger than ttl_days."""
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime < timedelta(days=ttl_days)


def cache_age_days(path: Path) -> float | None:
    """Return age of file in days, or None if missing."""
    if not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).total_seconds() / 86400
