"""Paths and tunable constants. Everything can be overridden with environment variables."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = Path(os.environ.get("PROFREC_CACHE_DIR", ROOT / "cache"))

# Source databases (treated as read-only sources of truth).
COURSE_DB = Path(os.environ.get("PROFREC_COURSE_DB", DATA_DIR / "courses_spring2027_itec.sqlite"))
RMP_DB = Path(os.environ.get("PROFREC_RMP_DB", DATA_DIR / "rmp_reviews_itec.sqlite"))

# Derived, disposable cache. Safe to delete: it is rebuilt from the sources.
CACHE_DB = CACHE_DIR / "cache.sqlite"
# Parsed-request cache lives in its own file so a cache rebuild does not erase it
# (a parsed request depends only on the text and the prompt, not on the data).
REQUEST_CACHE_DB = CACHE_DIR / "requests.sqlite"

INSTRUCTOR_ALIASES_FILE = DATA_DIR / "instructor_aliases.json"
COURSE_ALIASES_FILE = DATA_DIR / "course_aliases.json"

# --- LLM (request parsing only) ---
GEMINI_MODEL = os.environ.get("PROFREC_GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"  # the key is read from this environment variable only

# --- Scoring constants (documented in README) ---
SCHEMA_VERSION = 1
MIN_COURSE_REVIEWS = 3      # fewer reviews than this for a course -> fall back to all-course stats
SHRINK_K = 5                # pseudo-reviews pulling small samples toward the department average
RECENT_DAYS = 3 * 365       # a review is "recent" if within this many days of the newest review
STALE_DAYS = 3 * 365        # warn if a professor's newest review is older than this
# review-count -> confidence label thresholds (lower bound inclusive)
CONFIDENCE_MEDIUM = 5
CONFIDENCE_HIGH = 15


def connect_ro(path: Path) -> sqlite3.Connection:
    """Open a SQLite file read-only so nothing in this app can modify a source database."""
    conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn
