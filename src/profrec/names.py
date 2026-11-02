"""Name normalization used to join the course database to the RMP database."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

_PAREN = re.compile(r"\(.*?\)")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def schedule_display_name(raw: str | None) -> str | None:
    """'Anil(Anil)Pereira' -> 'Anil Pereira'. Blank/None (TBA) -> None."""
    if raw is None:
        return None
    s = _PAREN.sub(" ", raw)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def norm_key(name: str | None) -> str:
    """Case/accent/punctuation-insensitive key; parenthetical parts are dropped.

    'Jeffrey (J. P.) Zinn' and 'Jeffrey Zinn' both -> 'jeffrey zinn'.
    """
    if not name:
        return ""
    s = _PAREN.sub(" ", name)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = _NON_ALNUM.sub(" ", s.lower()).strip()
    return re.sub(r"\s+", " ", s)


def load_aliases(path: Path) -> dict:
    """Load a small JSON alias file. Missing file -> empty dict."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except FileNotFoundError:
        return {}


def review_course_code(raw_class: str | None) -> str | None:
    """Normalize an RMP 'class' label to 'ITEC####' (or None if it is not clearly an ITEC course).

    'ITEC2150', 'itec3300', 'ITEC-1001', 'ITECH2110' -> 'ITEC####'.
    Bare numbers ('1001'), 'ITEC' alone and free text ('DIGITALMEDIA') -> None: they are not
    guessed, so they never count toward a specific course.
    """
    if not raw_class:
        return None
    s = re.sub(r"[\s\-_]", "", raw_class).upper()
    m = re.fullmatch(r"ITECH?(\d{4})H?", s)
    return f"ITEC{m.group(1)}" if m else None


def course_base(course_number: str) -> str:
    """'2110H' -> '2110' (honors sections share reviews with the base course)."""
    return course_number.rstrip("Hh")
