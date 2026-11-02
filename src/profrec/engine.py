"""Recommendation engine. Pure code over cache.sqlite: no AI, no network.

Flow for recommend(course, request):
  1. load the course's sections and group them by instructor
  2. drop sections that break the schedule constraints (remembering why)
  3. one instructor left  -> "single" mode: show their sections and what reviews say, no ranking
     several              -> "ranked" mode: score each professor on the requested traits
  4. every statement in the result is a database value or a verbatim sentence from a review
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date

from . import config
from .cache_build import ensure_cache, read_meta
from .criteria import TRAITS
from .schema import ParsedRequest

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# --------------------------------------------------------------------------------------------
# result types
# --------------------------------------------------------------------------------------------
def fmt_time(hhmm: str | None) -> str:
    if not hhmm:
        return "?"
    h, m = (int(x) for x in hhmm.split(":"))
    return f"{(h % 12) or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


@dataclass
class SectionInfo:
    crn: str
    section_number: str
    course_number: str
    title: str
    instructor: str | None
    instructor_key: str | None
    method: str
    days: str | None
    start: str | None
    end: str | None
    building: str | None
    room: str | None
    start_date: str | None
    end_date: str | None
    seats_available: int | None
    enrollment: int | None
    max_enrollment: int | None
    open_section: bool

    @property
    def online(self) -> bool:
        return self.method == "Fully at a distance"

    @property
    def day_list(self) -> list[str]:
        return [d.strip() for d in (self.days or "").split(",") if d.strip()]

    @property
    def schedule_text(self) -> str:
        if not self.days or not self.start:
            return "No scheduled meeting times in the course data"
        return f"{', '.join(self.day_list)} · {fmt_time(self.start)}–{fmt_time(self.end)}"

    @property
    def place_text(self) -> str:
        if self.online:
            return "Online (fully at a distance)"
        bits = [b for b in (self.building, f"Room {self.room}" if self.room else None) if b]
        return ", ".join(bits) if bits else self.method


@dataclass
class Evidence:
    trait: str
    kind: str            # 'quote' | 'tag_stat' | 'keyword'
    supports: bool | None  # True supports the student's preference, False is a caution, None neutral
    text: str            # verbatim sentence (quote/keyword) or a count statement built from DB values
    review_id: int | None = None
    date: str | None = None
    course: str | None = None


@dataclass
class TraitLine:
    trait: str
    description: str
    direction: str
    weight: float
    score: float | None      # 0..1 percentile-style score, None if no evidence
    n_pos: int
    n_neg: int
    n_reviews: int


@dataclass
class Candidate:
    instructor_key: str | None
    name: str
    status: str                                  # ok | tba | no_rmp_profile | no_reviews
    sections: list[SectionInfo]
    rmp_url: str | None = None
    scope: str = "none"                          # course | overall | none
    scope_label: str = ""
    n_reviews: int = 0
    n_course_reviews: int = 0
    confidence: str = "none"
    last_review_date: str | None = None
    stale: bool = False
    rmp_site: dict = field(default_factory=dict)       # RMP's own aggregate for the professor
    scope_stats: dict = field(default_factory=dict)    # computed from collected reviews in scope
    trait_lines: list[TraitLine] = field(default_factory=list)
    combined: float | None = None
    keyword_hits: int = 0
    evidence: list[Evidence] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rank: int | None = None


@dataclass
class CourseOption:
    number: str
    title: str
    label: str
    aliases: list[str]
    n_sections: int
    n_instructors: int

    @property
    def display(self) -> str:
        base = f"{self.label} – {self.title}"
        return f"{base}  (also: {', '.join(self.aliases)})" if self.aliases else base


@dataclass
class Recommendation:
    course: CourseOption
    mode: str                                    # single | ranked | no_sections
    request: ParsedRequest | None
    candidates: list[Candidate]                  # ranked first, then those without review data
    excluded: list[tuple[SectionInfo, list[str]]] = field(default_factory=list)
    default_criteria_used: bool = False
    notes: list[str] = field(default_factory=list)
    as_of: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _confidence_note(c: Candidate) -> str:
    return {"none": "no reviews", "low": "very few reviews", "medium": "some reviews", "high": "many reviews"}[c.confidence]


def _violations(s: SectionInfo, cons) -> list[str]:
    why: list[str] = []
    if cons.delivery == "online" and not s.online:
        why.append("not online")
    if cons.delivery == "in_person" and s.online:
        why.append("online only")
    if s.days and s.start:
        if cons.earliest_start and s.start < cons.earliest_start:
            why.append(f"starts {fmt_time(s.start)} (before {fmt_time(cons.earliest_start)})")
        if cons.latest_end and s.end and s.end > cons.latest_end:
            why.append(f"ends {fmt_time(s.end)} (after {fmt_time(cons.latest_end)})")
        clash = [d for d in s.day_list if d in cons.avoid_days]
        if clash:
            why.append(f"meets on {', '.join(clash)}")
    return why


def _fts_query(keywords: list[str]) -> str | None:
    parts = []
    for k in keywords:
        words = re.findall(r"[a-z0-9']+", k.lower())
        words = [w.replace("'", "") for w in words if w.replace("'", "")]
        if words:
            parts.append('"' + " ".join(words) + '"')
    return " OR ".join(parts) if parts else None


def _sentence_with(text: str, rx: re.Pattern) -> str:
    m = rx.search(text)
    if not m:   # the search matched a stemmed form (e.g. "extension" for "extensions"): show the start of the review
        t = text.strip()
        return t if len(t) <= 300 else t[:300].rstrip() + "…"
    start = max(text.rfind(c, 0, m.start()) for c in ".!?") + 1
    ends = [i for i in (text.find(c, m.end()) for c in ".!?") if i != -1]
    end = (min(ends) + 1) if ends else len(text)
    return text[start:end].strip()


# --------------------------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------------------------
class Engine:
    def __init__(self, auto_build: bool = True, log=print):
        self.log = log
        self.build_info = ensure_cache(log=log) if auto_build else (read_meta() or {})

    def refresh_if_stale(self) -> bool:
        """Rebuild the cache if a source database or the builder changed. Returns True if rebuilt."""
        info = ensure_cache(log=self.log)
        rebuilt = bool(info.get("rebuilt"))
        if rebuilt:
            self.build_info = info
        return rebuilt

    def _conn(self) -> sqlite3.Connection:
        return config.connect_ro(config.CACHE_DB)

    # ---- catalog -----------------------------------------------------------------------
    def as_of(self) -> dict:
        m = read_meta() or {}
        return {
            "term": m.get("course_term_name", ""),
            "courses_retrieved": (m.get("course_retrieved_at_utc") or "")[:10],
            "reviews_through": m.get("rmp_latest_review_date", ""),
            "cache_built": m.get("built_at_utc", ""),
            "n_reviews": m.get("n_reviews", ""),
        }

    def courses(self) -> list[CourseOption]:
        conn = self._conn()
        try:
            rows = conn.execute("SELECT * FROM courses ORDER BY course_number").fetchall()
        finally:
            conn.close()
        return [CourseOption(r["course_number"], r["title"], r["label"], json.loads(r["aliases_json"] or "[]"),
                             r["n_sections"], r["n_instructors"]) for r in rows]

    # ---- main entry --------------------------------------------------------------------
    def recommend(self, course_number: str, request: ParsedRequest | None = None,
                  listing_only: bool = False) -> Recommendation:
        """listing_only=True returns every section grouped by instructor, unranked (no preferences yet)."""
        request = request or ParsedRequest()
        conn = self._conn()
        try:
            return self._recommend(conn, course_number, request, listing_only)
        finally:
            conn.close()

    def _recommend(self, conn, course_number: str, request: ParsedRequest, listing_only: bool = False) -> Recommendation:
        crow = conn.execute("SELECT * FROM courses WHERE course_number=?", (course_number,)).fetchone()
        if not crow:
            raise KeyError(f"unknown course {course_number!r}")
        course = CourseOption(crow["course_number"], crow["title"], crow["label"],
                              json.loads(crow["aliases_json"] or "[]"), crow["n_sections"], crow["n_instructors"])
        cbase = crow["base"]
        rec = Recommendation(course=course, mode="ranked", request=request, candidates=[], as_of=self.as_of())

        sections = self._sections(conn, course_number)
        eligible: list[SectionInfo] = []
        for s in sections:
            why = _violations(s, request.constraints)
            if why:
                rec.excluded.append((s, why))
            else:
                eligible.append(s)
        if rec.excluded:
            rec.notes.append(f"{len(rec.excluded)} of {len(sections)} sections were removed by your schedule filters.")
        if not eligible:
            rec.mode = "no_sections"
            return rec

        groups: dict[str | None, list[SectionInfo]] = {}
        for s in eligible:
            groups.setdefault(s.instructor_key, []).append(s)

        criteria = list(request.criteria)
        if not criteria and not request.keywords:
            rec.default_criteria_used = True

        ref = date.fromisoformat((read_meta() or {}).get("rmp_latest_review_date", "1970-01-01"))
        cands = [self._candidate(conn, ikey, secs, cbase, course.label, ref) for ikey, secs in groups.items()]
        if len(cands) == 1:
            rec.mode = "single"
            self._fill_lines(conn, cands[0], criteria, cbase)
            self._fill_evidence(conn, cands[0], criteria, request.keywords, cbase)
            rec.candidates = cands
            if cands[0].status == "tba":
                rec.notes.append("The instructor for this course is not listed yet (TBA).")
            return rec

        if listing_only:
            rec.mode = "list"
            rec.default_criteria_used = False
            rec.candidates = sorted(cands, key=lambda c: (c.status != "ok", c.name))
            return rec

        if rec.default_criteria_used:
            from .schema import Criterion
            criteria = [Criterion(name="overall_rating", direction="prefer", weight=1.0)]
        for c in cands:
            self._fill_lines(conn, c, criteria, cbase)
        self._score(conn, cands, criteria, request.keywords, cbase)
        for c in cands:
            if c.status == "ok":
                self._fill_evidence(conn, c, criteria, request.keywords, cbase)

        ranked = sorted([c for c in cands if c.combined is not None],
                        key=lambda c: (-c.combined, -c.n_reviews, c.name))
        for i, c in enumerate(ranked, 1):
            c.rank = i
        unranked = sorted([c for c in cands if c.combined is None], key=lambda c: c.name)
        rec.candidates = ranked + unranked
        if unranked:
            rec.notes.append("Instructors with no review data are listed last and are not ranked.")
        return rec

    # ---- loading -----------------------------------------------------------------------
    def _sections(self, conn, course_number: str) -> list[SectionInfo]:
        rows = conn.execute(
            """SELECT s.*, m.days, m.start_time, m.end_time, m.start_date, m.end_date, m.building, m.room
               FROM sections s LEFT JOIN meetings m ON m.crn = s.crn AND m.meeting_number = 1
               WHERE s.course_number = ? ORDER BY s.section_number""", (course_number,)).fetchall()
        return [SectionInfo(
            crn=r["crn"], section_number=r["section_number"], course_number=r["course_number"], title=r["title"],
            instructor=r["instructor_display"], instructor_key=r["instructor_key"], method=r["method"] or "",
            days=r["days"], start=r["start_time"], end=r["end_time"], building=r["building"], room=r["room"],
            start_date=r["start_date"], end_date=r["end_date"], seats_available=r["seats_available"],
            enrollment=r["enrollment"], max_enrollment=r["max_enrollment"], open_section=bool(r["open_section"]))
            for r in rows]

    def _candidate(self, conn, ikey, secs: list[SectionInfo], cbase: str, course_label: str, ref: date) -> Candidate:
        if ikey is None:
            return Candidate(None, "Instructor TBA", "tba", secs, notes=["No instructor is assigned yet, so no reviews can be shown."])
        inst = conn.execute("SELECT * FROM instructors WHERE key=?", (ikey,)).fetchone()
        c = Candidate(ikey, inst["display_name"], "ok", secs, rmp_url=inst["rmp_url"])
        if inst["status"] != "matched":
            c.status = "no_rmp_profile"
            c.notes.append("No RateMyProfessors profile was found for this instructor.")
            return c
        c.rmp_site = {"num_ratings": inst["rmp_num_ratings"], "avg_rating": inst["rmp_avg_rating"],
                      "avg_difficulty": inst["rmp_avg_difficulty"], "would_take_again_pct": inst["rmp_wta_pct"],
                      "department": inst["department"]}
        course_row = conn.execute("SELECT * FROM scope_stats WHERE instructor_key=? AND course_base=?", (ikey, cbase)).fetchone()
        all_row = conn.execute("SELECT * FROM scope_stats WHERE instructor_key=? AND course_base='*'", (ikey,)).fetchone()
        c.n_course_reviews = course_row["n_reviews"] if course_row else 0
        row, scope = None, "none"
        if course_row and course_row["n_reviews"] >= config.MIN_COURSE_REVIEWS:
            row, scope = course_row, "course"
        elif all_row:
            row, scope = all_row, "overall"
            if c.n_course_reviews:
                c.notes.append(f"Only {c.n_course_reviews} review(s) mention this course, so all-course reviews are used.")
            else:
                c.notes.append("No reviews are labeled with this course, so all-course reviews are used.")
        if not row:
            c.status = "no_reviews"
            c.notes.append("This instructor has an RMP profile but no collected reviews.")
            return c
        c.scope = scope
        c.scope_label = f"this course ({course_label})" if scope == "course" else "all courses"
        c.n_reviews = row["n_reviews"]
        c.confidence = row["confidence"]
        c.last_review_date = row["last_review_date"]
        if c.last_review_date:
            c.stale = (ref - date.fromisoformat(c.last_review_date)).days > config.STALE_DAYS
            if c.stale:
                c.notes.append(f"Newest review is from {c.last_review_date}; it may not reflect current teaching.")
        c.scope_stats = {"mean_quality": row["mean_quality"], "mean_difficulty": row["mean_difficulty"],
                         "wta_rate": row["wta_rate"], "n_wta": row["n_wta"], "n_recent": row["n_recent"]}
        if c.confidence in ("low", "medium"):
            c.notes.append(f"Based on {c.n_reviews} review(s): treat as a rough signal.")
        return c

    def _fill_lines(self, conn, c: Candidate, criteria, cbase: str) -> None:
        if c.status != "ok":
            return
        scope_cb = cbase if c.scope == "course" else "*"
        for cr in criteria:
            r = conn.execute("SELECT * FROM trait_scores WHERE instructor_key=? AND course_base=? AND trait=?",
                             (c.instructor_key, scope_cb, cr.name)).fetchone()
            c.trait_lines.append(TraitLine(
                cr.name, TRAITS[cr.name].description, cr.direction, cr.weight,
                r["score"] if r else None, r["n_pos"] if r else 0, r["n_neg"] if r else 0,
                r["n_reviews"] if r else c.n_reviews))

    # ---- scoring -----------------------------------------------------------------------
    def _score(self, conn, cands: list[Candidate], criteria, keywords: list[str], cbase: str) -> None:
        live = [c for c in cands if c.status == "ok"]
        kw_scores: dict[str, float] = {}
        q = _fts_query(keywords)
        if q and live:
            raw: dict[str, float] = {}
            for c in live:
                scope_sql, args = ("", [])
                if c.scope == "course":
                    scope_sql, args = " AND r.course_base = ?", [cbase]
                n = conn.execute(
                    "SELECT COUNT(*) FROM review_fts JOIN reviews r ON r.id = review_fts.rowid "
                    "WHERE review_fts MATCH ? AND r.instructor_key = ?" + scope_sql, [q, c.instructor_key, *args]).fetchone()[0]
                c.keyword_hits = n
                raw[c.instructor_key] = n / (c.n_reviews + config.SHRINK_K)
            lo, hi = min(raw.values()), max(raw.values())
            if hi > 0:
                kw_scores = {k: (v - lo) / (hi - lo) if hi > lo else 1.0 for k, v in raw.items()}
        total_w = sum(cr.weight for cr in criteria) or 1.0
        kw_weight = (0.25 * total_w) if criteria else 1.0
        for c in live:
            num = den = 0.0
            for ln in c.trait_lines:
                if ln.score is None:
                    continue
                v = ln.score if ln.direction == "prefer" else 1.0 - ln.score
                num += ln.weight * v
                den += ln.weight
            if c.instructor_key in kw_scores:
                num += kw_weight * kw_scores[c.instructor_key]
                den += kw_weight
            c.combined = (num / den) if den else None
            asked = sum(cr.weight for cr in criteria)
            if criteria and asked and den and (den - (kw_weight if c.instructor_key in kw_scores else 0)) / asked < 0.5:
                c.notes.append("Little review evidence exists for several of the traits you asked about.")

    # ---- evidence ----------------------------------------------------------------------
    def _quotes(self, conn, c: Candidate, trait: str, polarity: int, cbase: str, limit: int) -> list[Evidence]:
        scope_filter = " AND r.course_base = ?" if c.scope == "course" else ""
        args = [c.instructor_key, trait, polarity] + ([cbase] if scope_filter else []) + [cbase, limit]
        rows = conn.execute(
            f"""SELECT rt.review_id, rt.text, r.date, r.course_code FROM review_traits rt
                JOIN reviews r ON r.id = rt.review_id
                WHERE r.instructor_key=? AND rt.trait=? AND rt.polarity=? AND rt.source='lexicon'{scope_filter}
                GROUP BY rt.review_id
                ORDER BY (r.course_base = ?) DESC, r.date DESC LIMIT ?""", args).fetchall()
        return [Evidence(trait, "quote", None, r["text"], r["review_id"], r["date"], r["course_code"]) for r in rows]

    def _tag_stat(self, conn, c: Candidate, trait: str, cbase: str) -> list[Evidence]:
        scope_filter = " AND r.course_base = ?" if c.scope == "course" else ""
        extra = [cbase] if scope_filter else []
        tagged = conn.execute(f"SELECT COUNT(*) FROM reviews r WHERE r.instructor_key=? AND r.n_tags>0{scope_filter}",
                              [c.instructor_key, *extra]).fetchone()[0]
        if not tagged:
            return []
        rows = conn.execute(
            f"""SELECT rt.text, COUNT(DISTINCT rt.review_id) n FROM review_traits rt JOIN reviews r ON r.id=rt.review_id
                WHERE r.instructor_key=? AND rt.trait=? AND rt.source='tag'{scope_filter} GROUP BY rt.text ORDER BY n DESC""",
            [c.instructor_key, trait, *extra]).fetchall()
        return [Evidence(trait, "tag_stat", None, f"Tag “{r['text']}” appears on {r['n']} of {tagged} tagged reviews") for r in rows]

    def _fill_evidence(self, conn, c: Candidate, criteria, keywords: list[str], cbase: str) -> None:
        if c.status != "ok":
            return
        top = sorted(criteria, key=lambda x: -x.weight)[:3]
        for cr in top:
            want = +1 if cr.direction == "prefer" else -1
            for e in self._quotes(conn, c, cr.name, want, cbase, 1):
                e.supports = True
                c.evidence.append(e)
            for e in self._quotes(conn, c, cr.name, -want, cbase, 1):
                e.supports = False
                c.evidence.append(e)
            for e in self._tag_stat(conn, c, cr.name, cbase)[:2]:
                c.evidence.append(e)
        q = _fts_query(keywords)
        if q:
            scope_sql, args = ("", [])
            if c.scope == "course":
                scope_sql, args = " AND r.course_base = ?", [cbase]
            rows = conn.execute(
                "SELECT r.id, r.comment, r.date, r.course_code FROM review_fts JOIN reviews r ON r.id = review_fts.rowid "
                "WHERE review_fts MATCH ? AND r.instructor_key = ?" + scope_sql + " ORDER BY r.date DESC LIMIT 2",
                [q, c.instructor_key, *args]).fetchall()
            rx = re.compile("|".join(re.escape(k) for k in keywords), re.I)
            for r in rows:
                c.evidence.append(Evidence("keywords", "keyword", None, _sentence_with(r["comment"], rx), r["id"], r["date"], r["course_code"]))
