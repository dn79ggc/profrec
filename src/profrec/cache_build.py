"""Builds cache.sqlite from the two source databases.

No AI is involved. The cache is a pure function of (course DB, RMP DB, this code, alias files):
  * it is rebuilt only when the SHA-256 of either source file or the builder code changes
  * it is written to a temporary file and swapped in with an atomic rename, so a reader never
    sees a half-built cache
  * the source databases are opened read-only

CLI:  python -m profrec.cache_build [--force]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sqlite3
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from . import config
from .criteria import TRAIT_LIST, analyze_review, split_tags
from .names import (
    course_base,
    load_aliases,
    norm_key,
    review_course_code,
    schedule_display_name,
)

_PKG = Path(__file__).resolve().parent
_BUILDER_FILES = [_PKG / n for n in ("cache_build.py", "criteria.py", "names.py", "config.py")]

DDL = """
CREATE TABLE build_meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE courses(
  course_number TEXT PRIMARY KEY, subject TEXT, title TEXT, base TEXT, label TEXT,
  aliases_json TEXT, n_sections INT, n_instructors INT);
CREATE TABLE instructors(
  key TEXT PRIMARY KEY, display_name TEXT, status TEXT, teaching INT,
  rmp_id INT, rmp_url TEXT, department TEXT, rmp_num_ratings INT, rmp_avg_rating REAL,
  rmp_avg_difficulty REAL, rmp_wta_pct REAL, rmp_reviews_collected INT);
CREATE TABLE sections(
  crn TEXT PRIMARY KEY, term_code TEXT, course_number TEXT, section_number TEXT, title TEXT,
  instructor_key TEXT, instructor_display TEXT, method TEXT, campus TEXT, schedule_type TEXT,
  enrollment INT, max_enrollment INT, seats_available INT, open_section INT, part_of_term TEXT);
CREATE TABLE meetings(
  crn TEXT, meeting_number INT, days TEXT, start_time TEXT, end_time TEXT, start_date TEXT,
  end_date TEXT, building TEXT, room TEXT, meeting_type TEXT, PRIMARY KEY(crn, meeting_number));
CREATE TABLE reviews(
  id INT PRIMARY KEY, instructor_key TEXT, date TEXT, class_raw TEXT, course_code TEXT,
  course_base TEXT, quality REAL, difficulty REAL, helpful REAL, would_take_again INT,
  attendance INT, online INT, tags_json TEXT, n_tags INT, comment TEXT, thumbs_up INT, thumbs_down INT);
CREATE TABLE review_traits(review_id INT, trait TEXT, polarity INT, source TEXT, text TEXT);
CREATE TABLE scope_stats(
  instructor_key TEXT, course_base TEXT, n_reviews INT, n_recent INT, last_review_date TEXT,
  mean_quality REAL, mean_difficulty REAL, wta_rate REAL, n_wta INT, confidence TEXT,
  PRIMARY KEY(instructor_key, course_base));
CREATE TABLE trait_scores(
  instructor_key TEXT, course_base TEXT, trait TEXT, score REAL, n_pos INT, n_neg INT,
  n_reviews INT, comps_json TEXT, PRIMARY KEY(instructor_key, course_base, trait));
CREATE INDEX idx_sections_course ON sections(course_number);
CREATE INDEX idx_reviews_instr ON reviews(instructor_key, course_base);
CREATE INDEX idx_rt_review ON review_traits(review_id);
CREATE INDEX idx_rt_trait ON review_traits(trait, polarity);
"""


# --------------------------------------------------------------------------------------------
# fingerprint / freshness
# --------------------------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def builder_hash() -> str:
    h = hashlib.sha256()
    for p in _BUILDER_FILES + [config.INSTRUCTOR_ALIASES_FILE, config.COURSE_ALIASES_FILE]:
        h.update(p.name.encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()


def fingerprint() -> dict:
    return {
        "schema_version": str(config.SCHEMA_VERSION),
        "builder_sha256": builder_hash(),
        "course_db_sha256": _sha256(config.COURSE_DB),
        "rmp_db_sha256": _sha256(config.RMP_DB),
    }


def read_meta(path: Path | None = None) -> dict | None:
    path = path or config.CACHE_DB
    if not Path(path).exists():
        return None
    try:
        conn = config.connect_ro(path)
        try:
            return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM build_meta")}
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def is_fresh() -> bool:
    meta = read_meta()
    return bool(meta) and all(meta.get(k) == v for k, v in fingerprint().items())


@contextlib.contextmanager
def _build_lock(timeout: float = 90.0, stale_after: float = 180.0):
    """Portable lock (no fcntl) so two app sessions never build at the same time."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lock = config.CACHE_DIR / ".build.lock"
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            with contextlib.suppress(OSError):
                if time.time() - lock.stat().st_mtime > stale_after:
                    lock.unlink()
                    continue
            if time.time() > deadline:
                raise TimeoutError("another process is building the cache")
            time.sleep(0.2)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            lock.unlink()


def ensure_cache(force: bool = False, log=print) -> dict:
    """Return cache metadata, rebuilding first if the sources or the builder changed."""
    if not force and is_fresh():
        return {**read_meta(), "rebuilt": False}
    with _build_lock():
        if not force and is_fresh():          # someone else rebuilt while we waited
            return {**read_meta(), "rebuilt": False}
        t0 = time.time()
        meta = build_cache(config.CACHE_DB, config.COURSE_DB, config.RMP_DB, log=log)
        meta["rebuilt"] = True
        meta["build_seconds"] = round(time.time() - t0, 2)
        return meta


# --------------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------------
def _confidence(n: int) -> str:
    if n <= 0:
        return "none"
    if n < config.CONFIDENCE_MEDIUM:
        return "low"
    if n < config.CONFIDENCE_HIGH:
        return "medium"
    return "high"


def _attendance_flag(raw: str | None) -> int | None:
    v = (raw or "").strip().lower()
    if v in ("mandatory", "y"):
        return 1
    if v in ("non mandatory", "n"):
        return 0
    return None


def _norm_rating(m: float) -> float:
    return (m - 1.0) / 4.0   # RMP 1..5 -> 0..1


def _comp_values(trait, rows: list[dict]) -> list[tuple[str, float, int]]:
    """Raw component values for one trait over a set of reviews -> [(kind, value, n)]."""
    out: list[tuple[str, float, int]] = []
    for rf in trait.ratings:
        vals = [r[rf] for r in rows if r[rf] is not None]
        if vals:
            m = sum(vals) / len(vals)
            out.append((f"rating:{rf}", _norm_rating(m) if rf in ("difficulty", "quality", "helpful") else m, len(vals)))
    if trait.pos_tags or trait.neg_tags:
        tagged = [r for r in rows if r["n_tags"] > 0]
        if tagged:
            pos = sum(1 for r in tagged if +1 in r["tagflags"].get(trait.name, ()))
            neg = sum(1 for r in tagged if -1 in r["tagflags"].get(trait.name, ()))
            out.append(("tags", (pos - neg) / len(tagged), len(tagged)))
    if trait.pos_patterns or trait.neg_patterns:
        if rows:
            pos = sum(1 for r in rows if +1 in r["lexflags"].get(trait.name, ()))
            neg = sum(1 for r in rows if -1 in r["lexflags"].get(trait.name, ()))
            out.append(("lexicon", (pos - neg) / len(rows), len(rows)))
    return out


def _pct(x: float, population: list[float]) -> float:
    if not population:
        return 0.5
    lo = sum(1 for v in population if v < x)
    eq = sum(1 for v in population if v == x)
    return (lo + 0.5 * eq) / len(population)


def build_cache(out_path: Path, course_db: Path, rmp_db: Path, log=print) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".building")
    if tmp.exists():
        tmp.unlink()
    fp = fingerprint()
    cdb, rdb = config.connect_ro(course_db), config.connect_ro(rmp_db)
    out = sqlite3.connect(tmp)
    warnings: list[str] = []
    try:
        out.executescript(DDL)
        info = _fill(out, cdb, rdb, warnings, log)
        meta = {**fp, **info,
                "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "warnings_json": json.dumps(warnings)}
        out.executemany("INSERT INTO build_meta VALUES (?,?)", [(k, str(v)) for k, v in meta.items()])
        out.commit()
        out.execute("PRAGMA journal_mode=DELETE")
    finally:
        out.close()
        cdb.close()
        rdb.close()
    os.replace(tmp, out_path)   # atomic swap
    for w in warnings:
        log(f"  warning: {w}")
    return meta


def _fill(out: sqlite3.Connection, cdb, rdb, warnings: list[str], log) -> dict:
    cmeta = {r["key"]: r["value"] for r in cdb.execute("SELECT key, value FROM metadata")}
    course_aliases = load_aliases(config.COURSE_ALIASES_FILE)
    inst_aliases = load_aliases(config.INSTRUCTOR_ALIASES_FILE).get("schedule_to_rmp", {})

    # ---- RMP professors ------------------------------------------------------------------
    profs = list(rdb.execute("SELECT * FROM professors"))
    prof_by_key: dict[str, sqlite3.Row] = {}
    for p in profs:
        k = norm_key(p["instructor_name"])
        if k in prof_by_key:
            warnings.append(f"duplicate RMP professor key '{k}'")
        prof_by_key[k] = p
    unmatched_keys = {norm_key(r["instructor_name"]) for r in rdb.execute("SELECT instructor_name FROM unmatched_instructors")}

    # ---- schedule instructors ----------------------------------------------------------
    sec_rows = list(cdb.execute("SELECT * FROM sections"))
    sched_names = {schedule_display_name(r["instructors"]) for r in sec_rows} - {None}
    name_to_key: dict[str, str] = {}
    instructors: dict[str, dict] = {}
    for p in profs:
        k = norm_key(p["instructor_name"])
        instructors[k] = dict(key=k, display_name=p["instructor_name"], status="matched", teaching=0,
                              rmp_id=p["id"], rmp_url=p["rmp_url"], department=p["department"],
                              rmp_num_ratings=p["num_ratings"], rmp_avg_rating=p["avg_rating"],
                              rmp_avg_difficulty=p["avg_difficulty"], rmp_wta_pct=p["would_take_again_pct"],
                              rmp_reviews_collected=p["reviews_collected"])
    for n in sorted(sched_names):
        k = norm_key(inst_aliases.get(n, n))
        if k in prof_by_key:
            name_to_key[n] = k
            instructors[k]["display_name"] = n
            instructors[k]["teaching"] = 1
        else:
            k = norm_key(n)
            name_to_key[n] = k
            instructors[k] = dict(key=k, display_name=n, status="no_rmp_profile", teaching=1, rmp_id=None,
                                  rmp_url=None, department=None, rmp_num_ratings=None, rmp_avg_rating=None,
                                  rmp_avg_difficulty=None, rmp_wta_pct=None, rmp_reviews_collected=None)
            if k not in unmatched_keys:
                warnings.append(f"schedule instructor '{n}' is neither in RMP professors nor in the unmatched list "
                                f"(possible name mismatch; add it to data/instructor_aliases.json)")
    cols = ["key", "display_name", "status", "teaching", "rmp_id", "rmp_url", "department", "rmp_num_ratings",
            "rmp_avg_rating", "rmp_avg_difficulty", "rmp_wta_pct", "rmp_reviews_collected"]
    out.executemany(f"INSERT INTO instructors({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
                    [tuple(i[c] for c in cols) for i in instructors.values()])

    # ---- sections / meetings / courses ---------------------------------------------------
    for r in sec_rows:
        disp = schedule_display_name(r["instructors"])
        out.execute("INSERT INTO sections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            r["crn"], r["term_code"], r["course_number"], r["section_number"], r["title"],
            name_to_key.get(disp) if disp else None, disp, r["instructional_method"], r["campus"],
            r["schedule_type"], r["enrollment"], r["maximum_enrollment"], r["seats_available"],
            r["open_section"], r["part_of_term"]))
    for m in cdb.execute("SELECT * FROM meetings"):
        out.execute("INSERT INTO meetings VALUES (?,?,?,?,?,?,?,?,?,?)", (
            m["crn"], m["meeting_number"], m["days"], m["start_time"], m["end_time"], m["start_date"],
            m["end_date"], m["building_description"], m["room"], m["meeting_type"]))
    for c in cdb.execute("SELECT * FROM courses ORDER BY course_number"):
        ns = out.execute("SELECT COUNT(*) FROM sections WHERE course_number=?", (c["course_number"],)).fetchone()[0]
        ni = out.execute("SELECT COUNT(DISTINCT instructor_key) FROM sections WHERE course_number=?",
                         (c["course_number"],)).fetchone()[0]
        has_tba = out.execute("SELECT COUNT(*) FROM sections WHERE course_number=? AND instructor_key IS NULL",
                              (c["course_number"],)).fetchone()[0]
        aliases = course_aliases.get(c["course_number"], [])
        out.execute("INSERT INTO courses VALUES (?,?,?,?,?,?,?,?)", (
            c["course_number"], c["subject"], c["title"], course_base(c["course_number"]),
            f"{c['subject']} {c['course_number']}", json.dumps(aliases), ns, ni + (1 if has_tba else 0)))

    # ---- reviews -------------------------------------------------------------------------
    id_to_key = {p["id"]: norm_key(p["instructor_name"]) for p in profs}
    reviews: list[dict] = []
    for r in rdb.execute("SELECT * FROM reviews ORDER BY id"):
        tags = split_tags(r["tags"])
        code = review_course_code(r["class"])
        reviews.append(dict(
            id=r["id"], instructor_key=id_to_key.get(r["professor_id"]), date=(r["date"] or "")[:10],
            class_raw=r["class"], course_code=code, course_base=course_base(code[4:]) if code else None,
            quality=r["quality"], difficulty=r["difficulty"], helpful=r["helpful"],
            would_take_again=r["would_take_again"], attendance=_attendance_flag(r["attendance_mandatory"]),
            online=r["for_online_class"], tags=tags, n_tags=len(tags), comment=r["comment"] or "",
            thumbs_up=r["thumbs_up"], thumbs_down=r["thumbs_down"], tagflags={}, lexflags={}))
    n_orphans = sum(1 for r in reviews if r["instructor_key"] is None)
    if n_orphans:
        warnings.append(f"{n_orphans} reviews reference an unknown professor id and were skipped")
    reviews = [r for r in reviews if r["instructor_key"]]

    trait_rows = []
    for r in reviews:
        seen = set()
        for h in analyze_review(r["comment"], r["tags"]):
            bucket = r["tagflags"] if h.source == "tag" else r["lexflags"]
            bucket.setdefault(h.trait, set()).add(h.polarity)
            sig = (r["id"], h.trait, h.polarity, h.source, h.text)
            if sig not in seen:
                seen.add(sig)
                trait_rows.append(sig)
    out.executemany("INSERT INTO review_traits VALUES (?,?,?,?,?)", trait_rows)
    out.executemany("INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", [(
        r["id"], r["instructor_key"], r["date"], r["class_raw"], r["course_code"], r["course_base"], r["quality"],
        r["difficulty"], r["helpful"], r["would_take_again"], r["attendance"], r["online"], json.dumps(r["tags"]),
        r["n_tags"], r["comment"], r["thumbs_up"], r["thumbs_down"]) for r in reviews])
    out.executescript(
        "CREATE VIRTUAL TABLE review_fts USING fts5(comment, content='reviews', content_rowid='id', "
        "tokenize='porter unicode61');"
        "INSERT INTO review_fts(review_fts) VALUES('rebuild');")

    # ---- per-scope statistics and trait scores ----------------------------------------------
    latest = max((r["date"] for r in reviews if r["date"]), default="1970-01-01")
    ref = date.fromisoformat(latest)
    by_scope: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in reviews:
        by_scope[(r["instructor_key"], "*")].append(r)
        if r["course_base"]:
            by_scope[(r["instructor_key"], r["course_base"])].append(r)

    for (ik, cb), rows in by_scope.items():
        qs = [r["quality"] for r in rows if r["quality"] is not None]
        ds = [r["difficulty"] for r in rows if r["difficulty"] is not None]
        wt = [r["would_take_again"] for r in rows if r["would_take_again"] is not None]
        recent = sum(1 for r in rows if r["date"] and (ref - date.fromisoformat(r["date"])).days <= config.RECENT_DAYS)
        out.execute("INSERT INTO scope_stats VALUES (?,?,?,?,?,?,?,?,?,?)", (
            ik, cb, len(rows), recent, max(r["date"] for r in rows),
            sum(qs) / len(qs) if qs else None, sum(ds) / len(ds) if ds else None,
            sum(wt) / len(wt) if wt else None, len(wt), _confidence(len(rows))))

    K = config.SHRINK_K
    prior = {t.name: {kind: v for kind, v, _ in _comp_values(t, reviews)} for t in TRAIT_LIST}

    def shrunk_components(trait, rows):
        comps = []
        for kind, v, n in _comp_values(trait, rows):
            p = prior[trait.name][kind]
            comps.append({"kind": kind, "raw": round(v, 4), "n": n, "shrunk": (n * v + K * p) / (n + K)})
        return comps

    overall: dict[tuple[str, str], list[dict]] = {}
    for (ik, cb), rows in by_scope.items():
        if cb == "*":
            for t in TRAIT_LIST:
                overall[(ik, t.name)] = shrunk_components(t, rows)
    population: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (ik, tname), comps in overall.items():
        for c in comps:
            population[(tname, c["kind"])].append(c["shrunk"])

    for (ik, cb), rows in by_scope.items():
        for t in TRAIT_LIST:
            comps = overall[(ik, t.name)] if cb == "*" else shrunk_components(t, rows)
            for c in comps:
                c["pct"] = round(_pct(c["shrunk"], population[(t.name, c["kind"])]), 4)
                c["shrunk"] = round(c["shrunk"], 4)
            if not comps:
                continue
            score = sum(c["pct"] for c in comps) / len(comps)
            n_pos = sum(1 for r in rows if +1 in r["tagflags"].get(t.name, ()) or +1 in r["lexflags"].get(t.name, ()))
            n_neg = sum(1 for r in rows if -1 in r["tagflags"].get(t.name, ()) or -1 in r["lexflags"].get(t.name, ()))
            out.execute("INSERT INTO trait_scores VALUES (?,?,?,?,?,?,?,?)", (
                ik, cb, t.name, round(score, 4), n_pos, n_neg, len(rows), json.dumps(comps)))

    log(f"cache built: {len(instructors)} instructors, {len(sec_rows)} sections, {len(reviews)} reviews, "
        f"{len(trait_rows)} trait hits")
    return {
        "course_term_name": cmeta.get("term_name", ""), "course_term_code": cmeta.get("term_code", ""),
        "course_retrieved_at_utc": cmeta.get("retrieved_at_utc", ""),
        "rmp_latest_review_date": latest, "n_reviews": len(reviews), "n_sections": len(sec_rows),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build or refresh the profrec cache.")
    ap.add_argument("--force", action="store_true", help="rebuild even if the cache looks fresh")
    args = ap.parse_args(argv)
    meta = ensure_cache(force=args.force)
    if meta.get("rebuilt"):
        print(f"rebuilt in {meta.get('build_seconds')}s -> {config.CACHE_DB}")
    else:
        print(f"cache is fresh (built {meta.get('built_at_utc')}); use --force to rebuild")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
