import shutil
import sqlite3

import pytest

from profrec import config
from profrec.cache_build import build_cache, ensure_cache, is_fresh, read_meta


def test_build_has_expected_shape(built):
    meta = read_meta()
    assert int(meta["n_reviews"]) == 1822
    conn = config.connect_ro(config.CACHE_DB)
    assert conn.execute("SELECT COUNT(*) FROM courses").fetchone()[0] == 47
    assert conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0] == 161
    # FTS5 must be present and queryable
    assert conn.execute("SELECT COUNT(*) FROM review_fts WHERE review_fts MATCH 'helpful'").fetchone()[0] > 0
    conn.close()


def test_every_hit_text_is_verbatim_from_source(built):
    src = sqlite3.connect(config.RMP_DB)
    comments = {}
    cache = config.connect_ro(config.CACHE_DB)
    rows = cache.execute("SELECT r.id, r.comment, t.text, t.source FROM review_traits t "
                         "JOIN reviews r ON r.id=t.review_id WHERE t.source='lexicon'").fetchall()
    assert rows
    norm = lambda s: s.replace("’", "'")
    for r in rows:
        assert norm(r["text"]) in norm(r["comment"])
    # and every cached comment exists in the source DB
    src_comments = {row[0] for row in src.execute("SELECT comment FROM reviews")}
    for r in cache.execute("SELECT comment FROM reviews LIMIT 200"):
        assert r["comment"] in src_comments or r["comment"] is None
    cache.close(); src.close()


def test_fresh_after_build_and_stale_when_source_changes(built, tmp_path, monkeypatch):
    assert is_fresh()
    rmp = tmp_path / "rmp.sqlite"
    shutil.copy(config.RMP_DB, rmp)
    monkeypatch.setattr(config, "RMP_DB", rmp)
    assert is_fresh()                       # identical bytes -> same hash
    rmp.chmod(0o644)
    con = sqlite3.connect(rmp)
    con.execute("CREATE TABLE _touch(x)")   # any byte change must invalidate
    con.commit(); con.close()
    assert not is_fresh()


def test_atomic_build_leaves_old_cache_on_failure(built, tmp_path):
    before = config.CACHE_DB.read_bytes()
    bad = tmp_path / "bad.sqlite"
    bad.write_bytes(b"not a database")
    with pytest.raises(Exception):
        build_cache(config.CACHE_DB, config.COURSE_DB, bad, log=lambda *_: None)
    assert config.CACHE_DB.read_bytes() == before
    assert is_fresh()


def test_sources_are_never_modified(built, engine):
    """The app must never write to a source DB. Checked by behavior, not by file permission
    (permissions are not preserved by zip/git and are meaningless on Windows)."""
    from profrec.cache_build import _sha256

    for p in (config.COURSE_DB, config.RMP_DB):
        conn = config.connect_ro(p)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE _should_fail(x)")
        conn.close()

    before = {p: _sha256(p) for p in (config.COURSE_DB, config.RMP_DB)}
    ensure_cache(force=True, log=lambda *_: None)     # full rebuild reads both source
    engine.recommend("2150", listing_only=True)        # and a query path
    assert {p: _sha256(p) for p in before} == before
