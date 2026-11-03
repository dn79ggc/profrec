from profrec.engine import is_full
from profrec.parser import RuleBasedParser
from profrec.schema import Criterion, ParsedRequest, ScheduleConstraints


def test_course_list_and_alias(engine):
    cs = engine.courses()
    assert len(cs) == 47
    c = next(c for c in cs if c.number == "2150")
    assert "intermediate programming" in c.display.lower()


def test_listing_only_is_unranked(engine):
    rec = engine.recommend("2150", listing_only=True)
    assert rec.mode == "list"
    assert all(c.rank is None for c in rec.candidates)


def test_single_instructor_course_returns_all_sections(engine):
    single = [c for c in engine.courses() if c.n_instructors == 1 and c.n_sections > 1]
    assert single, "dataset should contain a multi-section single-instructor course"
    rec = engine.recommend(single[0].number, RuleBasedParser().parse("easy and helpful"))
    assert rec.mode == "single"
    assert len(rec.candidates) == 1
    assert len(rec.candidates[0].sections) == single[0].n_sections


def test_ranked_order_and_friday_filter(engine):
    req = RuleBasedParser().parse("easy-going and helpful, not nitpicky, no Friday classes")
    rec = engine.recommend("2150", req)
    assert rec.mode == "ranked"
    ranked = [c for c in rec.candidates if c.rank]
    assert [c.rank for c in ranked] == list(range(1, len(ranked) + 1))
    scores = [c.combined for c in ranked]
    assert scores == sorted(scores, reverse=True)
    for c in rec.candidates:
        for s in c.sections:
            assert "Fri" not in s.day_list
    assert rec.excluded, "the MWF section should be reported as hidden"


def test_all_sections_filtered_gives_no_sections(engine):
    req = ParsedRequest(constraints=ScheduleConstraints(earliest_start="23:00"))
    assert engine.recommend("2150", req).mode == "no_sections"


def test_quotes_are_verbatim_substrings_of_source_comments(engine):
    import sqlite3
    from profrec import config
    src = sqlite3.connect(config.RMP_DB)
    all_comments = [r[0] or "" for r in src.execute("SELECT comment FROM reviews")]
    norm = lambda s: s.replace("’", "'")
    blob = norm("\n".join(all_comments))
    req = ParsedRequest(criteria=[Criterion(name="helpfulness", direction="prefer"),
                                  Criterion(name="strict_grading", direction="avoid")], keywords=["office hours"])
    rec = engine.recommend("2150", req)
    seen = 0
    for c in rec.candidates:
        for e in c.evidence:
            if e.kind in ("quote", "keyword"):
                seen += 1
                import html
                assert norm(html.unescape(e.text)) in blob or norm(e.text) in blob
    assert seen > 0


def test_unreviewed_instructors_listed_not_ranked(engine):
    for course in engine.courses():
        rec = engine.recommend(course.number, ParsedRequest(criteria=[Criterion(name="helpfulness", direction="prefer")]))
        for c in rec.candidates:
            if c.status in ("tba", "no_rmp_profile", "no_reviews") and rec.mode == "ranked":
                assert c.rank is None


def test_no_instructor_name_invented(engine):
    import sqlite3
    from profrec import config
    names = {r[0] for r in sqlite3.connect(config.CACHE_DB).execute("SELECT display_name FROM instructors")}
    for course in engine.courses():
        for c in engine.recommend(course.number, listing_only=True).candidates:
            assert c.name in names or c.name.upper().startswith("TBA") or c.status == "tba"


def test_open_only_drops_full_sections_and_says_why(engine):
    everyone = engine.recommend("3600", listing_only=True)
    open_only = engine.recommend("3600", listing_only=True, open_only=True)
    assert sum(len(c.sections) for c in everyone.candidates) == 2
    assert open_only.open_only and all(not is_full(s) for c in open_only.candidates for s in c.sections)
    assert [why for _, why in open_only.excluded] == [["full (0 seats open)"]]
    assert open_only.mode == "single"            # one instructor still has seats


def test_open_only_with_every_section_full_is_no_sections(engine):
    assert engine.recommend("4310", open_only=True).mode == "no_sections"
    assert engine.recommend("4310").mode == "single"


def test_open_only_off_changes_nothing(engine):
    a = engine.recommend("2150", listing_only=True)
    b = engine.recommend("2150", listing_only=True, open_only=False)
    assert [c.name for c in a.candidates] == [c.name for c in b.candidates] and not b.excluded


def test_is_full_uses_seat_count_and_never_guesses():
    from profrec.engine import SectionInfo
    mk = lambda seats: SectionInfo("1", "01", "9999", "t", None, None, "", None, None, None, None, None, None, None,
                                   seats, 0, 0, True)
    assert is_full(mk(0)) and not is_full(mk(3)) and not is_full(mk(None))
