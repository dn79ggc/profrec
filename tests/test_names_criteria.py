from profrec.criteria import TRAIT_NAMES, analyze_review
from profrec.names import course_base, norm_key, review_course_code, schedule_display_name


def test_schedule_display_name():
    assert schedule_display_name("Anil(Anil)Pereira") == "Anil Pereira"


def test_norm_key_drops_parentheticals_and_accents():
    assert norm_key("Jeffrey (J. P.) Zinn") == norm_key("Jeffrey Zinn") == "jeffrey zinn"
    assert norm_key("José O'Neil") == "jose o neil"


def test_review_course_code():
    assert review_course_code("ITEC2150") == "ITEC2150"
    assert review_course_code("itec-3300") == "ITEC3300"
    assert review_course_code("ITECH2110") == "ITEC2110"
    assert review_course_code("1001") is None
    assert review_course_code("DIGITALMEDIA") is None
    assert review_course_code(None) is None


def test_course_base():
    assert course_base("2110H") == "2110"
    assert course_base("2150") == "2150"


def test_negation_is_respected():
    pos = analyze_review("He is a very lenient grader.", [])
    neg = analyze_review("He is not lenient at all.", [])
    assert any(h.trait == "strict_grading" and h.polarity < 0 for h in pos)
    assert not any(h.trait == "strict_grading" and h.polarity < 0 for h in neg)


def test_late_penalty_is_not_strict_grading_noise():
    hits = analyze_review("Points off for late work are small.", [])
    assert not any(h.trait == "strict_grading" for h in hits)


def test_tags_map_to_traits():
    hits = analyze_review("", ["tough grader"])
    assert any(h.trait == "strict_grading" and h.source == "tag" for h in hits)


def test_hit_text_is_verbatim_sentence():
    comment = "Great class. He is very helpful during office hours! Exams were fine."
    for h in analyze_review(comment, []):
        if h.source == "lexicon":
            assert h.text in comment


def test_trait_vocabulary_is_closed():
    assert len(set(TRAIT_NAMES)) == len(TRAIT_NAMES) == 14
