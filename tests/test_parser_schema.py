import pytest

from profrec.parser import FallbackParser, GeminiParser, ParserError, RequestCache, RuleBasedParser
from profrec.schema import LLM_RESPONSE_SCHEMA, ParsedRequest, from_llm_dict


def from_llm_dict_(d):
    return from_llm_dict(d, original_text='t', parsed_by='test')


def test_rule_parser_easygoing_no_friday():
    r = RuleBasedParser().parse("easy-going and helpful, not nitpicky, no Friday classes")
    names = {(c.name, c.direction) for c in r.criteria}
    assert ("helpfulness", "prefer") in names
    assert "Fri" in r.constraints.avoid_days


def test_rule_parser_morning_filter():
    r = RuleBasedParser().parse("nothing before 10am please")
    assert r.constraints.earliest_start == "10:00"


def test_invalid_trait_rejected():
    with pytest.raises(Exception):
        from_llm_dict_({"criteria": [{"name": "is_handsome", "direction": "prefer", "weight": 1}]})


def test_weight_clamped_and_keywords_sanitized():
    r = from_llm_dict_({"criteria": [{"name": "helpfulness", "direction": "prefer", "weight": 9}],
                       "keywords": ["python", "a b c d e f", "x" * 100, "", "ok"]})
    assert r.criteria[0].weight == 1.0
    assert len(r.keywords) <= 6 and all(len(k.split()) <= 3 for k in r.keywords)


def test_duplicate_traits_deduped():
    r = from_llm_dict_({"criteria": [{"name": "helpfulness", "direction": "prefer", "weight": 1},
                                    {"name": "helpfulness", "direction": "avoid", "weight": 0.5}]})
    assert len([c for c in r.criteria if c.name == "helpfulness"]) == 1


def test_bad_time_dropped():
    r = from_llm_dict_({"earliest_start": "25:99"})
    assert r.constraints.earliest_start is None


def test_gemini_parser_caches_and_fallback(tmp_path, monkeypatch):
    calls = []

    def fake(self, text):
        calls.append(text)
        return {"criteria": [{"name": "helpfulness", "direction": "prefer", "weight": 1}], "keywords": [],
                "constraints": {}, "unsupported": []}

    monkeypatch.setattr(GeminiParser, "_call", fake)
    g = GeminiParser(api_key="x", cache=RequestCache(tmp_path / "r.sqlite"))
    a = g.parse("Helpful please"); b = g.parse("  helpful   please ")
    assert len(calls) == 1 and a.criteria == b.criteria

    def boom(self, text):
        raise ParserError("down")

    monkeypatch.setattr(GeminiParser, "_call", boom)
    g2 = GeminiParser(api_key="x", cache=RequestCache(tmp_path / "r2.sqlite"))
    r = FallbackParser(g2, RuleBasedParser()).parse("no friday classes")
    assert "Fri" in r.constraints.avoid_days


def test_llm_schema_has_no_free_text_answer_field():
    assert "answer" not in LLM_RESPONSE_SCHEMA.get("properties", {})


def test_time_phrases():
    p = RuleBasedParser()
    assert p.parse("classes after 2pm").constraints.earliest_start == "14:00"
    assert p.parse("nothing after 3pm").constraints.latest_end == "15:00"
    assert p.parse("done by 1pm").constraints.latest_end == "13:00"
    assert p.parse("no classes before 11am").constraints.earliest_start == "11:00"
