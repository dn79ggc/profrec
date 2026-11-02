"""Headless smoke tests of the Streamlit page: it renders, the dropdown works, a request produces results."""
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

from profrec.examples import EXAMPLE_REQUESTS  # noqa: E402
from profrec.parser import RuleBasedParser  # noqa: E402
from profrec.render import nice_date  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def no_gemini_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def _app(built):
    return streamlit_testing.AppTest.from_file(APP, default_timeout=60).run()


def _pick_2150(at):
    at.selectbox[0].select(next(o for o in at.selectbox[0].options if "2150" in o)).run()


def test_page_renders_with_all_courses(built):
    at = _app(built)
    assert not at.exception
    assert len(at.selectbox[0].options) == 47


def test_request_flow_ranks_without_errors(built):
    at = _app(built)
    _pick_2150(at)
    assert not at.exception
    at.chat_input[0].set_value("easy-going and helpful, not nitpicky, no Friday classes").run()
    assert not at.exception
    titles = [s.value for s in at.subheader]
    assert titles and titles[0].startswith("#1")


def test_chat_box_waits_for_a_course(built):
    at = _app(built)
    assert at.chat_input[0].disabled
    _pick_2150(at)
    assert not at.chat_input[0].disabled


def test_example_chip_runs_a_request(built):
    at = _app(built)
    _pick_2150(at)
    at.session_state["queued_prompt"] = EXAMPLE_REQUESTS[0]   # what the chip's callback sets
    at.run()
    assert not at.exception
    assert [s.value for s in at.subheader][0].startswith("#1")


def test_comparison_table_lists_every_instructor(built):
    at = _app(built)
    _pick_2150(at)
    at.chat_input[0].set_value("helpful").run()
    frames = [d.value for d in at.dataframe]
    table = next(f for f in frames if "Instructor" in f.columns)
    assert len(table) >= 2 and "Fit" in table.columns


def test_sidebar_has_no_dead_radio_without_key(built):
    at = _app(built)
    assert len(at.radio) == 0
    assert any("GEMINI_API_KEY" in c.value for c in at.sidebar.caption)


@pytest.mark.parametrize("text", EXAMPLE_REQUESTS)
def test_every_example_works_offline(text):
    req = RuleBasedParser().parse(text)
    assert req.criteria and not req.unsupported


def test_nice_date():
    assert nice_date("2026-10-22") == "Oct 22, 2026"
    assert nice_date("2027-01-05") == "Jan 5, 2027"
    assert nice_date("") == ""
    assert nice_date("not a date") == "not a date"


def test_no_decorative_emoji_in_title(built):
    at = _app(built)
    assert [t.value for t in at.title] == ["Professor Match"]


def test_chat_input_is_inline_not_page_pinned():
    """A page-pinned st.chat_input makes Streamlit auto-scroll the whole page to the bottom on every rerun.
    It must stay inside a container (see app.py)."""
    src = Path(APP).read_text(encoding="utf-8")
    idx = src.index("st.chat_input(")
    assert "with st.container():" in src[max(0, idx - 120):idx]
