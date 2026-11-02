"""Streamlit front end.  Run:  streamlit run app.py

All matching logic lives in the profrec package (no Streamlit imports there). This file only
collects input, shows how the request was understood (editable), and displays engine results.
"""
from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from profrec import config  # noqa: E402
from profrec.criteria import TRAIT_NAMES, TRAITS  # noqa: E402
from profrec.engine import Candidate, Engine, Recommendation  # noqa: E402
from profrec.parser import FallbackParser, GeminiParser, ParserError, RuleBasedParser  # noqa: E402
from profrec.render import ordinal  # noqa: E402
from profrec.schema import Criterion, ParsedRequest, ScheduleConstraints  # noqa: E402

st.set_page_config(page_title="Professor Match", page_icon="🎓", layout="wide")

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CONF_LABEL = {"none": "no reviews", "low": "very few reviews", "medium": "some reviews", "high": "many reviews"}


# ---------------------------------------------------------------------------------------------
# resources
# ---------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing data…")
def get_engine() -> Engine:
    return Engine()


def gemini_key() -> str | None:
    key = os.environ.get(config.GEMINI_API_KEY_ENV)
    if key:
        return key
    try:
        return st.secrets.get(config.GEMINI_API_KEY_ENV)
    except Exception:  # no secrets file
        return None


@st.cache_resource
def get_parser(mode: str, key: str | None):
    if mode == "rules" or not key:
        return RuleBasedParser()
    return FallbackParser(GeminiParser(api_key=key), RuleBasedParser())


def esc(text: str) -> str:
    """Escape review text for st.markdown (stops $ being read as math, * as emphasis, etc.)."""
    return re.sub(r"([\\`*_{}\[\]()#+!$<>|~])", r"\\\1", html.unescape(text))


# ---------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------
def sections_table(c: Candidate) -> pd.DataFrame:
    return pd.DataFrame([{
        "Section": s.section_number, "CRN": s.crn, "Days & time": s.schedule_text, "Where": s.place_text,
        "Seats open": f"{s.seats_available} of {s.max_enrollment}" if s.max_enrollment is not None else s.seats_available,
        "Dates": f"{s.start_date} → {s.end_date}" if s.start_date else "",
    } for s in c.sections])


def render_candidate(c: Candidate, mode: str) -> None:
    with st.container(border=True):
        left, right = st.columns([3, 1])
        title = f"#{c.rank}  {c.name}" if c.rank else c.name
        left.subheader(title)
        if c.combined is not None and mode == "ranked":
            right.metric("Fit with your request", f"{round(c.combined * 100)}/100",
                         help="Relative score among the options for this course, built from the traits you asked for. "
                              "It estimates style fit from reviews, not teaching quality.")
        if c.status == "tba":
            st.info("Instructor not assigned yet (TBA), so there are no reviews to show.")
        elif c.status == "no_rmp_profile":
            st.info("No RateMyProfessors profile was found for this instructor, so there are no reviews to show.")
        elif c.status == "no_reviews":
            st.info("This instructor has an RMP profile but no collected reviews.")
        if c.rmp_site:
            r = c.rmp_site
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("RMP quality", f"{r['avg_rating']}/5")
            m2.metric("RMP difficulty", f"{r['avg_difficulty']}/5")
            m3.metric("Would retake", f"{r['would_take_again_pct']:.0f}%" if r["would_take_again_pct"] is not None else "n/a")
            m4.metric("RMP ratings", r["num_ratings"])
            st.caption(f"RateMyProfessors totals for this professor. This app analyzed {c.n_reviews} collected review(s) from "
                       f"{c.scope_label} ({CONF_LABEL[c.confidence]}); newest {c.last_review_date}."
                       + (f" [Open RMP profile]({c.rmp_url})" if c.rmp_url else ""))
        if c.trait_lines:
            for ln in c.trait_lines:
                want = "more" if ln.direction == "prefer" else "less"
                pct_txt = (ordinal(round(ln.score * 100)) + " percentile among IT instructors"
                           if ln.score is not None else "no evidence")
                counts = (f"{ln.n_pos} review(s) show it, {ln.n_neg} show the opposite, of {ln.n_reviews}"
                          if TRAITS[ln.trait].pos_tags or TRAITS[ln.trait].neg_tags
                          or TRAITS[ln.trait].pos_patterns or TRAITS[ln.trait].neg_patterns
                          else f"based on star ratings of {ln.n_reviews} review(s)")
                st.markdown(f"- **{ln.trait.replace('_', ' ')}** (you want {want}): {pct_txt} · {counts}")
        st.markdown("**Sections**")
        st.dataframe(sections_table(c), hide_index=True, width="stretch")
        quotes = [e for e in c.evidence if e.kind in ("quote", "keyword")]
        stats = [e for e in c.evidence if e.kind == "tag_stat"]
        if quotes or stats:
            with st.expander("Evidence from reviews"):
                for e in stats:
                    st.markdown(f"- {e.text}")
                for e in quotes:
                    icon = {True: "✅ supports your preference", False: "⚠️ caution", None: "🔎 matches your keywords"}[e.supports]
                    st.markdown(f"{icon} · *{e.trait.replace('_', ' ')}*\n\n> “{esc(e.text)}”")
                    st.caption(f"review {e.review_id} · {e.date} · {e.course or 'course not labeled'}")
        for n in c.notes:
            st.caption(f"ℹ️ {n}")


def render_result(rec: Recommendation) -> None:
    if rec.mode == "no_sections":
        st.warning("No sections of this course match your schedule filters.")
    elif rec.mode == "list":
        st.markdown(f"**{len(rec.candidates)} instructor group(s)** teach {rec.course.label} this term. "
                    f"Describe the professor you want below to rank them.")
    elif rec.mode == "single":
        st.success(f"Only one instructor teaches {rec.course.label} this term, so there is nothing to compare. "
                   f"Here are all matching sections and what reviews say.")
    else:
        if rec.default_criteria_used:
            st.info("I didn't recognize specific preferences, so the options are ordered by overall rating.")
        st.markdown(f"**{sum(1 for c in rec.candidates if c.rank)} ranked option(s)** for {rec.course.label}, best fit first.")
    for n in rec.notes:
        st.caption(f"ℹ️ {n}")
    for c in rec.candidates:
        render_candidate(c, rec.mode)
    if rec.excluded:
        with st.expander(f"{len(rec.excluded)} section(s) hidden by your schedule filters"):
            for s, why in rec.excluded:
                st.markdown(f"- Section {s.section_number} ({s.instructor or 'TBA'}), {s.schedule_text}: {'; '.join(why)}")


def interpretation_panel(req: ParsedRequest, rid: int) -> ParsedRequest:
    """Show how the request was understood and let the student edit it. Re-ranking needs no AI call."""
    with st.expander("How I understood your request (you can edit this)", expanded=True):
        st.caption(f"Parsed by: {req.parsed_by}")
        if req.unsupported:
            st.warning("Reviews and schedules can't answer: " + "; ".join(req.unsupported))
        df = pd.DataFrame([{"trait": c.name, "direction": c.direction, "weight": c.weight} for c in req.criteria],
                          columns=["trait", "direction", "weight"])
        edited = st.data_editor(
            df, key=f"crit_{rid}", num_rows="dynamic", hide_index=True, width="stretch",
            column_config={
                "trait": st.column_config.SelectboxColumn("Trait", options=list(TRAIT_NAMES), required=True),
                "direction": st.column_config.SelectboxColumn("You want", options=["prefer", "avoid"], required=True,
                                                              help="prefer = more of it, avoid = less of it"),
                "weight": st.column_config.NumberColumn("Importance (0–1)", min_value=0.0, max_value=1.0, step=0.1, default=1.0),
            })
        kw = st.text_input("Words to look for in reviews (comma separated)", ", ".join(req.keywords), key=f"kw_{rid}")
        c1, c2, c3, c4 = st.columns(4)
        es = c1.text_input("Earliest start (HH:MM)", req.constraints.earliest_start or "", key=f"es_{rid}")
        le = c2.text_input("Latest end (HH:MM)", req.constraints.latest_end or "", key=f"le_{rid}")
        ad = c3.multiselect("Avoid days", DAYS, default=req.constraints.avoid_days, key=f"ad_{rid}")
        dl = c4.selectbox("Delivery", ["any", "in_person", "online"],
                          index=["any", "in_person", "online"].index(req.constraints.delivery), key=f"dl_{rid}")
        with st.popover("What do these traits mean?"):
            for t in TRAITS.values():
                st.markdown(f"**{t.name.replace('_', ' ')}**: {t.description}")
    crits = []
    for _, row in edited.iterrows():
        if row.get("trait") in TRAIT_NAMES and row.get("direction") in ("prefer", "avoid"):
            w = row.get("weight")
            crits.append(Criterion(name=row["trait"], direction=row["direction"], weight=1.0 if pd.isna(w) else w))
    return ParsedRequest(
        criteria=crits, keywords=[k for k in kw.split(",") if k.strip()],
        constraints=ScheduleConstraints(earliest_start=es or None, latest_end=le or None, avoid_days=ad, delivery=dl),
        unsupported=req.unsupported, parsed_by=req.parsed_by, original_text=req.original_text)


# ---------------------------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------------------------
engine = get_engine()
engine.refresh_if_stale()       # cheap hash check; rebuilds the cache only if a source database changed
as_of = engine.as_of()

with st.sidebar:
    st.header("Settings")
    key = gemini_key()
    mode = st.radio("Request understanding", ["Gemini (falls back to offline rules)", "Offline rules only"],
                    index=0 if key else 1, disabled=not key,
                    help="Gemini only reads your typed request. It never sees professor or review data.")
    if not key:
        st.caption(f"No Gemini key found (set the {config.GEMINI_API_KEY_ENV} environment variable). Using offline rules.")
    st.divider()
    st.caption(f"Courses: {as_of['term']} (retrieved {as_of['courses_retrieved']})\n\n"
               f"Reviews: {as_of['n_reviews']} from RateMyProfessors, through {as_of['reviews_through']}")
parser = get_parser("rules" if (not key or mode.startswith("Offline")) else "gemini", key)

st.title("🎓 Professor Match")
st.caption("Pick a course, describe the professor you want, and see matching sections with the review evidence behind each recommendation.")

courses = engine.courses()
course = st.selectbox("Course", courses, index=None, format_func=lambda c: c.display,
                      placeholder="Type a course number or name, e.g. 2150 or intermediate programming")

if "req_id" not in st.session_state:
    st.session_state.req_id = 0
prompt = st.chat_input("What kind of professor do you want? e.g. easy-going and helpful, no Friday classes")
if prompt:
    if course is None:
        st.warning("Choose a course first, then describe the professor you want.")
    else:
        with st.spinner("Understanding your request…"):
            req = parser.parse(prompt)
        st.session_state.req_id += 1
        st.session_state.last = req.model_dump()

last = st.session_state.get("last")
if course is not None and last is None:
    render_result(engine.recommend(course.number, ParsedRequest(), listing_only=True))
elif course is not None and last is not None:
    req0 = ParsedRequest.model_validate(last)
    with st.chat_message("user"):
        st.write(req0.original_text)
    with st.chat_message("assistant"):
        req = interpretation_panel(req0, st.session_state.req_id)
        render_result(engine.recommend(course.number, req))
elif course is None:
    st.info("Start by choosing a course above.")

st.divider()
st.caption("Fit scores estimate how well review language matches your request; they do not measure teaching quality. "
           "Student reviews are self-selected and can reflect bias, and small review counts are noisy, so counts and quotes are always shown. "
           "Everything displayed comes from the course database or the RateMyProfessors reviews; AI is used only to read your request.")
