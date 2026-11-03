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
from profrec.examples import EXAMPLE_REQUESTS  # noqa: E402
from profrec.parser import FallbackParser, GeminiParser, ParserError, RuleBasedParser  # noqa: E402
from profrec.render import nice_date, ordinal  # noqa: E402
from profrec.schema import Criterion, ParsedRequest, ScheduleConstraints  # noqa: E402

st.set_page_config(page_title="Professor Match", page_icon="🎓", layout="wide")

# Theme switch: Streamlit swaps all colors at once, which reads as a flash. Easing color-like properties
# turns it into a ~0.4 s cross-fade. Skipped entirely for people who ask their OS for reduced motion.
# opacity/transform/width are listed too so Streamlit's own fades and slides keep working.
#
# Secondary text: Streamlit fades captions and quotes with opacity 0.6, which on the soft palette drops
# below WCAG AA (3.6:1 in light mode) and also fades any link inside them. Fading the text color instead
# keeps links at full strength; 80% keeps every pairing at 6:1 or better in both themes.
#
# Deploy button: only shows on localhost and does nothing for students, so it is hidden. The menu next to it
# stays, because Settings (Light / Dark / System) lives there. toolbarMode="viewer" would remove Settings too.
# The user's chat bubble is tinted from the text color, so it follows both themes instead of a fixed grey.
st.html("""
<style>
[data-testid="stCaptionContainer"], .stApp blockquote {
  opacity: 1 !important;
  color: color-mix(in srgb, currentColor 80%, transparent);
}
[data-testid="stAppDeployButton"] { display: none; }

/* Chat box: Streamlit draws it with a transparent border and a fill 1.1:1 against the page, so it
   disappears. A sage mixed with the theme's own text color gives 5:1 or better in both themes and
   fades with them (currentColor is the text color). Focus adds an offset ring for keyboard users. */
[data-testid="stChatInput"] > div:has(textarea) {
  border: 2px solid color-mix(in srgb, #5F8F76 55%, currentColor);
}
[data-testid="stChatInput"] > div:has(textarea):focus-within {
  outline: 2px solid color-mix(in srgb, #5F8F76 55%, currentColor);
  outline-offset: 2px;
}

/* New session: Streamlit has no API for toolbar items, so the button is fixed-positioned into the
   header, left of the menu (Deploy's old slot is empty). Out of flow, so it adds no gap to the page. */
.st-key-new_session {
  position: fixed;
  top: 0.7rem;
  right: 3.6rem;
  z-index: 1000000;
  width: auto;
}
.st-key-new_session button { min-height: 2.25rem; padding: 0.25rem 0.8rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background-color: color-mix(in srgb, currentColor 7%, transparent);
}
@media (prefers-reduced-motion: no-preference) {
  .stApp, .stApp *, .stApp *::before, .stApp *::after {
    transition: background-color 420ms ease, color 420ms ease, border-color 420ms ease,
                fill 420ms ease, stroke 420ms ease, box-shadow 420ms ease,
                opacity 200ms ease, transform 200ms ease, width 200ms ease;
  }
}
</style>
""")

def new_session() -> None:
    """Back to the first screen: no course, no request, no chip selected.

    A widget is reset by assigning its keyed value, not by deleting it (a deleted widget keeps the value
    the browser holds). req_id is deliberately NOT reset: the edit widgets are keyed by it, so reusing
    numbers would bring an earlier session's edits back. Cached resources (engine, parser) are untouched.
    """
    st.session_state["course"] = None
    st.session_state["example_pick"] = None
    for k in ("last", "queued_prompt"):
        st.session_state.pop(k, None)


# No help= tooltip on purpose: Streamlit then adds a hidden 0x0 copy of the button that still takes a Tab stop.
st.button("New session", key="new_session", on_click=new_session, icon=":material/restart_alt:")

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


def _retake(r: dict) -> str:
    return f"{r['would_take_again_pct']:.0f}%" if r["would_take_again_pct"] is not None else "n/a"


def comparison_table(rec: Recommendation) -> pd.DataFrame:
    """One row per instructor so options can be compared at a glance before reading any card."""
    ranked = rec.mode == "ranked"
    rows = []
    for c in rec.candidates:
        r = c.rmp_site
        row = {}
        if ranked:      # rank and fit mean nothing before a request, so those columns appear only with one
            row["Rank"] = c.rank if c.rank else ""
        row["Instructor"] = c.name
        if ranked:
            row["Fit"] = f"{round(c.combined * 100)}/100" if c.combined is not None else ""
        row.update({
            "RMP quality": f"{r['avg_rating']}/5" if r else "",
            "RMP difficulty": f"{r['avg_difficulty']}/5" if r else "",
            "Would retake": _retake(r) if r else "",
            "Reviews analyzed": f"{c.n_reviews} ({CONF_LABEL[c.confidence]})" if c.n_reviews else "none",
            "Sections": len(c.sections),
        })
        rows.append(row)
    return pd.DataFrame(rows)


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
            st.markdown(f"RateMyProfessors: **{r['avg_rating']}/5** quality · **{r['avg_difficulty']}/5** difficulty · "
                        f"**{_retake(r)}** would retake · **{r['num_ratings']}** ratings")
            newest = f"; newest {nice_date(c.last_review_date)}" if c.last_review_date else ""
            st.caption(f"This app analyzed {c.n_reviews} collected review(s) from {c.scope_label} "
                       f"({CONF_LABEL[c.confidence]}){newest}."
                       + (f" [Open RMP profile]({c.rmp_url})" if c.rmp_url else ""))
        st.markdown("**Sections**")
        st.dataframe(sections_table(c), hide_index=True, width="stretch")
        quotes = [e for e in c.evidence if e.kind in ("quote", "keyword")]
        stats = [e for e in c.evidence if e.kind == "tag_stat"]
        if c.trait_lines or quotes or stats:
            # The top pick (or the only instructor) opens by default; the rest stay folded so the page
            # does not become a wall of text when comparing four or five professors.
            with st.expander("Why this fit, with review evidence", expanded=(c.rank == 1 or mode == "single")):
                for ln in c.trait_lines:
                    want = "more" if ln.direction == "prefer" else "less"
                    pct_txt = (ordinal(round(ln.score * 100)) + " percentile among IT instructors"
                               if ln.score is not None else "no evidence")
                    counts = (f"{ln.n_pos} review(s) show it, {ln.n_neg} show the opposite, of {ln.n_reviews}"
                              if TRAITS[ln.trait].pos_tags or TRAITS[ln.trait].neg_tags
                              or TRAITS[ln.trait].pos_patterns or TRAITS[ln.trait].neg_patterns
                              else f"based on star ratings of {ln.n_reviews} review(s)")
                    st.markdown(f"- **{ln.trait.replace('_', ' ')}** (you want {want}): {pct_txt} · {counts}")
                for e in stats:
                    st.markdown(f"- {e.text}")
                for e in quotes:
                    label = {True: "Supports your preference", False: "Caution", None: "Matches your keywords"}[e.supports]
                    st.markdown(f"**{label}** · *{e.trait.replace('_', ' ')}*\n\n> “{esc(e.text)}”")
                    st.caption(f"review {e.review_id} · {e.date} · {e.course or 'course not labeled'}")
        for n in c.notes:
            st.caption(n)


def render_result(rec: Recommendation) -> None:
    if rec.mode == "no_sections":
        st.warning("No sections of this course match your schedule filters.")
    elif rec.mode == "list":
        st.markdown(f"**{len(rec.candidates)} instructor group(s)** teach {rec.course.label} this term. "
                    f"Describe the professor you want, or pick an example, to rank them.")
    elif rec.mode == "single":
        st.success(f"Only one instructor teaches {rec.course.label} this term, so there is nothing to compare. "
                   f"Here are all matching sections and what reviews say.")
    else:
        if rec.default_criteria_used:
            st.info("I didn't recognize specific preferences, so the options are ordered by overall rating.")
        st.markdown(f"**{sum(1 for c in rec.candidates if c.rank)} ranked option(s)** for {rec.course.label}, best fit first.")
    for n in rec.notes:
        st.caption(n)
    if len(rec.candidates) > 1:
        st.dataframe(comparison_table(rec), hide_index=True, width="stretch")
    for c in rec.candidates:
        render_candidate(c, rec.mode)
    if rec.excluded:
        with st.expander(f"{len(rec.excluded)} section(s) hidden by your schedule filters"):
            for s, why in rec.excluded:
                st.markdown(f"- Section {s.section_number} ({s.instructor or 'TBA'}), {s.schedule_text}: {'; '.join(why)}")


def understood_summary(req: ParsedRequest) -> str:
    """One readable line: what the request was turned into."""
    parts = [f"{'more' if c.direction == 'prefer' else 'less'} {c.name.replace('_', ' ')}" for c in req.criteria]
    parts += [f"“{k}” in reviews" for k in req.keywords]
    cons = req.constraints
    if cons.avoid_days:
        parts.append("no " + "/".join(cons.avoid_days))
    if cons.earliest_start:
        parts.append(f"start {cons.earliest_start} or later")
    if cons.latest_end:
        parts.append(f"end by {cons.latest_end}")
    if cons.delivery != "any":
        parts.append(cons.delivery.replace("_", " "))
    return ", ".join(parts) if parts else "no specific preferences recognized"


def interpretation_panel(req: ParsedRequest, rid: int) -> ParsedRequest:
    """Show how the request was understood and let the student edit it. Re-ranking needs no AI call.

    Collapsed by default: the one-line summary is enough for most requests, and the results should be
    the first thing on screen, not a table editor.
    """
    if req.unsupported:
        st.warning("Reviews and schedules can't answer: " + "; ".join(req.unsupported))
    summary_slot = st.empty()       # filled after the edit widgets run, so it reflects edits
    with st.expander("Edit how I understood your request", expanded=False):
        st.caption(f"Parsed by: {req.parsed_by}")
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
    final = ParsedRequest(
        criteria=crits, keywords=[k for k in kw.split(",") if k.strip()],
        constraints=ScheduleConstraints(earliest_start=es or None, latest_end=le or None, avoid_days=ad, delivery=dl),
        unsupported=req.unsupported, parsed_by=req.parsed_by, original_text=req.original_text)
    summary_slot.markdown(f"**Understood as:** {understood_summary(final)}")
    return final


# ---------------------------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------------------------
engine = get_engine()
engine.refresh_if_stale()       # cheap hash check; rebuilds the cache only if a source database changed
as_of = engine.as_of()

with st.sidebar:
    key = gemini_key()
    if key:
        st.subheader("Reading your request")
        mode = st.radio("Method", ["Gemini (falls back to offline rules)", "Offline rules only"], index=0,
                        help="Gemini only reads your typed request. It never sees professor or review data.")
    else:
        # A greyed-out radio with one live option is clutter; say what is happening and how to change it.
        mode = "Offline rules only"
        st.subheader("Reading your request")
        st.caption(f"Using built-in rules, which understand common phrases. Set {config.GEMINI_API_KEY_ENV} "
                   "to let Gemini read free-form wording.")
    st.subheader("About the data")
    st.caption(f"Schedule: {as_of['term']}, retrieved {nice_date(as_of['courses_retrieved'])}.")
    st.caption(f"Reviews: {int(as_of['n_reviews'] or 0):,} from RateMyProfessors, newest {nice_date(as_of['reviews_through'])}.")
parser = get_parser("rules" if (not key or mode.startswith("Offline")) else "gemini", key)

st.title("Professor Match")
st.caption("Pick a course, describe the professor you want, and see matching sections with the review evidence behind each recommendation.")

courses = engine.courses()
course = st.selectbox("Course", courses, index=None, format_func=lambda c: c.display, key="course",
                      placeholder="Type a course number or name, e.g. 2150 or intermediate programming")

if "req_id" not in st.session_state:
    st.session_state.req_id = 0
# Inside a container the chat box renders inline under the course picker. Pinned to the page bottom,
# Streamlit makes the whole page a "scroll to bottom" container and drags the view down on every rerun.


def _queue_example() -> None:
    """Chip clicked: hand its text to the normal request flow on the next run, then clear the chip."""
    pick = st.session_state.get("example_pick")
    if pick:
        st.session_state.queued_prompt = pick
        st.session_state.example_pick = None


with st.container():
    prompt = st.chat_input(
        "What kind of professor do you want? e.g. easy-going and helpful, no Friday classes" if course is not None
        else "Choose a course first", disabled=course is None)
    if course is not None:
        st.pills("Or try an example", EXAMPLE_REQUESTS, key="example_pick", on_change=_queue_example)
prompt = st.session_state.pop("queued_prompt", None) or prompt
if prompt and course is not None:
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
else:
    st.caption("Choose a course to see who teaches it this term and which sections are open.")

st.divider()
st.caption("Fit scores estimate how well review language matches your request; they do not measure teaching quality. "
           "Student reviews are self-selected and can reflect bias, and small review counts are noisy, so counts and quotes are always shown. "
           "Everything displayed comes from the course database or the RateMyProfessors reviews; AI is used only to read your request.")
