# Professor Match (MVP)

Pick an ITEC course, describe the professor you want, and get matching sections ranked by how well each instructor's
RateMyProfessors (RMP) reviews fit the request, with the review quotes behind every recommendation.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"                        # optional; PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

Without a key the app uses the offline rule-based parser (limited vocabulary). With a key it uses Gemini
(`gemini-3.5-flash-lite` by default; override with `PROFREC_GEMINI_MODEL`) and falls back to the rules if the call fails.
Verify the key without the UI:

```bash
python -c "import sys; sys.path.insert(0,'src'); from profrec.parser import GeminiParser; print(GeminiParser().parse('easy-going, helpful, no Friday classes').model_dump())"
```

Run the tests: `python -m pytest -q`.

## How it works

1. **Sources of truth** (`data/*.sqlite`, opened read-only): Spring 2027 ITEC schedule and RMP reviews.
2. **Cache** (`cache/cache.sqlite`, derived, disposable). Built deterministically, no AI. The app checks SHA-256 of both source
   files plus the builder code and alias files on every page load, and rebuilds (about 1-2 s) into a temp file that is swapped in atomically.
   Force a rebuild: `python -m profrec.cache_build --force`. To update data, replace the two files in `data/` and reload.
3. **Request parsing** (the only AI step). The LLM sees only the typed request and the fixed trait list. It returns JSON validated against
   a closed schema (trait enum, prefer/avoid, weight, keywords, schedule constraints). It never sees professor or review data and cannot
   write recommendations. Parsed requests are cached in `cache/requests.sqlite`. The parse is shown to the student and is editable;
   re-ranking after an edit uses no AI.
4. **Engine**: filter sections by schedule, group by instructor; one instructor -> all sections returned immediately; several -> rank.
5. **Scoring**: for each trait the cache holds a 0-1 score per instructor (per course when at least 3 course reviews exist, otherwise across all
   their courses). Each component (RMP numeric rating, RMP tags, phrase matches in comments) is shrunk toward the department mean
   (K=5 pseudo-reviews), converted to a percentile among IT instructors, and averaged. A request's fit is the weighted mean of trait scores
   (avoid -> 1 - score); keywords found by full-text search add a 25% component. Constants are in `src/profrec/config.py`.
6. **Evidence**: every quote is a sentence copied verbatim from a review; counts and confidence labels (<5 reviews low, <15 medium, else high) are always displayed.

## Where "easy-going" etc. are defined

`src/profrec/criteria.py` holds the 14 traits, their RMP tags and phrase patterns. Edit it, and the cache rebuilds itself on the next load.

| Trait | Meaning | RMP rating used | RMP tags used |
|---|---|---|---|
| difficulty | How hard the course feels | difficulty | none (rating only) |
| strict_grading | Strict / nitpicky grading | none | tough grader |
| heavy_workload | Amount of work | none | lots of homework, so many papers, get ready to read |
| helpfulness | Supportive, wants students to succeed | helpful | caring, gives good feedback |
| accessibility | Reachable outside class | none | accessible outside class |
| clear_expectations | Clear criteria, rubrics, feedback | none | clear grading criteria, gives good feedback |
| extra_credit | Extra credit offered | none | extra credit |
| flexibility | Extensions, second chances | none | none |
| engagement | Engaging, fun lectures | none | amazing lectures, hilarious, inspirational |
| lecture_heavy | Lecturing vs hands-on | none | lecture heavy |
| exam_heavy | Grade driven by tests | none | test heavy, beware of pop quizzes |
| group_work | Group projects | none | group projects |
| attendance_strict | Attendance counts | attendance flag | skip class? you won't pass, participation matters |
| overall_rating | Well liked overall | quality, would-take-again | none |

"Easy-going" is not its own trait: it maps to avoid `difficulty` + avoid `strict_grading` + prefer `flexibility` (+ `helpfulness` if asked).

## Look and feel

- Themes live in `.streamlit/config.toml` (`[theme.light]`, `[theme.dark]`): one muted sage accent on warm neutrals in both, body text about 11:1 contrast instead of pure black on white. Switch with the three-dot menu (System / Light / Dark). That menu loses the theme items under `toolbarMode = "viewer"`, so only the Deploy button is hidden (CSS in `app.py`).
- Theme switches cross-fade over about 0.4 s (CSS in `app.py`); users with "reduce motion" set in their OS get an instant switch.
- Captions and quotes fade their text color rather than using Streamlit's `opacity: 0.6`, which fell below WCAG AA on this palette.
- The chat box is inline under the course picker on purpose. A page-pinned `st.chat_input` makes Streamlit scroll the whole page to the bottom on every rerun (`tests/test_app_smoke.py` guards this). It stays disabled until a course is chosen.
- The chat box has a 2 px sage outline (about 5:1 against the page in both themes, WCAG asks 3:1 for UI edges) and a separate focus ring.
- "New session" sits top right, left of the three-dot menu (fixed-position CSS keyed on `new_session`). It clears the course, results and queued example without reloading the page. The course picker is keyed so it actually resets; the request counter is kept on purpose because the edit widgets are keyed by it. It has no `help=` tooltip because Streamlit then adds a hidden duplicate button that takes a Tab stop. Known quirk: Tab reaches the three-dot menu just before this button.
- Starter requests (`src/profrec/examples.py`) appear as chips under the chat box and run through the normal request flow. A test checks each one works offline.
- Results lead with a one-line "Understood as" summary and a comparison table. The editable parse panel is collapsed, and only the top pick's "Why this fit" panel opens by default.
- Known limit: the tables are drawn on a canvas by Streamlit, so they change color instantly instead of fading.

## Alias files

- `data/course_aliases.json`: extra names for a course in the dropdown search, e.g. `{"2150": ["intermediate programming"]}`.
- `data/instructor_aliases.json`: `{"schedule_to_rmp": {"Schedule Name": "RMP Name"}}` when a schedule name does not match an RMP profile automatically.

## Limitations

- Phrase matching is literal: a paraphrase no pattern covers is missed. "No mention" means little evidence, not the opposite.
- Review text is self-selected and can reflect gender, race and grade bias. Scores estimate style fit, not teaching quality.
- Small samples are noisy; the 1,822 collected reviews are capped per professor, and tags only exist from about 2016.
- 5 instructors have no RMP profile and 11 sections have TBA instructors; they are listed but never ranked.
- The Gemini free tier may use submitted content to improve Google products; only the student's typed request is sent.
- Check RateMyProfessors' terms before publishing anything built on their data.
- No live Gemini call was tested in development (sandbox network); use the verification command above.
