"""Plain-text rendering of a Recommendation (CLI, tests, logs). Every number comes from the engine result."""
from __future__ import annotations

from .engine import Recommendation


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def pct(score: float | None) -> str:
    """0..1 score -> '82nd percentile'. The percentile is among all IT instructors in the RMP data."""
    return "no evidence" if score is None else f"{ordinal(round(score * 100))} percentile"


def to_text(rec: Recommendation, max_quotes: int = 3) -> str:
    out: list[str] = []
    a = rec.as_of
    out.append(f"{rec.course.label} – {rec.course.title}   [{a.get('term', '')}; reviews through {a.get('reviews_through', '')}]")
    req = rec.request
    if req and req.criteria:
        out.append("Interpreted as: " + ", ".join(f"{c.direction} {c.name} ({c.weight:.1f})" for c in req.criteria))
    if rec.default_criteria_used:
        out.append("No specific preferences recognized: ranked by overall rating.")
    if req and req.unsupported:
        out.append("The data cannot answer: " + "; ".join(req.unsupported))
    out.extend(f"Note: {n}" for n in rec.notes)
    if rec.mode == "no_sections":
        out.append("No sections match your schedule filters.")
        for s, why in rec.excluded[:10]:
            out.append(f"   excluded {s.section_number} ({s.instructor or 'TBA'}): {'; '.join(why)}")
        return "\n".join(out)
    out.append(f"Mode: {rec.mode}")
    for c in rec.candidates:
        head = f"#{c.rank} " if c.rank else "   "
        score = f"  match={c.combined:.2f}" if c.combined is not None else ""
        out.append(f"\n{head}{c.name}{score}  [{c.status}; {c.n_reviews} reviews from {c.scope_label or 'n/a'}; confidence {c.confidence}]")
        if c.rmp_site:
            r = c.rmp_site
            out.append(f"   RMP site: {r['avg_rating']}/5 quality, {r['avg_difficulty']}/5 difficulty, "
                       f"{r['would_take_again_pct']:.0f}% would retake, {r['num_ratings']} ratings" if r['would_take_again_pct'] is not None else
                       f"   RMP site: {r['avg_rating']}/5 quality, {r['avg_difficulty']}/5 difficulty, {r['num_ratings']} ratings")
        for ln in c.trait_lines:
            out.append(f"   you want to {ln.direction:6s} {ln.trait:18s} (weight {ln.weight:.1f}): they sit at the {pct(ln.score)} "
                       f"for this trait; {ln.n_pos} review(s) show it, {ln.n_neg} show the opposite, of {ln.n_reviews}")
        for e in [x for x in c.evidence if x.kind != "tag_stat"][:max_quotes]:
            mark = {True: "supports", False: "caution", None: "mentions"}[e.supports]
            out.append(f"   [{mark}: {e.trait}] “{e.text}” — review {e.review_id}, {e.date}, {e.course or 'course not labeled'}")
        for e in [x for x in c.evidence if x.kind == "tag_stat"][:2]:
            out.append(f"   [{e.trait}] {e.text}")
        for s in c.sections:
            out.append(f"   sec {s.section_number} (CRN {s.crn}) · {s.schedule_text} · {s.place_text} · {s.seats_available} seats open")
        out.extend(f"   note: {n}" for n in c.notes)
    return "\n".join(out)
