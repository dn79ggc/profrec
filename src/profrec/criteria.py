"""The closed vocabulary of traits, and exactly how each one is measured from the RMP data.

This file is the single definition of what words like "easy-going" mean in this app.
Nothing here asks an AI anything. A trait is measured from three kinds of evidence that already
exist in the RMP database:

  rating   RMP's own numeric fields (difficulty, quality, helpful, would-take-again, attendance flag)
  tags     RMP's fixed tag vocabulary (e.g. "Tough grader", "Extra credit")
  lexicon  literal phrases found in the review comment (regex below); the matched sentence is
           kept verbatim so it can be quoted back as evidence

A score of 1.0 always means "this trait is strongly present" (difficulty -> hard; strict_grading
-> strict). "Easy-going" is therefore expressed as *avoid* difficulty / strict_grading / and *prefer*
flexibility, not as a separate trait. The LLM maps the student's words onto these traits; it never
decides anything about a professor.

Pattern strings beginning with '!' already contain their own negation ("not helpful"), so the
generic negation check ("not lenient" must not count as lenient) is skipped for them.
How a score is made (see cache_build.py): each evidence kind is turned into a value per professor,
pulled toward the department average when there are few reviews (shrinkage), converted to a
percentile among all IT instructors, and the percentiles are averaged. So a score of 0.8 means
"the reviews show this more strongly than for ~80% of IT instructors".

Known limitations:
  * The lexicon is literal. Paraphrases that no pattern covers are missed.
  * Text and tag signals measure how often a trait is *mentioned*. No mention means little
    evidence, not the opposite, so such professors sit low-to-middle. Review counts are always shown.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Trait:
    name: str
    description: str                     # plain language; shown in the UI and given to the LLM
    ratings: tuple[str, ...] = ()        # numeric components: 'difficulty','quality','helpful','would_take_again','attendance'
    pos_tags: tuple[str, ...] = ()       # tags that indicate the trait is present (lowercase)
    neg_tags: tuple[str, ...] = ()       # tags that indicate it is absent / opposite
    pos_patterns: tuple[str, ...] = ()   # regexes: comment shows the trait is present
    neg_patterns: tuple[str, ...] = ()   # regexes: comment shows it is absent / opposite
    _compiled: dict = field(default_factory=dict, compare=False, repr=False)

    def compiled(self) -> tuple[list, list]:
        if not self._compiled:
            def comp(pats):
                return [(re.compile(p.lstrip("!"), re.I), not p.startswith("!")) for p in pats]
            self._compiled["pos"] = comp(self.pos_patterns)
            self._compiled["neg"] = comp(self.neg_patterns)
        return self._compiled["pos"], self._compiled["neg"]


TRAIT_LIST: tuple[Trait, ...] = (
    Trait(
        "difficulty",
        "How hard or demanding the course feels. High = hard; low = easy.",
        ratings=("difficulty",),   # rating only: students write "easy A" far more often than "hard",
                                   # so a text signal here would be one-sided and misleading
    ),
    Trait(
        "strict_grading",
        "Grades strictly or nitpicks details. High = strict/picky; low = lenient/forgiving about grading.",
        pos_tags=("tough grader",),
        pos_patterns=(
            r"\b(strict|harsh|tough|hard|picky|unfair)\s+grad(?:er|ers|ing|es)\b",
            r"\bgrades?\s+(?:harshly|strictly|hard|unfairly)\b",
            r"\bnit[- ]?pick\w*",
            r"\bpicky\b",
            r"\bunforgiving\b",
        ),
        neg_patterns=(
            r"\blenient\b",
            r"\beasy\s+grad(?:er|ing)\b",
            r"\bgrades?\s+(?:easy|easily|generously|fairly)\b",
            r"\bgenerous\b",
            r"\bforgiving\b",
            r"\bpartial\s+credit\b",
            r"\beasy\s+(?:a|as|points?)\b",
            r"!\bdoesn'?t\s+need\s+perfect\b",
            r"!\bnot\s+(?:a\s+)?(?:strict|harsh|tough)\b",
        ),
    ),
    Trait(
        "heavy_workload",
        "Amount of homework, projects and reading. High = heavy; low = light.",
        pos_tags=("lots of homework", "so many papers", "get ready to read"),
        pos_patterns=(
            r"\b(?:lots?|tons?|alot|heavy|so\s+much|too\s+much)\s+(?:of\s+)?(?:homework|work|assignments?|projects?|coding|reading)\b",
            r"\ba\s+lot\s+of\s+(?:homework|work|assignments?|projects?|coding|reading)\b",
            r"\btime[- ]consuming\b",
            r"\bworkload\b",
        ),
        neg_patterns=(
            r"!\b(?:light|small|manageable|reasonable)\s+(?:workload|homework|amount\s+of\s+work)\b",
            r"!\bnot\s+(?:a\s+lot|much|too\s+much)\s+(?:of\s+)?(?:homework|work)\b",
            r"!\bno\s+homework\b",
        ),
    ),
    Trait(
        "helpfulness",
        "Supportive and caring; willing to help students succeed.",
        ratings=("helpful",),
        pos_tags=("caring", "gives good feedback"),
        pos_patterns=(
            r"\b(?:helpful|supportive|caring|patient)\b",
            r"\b(?:very|so|extremely|really)\s+kind\b",
            r"\bwilling\s+to\s+(?:help|work\s+with)\b",
            r"\bgoes?\s+(?:above\s+and\s+beyond|out\s+of\s+(?:his|her|their)\s+way)\b",
            r"\bwants?\s+(?:you|students|everyone|all\s+students)(?:\s+all)?\s+to\s+(?:succeed|learn|pass|do\s+well)\b",
            r"\bcares?\s+(?:about|for)\b",
        ),
        neg_patterns=(
            r"\bunhelpful\b",
            r"\bdismissive\b",
            r"\brude\b",
            r"!\bnot\s+(?:very\s+)?(?:helpful|supportive|caring)\b",
            r"!\b(?:doesn'?t|does\s+not|won'?t|never)\s+(?:help|care)\s+(?:students?|me|us|you|with|outside|about)\b",
        ),
    ),
    Trait(
        "accessibility",
        "Easy to reach outside class: office hours, quick replies, available for questions.",
        pos_tags=("accessible outside class",),
        pos_patterns=(
            r"\boffice\s+hours\b",
            r"\baccessible\b",
            r"\bresponsive\b",
            r"\b(?:responds?|replies|replied|answers?)\s+(?:to\s+)?(?:emails?|messages?|questions?)?\s*(?:quickly|fast|promptly|right\s+away|within)\b",
            r"\bhelps?\s+(?:you\s+)?(?:outside|after)\s+(?:of\s+)?(?:class|hours)\b",
            r"\bavailable\s+(?:outside|after|on|even)\b",
        ),
        neg_patterns=(
            r"\bunresponsive\b",
            r"!\bnever\s+(?:responds?|replies|answers?)\b",
            r"!\b(?:doesn'?t|does\s+not)\s+(?:respond|reply|answer)\b",
            r"!\bhard\s+to\s+(?:reach|get\s+(?:a\s+hold|ahold))\b",
            r"!\bnot\s+(?:very\s+)?(?:available|accessible|responsive)\b",
        ),
    ),
    Trait(
        "clear_expectations",
        "Clear grading criteria, rubrics, instructions and feedback.",
        pos_tags=("clear grading criteria", "gives good feedback"),
        pos_patterns=(
            r"\brubrics?\b",
            r"\bclear\s+(?:grading|expectations?|instructions?|criteria|assignments?|directions?)\b",
            r"\bwell[- ]organized\b",
            r"\bgrades?\s+(?:quickly|right\s+away|fast|promptly)\b",
            r"\bgood\s+feedback\b",
        ),
        neg_patterns=(
            r"\bunclear\b",
            r"\bvague\b",
            r"\bdisorganized\b",
            r"\bunorganized\b",
            r"\bconfusing\s+(?:instructions?|assignments?|grading|rubrics?|directions?)\b",
        ),
    ),
    Trait(
        "extra_credit",
        "Offers extra-credit opportunities.",
        pos_tags=("extra credit",),
        pos_patterns=(r"\bextra[- ]credit\b",),
        neg_patterns=(r"!\bno\s+extra[- ]credit\b",),
    ),
    Trait(
        "flexibility",
        "Flexible and understanding: deadline extensions, second chances, works with students.",
        pos_patterns=(
            r"\bextensions?\b",
            r"\bflexible\b",
            r"\b(?:very|extremely|so|really|super)\s+understanding\b",
            r"\bunderstanding\s+(?:of|and|professor|teacher|person|prof)\b",
            r"\blenient\s+(?:with|on)\s+(?:due\s+)?dates?\b",
            r"\blate\s+(?:work|assignments?|submissions?)\s+(?:is\s+|are\s+)?(?:accepted|allowed)\b",
            r"\bworks?\s+with\s+(?:you|students|your\s+schedule)\b",
            r"\bdrops?\s+(?:your\s+)?lowest\b",
            r"\bsecond\s+chance\b",
            r"\bresubmit\w*",
            r"\bgrace\s+period\b",
        ),
        neg_patterns=(
            r"\binflexible\b",
            r"!\bno\s+(?:late|extensions?|exceptions?)\b",
            r"!\bnot\s+(?:very\s+)?(?:flexible|understanding)\b",
            r"!\bstrict\s+(?:deadlines?|policy|policies)\b",
        ),
    ),
    Trait(
        "engagement",
        "Lectures are engaging, fun or inspiring.",
        pos_tags=("amazing lectures", "hilarious", "inspirational"),
        pos_patterns=(
            r"\b(?:engaging|entertaining|funny|interesting|passionate|enthusiastic|inspiring|inspirational)\b",
            r"\bfun\s+(?:class|professor|teacher|lectures?)\b",
        ),
        neg_patterns=(
            r"\b(?:boring|monotone|dull|uninspiring)\b",
            r"\bput\s+me\s+to\s+sleep\b",
        ),
    ),
    Trait(
        "lecture_heavy",
        "Mostly lecturing (high) versus hands-on work or live coding in class (low).",
        pos_tags=("lecture heavy",),
        pos_patterns=(
            r"\blecture[- ]heavy\b",
            r"\breads?\s+(?:off\s+)?(?:the\s+|from\s+the\s+)?(?:slides|powerpoints?|ppts?)\b",
        ),
        neg_patterns=(
            r"\bhands[- ]on\b",
            r"\blive\s+cod(?:e|ing)\b",
            r"\bin[- ]class\s+(?:work|coding|activities|projects?|labs?)\b",
        ),
    ),
    Trait(
        "exam_heavy",
        "Grade driven mainly by tests and quizzes.",
        pos_tags=("test heavy", "beware of pop quizzes"),
        pos_patterns=(r"\b(?:exam|test)[- ]heavy\b", r"\bpop\s+quizz?es\b"),
        neg_tags=("tests? not many",),
        neg_patterns=(r"!\bno\s+(?:exams?|tests?|quizz?es)\b",),
    ),
    Trait(
        "group_work",
        "Group projects and teamwork are part of the course.",
        pos_tags=("group projects",),
        pos_patterns=(
            r"\bgroup\s+(?:projects?|work|assignments?|activities)\b",
            r"\bteam\s+(?:projects?|work)\b",
            r"\bin\s+groups\b",
        ),
        neg_patterns=(r"!\bno\s+group\s+(?:projects?|work)\b",),
    ),
    Trait(
        "attendance_strict",
        "Attendance or participation counts toward the grade.",
        ratings=("attendance",),
        pos_tags=("skip class? you won't pass.", "participation matters"),
        pos_patterns=(
            r"\battendance\s+(?:is\s+)?(?:mandatory|required|matters|counts|graded|taken)\b",
            r"\bmandatory\s+attendance\b",
            r"\btakes?\s+attendance\b",
        ),
        neg_patterns=(
            r"!\battendance\s+(?:is\s+)?(?:not|optional)\b",
            r"!\b(?:doesn'?t|does\s+not|never)\s+(?:take|require|check)s?\s+attendance\b",
            r"!\bno\s+attendance\b",
            r"!\bcan\s+skip\b",
        ),
    ),
    Trait(
        "overall_rating",
        "Generally well liked: high overall rating and students would take the professor again.",
        ratings=("quality", "would_take_again"),
        pos_patterns=(
            r"\b(?:best|great|amazing|awesome|excellent|wonderful|fantastic)\s+(?:professor|prof|teacher|instructor)\b",
            r"\b(?:highly\s+)?recommend\w*",
            r"\bwould\s+take\s+(?:him|her|them|again)\b",
        ),
        neg_patterns=(
            r"\b(?:worst|terrible|awful|horrible)\s+(?:professor|prof|teacher|instructor)\b",
            r"!\b(?:do\s+not|don'?t|never)\s+take\b",
            r"!\bwould\s+not\s+(?:take|recommend)\b",
            r"!\b(?:do\s+not|don'?t|wouldn'?t|can'?t|cannot)\s+recommend\b",
            r"\bavoid\b",
        ),
    ),
)

TRAITS: dict[str, Trait] = {t.name: t for t in TRAIT_LIST}
TRAIT_NAMES: tuple[str, ...] = tuple(TRAITS)

# ----------------------------------------------------------------------------------------------
# Review analysis
# ----------------------------------------------------------------------------------------------
_NEGATION = re.compile(
    r"\b(?:not|no|never|isn'?t|wasn'?t|aren'?t|weren'?t|don'?t|doesn'?t|didn'?t|won'?t|can'?t|cannot|"
    r"without|hardly|barely)\b[^.!?;,]{0,20}$",
    re.I,
)
_SENT_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Hit:
    trait: str
    polarity: int     # +1 trait present, -1 trait absent/opposite
    source: str       # 'tag' | 'lexicon'
    text: str         # tag text, or the verbatim sentence containing the match


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _SENT_END.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def _sentence_at(text: str, pos: int) -> str:
    for s, e in _sentence_spans(text):
        if s <= pos <= e:
            return text[s:e].strip()
    return text.strip()


def normalize_tag(tag: str) -> str:
    return tag.strip().lower().replace("’", "'")


def split_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t for t in (normalize_tag(x) for x in raw.split("--")) if t]


def analyze_review(comment: str | None, tags: list[str]) -> list[Hit]:
    """Return every trait hit in one review (tags + lexicon). Pure function of the review."""
    hits: list[Hit] = []
    tagset = set(tags)
    for t in TRAIT_LIST:
        for tag in t.pos_tags:
            if tag in tagset:
                hits.append(Hit(t.name, +1, "tag", tag))
        for tag in t.neg_tags:
            if tag in tagset:
                hits.append(Hit(t.name, -1, "tag", tag))
    if comment:
        norm = comment.replace("’", "'")   # same length, so offsets line up with the original
        for t in TRAIT_LIST:
            pos, neg = t.compiled()
            for polarity, plist in ((+1, pos), (-1, neg)):
                for rx, check_neg in plist:
                    for m in rx.finditer(norm):
                        if check_neg and _NEGATION.search(norm[max(0, m.start() - 30):m.start()]):
                            continue
                        hits.append(Hit(t.name, polarity, "lexicon", _sentence_at(comment, m.start())))
                        break  # one hit per pattern per review is enough
    return hits
