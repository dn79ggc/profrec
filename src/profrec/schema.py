"""Data shapes for a parsed student request.

ParsedRequest is the ONLY thing the language model is allowed to produce, and it is a closed
vocabulary: trait names come from criteria.TRAIT_NAMES, directions and days are enums, and the
free-text fields (keywords, unsupported) are sanitized. The model never sees database content
and never writes an answer.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .criteria import TRAIT_NAMES

Day = Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TraitName = Literal[TRAIT_NAMES]  # type: ignore[valid-type]
_HHMM = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
MAX_CRITERIA = 8
MAX_KEYWORDS = 6


class Criterion(BaseModel):
    name: TraitName
    direction: Literal["prefer", "avoid"]
    weight: float = 1.0

    @field_validator("weight", mode="before")
    @classmethod
    def _clamp(cls, v):
        try:
            return min(1.0, max(0.0, float(v)))
        except (TypeError, ValueError):
            return 1.0


class ScheduleConstraints(BaseModel):
    earliest_start: str | None = None      # 'HH:MM' 24h; sections must start at or after this
    latest_end: str | None = None          # 'HH:MM' 24h; sections must end at or before this
    avoid_days: list[Day] = Field(default_factory=list)
    delivery: Literal["any", "in_person", "online"] = "any"

    @field_validator("earliest_start", "latest_end", mode="before")
    @classmethod
    def _time(cls, v):
        if v is None or v == "":
            return None
        m = _HHMM.match(str(v).strip())
        return f"{int(m.group(1)):02d}:{m.group(2)}" if m else None

    def is_empty(self) -> bool:
        return not (self.earliest_start or self.latest_end or self.avoid_days or self.delivery != "any")


class ParsedRequest(BaseModel):
    criteria: list[Criterion] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)       # words to look for in review text
    constraints: ScheduleConstraints = Field(default_factory=ScheduleConstraints)
    unsupported: list[str] = Field(default_factory=list)    # parts of the request the data cannot answer
    parsed_by: str = ""                                     # which parser produced this (shown in the UI)
    original_text: str = ""

    @field_validator("criteria")
    @classmethod
    def _dedupe(cls, v):
        best: dict[tuple[str, str], Criterion] = {}
        for c in v:
            key = (c.name, c.direction)
            if key not in best or c.weight > best[key].weight:
                best[key] = c
        # if both prefer and avoid were given for one trait, keep the heavier one
        by_trait: dict[str, Criterion] = {}
        for c in best.values():
            if c.name not in by_trait or c.weight > by_trait[c.name].weight:
                by_trait[c.name] = c
        return list(by_trait.values())[:MAX_CRITERIA]

    @field_validator("keywords")
    @classmethod
    def _keywords(cls, v):
        out: list[str] = []
        for k in v:
            k = re.sub(r"[^a-z0-9' -]", " ", str(k).lower())
            k = re.sub(r"\s+", " ", k).strip()
            if k and len(k.split()) <= 3 and len(k) <= 30 and k not in out:
                out.append(k)
        return out[:MAX_KEYWORDS]

    @field_validator("unsupported")
    @classmethod
    def _unsupported(cls, v):
        return [str(u).strip()[:120] for u in v if str(u).strip()][:5]

    def has_preferences(self) -> bool:
        return bool(self.criteria or self.keywords)


# ----------------------------------------------------------------------------------------------
# The flat schema actually sent to the model (kept simple on purpose: no $ref, no nulls, no anyOf).
# ----------------------------------------------------------------------------------------------
LLM_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": list(TRAIT_NAMES)},
                    "direction": {"type": "string", "enum": ["prefer", "avoid"]},
                    "weight": {"type": "number"},
                },
                "required": ["name", "direction", "weight"],
            },
        },
        "keywords": {"type": "array", "items": {"type": "string"}},
        "earliest_start": {"type": "string"},
        "latest_end": {"type": "string"},
        "avoid_days": {"type": "array", "items": {"type": "string", "enum": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]}},
        "delivery": {"type": "string", "enum": ["any", "in_person", "online"]},
        "unsupported": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["criteria", "keywords", "earliest_start", "latest_end", "avoid_days", "delivery", "unsupported"],
}


def from_llm_dict(data: dict, original_text: str, parsed_by: str) -> ParsedRequest:
    """Validate the model's flat JSON into a ParsedRequest. Raises pydantic.ValidationError if invalid."""
    return ParsedRequest(
        criteria=data.get("criteria", []),
        keywords=data.get("keywords", []),
        constraints=ScheduleConstraints(
            earliest_start=data.get("earliest_start") or None,
            latest_end=data.get("latest_end") or None,
            avoid_days=data.get("avoid_days", []),
            delivery=data.get("delivery") or "any",
        ),
        unsupported=data.get("unsupported", []),
        parsed_by=parsed_by,
        original_text=original_text,
    )
