from __future__ import annotations

import json
import re
from typing import Any

from .schemas import ISSUE_TYPES, InspectionResult

UNCERTAIN_PATTERNS = (
    "uncertain",
    "cannot determine",
    "can't determine",
    "unable to determine",
    "too blurry",
    "not clear enough",
    "insufficient",
)
NORMAL_PATTERNS = (
    "normal",
    "no obvious issue",
    "no obvious problem",
    "no clear issue",
    "no visible issue",
    "no visible issues",
    "no visible problem",
    "no visible problems",
    "appears safe",
    "looks safe",
)


def strict_json_object(raw: str) -> tuple[dict[str, Any] | None, bool]:
    """Parse an unmodified model response and report raw JSON validity."""
    try:
        value = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError):
        return None, False
    return (value, True) if isinstance(value, dict) else (None, False)


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """Best-effort parser used by the demo, not by the JSON-validity metric."""
    strict, valid = strict_json_object(raw)
    if valid:
        return strict

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_label(raw: str) -> str:
    text = raw.strip().lower()
    exact = text.strip(" .`'\"\n\t")
    if exact in (*ISSUE_TYPES, "normal", "uncertain"):
        return exact

    for issue_type in ISSUE_TYPES:
        if issue_type in text:
            return issue_type
    if "countertop" in text and any(word in text for word in ("clutter", "mess", "crowded")):
        return "countertop_clutter"
    if any(word in text for word in ("floor", "walkway", "walking path")) and any(
        word in text for word in ("obstruct", "block", "clutter", "object")
    ):
        return "floor_obstruction"
    unsafe_phrases = ("unsafe placement", "placed unsafely", "dangerous position")
    if any(phrase in text for phrase in unsafe_phrases):
        return "unsafe_object_placement"
    if any(pattern in text for pattern in UNCERTAIN_PATTERNS):
        return "uncertain"
    if any(pattern in text for pattern in NORMAL_PATTERNS):
        return "normal"
    return "uncertain"


def result_from_label(label: str, raw: str = "") -> InspectionResult:
    if label in ISSUE_TYPES:
        return InspectionResult("attention", label, evidence=raw.strip())
    if label == "normal":
        return InspectionResult("normal", evidence=raw.strip())
    return InspectionResult("uncertain", evidence=raw.strip())


def parse_structured(raw: str) -> tuple[InspectionResult, bool, str]:
    """Return normalized result, raw JSON validity, and parse strategy."""
    strict, raw_valid = strict_json_object(raw)
    value = strict if raw_valid else extract_json_object(raw)
    if value is not None:
        try:
            result = InspectionResult.from_mapping(value)
            return result, raw_valid, "strict_json" if raw_valid else "recovered_json"
        except (TypeError, ValueError):
            pass

    label = parse_label(raw)
    return result_from_label(label, raw), raw_valid, "label_fallback"


def parse_confirmation(raw: str) -> str:
    match = re.search(r"\b(yes|no|uncertain)\b", raw.strip().lower())
    return match.group(1) if match else "uncertain"
