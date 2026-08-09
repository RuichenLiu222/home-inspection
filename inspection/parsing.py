from __future__ import annotations

import json
import re
from typing import Any

from .schemas import ISSUE_TYPES, InspectionResult

LABEL_VALUE_PATTERN = (
    r"floor[_ ]obstruction|countertop[_ ]clutter|"
    r"unsafe[_ ]object[_ ]placement|normal|uncertain"
)

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
    "none",
    "no obvious issue",
    "no obvious problem",
    "no clear issue",
    "no visible issue",
    "no visible issues",
    "no visible problem",
    "no visible problems",
    "no listed issue",
    "no floor obstruction",
    "no visible floor obstruction",
    "no countertop clutter",
    "no unsafe object placement",
    "appears safe",
    "looks safe",
)


def strict_json_object(raw: str) -> tuple[dict[str, Any] | None, bool]:
    try:
        value = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError):
        return None, False
    return (value, True) if isinstance(value, dict) else (None, False)


def extract_json_object(raw: str) -> dict[str, Any] | None:
    # JSON validity is recorded before attempting recovery.
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


def extract_loose_mapping(raw: str) -> dict[str, str] | None:
    fields: dict[str, str] = {}
    pattern = re.compile(
        r"[\"']?(result|issue_type|evidence|suggestion)[\"']?\s*[:=]\s*"
        r"(?:[\"']([^\"']*)[\"']|([^,\n}\r]+))",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(raw):
        key = match.group(1).lower()
        value = (match.group(2) or match.group(3) or "").strip(" .\t")
        fields[key] = value

    if not fields:
        return None
    if not fields.get("result") and fields.get("issue_type"):
        fields["result"] = "attention"
    return fields


def _field_value(text: str, field_names: str, values: str) -> str | None:
    match = re.search(
        rf"[\"']?(?:{field_names})[\"']?\s*[:=]\s*[\"']?({values})\b",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).lower().replace(" ", "_") if match else None


def _unnegated_label_mentions(text: str) -> set[str]:
    mentions: set[str] = set()
    for issue_type in ISSUE_TYPES:
        phrase = issue_type.replace("_", r"[_ ]")
        for match in re.finditer(rf"\b{phrase}\b", text):
            prefix = text[max(0, match.start() - 28) : match.start()]
            if not re.search(r"\b(no|not|without|absent)\b[^.!?;]*$", prefix):
                mentions.add(issue_type)
    return mentions


def parse_label(raw: str) -> str:
    text = raw.strip().lower()
    exact = text.strip(" .`'\"\n\t")
    if exact in (*ISSUE_TYPES, "normal", "uncertain"):
        return exact

    decision = _field_value(text, "decision|label", LABEL_VALUE_PATTERN)
    if decision:
        return decision

    result = _field_value(
        text,
        "result",
        rf"{LABEL_VALUE_PATTERN}|attention",
    )
    if result in (*ISSUE_TYPES, "normal", "uncertain"):
        return result

    issue_type = _field_value(text, "issue_type", LABEL_VALUE_PATTERN)
    if result == "attention" and issue_type in ISSUE_TYPES:
        return issue_type
    if result is None and issue_type in ISSUE_TYPES:
        return issue_type

    assessments: dict[str, str] = {}
    for candidate in ISSUE_TYPES:
        phrase = candidate.replace("_", r"[_ ]")
        match = re.search(
            rf"\b{phrase}\b\s*[:=\-]\s*(yes|no|uncertain)\b",
            text,
        )
        if match:
            assessments[candidate] = match.group(1)
    positives = [label for label, decision_value in assessments.items() if decision_value == "yes"]
    if len(positives) == 1:
        return positives[0]
    if len(positives) > 1:
        return "uncertain"
    if len(assessments) == len(ISSUE_TYPES) and all(
        decision_value == "no" for decision_value in assessments.values()
    ):
        return "normal"

    if any(pattern in text for pattern in UNCERTAIN_PATTERNS):
        return "uncertain"

    mentions = _unnegated_label_mentions(text)
    if any(pattern in text for pattern in NORMAL_PATTERNS) and not mentions:
        return "normal"
    if len(mentions) == 1:
        return next(iter(mentions))
    if len(mentions) > 1:
        return "uncertain"

    inferred: set[str] = set()
    if "countertop" in text and any(word in text for word in ("clutter", "mess", "crowded")):
        inferred.add("countertop_clutter")
    floor_signal = any(word in text for word in ("floor", "walkway", "walking path")) and any(
        word in text for word in ("obstruct", "block", "clutter", "object")
    )
    floor_negated = re.search(
        r"\b(no|not|without)\b[^.!?;]*(floor|walkway|walking path)",
        text,
    )
    if floor_signal and not floor_negated:
        inferred.add("floor_obstruction")
    unsafe_phrases = ("unsafe placement", "placed unsafely", "dangerous position")
    if any(phrase in text for phrase in unsafe_phrases):
        inferred.add("unsafe_object_placement")
    if len(inferred) == 1:
        return next(iter(inferred))
    if len(inferred) > 1:
        return "uncertain"
    if exact == "floor":
        return "floor_obstruction"
    return "uncertain"


def result_from_label(label: str, raw: str = "") -> InspectionResult:
    if label in ISSUE_TYPES:
        return InspectionResult("attention", label, evidence=raw.strip())
    if label == "normal":
        return InspectionResult("normal", evidence=raw.strip())
    return InspectionResult("uncertain", evidence=raw.strip())


def parse_structured(raw: str) -> tuple[InspectionResult, bool, str]:
    strict, raw_valid = strict_json_object(raw)
    value = strict if raw_valid else extract_json_object(raw)
    if value is not None:
        try:
            result = InspectionResult.from_mapping(value)
            return result, raw_valid, "strict_json" if raw_valid else "recovered_json"
        except (TypeError, ValueError):
            pass

    loose = extract_loose_mapping(raw)
    if loose is not None:
        try:
            return InspectionResult.from_mapping(loose), raw_valid, "recovered_fields"
        except (TypeError, ValueError):
            pass

    label = parse_label(raw)
    return result_from_label(label, raw), raw_valid, "label_fallback"


def parse_confirmation(raw: str) -> str:
    exact = raw.strip().lower().strip(" .!`'\"")
    if exact in {"yes", "no", "uncertain"}:
        return exact
    decisions = set(re.findall(r"\b(yes|no|uncertain)\b", raw.strip().lower()))
    return next(iter(decisions)) if len(decisions) == 1 else "uncertain"


def parse_quality(raw: str) -> str:
    exact = raw.strip().lower().strip(" .!`'\"")
    if exact in {"clear", "uncertain"}:
        return exact
    decisions = set(re.findall(r"\b(clear|uncertain)\b", raw.lower()))
    return next(iter(decisions)) if len(decisions) == 1 else "uncertain"


def parse_region_response(raw: str) -> tuple[str, str]:
    text = raw.strip()
    # Without a leading decision token, the response is not forced into a label.
    match = re.match(r"^(yes|no|uncertain)\b", text, flags=re.IGNORECASE)
    if match is None:
        return "uncertain", ""

    decision = match.group(1).lower()
    if decision != "yes":
        return decision, ""

    parts = [part.strip(" .\t") for part in text.split("|")]
    if len(parts) >= 3 and parts[1] and parts[2]:
        return "yes", f"{parts[1]} — {parts[2]}"

    remainder = text[match.end() :].strip(" :|-.")
    return ("yes", remainder) if remainder else ("uncertain", "")
