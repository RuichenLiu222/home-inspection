from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

ISSUE_TYPES = (
    "floor_obstruction",
    "countertop_clutter",
    "unsafe_object_placement",
)
RESULT_TYPES = ("normal", "attention", "uncertain")
LABELS = (*ISSUE_TYPES, "normal", "uncertain")


@dataclass(frozen=True)
class InspectionResult:
    result: str
    issue_type: str = ""
    evidence: str = ""
    suggestion: str = ""

    def __post_init__(self) -> None:
        if self.result not in RESULT_TYPES:
            raise ValueError(f"Unsupported result: {self.result}")
        if self.issue_type and self.issue_type not in ISSUE_TYPES:
            raise ValueError(f"Unsupported issue type: {self.issue_type}")
        if self.result == "attention" and self.issue_type not in ISSUE_TYPES:
            raise ValueError("An attention result must contain a supported issue type")
        if self.result != "attention" and self.issue_type:
            raise ValueError("normal/uncertain results must not contain an issue type")

    @property
    def label(self) -> str:
        return self.issue_type if self.result == "attention" else self.result

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> InspectionResult:
        result = str(value.get("result", "")).strip().lower()
        issue_type = str(value.get("issue_type", "")).strip().lower()
        evidence = str(value.get("evidence", "")).strip()
        suggestion = str(value.get("suggestion", "")).strip()

        if result in ISSUE_TYPES and not issue_type:
            issue_type, result = result, "attention"
        if result in {"issue", "problem", "warning", "unsafe"}:
            result = "attention"
        if result in {"unknown", "cannot_determine", "cannot determine"}:
            result = "uncertain"
        if result in {"safe", "none", "no_issue", "no issue"}:
            result = "normal"

        if result != "attention":
            issue_type = ""

        return cls(
            result=result,
            issue_type=issue_type,
            evidence=evidence,
            suggestion=suggestion,
        )
