from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from PIL import Image

from .model import DEFAULT_MODEL_ID, SmolVLMRunner
from .parsing import (
    parse_confirmation,
    parse_label,
    parse_structured,
    result_from_label,
)
from .prompts import (
    CHECKLIST_PROMPT,
    DIRECT_PROMPT,
    STRUCTURED_PROMPT,
    confirmation_prompt,
)
from .schemas import InspectionResult

METHODS = ("direct", "checklist", "structured", "verified")


@dataclass(frozen=True)
class InspectionTrace:
    method: str
    raw_output: str
    parsed: InspectionResult
    latency_seconds: float
    raw_json_valid: bool | None = None
    parse_strategy: str = ""
    confirmation_output: str = ""
    confirmation_decision: str = ""
    confirmation_latency_seconds: float = 0.0

    def to_dict(self) -> dict:
        value = asdict(self)
        value["parsed"] = self.parsed.to_dict()
        value["prediction"] = self.parsed.label
        return value


class InspectionPipeline:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        max_new_tokens: int = 160,
        runner: SmolVLMRunner | None = None,
    ) -> None:
        self.runner = runner or SmolVLMRunner(
            model_id=model_id,
            device=device,
            max_new_tokens=max_new_tokens,
        )

    def inspect(self, image: Image.Image | str | Path, method: str) -> InspectionTrace:
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from {METHODS}")

        if method == "direct":
            raw, latency = self.runner.generate(image, DIRECT_PROMPT)
            parsed = result_from_label(parse_label(raw), raw)
            return InspectionTrace(method, raw, parsed, latency, parse_strategy="label_heuristic")

        if method == "checklist":
            raw, latency = self.runner.generate(image, CHECKLIST_PROMPT, max_new_tokens=32)
            parsed = result_from_label(parse_label(raw), raw)
            return InspectionTrace(method, raw, parsed, latency, parse_strategy="label_parser")

        structured = self.inspect_structured(image)
        return structured if method == "structured" else self.verify(image, structured)

    def inspect_structured(self, image: Image.Image | str | Path) -> InspectionTrace:
        raw, latency = self.runner.generate(image, STRUCTURED_PROMPT)
        parsed, raw_valid, strategy = parse_structured(raw)
        return InspectionTrace(
            method="structured",
            raw_output=raw,
            parsed=parsed,
            latency_seconds=latency,
            raw_json_valid=raw_valid,
            parse_strategy=strategy,
        )

    def verify(
        self,
        image: Image.Image | str | Path,
        initial: InspectionTrace,
    ) -> InspectionTrace:
        if initial.parsed.result != "attention":
            return replace(initial, method="verified")

        raw, latency = self.runner.generate(
            image,
            confirmation_prompt(initial.parsed.issue_type, initial.parsed.evidence),
            max_new_tokens=12,
        )
        decision = parse_confirmation(raw)
        if decision == "yes":
            final = initial.parsed
        elif decision == "no":
            final = InspectionResult(
                result="normal",
                evidence="The second-pass verifier found no clear supporting evidence.",
            )
        else:
            final = InspectionResult(
                result="uncertain",
                evidence="The second-pass verifier could not confirm the initial issue.",
            )

        return replace(
            initial,
            method="verified",
            parsed=final,
            confirmation_output=raw,
            confirmation_decision=decision,
            confirmation_latency_seconds=latency,
        )
