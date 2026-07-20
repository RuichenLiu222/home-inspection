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
    DECOMPOSED_PROMPTS,
    DIRECT_PROMPT,
    STRUCTURED_PROMPT,
    confirmation_prompt,
)
from .schemas import InspectionResult

METHODS = ("direct", "checklist", "structured", "verified", "decomposed")

ISSUE_PRIORITY = (
    "unsafe_object_placement",
    "floor_obstruction",
    "countertop_clutter",
)

SUGGESTIONS = {
    "floor_obstruction": "Remove the object from the walking area.",
    "countertop_clutter": "Clear and organize the countertop.",
    "unsafe_object_placement": "Move the object to a safer position.",
}


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
    component_outputs: dict[str, str] | None = None

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

        if method == "decomposed":
            return self.inspect_decomposed(image)

        structured = self.inspect_structured(image)
        return structured if method == "structured" else self.verify(image, structured)

    def inspect_decomposed(self, image: Image.Image | str | Path) -> InspectionTrace:
        outputs: dict[str, str] = {}
        decisions: dict[str, str] = {}
        total_latency = 0.0

        for issue_type, prompt in DECOMPOSED_PROMPTS.items():
            raw, latency = self.runner.generate(image, prompt, max_new_tokens=48)
            outputs[issue_type] = raw
            decisions[issue_type] = parse_confirmation(raw)
            total_latency += latency

        positive = [issue for issue in ISSUE_PRIORITY if decisions[issue] == "yes"]
        if not positive:
            all_uncertain = all(value == "uncertain" for value in decisions.values())
            result = InspectionResult(
                result="uncertain" if all_uncertain else "normal",
                evidence=(
                    "The three focused checks could not judge the image."
                    if all_uncertain
                    else "No listed issue was confirmed by the focused checks."
                ),
            )
            return InspectionTrace(
                method="decomposed",
                raw_output="",
                parsed=result,
                latency_seconds=total_latency,
                parse_strategy="decomposed_checks",
                component_outputs=outputs,
            )

        issue_type = positive[0]
        raw_evidence = outputs[issue_type]
        evidence = raw_evidence.split("|", 1)[1].strip() if "|" in raw_evidence else raw_evidence
        initial = InspectionResult(
            result="attention",
            issue_type=issue_type,
            evidence=evidence,
            suggestion=SUGGESTIONS[issue_type],
        )
        confirmation_raw, confirmation_latency = self.runner.generate(
            image,
            confirmation_prompt(issue_type, evidence),
            max_new_tokens=12,
        )
        confirmation = parse_confirmation(confirmation_raw)
        if confirmation == "no":
            final = InspectionResult(
                result="normal",
                evidence="The final verifier rejected the candidate issue.",
            )
        elif confirmation == "uncertain":
            final = InspectionResult(
                result="uncertain",
                evidence="The final verifier could not confirm the candidate issue.",
            )
        else:
            final = initial

        return InspectionTrace(
            method="decomposed",
            raw_output=raw_evidence,
            parsed=final,
            latency_seconds=total_latency,
            raw_json_valid=None,
            parse_strategy="decomposed_checks",
            confirmation_output=confirmation_raw,
            confirmation_decision=confirmation,
            confirmation_latency_seconds=confirmation_latency,
            component_outputs=outputs,
        )

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
