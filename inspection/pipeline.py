from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from PIL import Image

from .model import DEFAULT_MODEL_ID, SmolVLMRunner
from .parsing import (
    parse_confirmation,
    parse_label,
    parse_quality,
    parse_region_response,
    parse_structured,
    result_from_label,
)
from .prompts import (
    CHECKLIST_PROMPT,
    DIRECT_PROMPT,
    ISSUE_PRIORITY,
    ISSUE_SUGGESTIONS,
    QUALITY_PROMPT,
    REGION_PROMPTS,
    REGION_VERIFICATION_PROMPTS,
    STRUCTURED_PROMPT,
    confirmation_prompt,
)
from .schemas import InspectionResult

BASE_METHODS = ("direct", "checklist", "structured", "verified")
V2_METHODS = ("region_v2", "region_v2_verified")
METHODS = (*BASE_METHODS, *V2_METHODS)


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
    component_outputs: dict[str, str] = field(default_factory=dict)
    component_decisions: dict[str, str] = field(default_factory=dict)

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
            return self.inspect_checklist(image)

        if method == "verified":
            return self.verify(image, self.inspect_checklist(image))

        if method == "region_v2":
            return self.inspect_regions(image)

        if method == "region_v2_verified":
            return self.verify_regions(image, self.inspect_regions(image))

        return self.inspect_structured(image)

    @staticmethod
    def _region_result(issue_type: str, evidence: str) -> InspectionResult:
        return InspectionResult(
            result="attention",
            issue_type=issue_type,
            evidence=evidence or "A visible object satisfies the region check.",
            suggestion=ISSUE_SUGGESTIONS[issue_type],
        )

    def inspect_regions(self, image: Image.Image | str | Path) -> InspectionTrace:
        outputs: dict[str, str] = {}
        decisions: dict[str, str] = {}
        evidence: dict[str, str] = {}
        total_latency = 0.0

        quality_raw, quality_latency = self.runner.generate(
            image, QUALITY_PROMPT, max_new_tokens=8
        )
        outputs["quality"] = quality_raw
        decisions["quality"] = parse_quality(quality_raw)
        total_latency += quality_latency
        if decisions["quality"] == "uncertain":
            return InspectionTrace(
                method="region_v2",
                raw_output=quality_raw,
                parsed=InspectionResult(
                    result="uncertain",
                    evidence="The image is not clear enough for reliable inspection.",
                ),
                latency_seconds=total_latency,
                parse_strategy="region_aggregation",
                component_outputs=outputs,
                component_decisions=decisions,
            )

        for issue_type, prompt in REGION_PROMPTS.items():
            raw, latency = self.runner.generate(image, prompt, max_new_tokens=32)
            decision, visible_evidence = parse_region_response(raw)
            outputs[issue_type] = raw
            decisions[issue_type] = decision
            evidence[issue_type] = visible_evidence
            total_latency += latency

        positive = [
            issue_type
            for issue_type in ISSUE_PRIORITY
            if decisions.get(issue_type) == "yes"
        ]
        if positive:
            issue_type = positive[0]
            result = self._region_result(issue_type, evidence[issue_type])
        elif any(decisions.get(key) == "uncertain" for key in REGION_PROMPTS):
            result = InspectionResult(
                result="uncertain",
                evidence="At least one relevant region could not be judged reliably.",
            )
        else:
            result = InspectionResult(
                result="normal",
                evidence="No listed issue was supported by the region checks.",
            )

        raw_output = "\n".join(f"{key}: {value}" for key, value in outputs.items())
        return InspectionTrace(
            method="region_v2",
            raw_output=raw_output,
            parsed=result,
            latency_seconds=total_latency,
            parse_strategy="region_aggregation",
            component_outputs=outputs,
            component_decisions=decisions,
        )

    def verify_regions(
        self,
        image: Image.Image | str | Path,
        initial: InspectionTrace,
    ) -> InspectionTrace:
        candidates = [
            issue_type
            for issue_type in ISSUE_PRIORITY
            if initial.component_decisions.get(issue_type) == "yes"
        ]
        if not candidates:
            return replace(initial, method="region_v2_verified")

        verification_outputs: list[str] = []
        saw_uncertain = False
        total_latency = 0.0
        final_decision = "no"

        for issue_type in candidates:
            raw, latency = self.runner.generate(
                image,
                REGION_VERIFICATION_PROMPTS[issue_type],
                max_new_tokens=8,
            )
            decision = parse_confirmation(raw)
            verification_outputs.append(f"{issue_type}: {raw}")
            total_latency += latency
            if decision == "yes":
                _, visible_evidence = parse_region_response(
                    initial.component_outputs[issue_type]
                )
                return replace(
                    initial,
                    method="region_v2_verified",
                    parsed=self._region_result(issue_type, visible_evidence),
                    confirmation_output="\n".join(verification_outputs),
                    confirmation_decision="yes",
                    confirmation_latency_seconds=total_latency,
                )
            if decision == "uncertain":
                saw_uncertain = True

        if saw_uncertain:
            final_decision = "uncertain"
            result = InspectionResult(
                result="uncertain",
                evidence="The independent verifier could not confirm the candidate issue.",
            )
        else:
            result = InspectionResult(
                result="normal",
                evidence="The independent verifier rejected all candidate issues.",
            )

        return replace(
            initial,
            method="region_v2_verified",
            parsed=result,
            confirmation_output="\n".join(verification_outputs),
            confirmation_decision=final_decision,
            confirmation_latency_seconds=total_latency,
        )

    def inspect_checklist(self, image: Image.Image | str | Path) -> InspectionTrace:
        # Prefixing this response caused repeated labels in debug runs, so the
        # checklist starts from an empty answer.
        raw, latency = self.runner.generate(image, CHECKLIST_PROMPT, max_new_tokens=64)
        parsed = result_from_label(parse_label(raw), raw)
        return InspectionTrace(
            method="checklist",
            raw_output=raw,
            parsed=parsed,
            latency_seconds=latency,
            parse_strategy="label_parser",
        )

    def inspect_structured(self, image: Image.Image | str | Path) -> InspectionTrace:
        raw, latency = self.runner.generate(
            image,
            STRUCTURED_PROMPT,
            assistant_prefix='{"result":"',
        )
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
