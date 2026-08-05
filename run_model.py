from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from inspection.dataset import load_annotations, resolve_image, write_jsonl
from inspection.model import DEFAULT_MODEL_ID
from inspection.pipeline import METHODS, InspectionPipeline


def parse_methods(value: str) -> list[str]:
    required = ["direct", "checklist", "structured", "verified"]
    methods = required if value == "all" else [item.strip() for item in value.split(",")]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid methods: {METHODS}")
    return methods


def run(args: argparse.Namespace) -> list[dict]:
    project_root = Path(__file__).resolve().parent
    annotations = [
        item for item in load_annotations(args.annotations) if item.split == args.split
    ]
    if args.limit is not None:
        annotations = annotations[: args.limit]
    if not annotations:
        raise ValueError(f"No {args.split!r} annotations found in {args.annotations}")

    methods = parse_methods(args.methods)
    pipeline = InspectionPipeline(
        model_id=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    records: list[dict] = []

    for annotation in tqdm(annotations, desc="Inspecting images"):
        image_path = resolve_image(project_root, annotation)
        try:
            traces = []
            if "direct" in methods:
                traces.append(pipeline.inspect(image_path, "direct"))
            if "checklist" in methods or "verified" in methods:
                checklist = pipeline.inspect_checklist(image_path)
                if "checklist" in methods:
                    traces.append(checklist)
                if "verified" in methods:
                    traces.append(pipeline.verify(image_path, checklist))
            if "structured" in methods:
                traces.append(pipeline.inspect_structured(image_path))
            for trace in traces:
                records.append(
                    {
                        "image": image_path.relative_to(project_root).as_posix(),
                        "ground_truth": annotation.label,
                        **trace.to_dict(),
                    }
                )
            write_jsonl(args.output, records)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"Skipped {image_path}: {type(exc).__name__}: {exc}")

    print(f"Saved {len(records)} prediction records to {args.output}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SmolVLM prompt experiments")
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations.jsonl"))
    parser.add_argument("--split", choices=["debug", "test"], default="test")
    parser.add_argument("--methods", default="all", help="all or comma-separated method names")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/predictions.jsonl"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
