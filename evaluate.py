from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from inspection.metrics import evaluate_records


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evaluate(predictions: Path, output: Path) -> dict:
    metrics = evaluate_records(load_jsonl(predictions))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "total",
                "correct",
                "accuracy",
                "false_positive_count",
                "json_valid_count",
                "json_total",
                "json_validity",
            ],
        )
        writer.writeheader()
        for method, values in metrics.items():
            metric_values = {key: values.get(key) for key in writer.fieldnames[1:]}
            writer.writerow({"method": method, **metric_values})

    print(f"Metrics JSON: {output}")
    print(f"Summary CSV: {csv_path}")
    for method, values in metrics.items():
        print(
            f"{method:10s} accuracy={values['accuracy']:.4f} "
            f"false_positives={values['false_positive_count']} "
            f"json_validity={values['json_validity']}"
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate kitchen-inspection predictions")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/predictions.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/metrics.json"))
    args = parser.parse_args()
    evaluate(args.predictions, args.output)


if __name__ == "__main__":
    main()
