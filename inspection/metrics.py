from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from .schemas import ISSUE_TYPES, LABELS


def evaluate_records(records: Iterable[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["method"])].append(record)

    output: dict[str, dict] = {}
    for method, items in grouped.items():
        total = len(items)
        correct = sum(item["ground_truth"] == item["prediction"] for item in items)
        false_positives = sum(
            item["ground_truth"] == "normal" and item["prediction"] in ISSUE_TYPES
            for item in items
        )
        json_items = [item for item in items if item.get("raw_json_valid") is not None]
        json_valid = sum(bool(item.get("raw_json_valid")) for item in json_items)

        confusion = Counter(
            (item["ground_truth"], item["prediction"])
            for item in items
        )
        output[method] = {
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "false_positive_count": false_positives,
            "json_valid_count": json_valid if json_items else None,
            "json_total": len(json_items) if json_items else None,
            "json_validity": round(json_valid / len(json_items), 4) if json_items else None,
            "label_counts": dict(Counter(item["prediction"] for item in items)),
            "confusion": {
                truth: {
                    prediction: confusion.get((truth, prediction), 0)
                    for prediction in LABELS
                }
                for truth in LABELS
            },
        }
    return output
