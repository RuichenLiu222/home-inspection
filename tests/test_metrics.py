from inspection.metrics import evaluate_records


def test_required_metrics():
    records = [
        {
            "method": "structured",
            "ground_truth": "normal",
            "prediction": "normal",
            "raw_json_valid": True,
        },
        {
            "method": "structured",
            "ground_truth": "normal",
            "prediction": "floor_obstruction",
            "raw_json_valid": False,
        },
        {
            "method": "structured",
            "ground_truth": "countertop_clutter",
            "prediction": "countertop_clutter",
            "raw_json_valid": True,
        },
    ]
    metrics = evaluate_records(records)["structured"]
    assert metrics["accuracy"] == 0.6667
    assert metrics["false_positive_count"] == 1
    assert metrics["json_validity"] == 0.6667
