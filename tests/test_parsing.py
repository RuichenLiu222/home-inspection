import json

from inspection.parsing import (
    parse_confirmation,
    parse_label,
    parse_structured,
    strict_json_object,
)


def test_strict_json_validity_rejects_markdown_fence():
    payload = {
        "result": "attention",
        "issue_type": "floor_obstruction",
        "evidence": "A box blocks the floor.",
        "suggestion": "Move the box.",
    }
    raw = json.dumps(payload)
    assert strict_json_object(raw)[1] is True
    assert strict_json_object(f"```json\n{raw}\n```")[1] is False


def test_recovered_json_keeps_raw_validity_false():
    raw = """```json
    {"result":"normal","issue_type":"","evidence":"Clear path.","suggestion":""}
    ```"""
    result, raw_valid, strategy = parse_structured(raw)
    assert result.label == "normal"
    assert raw_valid is False
    assert strategy == "recovered_json"


def test_label_and_confirmation_parsers():
    assert parse_label("countertop_clutter") == "countertop_clutter"
    assert parse_label("The image is too blurry to determine.") == "uncertain"
    assert parse_confirmation("Yes, the evidence is clear.") == "yes"
    assert parse_confirmation("I cannot decide") == "uncertain"
