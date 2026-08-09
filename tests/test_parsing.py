import json

from inspection.parsing import (
    extract_loose_mapping,
    parse_confirmation,
    parse_label,
    parse_quality,
    parse_region_response,
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
    assert parse_label("No visible problem.") == "normal"
    assert parse_label("No visible issues.") == "normal"
    assert parse_label("None.") == "normal"
    assert (
        parse_label("Evidence: dishes cover the sink.\nDecision: countertop_clutter")
        == "countertop_clutter"
    )
    assert parse_label("Unsafe object placement.") == "unsafe_object_placement"
    assert parse_confirmation("Yes, the evidence is clear.") == "yes"
    assert parse_confirmation("I cannot decide") == "uncertain"


def test_explicit_decision_wins_without_position_bias():
    raw = (
        "Options: floor_obstruction, countertop_clutter, unsafe_object_placement.\n"
        "Evidence: dishes cover the sink.\nDecision: countertop_clutter"
    )
    assert parse_label(raw) == "countertop_clutter"


def test_multiple_labels_without_decision_are_uncertain():
    raw = "Possible labels are floor_obstruction and countertop_clutter."
    assert parse_label(raw) == "uncertain"


def test_negated_label_does_not_become_positive():
    assert parse_label("No visible floor obstruction or other obvious problem.") == "normal"


def test_loose_structured_fields_are_recovered_without_claiming_valid_json():
    raw = (
        "result: attention, issue_type: countertop_clutter\n"
        "evidence: dishes cover the counter\nsuggestion: clear the counter"
    )
    mapping = extract_loose_mapping(raw)
    assert mapping is not None
    assert mapping["issue_type"] == "countertop_clutter"
    result, raw_valid, strategy = parse_structured(raw)
    assert result.label == "countertop_clutter"
    assert raw_valid is False
    assert strategy == "recovered_fields"


def test_conflicting_confirmation_words_are_uncertain():
    assert parse_confirmation("yes or no") == "uncertain"


def test_region_response_parser_requires_a_leading_decision():
    assert parse_region_response("YES | plate | center of walking path") == (
        "yes",
        "plate — center of walking path",
    )
    assert parse_region_response("NO") == ("no", "")
    assert parse_region_response("The answer is YES") == ("uncertain", "")
    assert parse_region_response("YES") == ("uncertain", "")


def test_quality_parser_is_conservative():
    assert parse_quality("CLEAR") == "clear"
    assert parse_quality("UNCERTAIN") == "uncertain"
    assert parse_quality("clear or uncertain") == "uncertain"
