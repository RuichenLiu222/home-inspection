from app import _demo_method, _with_demo_suggestion
from inspection.schemas import InspectionResult


def test_demo_checkbox_enables_required_verification_method():
    assert _demo_method(False) == "structured"
    assert _demo_method(True) == "verified"


def test_demo_adds_category_suggestion_only_when_missing():
    issue = InspectionResult("attention", "floor_obstruction", "A plate is on the floor.")
    filled = _with_demo_suggestion(issue)
    assert "通道" in filled.suggestion

    supplied = InspectionResult(
        "attention",
        "countertop_clutter",
        "Dishes cover the counter.",
        "Clear the dishes.",
    )
    assert _with_demo_suggestion(supplied) == supplied

    normal = InspectionResult("normal", evidence="No listed issue.")
    assert _with_demo_suggestion(normal) == normal
