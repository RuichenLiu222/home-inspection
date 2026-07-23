DIRECT_PROMPT = (
    "Please inspect this kitchen image and determine whether there is any "
    "obvious problem. Briefly state your judgment and visible evidence."
)


CHECKLIST_PROMPT = """Inspect this kitchen image using this location-based checklist.

Compare all five outcomes before deciding:
- floor_obstruction: a concrete object is on the visible floor or walking path and blocks it.
  Never use this label for dishes or objects located only on a countertop.
- countertop_clutter: many dishes, containers, food items, or other objects visibly occupy
  and clutter the countertop or sink area.
- unsafe_object_placement: there is direct visual evidence that an object is dangerously
  placed, such as beside a flame, on an edge, or an electrical item beside water.
- normal: the image is clear enough and none of the three issues is visibly present.
- uncertain: the image is too blurry, severely occluded, or too incomplete to judge.

Do not guess hidden hazards. Do not repeat the checklist. Reply in exactly two short lines:
Evidence: name the visible object and its location, or state that no issue is visible.
Decision: write exactly one of floor_obstruction, countertop_clutter,
unsafe_object_placement, normal, or uncertain."""


STRUCTURED_PROMPT = """You are a kitchen safety inspector. Ignore logos and decorative text.
Locate the visible objects before deciding:
- floor_obstruction only when a concrete object is on and blocks the visible floor/path;
- countertop_clutter when many objects visibly clutter the countertop or sink;
- unsafe_object_placement only with directly visible danger such as flame, edge, or
  electricity beside water;
- normal when the image is clear and no listed issue is visible;
- uncertain only when blur, occlusion, or an incomplete view prevents judgment.

Reply with one JSON object. Never reply with a label or sentence by itself. Do not use
Markdown. Your first character must be { and your last character must be }.
Use exactly four keys: result, issue_type, evidence, suggestion.
Important: result must be one literal value, not a slash-separated list.
Use this shape, replacing every angle-bracket placeholder:
{
  "result": "<normal, attention, or uncertain>",
  "issue_type": "<one issue label, or empty>",
  "evidence": "<short visible evidence>",
  "suggestion": "<short action, or empty>"
}
For attention, issue_type must be exactly one of floor_obstruction, countertop_clutter,
or unsafe_object_placement. For normal or uncertain, issue_type must be empty.
Keep evidence and suggestion short and based only on visible content."""


def confirmation_prompt(issue_type: str, evidence: str) -> str:
    required_evidence = {
        "floor_obstruction": (
            "a named concrete object must be visibly located on and blocking the floor/path"
        ),
        "countertop_clutter": (
            "multiple named objects must visibly occupy and clutter the countertop/sink"
        ),
        "unsafe_object_placement": (
            "a named object and a directly visible dangerous location must both be present"
        ),
    }.get(issue_type, "the exact reported issue must be directly visible")
    return f"""Act as a skeptical visual-evidence verifier.

The first inspection reported:
Issue type: {issue_type}
Evidence: {evidence}

Required visual condition: {required_evidence}.

Check the image independently. Answer no when the condition is absent or the evidence merely
repeats the category. Answer uncertain when the relevant region cannot be seen.
Answer only one word: yes, no, or uncertain."""
