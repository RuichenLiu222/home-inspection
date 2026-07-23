DIRECT_PROMPT = (
    "Please inspect this kitchen image and determine whether there is any "
    "obvious problem. Briefly state your judgment and visible evidence."
)


CHECKLIST_PROMPT = """Inspect this kitchen image using the following location-based checklist.

First locate the visible objects, then choose one result:
- floor_obstruction: a concrete object is on the visible floor or walking path and blocks it.
  Never use this label for dishes or objects located only on a countertop.
- countertop_clutter: many dishes, containers, food items, or other objects visibly occupy
  and clutter the countertop or sink area.
- unsafe_object_placement: there is direct visual evidence that an object is dangerously
  placed, such as beside a flame, on an edge, or an electrical item beside water.
- normal: the image is clear enough and none of the three issues is visibly present.
- uncertain: the image is too blurry, severely occluded, or too incomplete to judge.

Do not guess hidden hazards and do not choose a category merely because it appears earlier
in the list. Return exactly one label and no other text:
floor_obstruction, countertop_clutter, unsafe_object_placement, normal, uncertain."""


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
The object must contain exactly these four keys:
{
  "result": "normal/attention/uncertain",
  "issue_type": "",
  "evidence": "",
  "suggestion": ""
}
For attention, issue_type must be exactly one of floor_obstruction, countertop_clutter,
or unsafe_object_placement. For normal or uncertain, issue_type must be empty.
Keep evidence and suggestion short and based only on visible content."""


def confirmation_prompt(issue_type: str, evidence: str) -> str:
    return f"""Act as a strict visual-evidence verifier.

The first inspection reported:
Issue type: {issue_type}
Evidence: {evidence}

Check the kitchen image again. Is there clear and directly visible evidence
supporting this exact issue?

Answer only one word: yes, no, or uncertain.
Use yes only when the evidence is explicit. Do not rely on assumptions."""
