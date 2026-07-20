DIRECT_PROMPT = (
    "Please inspect this kitchen image and determine whether there is any "
    "obvious problem. Briefly state your judgment and visible evidence."
)


CHECKLIST_PROMPT = """Inspect this kitchen image for exactly one of the following visible issues:

1. floor_obstruction: objects clearly obstruct the floor or walking path;
2. countertop_clutter: the countertop is visibly and excessively cluttered;
3. unsafe_object_placement: an object is placed in a visibly unsafe or unreasonable position.

If no obvious issue is visible, answer normal.
If the image is too blurry, seriously occluded, or insufficient for judgment, answer uncertain.
Only use evidence directly visible in the image.
Return exactly one label from this list and no other text:
floor_obstruction, countertop_clutter, unsafe_object_placement, normal, uncertain."""


STRUCTURED_PROMPT = """You are a kitchen safety inspector.
Ignore logos, watermarks, and decorative text.
Check only these visible issues:
- floor_obstruction: objects blocking the floor or walking path
- countertop_clutter: an obviously cluttered countertop
- unsafe_object_placement: an object in a clearly unsafe position

If none is visible, use normal. If the image cannot be judged, use uncertain.
Return ONLY this valid JSON object, with no Markdown or extra text:
{
  "result": "normal/attention/uncertain",
  "issue_type": "",
  "evidence": "",
  "suggestion": ""
}
Use attention only with clear visual evidence and report one issue. For normal or uncertain,
issue_type must be empty. Keep evidence and suggestion short."""


def confirmation_prompt(issue_type: str, evidence: str) -> str:
    return f"""Act as a strict visual-evidence verifier.

The first inspection reported:
Issue type: {issue_type}
Evidence: {evidence}

Check the kitchen image again. Is there clear and directly visible evidence
supporting this exact issue?

Answer only one word: yes, no, or uncertain.
Use yes only when the evidence is explicit. Do not rely on assumptions."""


DECOMPOSED_PROMPTS = {
    "floor_obstruction": """Inspect only the floor and walking path in this kitchen image.
Is there a concrete visible object physically blocking the walking area?
Do not count shadows, floor patterns, cabinets, or objects located only on a countertop.
Answer exactly: yes | visible object and location
or: no | none
or: uncertain | reason""",
    "countertop_clutter": """Inspect only the kitchen countertop in this image.
Is the countertop visibly and excessively cluttered with many items, dishes, or waste?
Do not count normal appliances or a few neatly arranged objects.
Answer exactly: yes | visible objects and location
or: no | none
or: uncertain | reason""",
    "unsafe_object_placement": """Inspect object placement in this kitchen image.
Is a visible object in a clearly unsafe position, such as near an open flame,
at a falling edge, or blocking safe appliance use? Do not infer hidden risks.
Answer exactly: yes | visible object and unsafe location
or: no | none
or: uncertain | reason""",
}
