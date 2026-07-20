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


STRUCTURED_PROMPT = """Inspect this kitchen image.

Allowed issue types:
- floor_obstruction
- countertop_clutter
- unsafe_object_placement

Return exactly one JSON object in this format:
{
  "result": "normal/attention/uncertain",
  "issue_type": "",
  "evidence": "",
  "suggestion": ""
}

Rules:
1. Use attention only when one listed issue has clear visual evidence.
2. Report only the single most obvious issue.
3. For normal or uncertain, issue_type must be an empty string.
4. Do not infer anything that is not directly visible.
5. Use concise English sentences for evidence and suggestion.
6. Output JSON only, without Markdown or additional text."""


def confirmation_prompt(issue_type: str, evidence: str) -> str:
    return f"""Act as a strict visual-evidence verifier.

The first inspection reported:
Issue type: {issue_type}
Evidence: {evidence}

Check the kitchen image again. Is there clear and directly visible evidence
supporting this exact issue?

Answer only one word: yes, no, or uncertain.
Use yes only when the evidence is explicit. Do not rely on assumptions."""
