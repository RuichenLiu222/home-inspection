DIRECT_PROMPT = (
    "Please inspect this kitchen image and determine whether there is any "
    "obvious problem. Briefly state your judgment and visible evidence."
)


CHECKLIST_PROMPT = """Inspect one kitchen image. Decide from visible evidence only.

Use this decision procedure silently:
1. If the relevant areas cannot be judged because of severe blur or occlusion: uncertain.
2. If no strong evidence below is present: normal.
3. Otherwise choose the single issue with the clearest visible evidence:
   - countertop_clutter: multiple loose items substantially occupy the counter or sink.
     One appliance or a few normally used items are not enough.
   - unsafe_object_placement: a named object has a directly visible dangerous relation,
     such as beside a flame, at a falling edge, or an electrical item beside water.
   - floor_obstruction: a named object is visibly on the floor or walking path.
     Never use this for an object located only on a counter, table, sink, or appliance.

Do not list several labels and do not copy these rules. Reply in exactly two short lines:
Evidence: name one visible object and its location, or say no listed issue is visible.
Decision: exactly one label from normal, countertop_clutter, unsafe_object_placement,
floor_obstruction, uncertain."""


STRUCTURED_PROMPT = """Inspect one kitchen image using visible evidence only.

Decision rule:
- Use normal when the image is clear and no strong listed issue is visible.
- Use countertop_clutter only when multiple loose items substantially occupy the counter/sink.
- Use unsafe_object_placement only for a directly visible dangerous object-location relation.
- Use floor_obstruction only when a named object is visibly on the floor/walking path.
- Use uncertain only when severe blur or occlusion prevents judgment.

Return one compact JSON object and nothing else. Use result=attention for an issue and put
exactly one issue label in issue_type. For normal or uncertain, issue_type must be empty.
Required keys: result, issue_type, evidence, suggestion. Keep the last two values short.
Allowed result values: normal, attention, uncertain.
Allowed issue_type values: countertop_clutter, unsafe_object_placement,
floor_obstruction, or an empty string."""


def confirmation_prompt(issue_type: str, evidence: str) -> str:
    required_evidence = {
        "floor_obstruction": (
            "a solid object is visibly on the floor or walking path; an object only on a "
            "counter, table, sink, or appliance does not count"
        ),
        "countertop_clutter": (
            "several loose items visibly crowd the usable counter or sink; one appliance "
            "or a few normally arranged items do not count"
        ),
        "unsafe_object_placement": (
            "both a named object and a directly visible dangerous spatial relation are "
            "present; an object by itself does not count"
        ),
    }.get(issue_type, "the exact reported issue must be directly visible")
    return f"""Independently verify one candidate issue in this kitchen image.

Candidate: {issue_type}
First-pass note (it may be wrong): {evidence}
Condition to check in the image: {required_evidence}.

The candidate label is not evidence. Silently locate the exact object and its location.
Answer yes only when the image itself clearly satisfies the whole condition.
Answer no when the relevant region is visible but the whole condition is not satisfied.
Answer uncertain if the relevant region cannot be judged.
Reply with exactly one word: yes, no, or uncertain."""


# Experimental V2 prompts.  Keep the frozen V1 prompts above unchanged so the
# published baseline remains reproducible.
QUALITY_PROMPT = """Look at this single image.
Is it a kitchen image clear enough to identify visible objects and where they are?
Reply with exactly one token: CLEAR or UNCERTAIN.
Use UNCERTAIN only for severe blur, heavy occlusion, or a non-kitchen image."""


REGION_PROMPTS = {
    "floor_obstruction": """Inspect only the visible floor and walking path.
Is a movable object lying on the floor/path and obstructing walking?
Count boxes, bags, buckets, stools, loose cables, dishes, trash, or scattered objects in the path.
Do not count fixed furniture, chairs normally placed around a table, rugs,
tile patterns, shadows, or anything on a counter.
Reply exactly: NO, UNCERTAIN, or YES | <object> | <floor location>.""",
    "countertop_clutter": """Inspect only the counter and sink work surfaces.
Are many loose items or unwashed dishes covering a substantial usable area?
Count piles of dishes, containers, food packages, or many unrelated scattered items.
Do not count fixed appliances, a few everyday items, one dish, one bottle,
decorations, or a normally used drying rack.
Reply exactly: NO, UNCERTAIN, or YES | <visible items> | <counter/sink location>.""",
    "unsafe_object_placement": """Inspect visible object-to-hazard relationships.
Is an object placed in a directly visible unsafe location?
Count only: cloth/paper/plastic on or next to a burner; an electrical device
or cable touching or immediately next to sink water; a knife, glass, or heavy
object at an edge where it may fall.
Both the object and the hazard source, water, or edge must be visible. Clutter alone does not count.
Reply exactly: NO, UNCERTAIN, or YES | <object> | <hazard relation>.""",
}


REGION_VERIFICATION_PROMPTS = {
    "floor_obstruction": """Independently re-check only the floor and walking path.
Is a clearly visible movable object actually on the path and obstructing walking?
Reject floor patterns, shadows, fixed furniture, and objects on counters.
Reply exactly: YES, NO, or UNCERTAIN.""",
    "countertop_clutter": """Independently re-check only the counter and sink.
Do many loose items clearly cover a substantial usable work area?
Reject normal appliances, a few everyday items, and a normally used drying rack.
Reply exactly: YES, NO, or UNCERTAIN.""",
    "unsafe_object_placement": """Independently re-check the object-to-hazard relationship.
Are both the object and a directly dangerous heat, water, sharp-edge, or
falling relation clearly visible?
Clutter without a visible hazardous relation is NO.
Reply exactly: YES, NO, or UNCERTAIN.""",
}


ISSUE_PRIORITY = (
    "unsafe_object_placement",
    "floor_obstruction",
    "countertop_clutter",
)


ISSUE_SUGGESTIONS = {
    "floor_obstruction": "Remove the object from the walking path.",
    "countertop_clutter": "Clear and organize the usable counter area.",
    "unsafe_object_placement": "Move the object away from the visible hazard.",
}
