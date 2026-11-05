"""Grade a fix beyond "the rule passes": margin past the limit, size of the
change, design-intent questions, and the engineering improvement it buys.

Grades
- good:       passes with >= 25% margin, no unresolved design-intent question
- acceptable: passes with 10-25% margin, or raises a design-intent question
- poor:       lands within 10% of the limit, or fails
"""

from . import engineering
from .rules import (DEEP_HOLE, DEEP_SLOT, MAX_HOLE_RATIO, MAX_SLOT_RATIO, MIN_WALL, MISSING_TOL,
                    SHARP_CORNER, THIN_WALL, TIGHT_TOL, check)

GOOD_MARGIN, OK_MARGIN = 0.25, 0.10


def margin(issue_type, part):
    """Fractional distance past the limit (0 = exactly at the limit, < 0 = violates)."""
    pk, bh, sl = part["pocket"], part.get("blind_hole"), part.get("slot")
    if issue_type == THIN_WALL:
        return pk["wall"] / MIN_WALL - 1
    if issue_type == SHARP_CORNER:
        return pk["corner_r"] / (pk["depth"] / 3) - 1
    if issue_type == DEEP_HOLE:
        return MAX_HOLE_RATIO / (bh["depth"] / bh["D"]) - 1
    if issue_type == DEEP_SLOT:
        return MAX_SLOT_RATIO / (sl["depth"] / sl["width"]) - 1
    return None  # tolerance issues are binary


def intent_questions(issue_type, before, after):
    """Design-intent questions an engineer must answer before accepting the fix."""
    q = []
    b, a = before, after
    if issue_type == DEEP_HOLE and b.get("blind_hole") and a.get("blind_hole"):
        if a["blind_hole"]["D"] != b["blind_hole"]["D"]:
            q.append(f"Hole Ø{b['blind_hole']['D']} → Ø{a['blind_hole']['D']}: is the diameter functional "
                     "(pin, press fit, thread)?")
        if a["blind_hole"]["depth"] < b["blind_hole"]["depth"]:
            q.append(f"Hole depth {b['blind_hole']['depth']} → {a['blind_hole']['depth']} mm: is the depth "
                     "functional (thread or pin engagement)?")
    if issue_type == DEEP_SLOT and b.get("slot") and a.get("slot") and a["slot"]["width"] != b["slot"]["width"]:
        q.append(f"Slot width {b['slot']['width']} → {a['slot']['width']} mm: does a mating part fit this slot?")
    pa_b = b["pocket"]["width"] * b["pocket"]["height"]
    pa_a = a["pocket"]["width"] * a["pocket"]["height"]
    if issue_type in (THIN_WALL, SHARP_CORNER) and pa_a < 0.9 * pa_b:
        q.append(f"Pocket area reduced {100 * (1 - pa_a / pa_b):.0f}%: does whatever sits in the pocket still fit?")
    if issue_type == SHARP_CORNER and a["pocket"]["corner_r"] > 0.25 * min(a["pocket"]["width"], a["pocket"]["height"]):
        q.append(f"Corner R{a['pocket']['corner_r']} is large for this pocket: does a square part need to seat in it?")
    return q


def _eng(issue_type, part):
    ev = engineering.evidence(issue_type, part)
    if ev:
        ev = {k: v for k, v in ev.items() if not k.startswith("_")}
    return ev


def score_fix(issue_type, before, after, with_engineering=True):
    """Score one issue's fix on a full before/after part pair."""
    passes = not any(i["type"] == issue_type for i in check(after))
    m = margin(issue_type, after)
    qs = intent_questions(issue_type, before, after)
    if not passes:
        grade = "poor"
    elif m is not None and m < OK_MARGIN:
        grade = "poor"
    elif (m is not None and m < GOOD_MARGIN) or qs:
        grade = "acceptable"
    else:
        grade = "good"
    out = {"issue": issue_type, "passes": passes, "grade": grade,
           "margin_pct": None if m is None else round(100 * m), "intent_questions": qs}
    if with_engineering and issue_type in (THIN_WALL, SHARP_CORNER, DEEP_HOLE, DEEP_SLOT):
        eb, ea = _eng(issue_type, before), _eng(issue_type, after)
        out["engineering"] = {"metric": eb["metric"], "before": eb["value"], "after": ea["value"],
                              "improvement_x": round(eb["value"] / ea["value"], 1) if ea["value"] else None,
                              "assumption": ea["assumption"]}
    return out


def score_all(before, after, issue_types, with_engineering=True):
    return [score_fix(t, before, after, with_engineering) for t in issue_types]
