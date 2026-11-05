"""Auto-fix loop: propose changes -> apply -> validate geometry -> re-check.

A proposal is the same shape whether it comes from the deterministic
`reference_fix` (baseline, no API) or from Muse Spark:

    {"type": "thin_wall",
     "changes": [{"field": "pocket.wall", "to": 2.0}, {"field": "pocket.height", "to": 18.5}],
     "rationale": "..."}
"""

import copy
import math

from .rules import (DEEP_HOLE, DEEP_SLOT, MAX_HOLE_RATIO, MAX_SLOT_RATIO, MIN_WALL,
                    MISSING_TOL, SHARP_CORNER, THIN_WALL, TIGHT_TOL, check)

# Fields a proposal may change. Anything else is rejected, so a model cannot
# "fix" a part by e.g. deleting the hole.
EDITABLE = {
    "pocket.wall", "pocket.height", "pocket.corner_r", "pocket.depth", "pocket.sharp_via_note",
    "blind_hole.D", "blind_hole.depth",
    "slot.width", "slot.depth",
    "tight_tol_on", "general_tol",
}


def ceil05(x):
    return math.ceil(x * 2 - 1e-9) / 2


def _get(part, field):
    obj = part
    for k in field.split("."):
        obj = obj[k]
    return obj


def _set(part, field, value):
    *path, last = field.split(".")
    obj = part
    for k in path:
        obj = obj[k]
    obj[last] = value


def normalize(part):
    """Recompute fields derived from the editable ones."""
    pk = part["pocket"]
    pk["y1"] = round(part["W"] - pk["wall"], 2)
    pk["y0"] = round(pk["y1"] - pk["height"], 2)
    pk["width"] = round(pk["x1"] - pk["x0"], 2)
    if pk.get("sharp_via_note") and pk["corner_r"] > 0:
        pk["sharp_via_note"] = False
    for key, size in (("blind_hole", "D"), ("slot", "width")):
        f = part.get(key)
        if f:
            f["ratio"] = round(f["depth"] / f[size], 2)
    return part


def validate(part):
    """Geometry sanity checks a fix must not break. Returns error strings."""
    errs = []
    pk, bh, sl = part["pocket"], part.get("blind_hole"), part.get("slot")
    if pk["height"] <= 2 * pk["corner_r"]:
        errs.append("pocket corner radius too large for pocket height")
    if pk["y0"] <= part["W"] / 2:
        errs.append("pocket grew into the lower feature band")
    if pk["depth"] >= part["T"] - 1:
        errs.append("pocket depth leaves < 1 mm floor")
    if bh and bh["depth"] >= part["T"] - 1:
        errs.append("blind hole breaks through the floor")
    if sl and sl["depth"] >= part["T"] - 1:
        errs.append("slot breaks through the floor")
    if sl and sl["width"] / 2 >= sl["y"] - 1:
        errs.append("slot too wide for its position")
    return errs


def apply_proposal(part, proposal):
    """Return (new_part, applied_changes, rejected_changes)."""
    new = copy.deepcopy(part)
    applied, rejected = [], []
    for ch in proposal.get("changes", []):
        field = ch.get("field")
        if field not in EDITABLE:
            rejected.append({**ch, "reason": "field not editable"})
            continue
        try:
            old = _get(new, field)
        except (KeyError, TypeError):
            rejected.append({**ch, "reason": "feature not present"})
            continue
        _set(new, field, ch["to"])
        applied.append({"field": field, "from": old, "to": ch["to"]})
    return normalize(new), applied, rejected


def reference_fix(part, issue, feedback=None):
    """Deterministic baseline fix for one issue (no model needed)."""
    t = issue["type"]
    pk, bh, sl = part["pocket"], part.get("blind_hole"), part.get("slot")
    if t == THIN_WALL:
        new_wall = max(2.0, MIN_WALL * 2)
        return {"type": t, "changes": [
            {"field": "pocket.wall", "to": new_wall},
            {"field": "pocket.height", "to": round(pk["height"] - (new_wall - pk["wall"]), 2)}],
            "rationale": f"Move the pocket edge to leave a {new_wall} mm wall; shrink the pocket "
                         "by the same amount so its floor area and bottom edge stay put."}
    if t == SHARP_CORNER:
        r = max(1.0, ceil05(pk["depth"] / 3))
        return {"type": t, "changes": [{"field": "pocket.corner_r", "to": r},
                                       {"field": "pocket.sharp_via_note", "to": False}],
                "rationale": f"R{r} (>= depth/3) lets a standard end mill cut the corner and removes "
                             "the stress concentration."}
    if t == DEEP_HOLE:
        d = ceil05(bh["depth"] / MAX_HOLE_RATIO)
        return {"type": t, "changes": [{"field": "blind_hole.D", "to": d}],
                "rationale": f"Open the hole to Ø{d} so depth is <= {MAX_HOLE_RATIO:g}x D. If the "
                             "diameter is functional, reduce depth or drill from both sides instead."}
    if t == DEEP_SLOT:
        w = ceil05(sl["depth"] / MAX_SLOT_RATIO)
        return {"type": t, "changes": [{"field": "slot.width", "to": w}],
                "rationale": f"Widen the slot to {w} mm so a stiffer cutter reaches the full depth."}
    if t == TIGHT_TOL:
        return {"type": t, "changes": [{"field": "tight_tol_on", "to": None}],
                "rationale": "Overall size is non-functional; fall back to the general tolerance."}
    if t == MISSING_TOL:
        return {"type": t, "changes": [{"field": "general_tol", "to": True}],
                "rationale": "State ISO 2768-m style general tolerances in the title block."}
    raise ValueError(f"no reference fix for {t}")


def fix_loop(part, propose=reference_fix, max_iters=3):
    """Iterate until `check` is clean, geometry is valid, or we give up.

    Returns {"fixed": part, "passed": bool, "rounds": [...]}.
    """
    cur = normalize(copy.deepcopy(part))
    pre_existing = set(validate(cur))
    rounds = []
    feedback = {}  # issue type -> why the last attempt failed, passed to the next proposal
    for _ in range(max_iters):
        issues = check(cur)
        if not issues:
            break
        rnd = {"issues_before": issues, "proposals": []}
        for iss in issues:
            prop = propose(cur, iss, feedback.get(iss["type"]))
            nxt, applied, rejected = apply_proposal(cur, prop)
            # Only reject a fix for geometry problems it introduces, not ones the part
            # already had (e.g. from a misread value).
            pre = set(validate(cur))
            errs = [e for e in validate(nxt) if e not in pre]
            still = any(i["type"] == iss["type"] for i in check(nxt))
            accepted = not errs and bool(applied)
            rnd["proposals"].append({"issue": iss["type"], "rationale": prop.get("rationale", ""),
                                     "assumption": prop.get("assumption"), "alternatives": prop.get("alternatives", []),
                                     "applied": applied, "rejected": rejected,
                                     "geometry_errors": errs, "accepted": accepted,
                                     "resolved": accepted and not still})
            if accepted:
                cur = nxt
            reasons = ([f"rejected changes: {rejected}"] if rejected else []) + errs
            if accepted and still:
                reasons.append(f"change applied but the rule is still violated: {check(nxt)}")
            if reasons or not applied:
                feedback[iss["type"]] = "; ".join(reasons) or "no applicable change was proposed"
        rnd["issues_after"] = check(cur)
        rounds.append(rnd)
    if rounds and cur.get("rev"):
        cur["rev"] = chr(ord(cur["rev"]) + 1)
    new_errors = [e for e in validate(cur) if e not in pre_existing]
    return {"fixed": cur, "passed": not check(cur) and not new_errors, "rounds": rounds,
            "pre_existing_geometry_warnings": sorted(pre_existing)}
