"""DFM rules and a deterministic checker over a structured part spec.

The checker is the ground truth for the eval set and the verifier in the
auto-fix loop: a fix is accepted only when `check()` on the fixed spec is clean.
"""

THIN_WALL = "thin_wall"
SHARP_CORNER = "sharp_internal_corner"
DEEP_HOLE = "deep_small_hole"
DEEP_SLOT = "deep_narrow_slot"
TIGHT_TOL = "overly_tight_tolerance"
MISSING_TOL = "missing_general_tolerance"

MIN_WALL = 1.0
MAX_HOLE_RATIO = 6.0
MAX_SLOT_RATIO = 4.0

# Planted values sit well past the limit and hard negatives well inside it, so
# the eval key has no borderline calls.
RULES = {
    THIN_WALL: {
        "rule": f"Minimum wall between a pocket and the outer edge must be >= {MIN_WALL} mm.",
        "planted_range": "0.3-0.6 mm", "negative_range": "2.0-6.0 mm",
        "severity": "high", "category": "machinability",
        "failure_modes": ["chatter_vibration", "wall_deflection_out_of_tolerance",
                          "cracking_under_load", "distortion_after_machining"],
    },
    SHARP_CORNER: {
        "rule": "Internal vertical pocket corners need a radius >= 1/3 of pocket depth (R0 cannot be cut with an end mill).",
        "planted_range": "R0", "negative_range": ">= depth/3, min R1.0",
        "severity": "high", "category": "machinability",
        "failure_modes": ["stress_concentration", "fatigue_crack_initiation",
                          "not_machinable_with_end_mill", "requires_edm_high_cost"],
    },
    DEEP_HOLE: {
        "rule": f"Blind drilled hole depth must be <= {MAX_HOLE_RATIO:g}x diameter.",
        "planted_range": "10x-14x D", "negative_range": "2x-3.5x D",
        "severity": "medium", "category": "machinability",
        "failure_modes": ["drill_breakage", "hole_drift_runout", "chip_packing", "scrap"],
    },
    DEEP_SLOT: {
        "rule": f"Slot depth must be <= {MAX_SLOT_RATIO:g}x slot width.",
        "planted_range": "6x-8x width", "negative_range": "1x-2.5x width",
        "severity": "medium", "category": "machinability",
        "failure_modes": ["tool_deflection", "chatter_vibration",
                          "poor_surface_finish", "long_reach_tool_breakage"],
    },
    TIGHT_TOL: {
        "rule": "Non-functional dimensions (e.g. overall outside size) should not be toleranced tighter than +/-0.01 mm.",
        "planted_range": "+/-0.005 on overall length or width", "negative_range": "+/-0.1 or general tolerance",
        "severity": "medium", "category": "cost",
        "failure_modes": ["high_cost", "high_scrap_rate", "extra_inspection_time"],
    },
    MISSING_TOL: {
        "rule": "Title block must state general tolerances when dimensions carry no explicit tolerance.",
        "planted_range": "general tolerance field left blank", "negative_range": "field filled in",
        "severity": "high", "category": "drawing_quality",
        "failure_modes": ["ambiguous_acceptance_criteria", "supplier_dispute", "fit_failure_at_assembly"],
    },
}


def check(part):
    """Return the DFM issues in a part spec as [{type, value, description}]."""
    issues = []
    pk, bh, sl = part["pocket"], part.get("blind_hole"), part.get("slot")
    if pk["wall"] < MIN_WALL:
        issues.append({"type": THIN_WALL, "value": pk["wall"],
                       "description": f"wall {pk['wall']} mm between pocket and top edge"})
    if pk["corner_r"] < pk["depth"] / 3 - 1e-9:
        issues.append({"type": SHARP_CORNER, "value": pk["corner_r"],
                       "description": f"pocket corners R{pk['corner_r']:g} at {pk['depth']} mm depth"
                                      + (" (specified in notes)" if pk.get("sharp_via_note") else "")})
    if bh and bh["depth"] / bh["D"] > MAX_HOLE_RATIO:
        r = round(bh["depth"] / bh["D"], 2)
        issues.append({"type": DEEP_HOLE, "value": r,
                       "description": f"Ø{bh['D']} hole {bh['depth']} mm deep = {r}x D"})
    if sl and sl["depth"] / sl["width"] > MAX_SLOT_RATIO:
        r = round(sl["depth"] / sl["width"], 2)
        issues.append({"type": DEEP_SLOT, "value": r,
                       "description": f"{sl['width']} mm slot {sl['depth']} mm deep = {r}x width"})
    if part.get("tight_tol_on"):
        dim = "length" if part["tight_tol_on"] == "L" else "width"
        issues.append({"type": TIGHT_TOL, "value": "+/-0.005",
                       "description": f"overall {dim} toleranced +/-0.005"})
    if not part.get("general_tol", True):
        issues.append({"type": MISSING_TOL, "value": None,
                       "description": "general tolerance field in title block is blank"})
    return issues
