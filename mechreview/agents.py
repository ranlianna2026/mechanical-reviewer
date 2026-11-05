"""Muse Spark steps around the deterministic core: verify, explain, propose.

Each step has an offline fallback so the pipeline runs without credits:
verify is skipped, explain uses the knowledge base text, propose uses the
reference fixer.
"""

import json
from pathlib import Path

from PIL import Image

from . import meta_api
from .extract import _crop
from .fix import EDITABLE, reference_fix
from .rules import (DEEP_HOLE, DEEP_SLOT, MISSING_TOL, RULES, SHARP_CORNER, THIN_WALL,
                    TIGHT_TOL)

ROOT = Path(__file__).resolve().parents[1]
RECALLS = ROOT / "data" / "recalls.jsonl"

# ---------------------------------------------------------------- knowledge
KB = {
    THIN_WALL: {
        "mechanism": "A wall this thin deflects under cutting force, so it chatters, ends up out of "
                     "tolerance, and can crack under clamping or service load or distort as residual "
                     "stress is released after machining.",
        "recall_tags": ["fracture", "fatigue_crack"], "keywords": ["thin", "wall thickness", "flange", "housing crack", "web"]},
    SHARP_CORNER: {
        "mechanism": "A near-zero internal radius is a stress concentration (Kt often > 3). Under cyclic "
                     "load, fatigue cracks initiate at the corner. It also cannot be cut with a rotating "
                     "end mill, so it forces EDM or gets silently radiused by the shop.",
        "recall_tags": ["fatigue_crack"], "keywords": ["bracket", "mount", "fatigue", "corner", "crack"]},
    DEEP_HOLE: {
        "mechanism": "Long thin drills wander and break; chips pack in the flutes. Results are drifted "
                     "holes, broken drill remnants and scrap.",
        "recall_tags": ["drilled_hole", "machining"], "keywords": ["machined hole", "bore", "drilled hole", "hole was", "holes were", "improperly drilled", "machining of the"]},
    DEEP_SLOT: {
        "mechanism": "A long, small-diameter cutter deflects and chatters, giving poor finish, tapered "
                     "walls and tool breakage.",
        "recall_tags": ["machining"], "keywords": ["machin", "slot", "groove", "finish"]},
    TIGHT_TOL: {
        "mechanism": "A ±0.005 mm band on a non-functional size forces grinding or extra passes plus "
                     "CMM inspection, multiplying cost and scrap with no functional benefit.",
        "recall_tags": ["tolerance"], "keywords": ["tolerance", "dimension", "specification"]},
    MISSING_TOL: {
        "mechanism": "With no general tolerance, untoleranced dimensions have no acceptance criteria: "
                     "suppliers and inspectors interpret them differently and out-of-spec parts reach "
                     "assembly.",
        "recall_tags": ["tolerance"], "keywords": ["tolerance", "dimension", "specification"]},
}

# Which extraction evidence fields support each issue type.
EVIDENCE_FOR = {
    THIN_WALL: ["pocket_wall_dim"], SHARP_CORNER: ["pocket_callout", "notes"],
    DEEP_HOLE: ["hole_callout"], DEEP_SLOT: ["slot_callout"],
    TIGHT_TOL: ["overall_length_dim", "overall_width_dim"],
    MISSING_TOL: ["general_tolerance_block", "title_block"],
}

_recalls = None


def search_recalls(issue_type, k=3):
    """Keyword/tag search over the local NHTSA index; best analogues first."""
    global _recalls
    if _recalls is None:
        _recalls = [json.loads(l) for l in RECALLS.read_text().splitlines()] if RECALLS.exists() else []
    kb = KB[issue_type]
    scored = []
    for r in _recalls:
        text = (r["defect"] + " " + r["component"]).lower()
        hits = sum(w in text for w in kb["keywords"])
        score = 2 * len(set(kb["recall_tags"]) & set(r["tags"])) + 3 * hits
        if hits and score >= 5:
            scored.append((score, r.get("units") or 0, r))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [{"campaign": r["campaign"], "vehicle": " ".join(x for x in (r["year"], r["make"], r["model"]) if x and x != "9999"),
             "component": r["component"], "defect": r["defect"][:400], "consequence": r["consequence"][:300],
             "url": f"https://www.nhtsa.gov/recalls?nhtsaId={r['campaign']}"} for _, _, r in scored[:k]]


# ------------------------------------------------------------------- verify
VERIFY_SCHEMA = {"type": "object", "additionalProperties": False,
                 "required": ["text_read", "supports_claim", "note"],
                 "properties": {"text_read": {"type": "string"}, "supports_claim": {"type": "boolean"},
                                "note": {"type": "string"}}}


def verify_issue(image_path, issue, evidence):
    """Re-read the evidence for one issue on zoomed crops (all relevant ones).

    Returns {"status": "confirmed"|"rejected"|"skipped", ...}.
    """
    if not meta_api.online():
        return {"status": "skipped", "reason": "offline"}
    boxes = [e for e in evidence if e["field"] in EVIDENCE_FOR[issue["type"]]
             and len(e.get("bbox", [])) == 4 and e["bbox"][2] > e["bbox"][0] and e["bbox"][3] > e["bbox"][1]]
    im = Image.open(image_path).convert("RGB")
    images = [_crop(im, e["bbox"], margin=60, scale=2) for e in boxes] or [im]
    where = (f"{len(images)} zoomed crop(s) ({', '.join(e['field'] for e in boxes)}) of"
             if boxes else "the full")
    res = meta_api.chat_json(
        "You check claims about engineering drawings. Read only what is printed.",
        f"These are {where} an engineering drawing; the evidence may be in any of them. "
        f"Claim to check: {issue['description']}. Rule: {RULES[issue['type']]['rule']} "
        "Read the relevant text exactly and decide whether the drawing supports the claim.",
        VERIFY_SCHEMA, "verdict", images=images, effort="low", tag="verify")
    return {"status": "confirmed" if res["supports_claim"] else "rejected", "crops": len(boxes), **res}


# ------------------------------------------------------------------ explain
EXPLAIN_SCHEMA = {"type": "object", "additionalProperties": False,
                  "required": ["summary", "failure_chain", "risk", "recall_refs", "recall_relevance"],
                  "properties": {
                      "summary": {"type": "string"},
                      "failure_chain": {"type": "array", "items": {"type": "string"}},
                      "risk": {"type": "string", "enum": ["high", "medium", "low"]},
                      "recall_refs": {"type": "array", "items": {"type": "string"}},
                      "recall_relevance": {"type": "string"}}}


def explain_issue(part, issue):
    kb, recalls = KB[issue["type"]], search_recalls(issue["type"])
    base = {"failure_modes": RULES[issue["type"]]["failure_modes"], "recalls": recalls}
    if not meta_api.online():
        return {**base, "summary": kb["mechanism"], "failure_chain": [], "risk": RULES[issue["type"]]["severity"],
                "recall_refs": [], "recall_relevance": "offline: candidates listed, not reviewed", "source": "kb"}
    res = meta_api.chat_json(
        "You are a failure-analysis engineer. Be specific and concise. Cite only recalls from the "
        "provided list, and only when the failure mechanism is genuinely analogous.",
        json.dumps({"part": {k: part[k] for k in ("name", "material", "L", "W", "T")},
                    "issue": issue, "rule": RULES[issue["type"]]["rule"], "mechanism": kb["mechanism"],
                    "candidate_recalls": recalls}, ensure_ascii=False)
        + "\nExplain how this issue leads to failure in manufacturing or in service, as a short "
          "failure chain, and rate the risk for this part and material.",
        EXPLAIN_SCHEMA, "explanation", effort="low", tag="explain")
    refs = {r["campaign"] for r in recalls}
    res["recall_refs"] = [c for c in res["recall_refs"] if c in refs]
    return {**base, **res, "source": meta_api.SPARK}


# ------------------------------------------------------------------ propose
PROPOSE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["changes", "rationale"],
                  "properties": {
                      "changes": {"type": "array", "items": {
                          "type": "object", "additionalProperties": False, "required": ["field", "to"],
                          "properties": {"field": {"type": "string", "enum": sorted(EDITABLE)},
                                         "to": {"anyOf": [{"type": "number"}, {"type": "boolean"},
                                                          {"type": "string", "enum": ["L", "W"]},
                                                          {"type": "null"}]}}}},
                      "rationale": {"type": "string"}}}


def propose_fix(part, issue, feedback=None):
    """Muse Spark fix proposal; same shape as fix.reference_fix."""
    if not meta_api.online():
        return reference_fix(part, issue, feedback)
    msg = {"part": part, "issue": issue, "rule": RULES[issue["type"]]["rule"],
           "editable_fields": sorted(EDITABLE)}
    if feedback:
        msg["previous_attempt_failed_because"] = feedback
    res = meta_api.chat_json(
        "You are a design-for-manufacturing engineer fixing a drawing. Make the smallest change that "
        "satisfies the rule while preserving design intent (keep feature locations and functional "
        "sizes unless the rule forces a change). Use standard values (0.5 mm steps, standard drill sizes).",
        json.dumps(msg, ensure_ascii=False)
        + "\nNotes: pocket.wall is the wall to the top edge; increasing it moves the pocket's top edge down, "
          "so reduce pocket.height by the same amount to keep the pocket's bottom edge in place. "
          "Set tight_tol_on to null to remove a planted tight tolerance; general_tol true adds the "
          "title-block general tolerance.",
        PROPOSE_SCHEMA, "fix", effort="medium", tag="propose")
    return {"type": issue["type"], **res}


# --------------------------------------------------------------- propose v2
CANDIDATE = {"type": "object", "additionalProperties": False,
             "required": ["changes", "rationale", "assumption"],
             "properties": {"changes": PROPOSE_SCHEMA["properties"]["changes"],
                            "rationale": {"type": "string"}, "assumption": {"type": "string"}}}
PROPOSE_V2_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["candidates"],
                     "properties": {"candidates": {"type": "array", "items": CANDIDATE}}}
_GRADE_RANK = {"good": 0, "acceptable": 1, "poor": 2}


def _rank(cand_eval):
    """Lower is better: valid and resolving first, then grade, then distance from a
    ~50% margin (rewards margin, penalizes overshoot), then fewer open questions."""
    q = cand_eval["quality"]
    m = (q["margin_pct"] if q["margin_pct"] is not None else 50) / 100
    return (not cand_eval["ok"], _GRADE_RANK[q["grade"]], abs(m - 0.5), len(q["intent_questions"]))


def propose_fix_v2(part, issue, feedback=None):
    """Several candidates from Muse Spark, each applied, validated and scored; best one wins."""
    from .fix import apply_proposal, validate
    from .quality import score_fix
    from .rules import check
    if not meta_api.online():
        return reference_fix(part, issue, feedback)
    msg = {"part": part, "issue": issue, "rule": RULES[issue["type"]]["rule"],
           "editable_fields": sorted(EDITABLE)}
    if feedback:
        msg["previous_attempt_failed_because"] = feedback
    res = meta_api.chat_json(
        "You are a senior design-for-manufacturing engineer fixing a drawing.\n"
        "- Target 25-50% margin past the limit (e.g. a 1.5 mm wall when the minimum is 1.0 mm; a corner "
        "radius about 1.25-1.5x depth/3). Never land exactly on a limit: tolerances and tool wear eat margin.\n"
        "- Do not overshoot either: keep the change small, keep feature locations, and keep sizes that are "
        "likely functional.\n"
        "- Use standard values (0.5 mm steps, standard drill sizes).\n"
        "- When there is a real design-intent trade-off (e.g. widen a hole vs. make it shallower), return "
        "2-3 candidates, each stating its assumption about what is functional. Otherwise return 1.",
        json.dumps(msg, ensure_ascii=False)
        + "\nNotes: pocket.wall is the wall to the top edge; increasing it moves the pocket's top edge down, "
          "so reduce pocket.height by the same amount to keep the pocket's bottom edge in place. "
          "Set tight_tol_on to null to remove a planted tight tolerance; general_tol true adds the "
          "title-block general tolerance.",
        PROPOSE_V2_SCHEMA, "fix_candidates", effort="medium", tag="propose_v2")
    evals = []
    for cand in res.get("candidates", [])[:3]:
        nxt, applied, rejected = apply_proposal(part, cand)
        ok = bool(applied) and not validate(nxt) and not any(i["type"] == issue["type"] for i in check(nxt))
        evals.append({"cand": cand, "ok": ok, "quality": score_fix(issue["type"], part, nxt, with_engineering=False)})
    if not evals:
        return {"type": issue["type"], "changes": [], "rationale": "no candidates returned"}
    evals.sort(key=_rank)
    best = evals[0]
    alts = [{"changes": e["cand"]["changes"], "assumption": e["cand"]["assumption"],
             "grade": e["quality"]["grade"], "margin_pct": e["quality"]["margin_pct"], "ok": e["ok"]}
            for e in evals[1:]]
    return {"type": issue["type"], "changes": best["cand"]["changes"],
            "rationale": best["cand"]["rationale"], "assumption": best["cand"]["assumption"],
            "alternatives": alts}
