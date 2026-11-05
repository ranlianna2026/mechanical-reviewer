"""Drawing image -> structured part spec (Muse Spark vision, JSON schema).

The output matches eval/specs/*.json so the checker, renderer and fix loop
work on extracted parts exactly as on ground truth. Offline, `oracle=True`
returns the ground-truth spec for eval drawings so the pipeline can be
exercised end to end without API calls.
"""

import json
import re
from pathlib import Path

from PIL import Image

from . import meta_api
from .fix import normalize

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "eval" / "specs"

NUM = {"type": "number"}
STR = {"type": "string"}
NSTR = {"anyOf": [{"type": "string"}, {"type": "null"}]}


def _obj(props, required=None):
    return {"type": "object", "additionalProperties": False,
            "required": required or list(props), "properties": props}


def _nullable(o):
    return {"anyOf": [o, {"type": "null"}]}


EVIDENCE_FIELDS = ["overall_length_dim", "overall_width_dim", "pocket_wall_dim", "pocket_callout",
                   "hole_callout", "slot_callout", "notes", "general_tolerance_block", "title_block",
                   "pocket_outline_top_view", "blind_hole_circle_top_view", "slot_outline_top_view"]

SCHEMA = _obj({
    "name": STR, "number": STR, "rev": STR, "material": STR,
    "L": NUM, "W": NUM, "T": NUM,
    "L_tolerance": NSTR, "W_tolerance": NSTR,
    "general_tolerance_text": NSTR,
    "pocket": _obj({"x0": NUM, "width": NUM, "height": NUM, "depth": NUM, "wall": NUM,
                    "corner_r": NUM, "sharp_via_note": {"type": "boolean"}}),
    "blind_hole": _nullable(_obj({"D": NUM, "depth": NUM, "x": NUM, "y": NUM})),
    "slot": _nullable(_obj({"width": NUM, "depth": NUM, "length": NUM, "x": NUM, "y": NUM})),
    "mount_holes": _obj({"D": NUM, "inset": NUM, "count": NUM}),
    "evidence": {"type": "array", "items": _obj({
        "field": {"type": "string", "enum": EVIDENCE_FIELDS},
        "text_read": STR,
        "bbox": {"type": "array", "items": NUM}})},
    "unreadable": {"type": "array", "items": STR},
})

SYSTEM = """You are a senior manufacturing engineer reading 2D mechanical drawings.
Read values exactly as printed. Never guess or "correct" a value; if something is
not printed, say so in `unreadable`. All dimensions are millimeters."""

PROMPT = """Extract this drawing of a CNC-machined plate into the JSON schema.

How to read it:
- Third-angle projection: TOP VIEW above, FRONT VIEW below (hidden features dashed).
- L, W: overall length and width of the plate in the top view. T: plate thickness from the front view.
- L_tolerance / W_tolerance: any tolerance printed next to the overall L or W dimension
  (e.g. "±0.005"); null if the dimension has no explicit tolerance.
- general_tolerance_text: the title-block general tolerance values (e.g. "LINEAR X.X ±0.2 ...");
  null if the field is blank or missing.
- Pocket: callout "POCKET <width> x <height> / <depth> DEEP / 4X R<r>".
  x0 = horizontal dimension from the plate's left edge to the pocket's left edge.
  wall = vertical dimension between the plate's top edge and the pocket's top edge (can be very small).
  If corners are specified sharp in a note (e.g. "SHARP (R0)") or as R0, corner_r = 0;
  sharp_via_note = true only when a NOTE (not the callout) specifies it.
- Blind hole: callout "Ø<D> x <depth> DEEP"; x, y = its center from the left and bottom edges. null if none.
- Slot: callout "SLOT <width> WIDE x <length> LG / <depth> DEEP"; x, y = its center from the
  left and bottom edges. null if none.
- mount_holes: callout like "4X Ø6.6 THRU"; inset = distance from the edges to their centers.
- evidence: for each item you used (one entry per field in the enum that exists on the drawing),
  give the exact text you read and its pixel bounding box [x1, y1, x2, y2] in this image
  (image is {w}x{h} px, origin top-left). For the *_top_view fields, box the drawn geometry
  itself in the top view (text_read: short description).
"""

# Fields worth a second, zoomed read: small or rotated text that drives the checks.
NNUM = {"anyOf": [{"type": "number"}, {"type": "null"}]}
REREAD_KEYS = {  # evidence field -> (values to read, spec paths they fill)
    "overall_length_dim": {"value": ["L"], "tolerance": ["L_tolerance"]},
    "overall_width_dim": {"value": ["W"], "tolerance": ["W_tolerance"]},
    "pocket_wall_dim": {"value": ["pocket", "wall"]},
    "pocket_callout": {"width": ["pocket", "width"], "height": ["pocket", "height"],
                       "depth": ["pocket", "depth"], "corner_r": ["pocket", "corner_r"]},
    "hole_callout": {"D": ["blind_hole", "D"], "depth": ["blind_hole", "depth"]},
    "slot_callout": {"width": ["slot", "width"], "length": ["slot", "length"], "depth": ["slot", "depth"]},
}
REREAD_FIELDS = list(REREAD_KEYS)
_VALUE_KEYS = sorted({k for m in REREAD_KEYS.values() for k in m})
REREAD_SCHEMA = _obj({"readings": {"type": "array", "items": _obj({
    "field": {"type": "string", "enum": REREAD_FIELDS},
    **{k: (NSTR if k == "tolerance" else NNUM) for k in _VALUE_KEYS}})}})
TIEBREAK_SCHEMA = _obj({"value": NNUM, "printed_text": STR})
REREAD_HELP = {
    "overall_length_dim": "value = the dimension number; tolerance = e.g. '±0.005' if printed next to it, else null",
    "overall_width_dim": "value = the dimension number; tolerance = e.g. '±0.1' if printed next to it, else null",
    "pocket_wall_dim": "value = the small dimension number (the gap between the pocket and the top edge)",
    "pocket_callout": "the callout 'POCKET <width> x <height> / <depth> DEEP / 4X R<corner_r>' (corner_r null if absent)",
    "hole_callout": "the callout 'Ø<D> x <depth> DEEP' for the single blind hole (ignore any '4X Ø... THRU' callout)",
    "slot_callout": "the callout 'SLOT <width> WIDE x <length> LG / <depth> DEEP'",
}


def _crop(image, bbox, margin=40, scale=3):
    x1, y1, x2, y2 = bbox
    c = image.crop((max(0, int(x1 - margin)), max(0, int(y1 - margin)),
                    min(image.width, int(x2 + margin)), min(image.height, int(y2 + margin))))
    if c.height > 1.5 * c.width:  # vertical dimension text: turn it upright
        c = c.rotate(-90, expand=True)
    return c.resize((c.width * scale, c.height * scale), Image.LANCZOS)


def _get_path(raw, path):
    obj = raw
    for k in path:
        obj = obj.get(k) if isinstance(obj, dict) else None
    return obj


def _set_path(raw, path, val):
    obj = raw
    for k in path[:-1]:
        obj = obj.get(k) if isinstance(obj, dict) else None
    if isinstance(obj, dict):
        obj[path[-1]] = val


def reread(image_path, raw, effort="low"):
    """Second pass on tight, upright, zoomed crops of key evidence.

    A zoomed value replaces the first read only when a third, higher-zoom
    tie-break read agrees with it; otherwise the first read is kept and the
    field is reported as uncertain.
    """
    im = Image.open(image_path).convert("RGB")
    items = [e for e in raw.get("evidence", []) if e["field"] in REREAD_FIELDS and len(e.get("bbox", [])) == 4
             and e["bbox"][2] > e["bbox"][0] and e["bbox"][3] > e["bbox"][1]]
    seen, uniq = set(), []
    for e in items:
        if e["field"] not in seen:
            seen.add(e["field"])
            uniq.append(e)
    if not uniq:
        return []
    listing = "\n".join(f"image {k + 1}: {e['field']} -- {REREAD_HELP[e['field']]}" for k, e in enumerate(uniq))
    res = meta_api.chat_json(
        SYSTEM, "Each image is a tight zoomed crop centered on one dimension or callout of the same drawing, in "
                f"this order:\n{listing}\nCrops may show parts of neighboring dimensions at the edges: read only "
                "the item described, at the center. Return one reading per image; null for anything not printed.",
        REREAD_SCHEMA, "readings", images=[_crop(im, e["bbox"], margin=14, scale=3) for e in uniq],
        effort=effort, tag="reread")
    by_field = {}
    for r in res.get("readings", []):
        by_field.setdefault(r["field"], r)  # first reading per field
    boxes = {e["field"]: e["bbox"] for e in uniq}

    changes = []
    for field, reading in by_field.items():
        for key, path in REREAD_KEYS[field].items():
            new, old = reading.get(key), _get_path(raw, path)
            if new is None or old is None or new == old:
                continue
            if key == "tolerance":  # text; take the zoomed read only if it parses to a band
                if _tol_band(new) is not None and _tol_band(new) != _tol_band(old):
                    _set_path(raw, path, new)
                    changes.append({"field": ".".join(path), "first_read": old, "zoomed_read": new,
                                    "resolution": "zoomed"})
                continue
            tb = meta_api.chat_json(
                SYSTEM, f"This crop is centered on the {field.replace('_', ' ')} ({REREAD_HELP[field]}). "
                        f"Two earlier reads of the '{key}' value disagree: {old} vs {new}. "
                        "Read the printed value at the center of the crop exactly.",
                TIEBREAK_SCHEMA, "tiebreak", images=[_crop(im, boxes[field], margin=8, scale=5)],
                effort="medium", tag="reread:tiebreak")
            v = tb.get("value")
            if v == new:
                _set_path(raw, path, new)
                res_ = "zoomed (confirmed by tie-break)"
            elif v == old:
                res_ = "kept first read (tie-break agreed)"
            else:
                res_ = f"uncertain: kept first read (tie-break read {v})"
            changes.append({"field": ".".join(path), "first_read": old, "zoomed_read": new,
                            "tiebreak": v, "resolution": res_})
    return changes


def _tol_band(text):
    if not text:
        return None
    m = re.search(r"(?:±|\+/-)\s*([0-9]*\.?[0-9]+)", text)
    return float(m.group(1)) if m else None


def to_spec(raw):
    """Model output -> canonical part spec (same shape as eval/specs)."""
    pk = raw["pocket"]
    spec = {"L": raw["L"], "W": raw["W"], "T": raw["T"], "material": raw["material"],
            "name": raw["name"], "number": raw["number"], "rev": raw.get("rev") or "A",
            "pocket": {"x0": pk["x0"], "x1": round(pk["x0"] + pk["width"], 2), "width": pk["width"],
                       "height": pk["height"], "depth": pk["depth"], "corner_r": pk["corner_r"],
                       "wall": pk["wall"], "sharp_via_note": bool(pk["sharp_via_note"])},
            "blind_hole": dict(raw["blind_hole"]) if raw.get("blind_hole") else None,
            "slot": dict(raw["slot"]) if raw.get("slot") else None,
            "mount_holes": {"D": raw["mount_holes"]["D"], "inset": raw["mount_holes"]["inset"]}}
    lb, wb = _tol_band(raw.get("L_tolerance")), _tol_band(raw.get("W_tolerance"))
    spec["tight_tol_on"] = "L" if lb is not None and lb <= 0.01 else "W" if wb is not None and wb <= 0.01 else None
    spec["neg_tol_W"] = wb is not None and wb > 0.01
    spec["general_tol"] = _tol_band(raw.get("general_tolerance_text")) is not None
    return normalize(spec)


def extract(image_path, oracle=False, effort="medium", second_read=True):
    """Returns {"spec", "evidence", "unreadable", "corrections", "source"}."""
    image_path = Path(image_path)
    if oracle or not meta_api.online():
        gt = SPECS / f"{image_path.stem}.json"
        if not gt.exists():
            raise meta_api.APIError("offline and no ground-truth spec for this drawing")
        return {"spec": json.loads(gt.read_text()), "evidence": [], "unreadable": [], "corrections": [],
                "source": "oracle"}
    w, h = Image.open(image_path).size
    raw = meta_api.chat_json(SYSTEM, PROMPT.replace("{w}", str(w)).replace("{h}", str(h)), SCHEMA,
                             "part_spec", images=[image_path], effort=effort, tag="extract")
    first = json.loads(json.dumps(raw))
    corrections = reread(image_path, raw) if second_read else []
    return {"spec": to_spec(raw), "evidence": raw.get("evidence", []), "unreadable": raw.get("unreadable", []),
            "corrections": corrections, "source": meta_api.SPARK, "raw": first}
