"""Reverse mode: broken-part photo -> failure mode -> design feature -> fix.

A bounded, tool-using agent. Each turn Muse Spark picks one action:
  segment(prompt)  SAM 3.1 boxes for a noun phrase on the photo ("crack", "fracture surface")
  zoom(bbox)       a high-resolution crop is added to the conversation
  recalls(type)    analogous NHTSA recalls for a DFM issue type
  conclude(...)    structured verdict
It must conclude within MAX_STEPS. If a part spec (drawing) is given, the
verdict names the feature that caused the failure and, when that maps to a
DFM rule, the fix loop runs on it.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from . import meta_api
from .agents import propose_fix, search_recalls
from .fix import fix_loop
from .rules import RULES

MAX_STEPS = 6
MODES = ["fatigue", "ductile_overload", "brittle_overload", "corrosion_assisted", "wear", "unknown"]

ACTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["thought", "action", "prompt", "bbox", "issue_type", "verdict"],
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string", "enum": ["segment", "zoom", "recalls", "conclude"]},
        "prompt": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "bbox": {"anyOf": [{"type": "array", "items": {"type": "number"}}, {"type": "null"}]},
        "issue_type": {"anyOf": [{"type": "string", "enum": sorted(RULES)}, {"type": "null"}]},
        "verdict": {"anyOf": [{"type": "null"}, {
            "type": "object", "additionalProperties": False,
            "required": ["failure_mode", "confidence", "visual_evidence", "origin", "origin_bbox",
                         "design_cause", "linked_issue_type", "recommendation"],
            "properties": {
                "failure_mode": {"type": "string", "enum": MODES},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "visual_evidence": {"type": "array", "items": {"type": "string"}},
                "origin": {"type": "string"},
                "origin_bbox": {"anyOf": [{"type": "array", "items": {"type": "number"}}, {"type": "null"}]},
                "design_cause": {"type": "string"},
                "linked_issue_type": {"anyOf": [{"type": "string", "enum": sorted(RULES)}, {"type": "null"}]},
                "recommendation": {"type": "string"}}}]},
    }}

SYSTEM = f"""You are a failure-analysis engineer examining photos of a broken mechanical part.
Work like a fractographer: locate the fracture, find the origin, read the surface features
(beach marks / ratchet marks / smooth fatigue zone vs rough final-fracture zone = fatigue;
necking, shear lips, dull fibrous surface = ductile overload; flat, granular, chevrons =
brittle; pitting or deposits at the origin = corrosion-assisted). Say what you actually see;
if the photo cannot show something, lower your confidence rather than guess.

Each turn return ONE action:
- segment: set `prompt` to a short noun phrase for SAM 3.1 (e.g. "crack", "fracture surface")
- zoom: set `bbox` [x1, y1, x2, y2] in pixels of the ORIGINAL photo to inspect closely
- recalls: set `issue_type` to look up analogous field failures
- conclude: fill `verdict` (origin_bbox: [x1, y1, x2, y2] around the crack origin in ORIGINAL photo
  pixels, or null if it is not visible)
You have at most {MAX_STEPS} actions; conclude by the last one. Unused fields are null."""


def _mark(photo, boxes, path, origin=None):
    im = Image.open(photo).convert("RGB")
    dr = ImageDraw.Draw(im)
    lw = max(3, im.width // 300)
    for b in boxes:
        dr.rectangle(b["bbox"], outline=(220, 30, 30), width=lw)
    if origin and len(origin) == 4:
        x1, y1, x2, y2 = origin
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        r = max(x2 - x1, y2 - y1) / 2 + 3 * lw
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 230, 120), width=lw * 2)
        dr.text((cx + r + 4, cy - 10), "crack origin", fill=(0, 230, 120))
    im.save(path)


def diagnose(photo, out_dir, part=None, progress=lambda step, detail="": None):
    photo, out_dir = Path(photo), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    im = Image.open(photo).convert("RGB")
    ctx = f"Photo size {im.width}x{im.height} px."
    if part:
        ctx += ("\nThe part's drawing spec (mm) is below; if a feature on it explains the origin, name it "
                "and set linked_issue_type to the DFM rule it breaks.\n"
                + json.dumps({k: part[k] for k in ("name", "material", "L", "W", "T", "pocket",
                                                    "blind_hole", "slot")}, ensure_ascii=False))
    messages = [{"role": "system", "content": SYSTEM},
                meta_api.user_message(ctx + "\nHere is the broken part. Start your analysis.", [im])]
    trace, found_boxes, verdict = [], [], None

    for step in range(MAX_STEPS):
        last = step == MAX_STEPS - 1
        if last:
            messages.append(meta_api.user_message("Final step: you must conclude now."))
        act = meta_api.chat_json_messages(messages, ACTION_SCHEMA, "action", effort="medium",
                                          tag=f"diagnose:{step}")
        act = {k: act.get(k) for k in ACTION_SCHEMA["required"]}  # tolerate missing keys
        if act["verdict"]:
            act["verdict"] = {k: act["verdict"].get(k) for k in
                              ACTION_SCHEMA["properties"]["verdict"]["anyOf"][1]["required"]}
        act["thought"] = act["thought"] or ""
        messages.append({"role": "assistant", "content": json.dumps(act)})
        entry = {"step": step + 1, "thought": act["thought"], "action": act["action"]}
        progress(act["action"], act["thought"][:120])

        if act["action"] == "conclude" or last:
            verdict = act["verdict"]
            trace.append(entry)
            break
        if act["action"] == "segment" and act["prompt"]:
            boxes = meta_api.sam_boxes(im, act["prompt"], tag="diagnose:sam")
            found_boxes += boxes
            entry.update(prompt=act["prompt"], boxes=[b["bbox"] for b in boxes[:5]])
            messages.append(meta_api.user_message(
                f"SAM 3.1 found {len(boxes)} region(s) for '{act['prompt']}': "
                + json.dumps([{"bbox": [int(v) for v in b["bbox"]], "conf": b["conf"]} for b in boxes[:5]])))
        elif act["action"] == "zoom" and act["bbox"] and len(act["bbox"]) == 4:
            x1, y1, x2, y2 = [int(v) for v in act["bbox"]]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(im.width, x2), min(im.height, y2)
            if x2 - x1 > 8 and y2 - y1 > 8:
                crop = im.crop((x1, y1, x2, y2))
                if max(crop.size) < 1024:
                    k = 1024 / max(crop.size)
                    crop = crop.resize((int(crop.width * k), int(crop.height * k)), Image.LANCZOS)
                entry["bbox"] = [x1, y1, x2, y2]
                entry["crop"] = f"zoom_{step + 1}.png"
                crop.save(out_dir / entry["crop"])
                messages.append(meta_api.user_message(f"Zoomed crop of {entry['bbox']}:", [crop]))
            else:
                messages.append(meta_api.user_message("That box was empty; give a box inside the photo."))
        elif act["action"] == "recalls" and act["issue_type"]:
            rec = search_recalls(act["issue_type"])
            entry["issue_type"] = act["issue_type"]
            messages.append(meta_api.user_message("Analogous NHTSA recalls: " + json.dumps(rec, ensure_ascii=False)))
        else:
            messages.append(meta_api.user_message("That action was missing its argument; try again."))
        trace.append(entry)

    _mark(photo, found_boxes[:5], out_dir / "photo_marked.png", origin=(verdict or {}).get("origin_bbox"))
    result = {"photo": str(photo), "verdict": verdict, "trace": trace, "steps_used": len(trace)}

    issue_type = verdict and verdict.get("linked_issue_type")
    if part and issue_type:
        progress("fix", issue_type)
        res = fix_loop(part, propose=propose_fix)
        result["fix"] = {"passed": res["passed"], "rounds": res["rounds"], "fixed_spec": res["fixed"]}
    (out_dir / "diagnosis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    progress("done", "")
    return result
