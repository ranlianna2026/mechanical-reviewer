"""End-to-end review of one drawing.

extract -> check -> verify -> highlight -> explain -> fix loop -> render fixed
-> final verify (re-extract the fixed image) -> report.json + images.

`progress(step, detail)` is called as steps run, for the web UI.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import meta_api
from . import engineering
from .agents import EVIDENCE_FOR, explain_issue, propose_fix_v2, verify_issue
from .drawing import overlay, render
from .extract import extract
from .fix import fix_loop
from .quality import score_all
from .rules import DEEP_HOLE, DEEP_SLOT, RULES, SHARP_CORNER, THIN_WALL, check

RED, GREEN, ORANGE = (220, 30, 30), (20, 150, 60), (230, 140, 20)
# Geometry boxes Muse Spark returns with its extraction.
GEOMETRY_FOR = {THIN_WALL: ["pocket_outline_top_view"], SHARP_CORNER: ["pocket_outline_top_view"],
                DEEP_HOLE: ["blind_hole_circle_top_view"], DEEP_SLOT: ["slot_outline_top_view"]}
# SAM 3.1 finds circles on line drawings reliably but not rectangles or slots (probed on
# D03/D05), so it is used for holes only; the blind hole is the smallest circle.
SAM_PROMPT = {DEEP_HOLE: "tiny circle"}


def _oracle_regions(image_path, issue_type):
    """Answer-key regions for eval drawings; only used in offline (oracle) mode."""
    key = Path(__file__).resolve().parents[1] / "eval" / "answer_key.json"
    for d in json.loads(key.read_text()):
        if d["id"] == Path(image_path).stem:
            return [{**r, "source": "ground-truth"} for i in d["issues"] if i["type"] == issue_type
                    for r in i["regions_px"]]
    return []


def _highlights(image_path, issue_type, evidence, use_sam, oracle=False):
    if oracle:
        return _oracle_regions(image_path, issue_type)
    fields = EVIDENCE_FOR[issue_type] + GEOMETRY_FOR.get(issue_type, [])
    regs = [{"view": "geometry" if e["field"].endswith("_top_view") else "evidence",
             "bbox": [int(v) for v in e["bbox"]], "source": "muse-spark"}
            for e in evidence if e["field"] in fields and len(e.get("bbox", [])) == 4]
    if use_sam and meta_api.online() and issue_type in SAM_PROMPT:
        try:
            boxes = meta_api.sam_boxes(image_path, SAM_PROMPT[issue_type], tag=f"sam:{issue_type}")
            boxes.sort(key=lambda b: (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]))
            regs += [{"view": "geometry", "bbox": [int(v) for v in b["bbox"]], "source": "sam-3.1",
                      "conf": b["conf"]} for b in boxes[:1]]
        except meta_api.APIError as e:
            regs.append({"view": "error", "bbox": None, "source": "sam-3.1", "error": str(e)[:200]})
    return [r for r in regs if r["bbox"]]


def review(image_path, out_dir, oracle=False, use_sam=True, do_fix=True, final_verify=True,
           progress=lambda step, detail="": None):
    image_path, out_dir = Path(image_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, timings = time.time(), {}

    def step(name):
        timings[name] = round(time.time() - t0, 1)

    progress("extract", "reading the drawing")
    ex = extract(image_path, oracle=oracle)
    spec, evidence = ex["spec"], ex["evidence"]
    step("extract")

    progress("check", "applying DFM rules")
    found = check(spec)
    step("check")

    # Per-issue model calls are independent, so run them concurrently.
    pool = ThreadPoolExecutor(max_workers=6)
    issues, rejected = [], []
    progress("verify", ", ".join(i["type"] for i in found))
    verdicts = list(pool.map(lambda i: verify_issue(image_path, i, evidence) if not oracle
                             else {"status": "skipped", "reason": "oracle"}, found))
    for iss, v in zip(found, verdicts):
        (rejected if v["status"] == "rejected" else issues).append({**iss, "verification": v})
    step("verify")

    progress("highlight", ", ".join(i["type"] for i in issues))
    hls = list(pool.map(lambda i: _highlights(image_path, i["type"], evidence, use_sam,
                                              oracle=ex["source"] == "oracle"), issues))
    for iss, h in zip(issues, hls):
        iss["highlights"] = h
    overlay(image_path, [(i["type"], i["highlights"], RED) for i in issues], out_dir / "before_marked.png")
    step("highlight")

    progress("explain", ", ".join(i["type"] for i in issues))
    for iss in issues:
        iss["rule"], iss["severity"] = RULES[iss["type"]]["rule"], RULES[iss["type"]]["severity"]
    for iss, e in zip(issues, pool.map(lambda i: explain_issue(spec, i), issues)):
        iss["explanation"] = e
    pool.shutdown()
    step("explain")

    report = {"image": str(image_path), "extraction_source": ex["source"], "spec": spec,
              "unreadable": ex["unreadable"], "second_read_corrections": ex.get("corrections", []),
              "issues": issues, "rejected_by_verify": rejected}

    if do_fix and issues:
        progress("fix", "proposing and checking fixes")
        # Fix only what survived verification; the loop re-checks the whole part.
        res = fix_loop(spec, propose=propose_fix_v2)
        report["fix"] = {"passed": res["passed"], "rounds": res["rounds"], "fixed_spec": res["fixed"],
                         "quality": score_all(spec, res["fixed"], [i["type"] for i in issues])}
        if any(i["type"] == SHARP_CORNER for i in issues):
            progress("fix", "stress estimate before/after")
            fb, fa = engineering.plane_stress(spec), engineering.plane_stress(res["fixed"])
            vmax = max(3.0, min(fb["corner_kt"], 12.0))
            engineering.stress_heatmap(spec, fb, out_dir / "stress_before.png", vmax=vmax,
                                       title=f"Before: pocket corners R{spec['pocket']['corner_r']:g}")
            engineering.stress_heatmap(res["fixed"], fa, out_dir / "stress_after.png", vmax=vmax,
                                       title=f"After: pocket corners R{res['fixed']['pocket']['corner_r']:g}")
        _, regions = render(res["fixed"], out_dir / "after.png")
        fixed = [p["issue"] for r in res["rounds"] for p in r["proposals"] if p["accepted"]]
        overlay(out_dir / "after.png", [(f"fixed: {t}", regions[t], GREEN) for t in fixed if t in regions],
                out_dir / "after_marked.png")
        step("fix")

        if final_verify and not oracle and meta_api.online():
            progress("final_verify", "re-reading the fixed drawing")
            ex2 = extract(out_dir / "after.png", second_read=False)
            remaining = check(ex2["spec"])
            report["final_verify"] = {"clean": not remaining, "remaining": remaining}
            step("final_verify")

    report["timings_s"] = timings
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    progress("done", "")
    return report
