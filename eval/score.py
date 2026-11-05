"""Score the pipeline against the answer key.

Metrics
- extraction: share of numeric spec fields read exactly (vs eval/specs)
- detection: per-type recall, overall precision (after verify)
- localization: for each true positive, best IoU of any highlight vs any answer-key region
- fix: share of drawings whose auto-fix passes; final verify clean rate (online only)

Usage:
  .venv/bin/python eval/score.py                 # online if MODEL_API_KEY is set
  .venv/bin/python eval/score.py --oracle        # offline sanity run (perfect extraction)
  .venv/bin/python eval/score.py --ids D01 D05 --no-fix --no-sam
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mechreview import meta_api  # noqa: E402
from mechreview.pipeline import review  # noqa: E402
from mechreview.rules import RULES  # noqa: E402

EVAL = ROOT / "eval"
NUMERIC = ["L", "W", "T", "pocket.x0", "pocket.width", "pocket.height", "pocket.depth", "pocket.wall",
           "pocket.corner_r", "blind_hole.D", "blind_hole.depth", "slot.width", "slot.depth",
           "mount_holes.D", "mount_holes.inset"]
FLAGS = ["tight_tol_on", "general_tol", "blind_hole", "slot"]


def _get(d, path):
    for k in path.split("."):
        if d is None:
            return None
        d = d.get(k)
    return d


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def extraction_score(pred, gt):
    ok = total = 0
    for f in NUMERIC:
        g = _get(gt, f)
        if g is None:
            continue
        total += 1
        p = _get(pred, f)
        ok += p is not None and abs(float(p) - float(g)) < 1e-6
    for f in FLAGS:
        total += 1
        g, p = gt.get(f), pred.get(f)
        ok += (g is None) == (p is None) if f in ("blind_hole", "slot") else g == p
    return ok, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--no-fix", action="store_true")
    ap.add_argument("--no-sam", action="store_true")
    ap.add_argument("--tag", default=None, help="output folder name under out/eval/")
    args = ap.parse_args()

    key = {d["id"]: d for d in json.loads((EVAL / "answer_key.json").read_text())}
    ids = args.ids or sorted(key)
    mode = "oracle" if args.oracle or not meta_api.online() else meta_api.SPARK
    out = ROOT / "out" / "eval" / (args.tag or mode)
    per_type = {t: {"tp": 0, "fn": 0, "fp": 0} for t in RULES}
    ext_ok = ext_total = 0
    ious, fix_pass, fix_n, final_clean, final_n, rejected_true, rejected_false = [], 0, 0, 0, 0, 0, 0
    rows = []

    for did in ids:
        gt = key[did]
        rep = review(EVAL / gt["image"], out / did, oracle=mode == "oracle",
                     use_sam=not args.no_sam, do_fix=not args.no_fix)
        gt_spec = json.loads((EVAL / "specs" / f"{did}.json").read_text())
        ok, tot = extraction_score(rep["spec"], gt_spec)
        ext_ok, ext_total = ext_ok + ok, ext_total + tot

        gt_types = {i["type"]: i for i in gt["issues"]}
        pred_types = {i["type"]: i for i in rep["issues"]}
        for t in RULES:
            if t in gt_types and t in pred_types:
                per_type[t]["tp"] += 1
                boxes = [h["bbox"] for h in pred_types[t].get("highlights", [])
                         if h.get("source") != "ground-truth"]
                regions = [r["bbox"] for r in gt_types[t]["regions_px"]]
                if boxes:
                    ious.append(max(iou(b, r) for b in boxes for r in regions))
            elif t in gt_types:
                per_type[t]["fn"] += 1
            elif t in pred_types:
                per_type[t]["fp"] += 1
        for r in rep["rejected_by_verify"]:
            if r["type"] in gt_types:
                rejected_true += 1
            else:
                rejected_false += 1
        if "fix" in rep:
            fix_n += 1
            fix_pass += rep["fix"]["passed"]
        if "final_verify" in rep:
            final_n += 1
            final_clean += rep["final_verify"]["clean"]
        rows.append({"id": did, "gt": sorted(gt_types), "pred": sorted(pred_types),
                     "extraction": f"{ok}/{tot}", "timings_s": rep["timings_s"]})
        print(f"{did}: gt={sorted(gt_types)} pred={sorted(pred_types)} extraction {ok}/{tot}", flush=True)

    tp = sum(v["tp"] for v in per_type.values())
    fp = sum(v["fp"] for v in per_type.values())
    fn = sum(v["fn"] for v in per_type.values())
    summary = {
        "mode": mode, "drawings": len(ids),
        "extraction_field_accuracy": round(ext_ok / ext_total, 3) if ext_total else None,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "per_type_recall": {t: (round(v["tp"] / (v["tp"] + v["fn"]), 2) if v["tp"] + v["fn"] else None)
                            for t, v in per_type.items()},
        "false_positives": fp,
        "verify_dropped_false_positives": rejected_false, "verify_dropped_true_issues": rejected_true,
        "localization_mean_best_iou": round(sum(ious) / len(ious), 3) if ious else None,
        "fix_pass_rate": round(fix_pass / fix_n, 3) if fix_n else None,
        "final_verify_clean_rate": round(final_clean / final_n, 3) if final_n else None,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"summary": summary, "per_type": per_type, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
