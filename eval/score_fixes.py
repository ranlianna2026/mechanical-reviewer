"""Compare fix proposers on fix quality, independent of extraction.

Runs the fix loop on ground-truth specs (eval/specs) with each proposer and
grades every issue's fix (mechreview/quality.py): pass rate, grade mix, margin past
the limit, design-intent questions, and engineering improvement.

Usage:  .venv/bin/python eval/score_fixes.py [--proposers reference spark_v1 spark_v2]
"""

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mechreview.agents import propose_fix, propose_fix_v2  # noqa: E402
from mechreview.fix import fix_loop, reference_fix  # noqa: E402
from mechreview.quality import score_all  # noqa: E402
from mechreview.rules import check  # noqa: E402

PROPOSERS = {"reference": reference_fix, "spark_v1": propose_fix, "spark_v2": propose_fix_v2}


def run_one(name, did):
    part = json.loads((ROOT / "eval" / "specs" / f"{did}.json").read_text())
    issues = [i["type"] for i in check(part)]
    if not issues:
        return did, None
    res = fix_loop(part, propose=PROPOSERS[name])
    props = [p for r in res["rounds"] for p in r["proposals"]]
    return did, {"passed": res["passed"], "rounds": len(res["rounds"]),
                 "scores": score_all(part, res["fixed"], issues),
                 "changes": [{"issue": p["issue"], "applied": p["applied"], "assumption": p.get("assumption"),
                              "alternatives": len(p.get("alternatives") or [])} for p in props if p["accepted"]]}


def summarize(rows):
    scores = [s for r in rows.values() if r for s in r["scores"]]
    margins = [s["margin_pct"] for s in scores if s["margin_pct"] is not None]
    impr = [s["engineering"]["improvement_x"] for s in scores if s.get("engineering", {}).get("improvement_x")]
    grades = {g: sum(s["grade"] == g for s in scores) for g in ("good", "acceptable", "poor")}
    return {"fixes": len(scores),
            "drawings_passed": f"{sum(r['passed'] for r in rows.values() if r)}/{sum(1 for r in rows.values() if r)}",
            "grades": grades,
            "good_pct": round(100 * grades["good"] / len(scores)) if scores else None,
            "at_limit_lt10pct": sum(m < 10 for m in margins),
            "median_margin_pct": sorted(margins)[len(margins) // 2] if margins else None,
            "intent_questions": sum(len(s["intent_questions"]) for s in scores),
            "geo_mean_engineering_improvement_x": round(math.exp(sum(map(math.log, impr)) / len(impr)), 1) if impr else None,
            "mean_rounds": round(sum(r["rounds"] for r in rows.values() if r) / max(1, sum(1 for r in rows.values() if r)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposers", nargs="*", default=list(PROPOSERS))
    ap.add_argument("--ids", nargs="*")
    args = ap.parse_args()
    ids = args.ids or sorted(p.stem for p in (ROOT / "eval" / "specs").glob("D*.json"))
    out = ROOT / "out" / "eval" / "fix_quality"
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in args.proposers:
        with ThreadPoolExecutor(6) as pool:
            rows = dict(pool.map(lambda d: run_one(name, d), ids))
        report[name] = {"summary": summarize(rows), "rows": rows}
        print(name, json.dumps(report[name]["summary"]), flush=True)
    (out / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
