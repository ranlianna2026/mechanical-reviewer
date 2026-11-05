"""Run the auto-fix loop on eval specs and render before/after drawings.

Uses the deterministic reference fixer, so it runs without API credits.
Swap `propose=` for the Muse Spark proposer once it exists.

Usage:  .venv/bin/python scripts/fix_demo.py [D05 D18 ...]   (default: all)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mechreview.drawing import overlay, render  # noqa: E402
from mechreview.fix import fix_loop  # noqa: E402

SPECS = ROOT / "eval" / "specs"
OUT = ROOT / "out" / "fix_demo"
RED, GREEN = (220, 30, 30), (20, 150, 60)


def run(did):
    part = json.loads((SPECS / f"{did}.json").read_text())
    result = fix_loop(part)
    d = OUT / did
    d.mkdir(parents=True, exist_ok=True)

    _, before_regions = render(part, d / "before.png")
    flagged = [p["issue"] for r in result["rounds"] for p in r["proposals"]]
    overlay(d / "before.png", [(t, before_regions[t], RED) for t in flagged], d / "before_marked.png")

    _, after_regions = render(result["fixed"], d / "after.png")
    fixed_types = [p["issue"] for r in result["rounds"] for p in r["proposals"] if p["accepted"]]
    overlay(d / "after.png", [(f"fixed: {t}", after_regions[t], GREEN)
                              for t in fixed_types if t in after_regions], d / "after_marked.png")

    (d / "report.json").write_text(json.dumps(
        {"id": did, "passed": result["passed"], "rounds": result["rounds"],
         "rev": f"{part.get('rev', 'A')} -> {result['fixed'].get('rev')}"}, indent=2, ensure_ascii=False))
    return did, result


def main():
    ids = sys.argv[1:] or sorted(p.stem for p in SPECS.glob("D*.json"))
    passed = 0
    for did in ids:
        did, res = run(did)
        n = sum(len(r["issues_before"]) for r in res["rounds"][:1])
        passed += res["passed"]
        changes = [f"{c['field']} {c['from']}->{c['to']}" for r in res["rounds"]
                   for p in r["proposals"] if p["accepted"] for c in p["applied"]]
        print(f"{did}: {n} issue(s) -> {'PASS' if res['passed'] else 'FAIL'}  {'; '.join(changes)}")
    print(f"\n{passed}/{len(ids)} drawings pass after auto-fix. Output: {OUT}")


if __name__ == "__main__":
    main()
