"""Score reverse-mode diagnosis on the labeled fracture-photo set.

Usage:  .venv/bin/python scripts/fetch_photos.py && .venv/bin/python eval/score_photos.py
"""

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mechreview.diagnose import diagnose  # noqa: E402

PHOTOS, OUT = ROOT / "out" / "photos", ROOT / "out" / "eval" / "photos"


def run(item):
    photo = PHOTOS / f"{item['id']}.jpg"
    if not photo.exists():
        return item, None
    try:
        r = diagnose(photo, OUT / item["id"])
    except Exception as e:  # keep scoring the rest
        return item, {"error": f"{type(e).__name__}: {e}"}
    return item, r


def main():
    items = json.loads((ROOT / "eval" / "photos" / "manifest.json").read_text())
    if len(sys.argv) > 1:
        items = [i for i in items if i["id"] in sys.argv[1:]]
    rows, conf = [], Counter()
    with ThreadPoolExecutor(5) as pool:
        for item, r in pool.map(run, items):
            if r is None:
                print(item["id"], "missing photo")
                continue
            v = (r.get("verdict") or {}) if "error" not in r else {}
            pred = v.get("failure_mode", "error")
            ok = pred == item["label"]
            conf[(item["label"], pred)] += 1
            rows.append({"id": item["id"], "label": item["label"], "pred": pred, "ok": ok,
                         "confidence": v.get("confidence"), "steps": r.get("steps_used"),
                         "evidence": v.get("visual_evidence"), "error": r.get("error"), "title": item["title"]})
            print(f"{item['id']} {item['label']:18s} -> {pred:18s} {'OK ' if ok else 'XX '} {v.get('confidence')}  {item['title'][:40]}",
                  flush=True)
    n = len(rows)
    by_label = {}
    for lab in sorted({r["label"] for r in rows}):
        rs = [r for r in rows if r["label"] == lab]
        by_label[lab] = f"{sum(r['ok'] for r in rs)}/{len(rs)}"
    summary = {"photos": n, "accuracy": round(sum(r["ok"] for r in rows) / n, 3) if n else None,
               "by_label": by_label,
               "high_conf_accuracy": (lambda hs: f"{sum(r['ok'] for r in hs)}/{len(hs)}")(
                   [r for r in rows if r["confidence"] == "high"]),
               "confusions": {f"{a}->{b}": c for (a, b), c in conf.items() if a != b}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
