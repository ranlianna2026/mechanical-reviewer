"""Build data/recalls.jsonl: NHTSA recalls whose defect text describes the
mechanical failure modes our DFM issues lead to (fatigue cracks, thin walls,
drilled holes, tolerances, machining defects).

Source: NHTSA FLAT_RCL_POST_2010 (US government, public domain).
  curl -O https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_POST_2010.zip && unzip FLAT_RCL_POST_2010.zip

Usage:  .venv/bin/python scripts/build_recall_index.py path/to/FLAT_RCL_POST_2010.txt
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "recalls.jsonl"

# Field positions from https://static.nhtsa.gov/odi/ffdd/rcl/RCL.txt (1-based in the spec).
F = {"campno": 1, "make": 2, "model": 3, "year": 4, "component": 6, "mfr": 7,
     "units": 11, "report_date": 15, "defect": 19, "consequence": 20, "remedy": 21}

TAGS = {
    "fatigue_crack": r"\bfatigue\b|stress (concentration|riser)|crack(s|ed|ing)? (may|can|could) (form|develop|propagate|initiate)",
    "fracture": r"\bfractur|\bbreak(s|ing)? (due|from)|\bcrack(ed|ing|s)?\b",
    "thin_wall": r"(insufficient|inadequate|thin|reduced) (wall|material) thickness|thin[- ]wall",
    "drilled_hole": r"\b(drill|drilled|drilling)\b|\bhole(s)? (was|were|may|not)\b|\bmis-?drilled\b",
    "tolerance": r"out of (tolerance|specification)|dimension(al|s)? (out|not|outside)|tolerance stack|incorrect(ly)? (machined|dimension)",
    "machining": r"\bmachin(ed|ing)\b|\bburr(s)?\b|tool mark|surface finish|undercut",
}
# Mechanical parts only; drop airbag/software/label recalls.
COMPONENT_OK = re.compile(r"SUSPENSION|STEERING|STRUCTURE|ENGINE|POWER TRAIN|WHEELS|BRAKES|"
                          r"FUEL|EQUIPMENT|LATCHES|SEATS|TRAILER|HITCH|EXHAUST", re.I)


def main(src):
    seen, kept = set(), []
    with open(src, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) <= F["remedy"]:
                continue
            camp = cols[F["campno"]].strip()
            comp = cols[F["component"]].strip()
            if camp in seen or not COMPONENT_OK.search(comp):
                continue
            defect = cols[F["defect"]].strip()
            tags = [t for t, rx in TAGS.items() if re.search(rx, defect, re.I)]
            if not tags or tags == ["fracture"] and not re.search(r"\bcrack|fractur", defect, re.I):
                continue
            seen.add(camp)
            kept.append({"campaign": camp, "make": cols[F["make"]].strip(), "model": cols[F["model"]].strip(),
                         "year": cols[F["year"]].strip(), "component": comp, "tags": tags,
                         "report_date": cols[F["report_date"]].strip(),
                         "units": int(cols[F["units"]] or 0) if cols[F["units"]].strip().isdigit() else None,
                         "defect": defect[:1200], "consequence": cols[F["consequence"]].strip()[:600],
                         "remedy": cols[F["remedy"]].strip()[:600]})
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    counts = {t: sum(t in r["tags"] for r in kept) for t in TAGS}
    print(f"{len(kept)} recalls -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(counts)


if __name__ == "__main__":
    main(sys.argv[1])
