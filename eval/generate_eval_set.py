"""Generate the DFM evaluation set.

Synthetic 2D engineering drawings (top + front view, title block, notes) of
CNC-machined plate parts. Each drawing has zero to three planted DFM issues;
the answer key records every issue with its measured value, the rule it
breaks, expected failure modes, and pixel regions on the image.

Hard negatives (features close to, but inside, the limits) are included on
purpose so false positives can be measured too.

Usage:  .venv/bin/python eval/generate_eval_set.py
"""

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mechreview.drawing import overlay, render  # noqa: E402
from mechreview.rules import (DEEP_HOLE as H, DEEP_SLOT as S, MISSING_TOL as M, RULES,  # noqa: E402
                       SHARP_CORNER as C, THIN_WALL as W_, TIGHT_TOL as T, check)

OUT = Path(__file__).resolve().parent
SEED = 20261028

# 5 clean, 12 single, 6 double, 2 triple -> every issue type appears exactly 5 times.
SCHEDULE = (
    [()] * 5
    + [(C,), (C,), (W_,), (W_,), (H,), (H,), (S,), (S,), (T,), (T,), (M,), (M,)]
    + [(T, C), (T, W_), (T, H), (W_, S), (H, M), (S, M)]
    + [(C, W_, H), (S, M, C)]
)

PART_NAMES = ["SENSOR MOUNT PLATE", "MOTOR BRACKET BASE", "COVER PLATE", "MANIFOLD BLOCK",
              "FIXTURE PLATE", "ELECTRONICS LID", "CLAMP BLOCK", "ADAPTER PLATE",
              "VALVE BODY BLANK", "GEARBOX COVER", "CAMERA MOUNT", "PUMP BASE PLATE"]
MATERIALS = ["ALUMINUM 6061-T6", "ALUMINUM 7075-T6", "STAINLESS STEEL 304", "STEEL 1018"]


def r05(x):
    return round(x * 2) / 2


def design_part(rng, idx, planted):
    """Pick part geometry so the planted issues (and only those) are present."""
    L = rng.randrange(80, 151, 5)
    Wd = rng.randrange(55, 91, 5)
    part = {"L": L, "W": Wd, "T": rng.choice([10, 12, 15, 18, 20]), "material": rng.choice(MATERIALS),
            "name": PART_NAMES[idx % len(PART_NAMES)], "number": f"DFM-EVAL-D{idx + 1:02d}", "rev": "A"}
    hard_negs = []

    # Blind hole (lower band).
    if H in planted:
        D, ratio = rng.choice([1.5, 2.0, 2.5]), rng.uniform(10, 14)
    elif rng.random() < 0.7:
        D, ratio = float(rng.choice([3, 4, 5, 6])), rng.uniform(2, 3.5)
    else:
        D = None
    if D:
        depth = r05(D * ratio)
        part["blind_hole"] = {"D": D, "depth": depth, "x": r05(L * 0.3), "y": r05(Wd * 0.33),
                              "ratio": round(depth / D, 2)}
        part["T"] = max(part["T"], int(math.ceil(depth + 3)))
        if H not in planted:
            hard_negs.append({"type": H, "feature": "blind_hole", "value": round(depth / D, 2),
                              "note": "depth/diameter within 6x limit"})
    else:
        part["blind_hole"] = None

    # Slot (lower band, horizontal).
    if S in planted:
        sw = rng.choice([2.0, 2.5, 3.0])
        sd = r05(sw * rng.uniform(6, 8))
    elif rng.random() < 0.6:
        sw = float(rng.choice([6, 8, 10]))
        sd = r05(sw * rng.uniform(1, 2.5))
    else:
        sw = None
    if sw:
        part["slot"] = {"width": sw, "depth": sd, "length": float(min(30, int(L * 0.3))),
                        "x": r05(L * 0.65), "y": r05(Wd * 0.17), "ratio": round(sd / sw, 2)}
        part["T"] = max(part["T"], int(math.ceil(sd + 3)))
        if S not in planted:
            hard_negs.append({"type": S, "feature": "slot", "value": round(sd / sw, 2),
                              "note": "depth/width within 4x limit"})
    else:
        part["slot"] = None

    # Pocket (upper band) -- always present.
    T_ = part["T"]
    wall = rng.choice([0.3, 0.4, 0.5, 0.6]) if W_ in planted else r05(rng.uniform(2.0, 6.0))
    d = r05(rng.uniform(3, min(10, T_ - 3)))
    px0 = r05(rng.uniform(16, 22))
    px1 = L - r05(rng.uniform(16, 22))
    top = Wd - wall
    ph = r05(rng.uniform(10, top - (Wd / 2 + 2)))
    r = 0.0 if C in planted else max(1.0, math.ceil(d / 3 / 0.5) * 0.5)
    r = min(r, ph / 2 - 0.5)
    part["pocket"] = {"x0": px0, "x1": px1, "y0": round(top - ph, 1), "y1": top,
                      "width": px1 - px0, "height": ph, "depth": d, "corner_r": r, "wall": wall,
                      "sharp_via_note": C in planted and rng.random() < 0.5}
    if W_ not in planted:
        hard_negs.append({"type": W_, "feature": "pocket", "value": wall,
                          "note": "wall above 1.0 mm minimum"})
    if C not in planted:
        hard_negs.append({"type": C, "feature": "pocket", "value": r,
                          "note": f"corner radius >= depth/3 ({d}/3)"})

    part["mount_holes"] = {"D": 6.6, "inset": 8.0}
    part["tight_tol_on"] = rng.choice(["L", "W"]) if T in planted else None
    part["general_tol"] = M not in planted
    part["neg_tol_W"] = (T not in planted) and rng.random() < 0.5
    if part["neg_tol_W"]:
        hard_negs.append({"type": T, "feature": "overall_width", "value": "+/-0.1",
                          "note": "ordinary tolerance on overall width"})
    return part, hard_negs


def main():
    rng = random.Random(SEED)
    schedule = list(SCHEDULE)
    rng.shuffle(schedule)
    for sub in ("drawings", "qa_overlays", "specs"):
        (OUT / sub).mkdir(exist_ok=True)
    key = []
    for idx, planted in enumerate(schedule):
        did = f"D{idx + 1:02d}"
        part, hard_negs = design_part(rng, idx, set(planted))
        found = {i["type"] for i in check(part)}
        assert found == set(planted), f"{did}: checker found {found}, planted {set(planted)}"

        img = OUT / "drawings" / f"{did}.png"
        scale, regions = render(part, img, strict=True)
        issues = []
        for k, iss in enumerate(check(part), 1):
            r = RULES[iss["type"]]
            issues.append({"issue_id": f"{did}-I{k}", **iss, "rule": r["rule"], "severity": r["severity"],
                           "category": r["category"], "failure_modes": r["failure_modes"],
                           "regions_px": regions[iss["type"]]})
        features = {k: part[k] for k in ("pocket", "blind_hole", "slot", "mount_holes")}
        features["pocket"] = {k: v for k, v in features["pocket"].items() if k != "sharp_via_note"}
        key.append({"id": did, "image": f"drawings/{did}.png", "part_name": part["name"],
                    "part_number": part["number"], "material": part["material"], "units": "mm",
                    "scale": f"{scale:g}:1", "plate": {"L": part["L"], "W": part["W"], "T": part["T"]},
                    "features": features, "general_tolerance_present": part["general_tol"],
                    "is_clean": not planted, "issues": issues, "hard_negatives": hard_negs})
        (OUT / "specs" / f"{did}.json").write_text(json.dumps(part, indent=2))
        overlay(img, [(i["type"], i["regions_px"], (220, 30, 30)) for i in issues],
                OUT / "qa_overlays" / f"{did}.png")

    (OUT / "answer_key.json").write_text(json.dumps(key, indent=2, ensure_ascii=False))
    (OUT / "rules.json").write_text(json.dumps(RULES, indent=2))

    counts = {t: 0 for t in RULES}
    for d in key:
        for i in d["issues"]:
            counts[i["type"]] += 1
    print(f"{len(key)} drawings, {sum(counts.values())} issues, "
          f"{sum(d['is_clean'] for d in key)} clean, "
          f"{sum(len(d['hard_negatives']) for d in key)} hard negatives")
    for t, n in counts.items():
        print(f"  {t:28s} {n}")


if __name__ == "__main__":
    main()
