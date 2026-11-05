# DFM evaluation set

25 synthetic 2D engineering drawings of CNC-machined plate parts, each with a
top view, a front view (hidden features dashed), notes and a title block.
Every drawing has 0–3 **planted** design-for-manufacturing issues, and the
answer key records them exactly.

Built during the hackathon with `generate_eval_set.py` (deterministic, seed
`20261028`). Regenerate with:

```bash
.venv/bin/python eval/generate_eval_set.py
```

## Contents

| Path | What |
|---|---|
| `drawings/D01..D25.png` | Drawings (1984×1403 px, A3 at 120 dpi) — the model input |
| `answer_key.json` | Ground truth per drawing |
| `rules.json` | The six rules, thresholds, severity, failure-mode tags |
| `specs/D01..D25.json` | Full structured part spec per drawing (target output for extraction; input to the fix loop) |
| `qa_overlays/` | Same drawings with answer-key regions boxed in red, for humans only |

## Composition

- 5 clean drawings, 12 with one issue, 6 with two, 2 with three
- 30 planted issues, exactly 5 of each type
- 77 hard negatives: features near a rule but inside it (e.g. 2.5 mm wall,
  R2.0 corners, 3× D hole, ±0.1 on overall width), used to measure false positives

## Issue types

| Type | Rule | Planted | Hard negative |
|---|---|---|---|
| `thin_wall` | pocket-to-edge wall ≥ 1.0 mm | 0.3–0.6 mm | 2.0–6.0 mm |
| `sharp_internal_corner` | pocket corner R ≥ depth/3 | R0 (callout or note) | ≥ depth/3, min R1.0 |
| `deep_small_hole` | blind hole depth ≤ 6× D | 10–14× D | 2–3.5× D |
| `deep_narrow_slot` | slot depth ≤ 4× width | 6–8× width | 1–2.5× width |
| `overly_tight_tolerance` | non-functional dims not tighter than ±0.01 | ±0.005 on overall L or W | ±0.1 |
| `missing_general_tolerance` | title block states general tolerances | field blank | field filled |

Planted values sit well past each limit and hard negatives sit well inside, so
the key has no borderline judgment calls.

## Answer key schema (per drawing)

```jsonc
{
  "id": "D05", "image": "drawings/D05.png",
  "part_name": "...", "material": "ALUMINUM 6061-T6", "scale": "1.5:1",
  "plate": {"L": 150, "W": 85, "T": 28},
  "features": {"pocket": {...}, "blind_hole": {...} | null, "slot": {...} | null, "mount_holes": {...}},
  "general_tolerance_present": true,
  "is_clean": false,
  "issues": [{
    "issue_id": "D05-I1", "type": "thin_wall", "value": 0.3,
    "description": "wall 0.3 mm between pocket and top edge",
    "rule": "...", "severity": "high", "category": "machinability",
    "failure_modes": ["chatter_vibration", "..."],
    "regions_px": [{"view": "top", "bbox": [x0, y0, x1, y1]}, {"view": "dimension", "bbox": [...]}]
  }],
  "hard_negatives": [{"type": "deep_small_hole", "feature": "blind_hole", "value": 3.1, "note": "..."}]
}
```

`regions_px` bboxes are in image pixels, origin top-left. One issue can have
several regions (geometry in the top view, the front view, and the callout or
dimension text that states the value).

## Scoring (suggested)

Match a predicted issue to ground truth by `type` within the same drawing.
Report per-type recall, overall precision (false positives include flagged hard
negatives and anything on clean drawings), and optionally localization as IoU
between a predicted box and any `regions_px` box of the matched issue.

## License

Generated for this project; released under CC-BY-4.0. No third-party data.
