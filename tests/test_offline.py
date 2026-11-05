"""Offline tests: run with  .venv/bin/python -m unittest discover tests"""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mechreview import meta_api  # noqa: E402
from mechreview.extract import to_spec  # noqa: E402
from mechreview.fix import apply_proposal, fix_loop  # noqa: E402
from mechreview.rules import check  # noqa: E402

SPECS = ROOT / "eval" / "specs"


def raw_from_spec(s):
    """What a perfect Muse Spark extraction would return for a spec."""
    pk = s["pocket"]
    return {"name": s["name"], "number": s["number"], "rev": s["rev"], "material": s["material"],
            "L": s["L"], "W": s["W"], "T": s["T"],
            "L_tolerance": "±0.005" if s["tight_tol_on"] == "L" else None,
            "W_tolerance": "±0.005" if s["tight_tol_on"] == "W" else ("±0.1" if s["neg_tol_W"] else None),
            "general_tolerance_text": "LINEAR X.X ±0.2 X.XX ±0.1" if s["general_tol"] else None,
            "pocket": {k: pk[k] for k in ("x0", "width", "height", "depth", "wall", "corner_r", "sharp_via_note")},
            "blind_hole": {k: s["blind_hole"][k] for k in ("D", "depth", "x", "y")} if s["blind_hole"] else None,
            "slot": {k: s["slot"][k] for k in ("width", "depth", "length", "x", "y")} if s["slot"] else None,
            "mount_holes": {"D": 6.6, "inset": 8.0, "count": 4}, "evidence": [], "unreadable": []}


class ExtractionPostprocess(unittest.TestCase):
    def test_perfect_raw_reproduces_issues(self):
        for p in sorted(SPECS.glob("D*.json")):
            gt = json.loads(p.read_text())
            pred = to_spec(raw_from_spec(gt))
            self.assertEqual({i["type"] for i in check(pred)}, {i["type"] for i in check(gt)}, p.stem)


class SamParser(unittest.TestCase):
    def test_box_grammar(self):
        delta = ("<0f>7<|box;x1=10;y1=20;x2=110;y2=60;w=1984;h=1403;c=0.91|><|mask;x=0;y=0;data=4,4,~abc|>,"
                 "12<|box;x1=5;y1=5;x2=9;y2=9;w=1984;h=1403|><|mask;x=0;y=0;data=2,2,~xy|>")
        lines = ['data: ' + json.dumps({"type": "response.output_text.delta", "delta": delta}),
                 'data: ' + json.dumps({"type": "response.completed"})]
        with mock.patch.object(meta_api, "_request", return_value=lines), \
             mock.patch.object(meta_api, "_log_usage"):
            boxes = meta_api.sam_boxes(ROOT / "eval" / "drawings" / "D01.png", "small circle")
        self.assertEqual([b["bbox"] for b in boxes], [[10, 20, 110, 60], [5, 5, 9, 9]])
        self.assertEqual(boxes[0]["conf"], 0.91)
        self.assertIsNone(boxes[1]["conf"])


class FixLoop(unittest.TestCase):
    def test_rejects_non_editable_fields(self):
        gt = json.loads((SPECS / "D05.json").read_text())
        _, applied, rejected = apply_proposal(gt, {"changes": [{"field": "blind_hole", "to": None}]})
        self.assertEqual(applied, [])
        self.assertEqual(rejected[0]["reason"], "field not editable")

    def test_feedback_reaches_proposer_on_retry(self):
        gt = json.loads((SPECS / "D04.json").read_text())  # thin wall
        seen = []

        def bad_then_good(part, issue, feedback=None):
            seen.append(feedback)
            to = 0.8 if feedback is None else 2.0  # first try still under the 1.0 mm limit
            return {"changes": [{"field": "pocket.wall", "to": to},
                                {"field": "pocket.height", "to": part["pocket"]["height"] - (to - part["pocket"]["wall"])}]}

        res = fix_loop(gt, propose=bad_then_good)
        self.assertTrue(res["passed"])
        self.assertIsNone(seen[0])
        self.assertIn("still violated", seen[1])


if __name__ == "__main__":
    unittest.main()
