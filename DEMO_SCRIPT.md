# Demo video script — Mechanical Reviewer (2:37)

Narration for the submitted demo video. Footage is a live run (`Live · muse-spark-1.3 + sam-3.1`):
sample **D05 · Fixture Plate** in *Review drawing*, and photo **P04** (bolt fatigue fracture) in
*Diagnose broken part*. Slow stretches (the ~3-minute fix loop) are time-lapsed to fit the narration.

| Chapter | Narration |
|---|---|
| Title card | Every recall starts as a line on a drawing. |
| DETECT | A wall too thin. A corner too sharp. A hole too deep. Mechanical Reviewer finds them, fixes them, and proves the fix. Here's a live run on a fixture plate. |
| DETECT | Muse Spark reads the drawing into a structured spec. Deterministic manufacturing rules check it. The model never gets the last word. Every finding is re-read on a zoomed crop, and SAM 3.1 pins each issue on the drawing. |
| FIX | Then it fixes the design. Muse Spark proposes edits. Code applies them and re-runs every rule. If a fix fails, the reason goes back to the model, and it tries again. |
| VERIFY | And we don't trust our own fix. The new drawing is re-rendered and read again, from scratch. |
| RESULT | Red is what was wrong. Green is what changed. Revision A becomes revision B, and the re-read comes back clean. |
| ISSUES | Three real issues. A wall just 0.3 mm thick. Sharp pocket corners no end mill can cut. And a hole drilled 12x deeper than its diameter. |
| EVIDENCE | Each one is tied to how it fails in the field, backed by real NHTSA recalls with the same failure mechanism. |
| IMPACT | And every fix is quantified: wall deflection 81x lower, corner stress 3x lower, drill deflection 53x lower. |
| Card | But what if the part has already failed? |
| DIAGNOSE | Upload a photo of the fracture. The agent segments it with SAM 3.1, zooms into the fracture surface, reads it like a fractographer, and checks field failures with the same pattern. |
| DIAGNOSE | Verdict: fatigue, high confidence. It finds the origin at the thread root, and links it back to the sharp-corner design rule. The fix: a larger root radius and rolled threads. |
| BENCHMARK | And we measure it. 25 drawings, 30 planted issues, 77 hard negatives. Latest live run: 100% recall, 100% precision, 0 false positives, and 95% of designs fixed and re-verified clean. |
| End card | Mechanical Reviewer. From drawing, to failure, to verified fix. |

Numbers in the review chapters come from the recorded D05 run; benchmark numbers are run `live-full-v2`.
