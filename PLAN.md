# Plan — drawing ↔ failure loop

Meta Global AI Developer Hackathon · Indie track · Push the Frontier (manufacturing)
Submission due **Wed Nov 4, 9:00 PM ET** (public repo + 3-minute demo video).

**Pitch:** from a broken part to a fixed drawing. The agent reviews a drawing
for manufacturability, explains how each flaw fails in the field, fixes the
design and proves the fix passes. In reverse, it starts from a photo of a
broken part, finds the failure mode and traces it back to the feature on the
drawing that caused it.

## Pipeline

| Step | What | Model |
|---|---|---|
| 1. Extract | drawing image → structured part spec (same schema as `eval/specs/`) | Muse Spark (vision) |
| 2. Check | deterministic rules (`mechreview/rules.py`) + model reasoning for anything rules miss | Muse Spark + code |
| 3. Highlight | outline each issue on the drawing | SAM 3.1 |
| 4. Explain | issue → failure mode → similar real recalls (NHTSA) | Muse Spark + recall index |
| 5. Fix | propose changes → apply → re-render → re-check (`mechreview/fix.py`) | Muse Spark |
| 6. Verify | re-extract the *fixed image* and confirm it is clean | Muse Spark (vision) |
| Reverse | broken-part photo → fracture region → failure mode → design feature → step 5 | SAM 3.1 + Muse Spark |

## Status

- [x] Eval set: 25 drawings, 30 planted issues, 77 hard negatives, answer key with pixel regions
- [x] Shared render / rules / fix modules; checker reproduces the answer key exactly
- [x] Auto-fix loop with deterministic baseline fixer: 25/25 pass (`scripts/fix_demo.py`)
- [x] Meta Model API client (Muse Spark chat + JSON schema + vision; SAM 3.1 streamed boxes) — untested live until credits land
- [ ] Feasibility: Muse Spark extraction accuracy on eval drawings — **go/no-go**
- [x] Muse Spark fix proposer with retry feedback (live comparison vs baseline pending)
- [ ] SAM 3.1 highlighting, scored by IoU against `regions_px`
- [x] Scoring harness: extraction accuracy, per-type recall, precision, IoU, fix pass, re-verify
- [x] NHTSA recall index (1,067 mechanical recalls) + explain step
- [ ] Reverse mode: 2–3 real broken-part photos (taken during the event)
- [x] Web UI: sample/upload → step progress → marked-up review → failure evidence → fix → before/after
- [ ] README, demo video, submission

## Days

| Day | Date | Goal |
|---|---|---|
| 1 | Wed Oct 28 | Eval set + fix loop (done). API client as soon as credits land |
| 2 | Thu Oct 29 | Extraction feasibility on 25 drawings → go/no-go; scoring harness |
| 3 | Fri Oct 30 | Muse Spark checker + fix proposer; SAM 3.1 highlighting |
| 4 | Sat Oct 31 | Failure mapping with NHTSA evidence; verify-by-re-extraction |
| 5 | Sun Nov 1 | Reverse mode: broken-part photos → failure mode → feature |
| 6 | Mon Nov 2 | Web UI, end-to-end runs, numbers for the video |
| 7 | Tue Nov 3 | README, 3-min video, buffer; submit by Wed Nov 4 9 PM ET |

**Fallback:** if extraction fails the go/no-go on Day 2, switch to machine
observability on NASA C-MAPSS (CC0).

## Rules to keep in mind

- New work only: everything here is built Oct 26 – Nov 4. Don't reuse older code.
- Public repo; demo must make clear what was built during the event.
- Don't call the Meta API until the $150 credits show on Billing (earlier usage is billed to the card).
