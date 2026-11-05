"""Demo web app.

  .venv/bin/python -m uvicorn app.server:app --port 8000
  open http://127.0.0.1:8000

Runs reviews in a background thread and streams step progress by polling.
Without MODEL_API_KEY, sample drawings run with ground-truth extraction
(clearly labeled "offline") so the UI can be demoed before credits land.
"""

import json
import shutil
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mechreview import meta_api  # noqa: E402
from mechreview.diagnose import diagnose  # noqa: E402
from mechreview.pipeline import review  # noqa: E402

EVAL, OUT = ROOT / "eval", ROOT / "out"
JOBS_DIR = OUT / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD = 15 * 1024 * 1024
STEPS = ["extract", "check", "verify", "highlight", "explain", "fix", "final_verify"]

app = FastAPI(title="Mechanical Reviewer")
app.mount("/eval", StaticFiles(directory=EVAL), name="eval")
app.mount("/out", StaticFiles(directory=OUT), name="out")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")

jobs, lock = {}, threading.Lock()
pool = ThreadPoolExecutor(max_workers=2)


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/status")
def status():
    return {"online": meta_api.online(), "spark_model": meta_api.SPARK, "sam_model": meta_api.SAM}


@app.get("/api/samples")
def samples():
    key = json.loads((EVAL / "answer_key.json").read_text())
    return [{"id": d["id"], "name": d["part_name"], "material": d["material"],
             "image": f"/eval/{d['image']}", "planted": [i["type"] for i in d["issues"]]} for d in key]


@app.get("/api/eval")
def eval_results():
    res = []
    for p in sorted((OUT / "eval").glob("*/results.json")):
        data = json.loads(p.read_text())
        if "summary" in data:  # detection runs; fix-quality runs are served by /api/fix_quality
            res.append({"run": p.parent.name, **data["summary"]})
    return res


@app.get("/api/fix_quality")
def fix_quality():
    p = OUT / "eval" / "fix_quality" / "results.json"
    return {k: v["summary"] for k, v in json.loads(p.read_text()).items()} if p.exists() else {}


def _run(job_id, image, oracle):
    job = jobs[job_id]

    def progress(step, detail=""):
        with lock:
            job["step"], job["detail"] = step, detail
            if step not in job["done"] and step != "done":
                job["done"].append(step)

    try:
        job["report"] = review(image, JOBS_DIR / job_id, oracle=oracle, progress=progress)
        job["status"] = "done"
    except Exception as e:  # surfaced to the UI
        job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
        traceback.print_exc()
    job["seconds"] = round(time.time() - job["started"], 1)


@app.post("/api/review")
async def start_review(sample: str = Form(None), file: UploadFile = File(None)):
    job_id = uuid.uuid4().hex[:10]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    if sample:
        src = EVAL / "drawings" / f"{sample}.png"
        if not src.exists():
            raise HTTPException(404, "unknown sample")
        image = job_dir / f"{sample}.png"  # keep the stem so offline mode finds its spec
        shutil.copy(src, image)
    elif file:
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            raise HTTPException(400, "upload a PNG or JPG drawing")
        data = await file.read()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(400, "file too large (15 MB max)")
        if not meta_api.online():
            raise HTTPException(400, "uploads need the Meta Model API (set MODEL_API_KEY); try a sample")
        image = job_dir / f"upload{Path(file.filename).suffix.lower()}"
        image.write_bytes(data)
    else:
        raise HTTPException(400, "choose a sample or upload a drawing")
    oracle = not meta_api.online()
    jobs[job_id] = {"id": job_id, "status": "running", "step": "queued", "detail": "", "done": [],
                    "started": time.time(), "oracle": oracle, "input": f"/out/jobs/{job_id}/{image.name}"}
    pool.submit(_run, job_id, image, oracle)
    return {"job": job_id}


def _run_diag(job_id, photo, part):
    job = jobs[job_id]

    def progress(step, detail=""):
        with lock:
            job["step"], job["detail"] = step, detail
            job["done"].append({"action": step, "thought": detail})

    try:
        job["report"] = diagnose(photo, JOBS_DIR / job_id, part=part, progress=progress)
        job["status"] = "done"
    except Exception as e:
        job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
        traceback.print_exc()
    job["seconds"] = round(time.time() - job["started"], 1)


@app.post("/api/diagnose")
async def start_diagnose(file: UploadFile = File(...), drawing: str = Form(None)):
    if not meta_api.online():
        raise HTTPException(400, "diagnosis needs the Meta Model API (set MODEL_API_KEY)")
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        raise HTTPException(400, "upload a photo (PNG/JPG/WEBP)")
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(400, "file too large (15 MB max)")
    part = None
    if drawing:
        spec = EVAL / "specs" / f"{drawing}.json"
        if not spec.exists():
            raise HTTPException(404, "unknown drawing")
        part = json.loads(spec.read_text())
    job_id = uuid.uuid4().hex[:10]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    photo = job_dir / f"photo{Path(file.filename).suffix.lower()}"
    photo.write_bytes(data)
    jobs[job_id] = {"id": job_id, "kind": "diagnose", "status": "running", "step": "start", "detail": "",
                    "done": [], "started": time.time(), "input": f"/out/jobs/{job_id}/{photo.name}"}
    pool.submit(_run_diag, job_id, photo, part)
    return {"job": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    base = f"/out/jobs/{job_id}"
    files = {n: f"{base}/{n}.png" for n in ("before_marked", "after", "after_marked", "photo_marked",
                                                    "stress_before", "stress_after")
             if (JOBS_DIR / job_id / f"{n}.png").exists()}
    return {**{k: v for k, v in job.items() if k != "started"}, "files": files, "steps": STEPS}
