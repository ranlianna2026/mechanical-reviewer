"""Meta Model API client: Muse Spark (chat completions, vision, JSON schema)
and SAM 3.1 (segmentation via /v1/responses, streamed).

Docs: https://dev.meta.ai/docs/image-understanding, /docs/structured-output,
/docs/reasoning, /docs/media-segmentation, /docs/sam/reading-segmentation.

Set MODEL_API_KEY in the environment or in .env at the repo root. With no
key (or DFM_OFFLINE=1) `online()` is False and callers fall back to offline
behavior, so nothing is billed by accident.
"""

import base64
import io
import json
import os
import random
import re
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.environ.get("META_BASE_URL", "https://api.meta.ai/v1")
USAGE_LOG = ROOT / "out" / "usage.jsonl"


def _load_dotenv():
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
SPARK = os.environ.get("DFM_SPARK_MODEL", "muse-spark-1.3")
SAM = os.environ.get("DFM_SAM_MODEL", "sam-3.1")
RETRY_STATUS = {429, 500, 502, 503, 504}


def online():
    return bool(os.environ.get("MODEL_API_KEY")) and os.environ.get("DFM_OFFLINE") != "1"


class APIError(RuntimeError):
    pass


def image_part_url(img):
    """Path or PIL image -> data URL."""
    if hasattr(img, "save"):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data, mime = buf.getvalue(), "image/png"
    else:
        data = Path(img).read_bytes()
        mime = "image/png" if str(img).lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _log_usage(tag, model, usage, seconds):
    USAGE_LOG.parent.mkdir(exist_ok=True)
    with USAGE_LOG.open("a") as fh:
        fh.write(json.dumps({"ts": time.time(), "tag": tag, "model": model,
                             "seconds": round(seconds, 2), "usage": usage}) + "\n")


def _request(path, payload, stream=False, timeout=240):
    if not online():
        raise APIError("offline: set MODEL_API_KEY (and unset DFM_OFFLINE) to call the API")
    headers = {"Authorization": f"Bearer {os.environ['MODEL_API_KEY']}", "Content-Type": "application/json"}
    if stream:
        headers["Accept"] = "text/event-stream"
    delay = 0.5
    for _ in range(6):
        retry_after = None
        try:
            if stream:
                with httpx.stream("POST", BASE_URL + path, json=payload, headers=headers, timeout=timeout) as r:
                    if r.status_code not in RETRY_STATUS:
                        if r.status_code >= 400:
                            r.read()
                            raise APIError(f"{r.status_code}: {r.text[:800]}")
                        return list(r.iter_lines())
                    retry_after = r.headers.get("retry-after")
            else:
                r = httpx.post(BASE_URL + path, json=payload, headers=headers, timeout=timeout)
                if r.status_code not in RETRY_STATUS:
                    if r.status_code >= 400:
                        raise APIError(f"{r.status_code}: {r.text[:800]}")
                    return r.json()
                retry_after = r.headers.get("retry-after")
        except httpx.TransportError:
            pass
        time.sleep(float(retry_after) if retry_after else delay * (1 + random.random()))
        delay *= 2
    raise APIError(f"{path}: gave up after retries")


def chat_json(system, text, schema, schema_name, images=(), effort="medium", tag="chat", max_tokens=None):
    """One Muse Spark call with images and a JSON schema; returns the parsed object."""
    content = [{"type": "text", "text": text}]
    content += [{"type": "image_url", "image_url": {"url": image_part_url(i)}} for i in images]
    payload = {"model": SPARK,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
               "response_format": {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema}},
               "reasoning_effort": effort}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    return _complete(payload, tag)


def _complete(payload, tag, attempts=2):
    """POST a chat completion and parse its JSON content; retry once on an empty
    or unparseable message (seen occasionally on long multi-image turns)."""
    last = None
    for _ in range(attempts):
        t0 = time.time()
        resp = _request("/chat/completions", payload)
        _log_usage(tag, SPARK, resp.get("usage"), time.time() - t0)
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):  # content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        try:
            return json.loads(content)
        except (TypeError, json.JSONDecodeError):
            pass
        # Later turns of a multi-turn chat sometimes prefix the JSON with a sentence.
        start = content.find("{") if isinstance(content, str) else -1
        if start >= 0:
            try:
                return json.JSONDecoder().raw_decode(content[start:])[0]
            except json.JSONDecodeError:
                pass
        last = (f"finish_reason={choice.get('finish_reason')!r} refusal={msg.get('refusal')!r} "
                f"content={str(content)[:200]!r}")
    raise APIError(f"{tag}: no JSON in response ({last})")


def chat_json_messages(messages, schema, schema_name, effort="medium", tag="chat"):
    """Multi-turn variant: `messages` already in Chat Completions format."""
    payload = {"model": SPARK, "messages": messages, "reasoning_effort": effort,
               "response_format": {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema}}}
    return _complete(payload, tag)


def user_message(text, images=()):
    return {"role": "user", "content": [{"type": "text", "text": text}]
            + [{"type": "image_url", "image_url": {"url": image_part_url(i)}} for i in images]}


_BOX = re.compile(r"(\d+)<\|box;([^|]+)\|>")


def sam_boxes(image, prompt, tag="sam"):
    """Segment `prompt` (a short noun phrase) in an image with SAM 3.1.

    Returns [{"id", "bbox": [x1, y1, x2, y2], "conf"}] in source pixels.
    Masks are ignored; boxes are enough for highlighting and IoU scoring.
    """
    payload = {"model": SAM, "stream": True,
               "metadata": {"include_confidence": "true"},
               "input": [{"type": "message", "role": "user", "content": [
                   {"type": "input_text", "text": prompt},
                   {"type": "input_image", "image_url": image_part_url(image)}]}]}
    t0 = time.time()
    try:
        lines = _request("/responses", payload, stream=True)
    except APIError as e:
        if not str(e).startswith("400"):
            raise
        payload.pop("metadata")
        lines = _request("/responses", payload, stream=True)
    text = []
    for line in lines:
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "response.output_text.delta":
            text.append(ev.get("delta", ""))
    _log_usage(tag, SAM, None, time.time() - t0)
    out = []
    for oid, body in _BOX.findall("".join(text)):
        kv = dict(p.split("=", 1) for p in body.split(";") if "=" in p)
        try:
            box = [float(kv[k]) for k in ("x1", "y1", "x2", "y2")]
        except KeyError:
            continue
        out.append({"id": oid, "bbox": box, "conf": float(kv["c"]) if "c" in kv else None})
    return out
