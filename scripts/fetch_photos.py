"""Download the labeled fracture-photo set (eval/photos/manifest.json) into out/photos/.

Photos are Wikimedia Commons files under public-domain or Creative Commons licenses
(attribution in the manifest). They are not committed; this script fetches them.

Usage:  .venv/bin/python scripts/fetch_photos.py
"""

import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "photos"
UA = {"User-Agent": "dfm-hackathon-eval/0.1 (fracture photo benchmark)"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for item in json.loads((ROOT / "eval" / "photos" / "manifest.json").read_text()):
        dest = OUT / f"{item['id']}.jpg"
        if dest.exists() and dest.stat().st_size > 10_000:
            continue
        for attempt in range(4):
            r = httpx.get(item["download"], headers=UA, timeout=60, follow_redirects=True)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
                dest.write_bytes(r.content)
                print(item["id"], len(r.content) // 1024, "KB", item["title"][:50])
                break
            time.sleep(5 * (attempt + 1))
        else:
            print(item["id"], "FAILED", r.status_code)
        time.sleep(3)  # be polite to Wikimedia


if __name__ == "__main__":
    main()
