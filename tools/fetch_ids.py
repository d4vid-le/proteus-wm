#!/usr/bin/env python3
"""Fetch abstracts for a list of arXiv IDs with patient retry on 429."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import re, sys, time, urllib.request, urllib.parse

IDS = sys.argv[1].split(",") if len(sys.argv) > 1 else []
BATCH_SIZE = 18

def fetch_ids(id_list, tries=10):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"id_list": id_list, "max_results": len(id_list)+5})
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-assistant/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            msg = str(e)
            wait = 45 + attempt*15
            print(f"  [attempt {attempt+1} err: {msg[:60]} waiting {wait}s]", flush=True)
            time.sleep(wait)
    return ""

def parse(data):
    out = []
    for e in re.findall(r"<entry>.*?</entry>", data, re.S):
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        i = re.search(r"<id>(.*?)</id>", e, re.S)
        p = re.search(r"<published>(.*?)</published>", e, re.S)
        s = re.search(r"<summary>(.*?)</summary>", e, re.S)
        out.append({k: " ".join(v.group(1).split()) if v else "" for k, v in
                    [("title", t), ("id", i), ("year", p), ("sum", s)]})
    return out

for start in range(0, len(IDS), BATCH_SIZE):
    chunk = ",".join(IDS[start:start+BATCH_SIZE])
    print(f"\n+++++ CHUNK {chunk} +++++", flush=True)
    data = fetch_ids(chunk)
    if not data:
        print("  FAILED after retries", flush=True); continue
    for r in parse(data):
        print(f"\n### {r['year']} | {r['id'].split('/abs/')[-1]} | {r['title']}", flush=True)
        print(f"ABSTRACT: {r['sum']}", flush=True)
    time.sleep(6)
print("\nALL DONE", flush=True)
