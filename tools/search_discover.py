#!/usr/bin/env python3
"""Search arXiv by title/abstract with patient retry; print title+id."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import re, sys, time, urllib.request, urllib.parse

QUERIES = [
    "ti:\"continual learning\" AND all:\"LoRA\"",
    "all:\"continual learning\" AND all:\"adapters\"",
    "all:\"catastrophic forgetting\" AND all:\"LoRA\"",
    "all:\"gated\" AND all:\"adapter\" AND all:\"continual\"",
]

def fetch(q, max_results=20, tries=8):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": q, "max_results": max_results})
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-assistant/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            wait = 40 + attempt*12
            print(f"  [retry {attempt+1}: {str(e)[:50]} -> wait {wait}s]", flush=True)
            time.sleep(wait)
    return ""

def parse(data):
    out = []
    for e in re.findall(r"<entry>.*?</entry>", data, re.S):
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        i = re.search(r"<id>(.*?)</id>", e, re.S)
        p = re.search(r"<published>(.*?)</published>", e, re.S)
        out.append((p.group(1)[:4] if p else "", i.group(1).split('/abs/')[-1] if i else "",
                    " ".join(t.group(1).split()) if t else ""))
    return out

for q in QUERIES:
    print(f"\n##### SEARCH: {q} #####", flush=True)
    d = fetch(q)
    if not d:
        print("  FAILED", flush=True); continue
    res = parse(d)
    if not res: print("  (no matches)")
    for y, a, t in res:
        print(f"  {y} | {a} | {t}", flush=True)
    time.sleep(12)
print("\nDONE", flush=True)
