#!/usr/bin/env python3
"""Discovery v2: focused on copy-init, function-preserving, dynamics/world-model CL, scalar/domain gates."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import re, sys, time, urllib.request, urllib.parse

QUERIES = [
    'all:"function-preserving" AND all:"continual"',
    'all:"zero-initialized" AND all:"gate" AND all:"continual"',
    'all:"world model" AND all:"continual learning"',
    'all:"dynamics model" AND all:"continual learning" AND all:"adaptation"',
    'all:"state-transition" AND all:"continual adaptation"',
    'all:"copy" AND all:"initialization" AND all:"residual"',
]

def fetch(q, max_results=15, tries=8):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": q, "max_results": max_results})
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-assistant/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            wait = 40 + attempt*12
            print(f"  [retry {attempt+1}: {str(e)[:45]} wait {wait}s]", flush=True)
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
    if not res: print("  (no matches)", flush=True)
    for y, a, t in res:
        print(f"  {y} | {a} | {t}", flush=True)
    time.sleep(12)
print("\nDONE", flush=True)
