#!/usr/bin/env python3
"""arXiv API query driver with rate-limit backoff. Prints title + id."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import re, sys, time, urllib.parse, urllib.request

QUERIES = [
    # (label, search_query, max_results)
    ("residual adapter continual", 'all:"residual adapter" AND all:"continual learning"', 15),
    ("adapter fusion continual", 'all:"adapter fusion" AND all:"continual learning"', 15),
    ("task adapters incremental", 'all:"task adapters"', 20),
    ("LoRA continual learning", 'all:"continual learning" AND all:"low-rank adaptation"', 20),
    ("O-LoRA", 'all:"O-LoRA"', 10),
    ("InfLoRA", 'all:"InfLoRA"', 10),
    ("zero-init residual adapter", 'all:"zero-initialized" AND all:"adapter"', 15),
    ("LoRA catastrophic forgetting", 'all:"forgetting" AND all:"low-rank adaptation"', 20),
]

def fetch(search_query, max_results, tries=6):
    url = ("https://export.arxiv.org/api/query?search_query=" +
           urllib.parse.quote(search_query) + f"&max_results={max_results}")
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-assistant/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            msg = str(e)
            if "429" in msg or "Rate" in msg or attempt == tries-1:
                time.sleep(8 + attempt*4)
            else:
                # other error: wait then retry
                time.sleep(5)
    return ""

def parse(data):
    out = []
    for e in re.findall(r"<entry>.*?</entry>", data, re.S):
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        i = re.search(r"<id>(.*?)</id>", e, re.S)
        p = re.search(r"<published>(.*?)</published>", e, re.S)
        s = re.search(r"<summary>(.*?)</summary>", e, re.S)
        out.append({
            "title": " ".join(t.group(1).split()) if t else "",
            "id": i.group(1).strip() if i else "",
            "year": p.group(1)[:4] if p else "",
            "sum": " ".join(s.group(1).split()) if s else "",
        })
    return out

for label, q, mx in QUERIES:
    print(f"\n########## {label} ##########")
    print(f"  query: {q}")
    data = fetch(q, mx)
    if not data:
        print("  (failed/rate-limited)")
        time.sleep(10)
        continue
    res = parse(data)
    if not res:
        print("  (no entries)")
    for r in res:
        print(f"  {r['year']} | {r['id'].replace('http://arxiv.org/abs/','').replace('https://arxiv.org/abs/','')} | {r['title']}")
    time.sleep(9)
