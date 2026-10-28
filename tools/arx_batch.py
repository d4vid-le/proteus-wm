#!/usr/bin/env python3
"""arXiv query driver v2: robust, slow-paced, unbuffered."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import re, sys, time, urllib.parse, urllib.request

BATCH = sys.argv[1] if len(sys.argv) > 1 else "1"

# (label, search_query)
QUERIES = {
 "1": [
   ("residual_adapter_CL", 'all:"residual adapter" AND all:"continual"'),
   ("adapter_fusion", 'all:"adapter fusion"'),
   ("task_adapters_incr", 'all:"task adapters" AND all:"incremental"'),
   ("LoRA_CL", 'all:"low-rank adaptation" AND all:"continual"'),
 ],
 "2": [
   ("O_LoRA", 'all:"low-rank" AND all:"orthogonal" AND all:"continual"'),
   ("InfLoRA", 'all:"incremental" AND all:"low-rank" AND all:"continual"'),
   ("zero_init_residual", 'all:"zero-initialized" AND all:"residual"'),
   ("mixture_of_adapters", 'all:"mixture-of-adapters" OR all:"mixture of adapters"'),
 ],
 "3": [
   ("adapter_forgetting", 'all:"forgetting" AND all:"adapters"'),
   ("peft_lifelong", 'all:"parameter-efficient" AND all:"lifelong"'),
   ("replay_free_adapter", 'all:"replay" AND all:"incremental" AND all:"adapter"'),
   ("clip_adapter_cl", 'all:"CLIP" AND all:"continual" AND all:"adapter"'),
 ],
}

def fetch(search_query, max_results=12, tries=5):
    url = ("https://export.arxiv.org/api/query?search_query=" +
           urllib.parse.quote(search_query) + f"&max_results={max_results}")
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-assistant/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            msg = str(e)
            time.sleep(6 + attempt*3)
    return ""

def parse(data):
    out = []
    for e in re.findall(r"<entry>.*?</entry>", data, re.S):
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        i = re.search(r"<id>(.*?)</id>", e, re.S)
        p = re.search(r"<published>(.*?)</published>", e, re.S)
        out.append({"title": " ".join(t.group(1).split()) if t else "",
                    "id": i.group(1).strip() if i else "",
                    "year": p.group(1)[:4] if p else ""})
    return out

for label, q in QUERIES[BATCH]:
    print(f"\n########## {label}: {q}", flush=True)
    data = fetch(q)
    if not data:
        print("  (failed/rate-limited)", flush=True); time.sleep(8); continue
    res = parse(data)
    if not res:
        print("  (no entries)", flush=True)
    for r in res:
        print(f"  {r['year']} | {r['id'].split('/abs/')[-1]} | {r['title']}", flush=True)
    print("  --done--", flush=True)
    time.sleep(7)
print("\nALL DONE", flush=True)
