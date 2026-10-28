import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import sys, time, urllib.parse, urllib.request, json
def s2(query,limit=15):
    q=urllib.parse.quote(query)
    url="https://api.semanticscholar.org/graph/v1/paper/search?query=%s&limit=%d&fields=title,year,abstract,externalIds,venue"%(q,limit)
    for _ in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'lit/1.0'})
            data=urllib.request.urlopen(req,timeout=25).read()
            j=json.loads(data)
            if 'data' not in j: raise Exception('no data')
            return j['data']
        except Exception as e:
            time.sleep(6)
    return None
for q in sys.argv[1:]:
    r=s2(q)
    print("\n### S2 QUERY:",q,flush=True)
    if r is None:
        print("  RATE_LIMIT/fail",flush=True); continue
    for p in r:
        ex=p.get('externalIds') or {}
        doi=ex.get('DOI','')
        arx=ex.get('ArXiv','')
        ab=(p.get('abstract') or '')[:180]
        print(f"  [{p.get('year')}] DOI:{doi} ARX:{arx} :: {p.get('title','')[:90]} :: ABS:{ab}",flush=True)
