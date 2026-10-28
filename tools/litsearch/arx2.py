import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import sys, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET
NS={'a':'http://www.w3.org/2005/Atom'}
def arxiv(query,max_results=12):
    url="https://export.arxiv.org/api/query?search_query="+urllib.parse.quote(query)+"&max_results=%d"%max_results
    for _ in range(2):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'lit/1.0'})
            data=urllib.request.urlopen(req,timeout=25).read()
            if data[:20].lower().startswith(b'rate'): raise Exception('rate')
            return ET.fromstring(data)
        except Exception:
            time.sleep(8)
    return None
def parse(root):
    out=[]
    for e in root.findall('a:entry',NS):
        t=e.find('a:title',NS).text.strip().replace('\n',' ')
        idd=e.find('a:id',NS).text.strip()
        pub=e.find('a:published',NS).text[:10]
        summ=e.find('a:summary',NS).text.strip().replace('\n',' ')
        out.append((pub[:4],idd.split('/abs/')[-1],t,summ))
    return out
qlist=sys.argv[1:]
for i,q in enumerate(qlist):
    if i>0: time.sleep(14)
    r=arxiv(q)
    print("\n### QUERY:",q,flush=True)
    if r is None: print("  RATE_LIMIT",flush=True); continue
    for y,i_,t,s in parse(r):
        print(f"  [{y}] {i_} :: {t[:100]} || {s[:220]}",flush=True)
