import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import sys, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET, json

NS = {'a':'http://www.w3.org/2005/Atom'}
def arxiv(query, max_results=15, sleep=3.2):
    url = "https://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(query) + "&max_results=%d" % max_results
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'lit-search/1.0'})
            data = urllib.request.urlopen(req).read()
            if data.startswith(b'Rate') or b'Rate exceeded' in data[:20]:
                raise Exception('rate-exceeded')
            root = ET.fromstring(data)
            return root
        except Exception as e:
            time.sleep(8)
    return None

def parse(root):
    out=[]
    for e in root.findall('a:entry', NS):
        t=e.find('a:title',NS).text.strip().replace('\n',' ')
        idd=e.find('a:id',NS).text.strip()
        pub=e.find('a:published',NS).text[:10]
        summ=e.find('a:summary',NS).text.strip().replace('\n',' ')
        aid=idd.split('/abs/')[-1]
        out.append({"title":t,"id":aid,"year":pub[:4],"abstract":summ})
    return out

queries = sys.argv[1:]
res={}
for q in queries:
    r=arxiv(q)
    res[q]=parse(r) if r is not None else [{"title":"RATE_LIMIT_FAIL","id":"","year":"","abstract":""}]
    print("QUERY DONE: "+q, flush=True)
    time.sleep(1.2)
with open('litsearch/arxiv_raw.json','a') as f:
    json.dump(res,f,indent=2)
    f.write("\n")
# print compact
for q,items in res.items():
    print("\n### QUERY:",q)
    for it in items:
        print(f"  [{it['year']}] {it['id']} :: {it['title'][:110]}")
