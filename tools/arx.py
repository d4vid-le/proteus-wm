#!/usr/bin/env python3
"""Query arXiv API and print id/title/abstract for each entry."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import sys, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET

NS = {'a': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

def query(search_query, max_results=15):
    base = "https://export.arxiv.org/api/query?"
    params = urllib.parse.urlencode({
        'search_query': search_query,
        'start': 0,
        'max_results': max_results,
        'sortBy': 'relevance',
    })
    url = base + params
    for attempt in range(6):
        req = urllib.request.Request(url, headers={'User-Agent': 'research-litscan/1.0'})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(4 * (attempt + 1))
                continue
            raise
    else:
        raise RuntimeError("persistent 429 rate-limit for: " + search_query)
    root = ET.fromstring(data)
    out = []
    for e in root.findall('a:entry', NS):
        eid = e.find('a:id', NS).text.strip()
        title = ' '.join(e.find('a:title', NS).text.split())
        summ = ' '.join(e.find('a:summary', NS).text.split())
        pub = e.find('a:published', NS).text[:4]
        out.append((eid, pub, title, summ))
    return out

def main():
    query_str = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    for eid, pub, title, summ in query(query_str, n):
        print("=" * 90)
        print(f"ID: {eid}")
        print(f"YEAR: {pub}")
        print(f"TITLE: {title}")
        print(f"ABS: {summ}")
    print("=" * 90)

if __name__ == '__main__':
    main()
