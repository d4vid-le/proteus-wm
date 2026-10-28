#!/usr/bin/env python3
"""Fetch specific arXiv IDs in one bulk id_list request."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import sys, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET

NS = {'a': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

def fetch(ids):
    base = "https://export.arxiv.org/api/query?"
    params = urllib.parse.urlencode({
        'id_list': ','.join(ids),
        'max_results': len(ids),
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
                time.sleep(6 * (attempt + 1))
                continue
            raise
    else:
        raise RuntimeError("persistent 429 for id_list")
    root = ET.fromstring(data)
    for e in root.findall('a:entry', NS):
        eid = e.find('a:id', NS).text.strip()
        title = ' '.join(e.find('a:title', NS).text.split())
        summ = ' '.join(e.find('a:summary', NS).text.split())
        pub = e.find('a:published', NS).text[:4]
        print("=" * 90)
        print(f"ID: {eid}")
        print(f"YEAR: {pub}")
        print(f"TITLE: {title}")
        print(f"ABS: {summ}")

if __name__ == '__main__':
    fetch(sys.argv[1:])
