#!/bin/bash
# arXiv API search helper: prints title + arXiv id per result.
# Usage: arx_search.sh "query text" [max_results]
Q="$1"
MAX="${2:-15}"
# percent-encode spaces as +, keep quotes around the phrase
ENC=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$Q")
URL="https://export.arxiv.org/api/query?search_query=all:${ENC}&max_results=${MAX}"
echo "=== QUERY: $Q ==="
curl -sSL "$URL" | python3 -c "
import sys,re
data=sys.stdin.read()
entries=re.findall(r'<entry>.*?</entry>', data, re.S)
for e in entries:
    t=re.search(r'<title>(.*?)</title>', e, re.S)
    i=re.search(r'<id>(.*?)</id>', e, re.S)
    p=re.search(r'<published>(.*?)</published>', e, re.S)
    title=' '.join(t.group(1).split()) if t else ''
    aid=i.group(1).strip() if i else ''
    yr=p.group(1)[:4] if p else ''
    print(f'  {yr} | {aid} | {title}')
"
