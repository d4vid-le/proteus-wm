#!/usr/bin/env python3
"""interaction_daemon.py — Continuously record user interactions + feed the JEPA.

Runs the Swift input tap (clicks, typing, mouse, scroll) continuously, appends
to data/interactions.jsonl, and periodically summarizes the stream so the JEPA
can learn from real user behavior. This is the "feed the model all my data"
pipeline: every click/keystroke with app+window context becomes a learning
signal.

PRIVACY / SECURITY:
  - Keycodes are NOT recorded by default (only that a key was pressed). Pass
    --log-keycodes to enable them; even then, keystrokes in secure text fields
    (password fields, detected via the macOS Accessibility API) are redacted.
  - The interaction file is bounded: it is truncated to the last --max-lines
    lines after every chunk, so it never accumulates into a permanent log.
  - A redaction pass masks obvious secret patterns (api keys, tokens, long hex)
    as defense-in-depth for anything that slips through.

Usage:
  python3 interaction_daemon.py               # run forever (bounded chunks)
  python3 interaction_daemon.py --seconds 30  # record 30s then summarize
  python3 interaction_daemon.py --summary     # summarize existing interactions
"""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

INPUT_TAP = "/tmp/input_tap"  # compiled Swift tap
OUT = ROOT / "data" / "interactions.jsonl"
SOURCE = ROOT / "input_tap.swift"

# Defense-in-depth: mask obvious secret patterns in the raw JSONL lines.
SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|secret|token|passwd|password|pwd|authorization|bearer)"
    r"\s*[=:]\s*[\"']?[^\s\"']+|"
    r"\b(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{16,}|gho_[A-Za-z0-9]{16,}|"
    r"AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"[0-9a-fA-F]{32,64})\b"
)


def redact_line(line):
    def _mask(m):
        s = m.group(0)
        return s[:4] + "…[REDACTED]"
    return SECRET_RE.sub(_mask, line)


def compile_tap():
    if pathlib.Path(INPUT_TAP).exists():
        return True
    r = subprocess.run(["swiftc", "-O", str(SOURCE), "-o", INPUT_TAP],
                       capture_output=True, text=True)
    return r.returncode == 0


def summarize():
    from collections import Counter
    kinds = Counter()
    apps = Counter()
    n = 0
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                e = json.loads(line)
                kinds[e.get("kind", "?")] += 1
                apps[e.get("app", "?")] += 1
                n += 1
            except Exception:
                pass
    return {"total": n, "by_kind": dict(kinds), "top_apps": apps.most_common(5)}


def truncate(max_lines=20000):
    """Keep the interaction file bounded: redact any leaked secrets and drop
    old lines so the file never accumulates into a permanent log."""
    if not OUT.exists():
        return
    lines = OUT.read_text().splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    redacted = [redact_line(l) for l in lines]
    OUT.write_text("\n".join(redacted) + ("\n" if redacted else ""))


def run(seconds=0, chunk=300, max_lines=20000, log_keycodes=False):
    if not compile_tap():
        print("ERROR: couldn't compile input tap")
        return
    print(f"input tap: {INPUT_TAP}")
    base = [INPUT_TAP, str(OUT)]
    if log_keycodes:
        base += ["--keycodes"]
    print(f"recording interactions -> {OUT}  "
          f"(keycodes={'ON' if log_keycodes else 'OFF — secure by default'})")
    try:
        if seconds:
            subprocess.run(base + ["--seconds", str(seconds)])
        else:
            # run forever in bounded chunks so the file never accumulates
            while True:
                subprocess.run(base + ["--seconds", str(chunk)])
                truncate(max_lines)
    except KeyboardInterrupt:
        pass
    truncate(max_lines)
    s = summarize()
    print(f"\n=== INTERACTION SUMMARY ===")
    print(f"  total: {s['total']} events")
    for k, v in sorted(s["by_kind"].items(), key=lambda x: -x[1]):
        print(f"    {k}: {v}")
    print(f"  top apps: {s['top_apps']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0, help="record N seconds (0=forever)")
    ap.add_argument("--chunk", type=float, default=300,
                    help="seconds per recording chunk before the file is truncated (default 300)")
    ap.add_argument("--max-lines", type=int, default=20000,
                    help="keep at most this many interaction lines (default 20000)")
    ap.add_argument("--log-keycodes", action="store_true",
                    help="record keycodes (OFF by default; secure fields are still redacted)")
    ap.add_argument("--summary", action="store_true", help="summarize existing data")
    args = ap.parse_args()
    if args.summary:
        s = summarize()
        print(f"Interactions: {s['total']}")
        for k, v in sorted(s["by_kind"].items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
        print(f"  top apps: {s['top_apps']}")
        return
    run(args.seconds, chunk=args.chunk, max_lines=args.max_lines,
        log_keycodes=args.log_keycodes)


if __name__ == "__main__":
    main()
