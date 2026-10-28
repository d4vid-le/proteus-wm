#!/usr/bin/env python3
"""monitor.py — Comprehensive live tracker for proteus.

Tails the agent's journals and renders a real-time dashboard:
  - current context (app, windows, shell, z_vis vision energy)
  - recent actions + whether verified
  - learning events (gates added, learned, shifts detected)
  - surprise/energy stream (the reflex)
  - aggregated session stats

Usage:
  python -m proteus.eval.monitor              # live dashboard (follow)
  python -m proteus.eval.monitor --once       # one snapshot, then exit
  python -m proteus.eval.monitor --tail 20    # show last N events
"""
from proteus.paths import ROOT
import argparse
import json
import pathlib
import sys
import time

AGENT = ROOT / "data" / "autonomous_journal.jsonl"
TRACE = ROOT / "data" / "eris_trace.jsonl"
MEMORY = ROOT / "data" / "eris_memory.jsonl"
TELEM = ROOT / "data" / "telemetry.jsonl"
INTERACTIONS = ROOT / "data" / "interactions.jsonl"


def read_journal(path, offset=0):
    """Read new lines from a JSONL journal since offset. Returns (lines, new_offset)."""
    if not path.exists():
        return [], offset
    lines = path.read_text().splitlines()
    return lines[offset:], len(lines)


def fmt_ts(t):
    try:
        import datetime
        return datetime.datetime.fromtimestamp(float(t)).strftime("%H:%M:%S")
    except Exception:
        return str(t)[:8]


def snapshot():
    """Aggregate current agent state from all journals."""
    s = {"context": None, "last_actions": [], "learning": [], "surprise": [],
         "verified": 0, "learned": 0, "gates": 0, "shifts": 0, "errors": 0,
         "fs_events": 0, "shell": 0, "memories": 0, "session": None,
         "session_verified": 0, "session_learned": 0,
         "interactions": 0, "interaction_kinds": {}}

    last_session = None
    if AGENT.exists():
        for line in AGENT.read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            ev = e.get("event")
            last_session = e.get("session", last_session)
            if ev == "context":
                s["context"] = e
            elif ev == "verified_action":
                s["verified"] += 1
                s["last_actions"].append(e)
            elif ev == "learned" or ev == "learned_gated":
                s["learned"] += 1
                s["learning"].append(e)
            elif ev == "gates_added":
                s["gates"] += 1
            elif ev == "shift_detected":
                s["shifts"] += 1
            elif ev == "error":
                s["errors"] += 1
            elif ev == "shell_action":
                s["shell"] += 1
        s["session"] = last_session

    # current-session counts: only events from the last session (or last ~2h)
    if AGENT.exists():
        cutoff = time.time() - 7200  # last 2 hours = "active"
        for line in AGENT.read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("session") != s["session"]:
                continue
            if float(e.get("t", 0)) < cutoff:
                continue
            if e.get("event") == "verified_action":
                s["session_verified"] += 1
            elif e.get("event") in ("learned", "learned_gated"):
                s["session_learned"] += 1

    if TRACE.exists():
        for line in TRACE.read_text().splitlines():
            try:
                s["surprise"].append(json.loads(line))
            except Exception:
                continue

    if MEMORY.exists():
        s["memories"] = len(MEMORY.read_text().splitlines())

    if TELEM.exists():
        s["fs_events"] = sum(1 for l in TELEM.read_text().splitlines()
                             if '"stream": "fsevents"' in l)

    if INTERACTIONS.exists():
        from collections import Counter
        kinds = Counter()
        for line in INTERACTIONS.read_text().splitlines():
            try:
                kinds[json.loads(line).get("kind", "?")] += 1
            except Exception:
                continue
        s["interactions"] = sum(kinds.values())
        s["interaction_kinds"] = dict(kinds.most_common(6))

    # janitor: sum reclaimed bytes from verified janitor actions in the journal
    s["janitor_reclaimed"] = 0
    s["janitor_actions"] = 0
    if AGENT.exists():
        for line in AGENT.read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") == "verified_action" and "reclaimed_bytes" in e:
                s["janitor_reclaimed"] += int(e.get("reclaimed_bytes", 0))
                s["janitor_actions"] += 1

    # keep last N
    s["last_actions"] = s["last_actions"][-6:]
    s["learning"] = s["learning"][-4:]
    s["surprise"] = s["surprise"][-6:]
    return s


def render(s):
    """Render the dashboard as text."""
    print("\033[2J\033[H")  # clear screen
    bar = "=" * 60
    print(f"\033[95mPROTEUS AGENT MONITOR\033[0m  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(bar)

    # context
    c = s["context"]
    if c:
        app = c.get("app", "?")
        wins = c.get("windows", "?")
        zvis = c.get("zvis_std")
        shell = c.get("shell_new", 0)
        print(f"\033[96mCONTEXT\033[0m  app={app}  windows={wins}  "
              f"shell_new={shell}  z_vis(std)={zvis if zvis is not None else 'n/a'}")

    # counts
    print(f"\n\033[96mCOUNTS\033[0m  (session={s['session']})")
    print(f"  verified_actions={s['verified']}  (active: {s['session_verified']} in last 2h)")
    print(f"  learning_updates={s['learned']}  (active: {s['session_learned']})")
    print(f"  gates={s['gates']}  shifts={s['shifts']}  errors={s['errors']}  "
          f"fs_events={s['fs_events']}  shell={s['shell']}  memories={s['memories']}")

    # interactions (the input tap stream)
    if s["interactions"]:
        kinds = ", ".join(f"{k}={v}" for k, v in s["interaction_kinds"].items())
        print(f"\n\033[96mINTERACTIONS\033[0m  total={s['interactions']}  {kinds}")

    # janitor (developer artifact reclaim)
    if s.get("janitor_reclaimed"):
        print(f"\n\033[96mJANITOR\033[0m  reclaimed={s['janitor_reclaimed']/1e6:.1f} MB "
              f"across {s['janitor_actions']} verified action(s)")

    # recent actions
    print(f"\n\033[96mRECENT ACTIONS\033[0m")
    for a in reversed(s["last_actions"]):
        t = fmt_ts(a.get("t"))
        act = a.get("action", "?")
        ok = "OK" if a.get("verified") else "FAIL"
        detail = a.get("detail", "")
        print(f"  {t}  [{ok}] {act}  {detail[:50]}")

    # learning
    if s["learning"]:
        print(f"\n\033[96mLEARNING\033[0m")
        for l in reversed(s["learning"]):
            t = fmt_ts(l.get("t"))
            kind = "gated" if "gated" in l.get("event", "") else "update"
            mse = l.get("macro_mse")
            print(f"  {t}  {kind}  macro_mse={mse if mse is not None else '?'}")

    # surprise / energy
    if s["surprise"]:
        print(f"\n\033[96mSURPRISE / ENERGY\033[0m")
        for x in reversed(s["surprise"][-4:]):
            t = fmt_ts(x.get("t"))
            print(f"  {t}  surprise={x.get('surprise'):.5f}  {x.get('action')}")
        # sparkline of surprise
        vals = [x.get("surprise", 0) for x in s["surprise"]]
        mx = max(vals) if vals else 1
        bar_chars = "".join("#" if v > 0.5 * mx else "." for v in vals[-40:])
        print(f"  trend: {bar_chars}")

    print(bar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single snapshot")
    ap.add_argument("--tail", type=int, default=0,
                    help="show last N events then exit")
    ap.add_argument("--refresh", type=float, default=2.0, help="refresh seconds")
    args = ap.parse_args()

    if args.tail:
        # just dump the last N journal events
        lines = AGENT.read_text().splitlines() if AGENT.exists() else []
        for line in lines[-args.tail:]:
            print(line)
        return

    try:
        while True:
            s = snapshot()
            render(s)
            if args.once:
                break
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nmonitor stopped")


if __name__ == "__main__":
    main()