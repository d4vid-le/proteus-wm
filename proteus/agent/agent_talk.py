#!/usr/bin/env python3
"""agent_talk.py — Let PROTEUS speak out loud, agent-initiated.

Two self-triggered "voices":
  SURPRISE  (it noticed something)  — grounded JEPA->Qwen explanation of a
             surprise spike. Uses `SurpriseReflex._jepa_hypothesis`. This path
             needs LIVE obs in autonomous_journal.jsonl; without it the reflex
             can't ground a hypothesis and stays silent (the "no private
             language" rule). Verified: 0/12462 journal rows currently carry obs.
  ACTION    (it did something)      — a concise natural-language report grounded
             in the VERIFIED action label (real ground truth), so it speaks NOW
             even before live obs flow. Upgrades to full latent-grounded phrasing
             once obs starts being logged.

Reuses the existing "mouth" (jepa_explain / jepa_llm_bridge). No new model.
Run INSTEAD of chat.py (both load Qwen once; two loads OOM your Mac).

Usage:
  python -m proteus.agent.agent_talk --watch                  # surprise + after-action
  python -m proteus.agent.agent_talk --watch --voices action  # only after-action
  python -m proteus.agent.agent_talk --watch --voices surprise # only surprise
  python -m proteus.agent.agent_talk --once --surprise 0.05   # headless test (surprise)
  python -m proteus.agent.agent_talk --once --action 'moved photo.jpg -> /jpg/'  # headless test (action)
  python -m proteus.agent.agent_talk --emit none              # don't print, just log
"""
from proteus.paths import ROOT
import argparse
import json
import pathlib
import sys
import time

TRACE = ROOT / "data" / "eris_trace.jsonl"
JOURNAL = ROOT / "data" / "autonomous_journal.jsonl"
MEM = ROOT / "data" / "eris_memory.jsonl"
COMM = ROOT / "data" / "comm.jsonl"          # agent-initiated messages (this module)

_loaded = False


def _load():
    """Load the JEPA->Qwen bridge + Qwen ONCE (2-4 min first time). Don't run a
    second Qwen process simultaneously (MPS memory)."""
    global _loaded
    if _loaded:
        return
    import proteus.bridge.explain as jepa_explain
    jepa_explain._load()
    _loaded = True


def _log(rec):
    with open(COMM, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _make_speaker(emit):
    """Returns (maybe_surprise, report_action). Both pace via a shared min-gap."""
    from proteus.agent.reflex import SurpriseReflex
    r = SurpriseReflex()
    last = [0.0]

    def _pace(min_interval):
        now = time.time()
        if now - last[0] < min_interval:
            return False
        last[0] = now
        return True

    def maybe_surprise(surprise, min_interval=30.0):
        if not _pace(min_interval):
            return None
        triggered, th, s = r.ingest({"surprise": surprise})
        if not triggered:
            return None
        try:
            hyp = r._jepa_hypothesis(s)   # grounded natural-language explanation
        except Exception:
            hyp = None
        if not hyp:
            return None                    # can't ground -> stay silent
        msg = f"[surprise {s:.4f}] {hyp}"
        if emit:
            emit(f"\nPROTEUS > {msg}")
        _log({"t": time.time(), "voice": "surprise", "surprise": s, "msg": msg})
        return msg

    def report_action(detail, action, min_interval=15.0):
        """Natural-language report grounded in the VERIFIED action label."""
        if not detail:
            return None
        low = detail.lower()
        if "moved" not in low and "deleted" not in low and "created" not in low:
            return None
        if not _pace(min_interval):
            return None
        msg = f"Done — I just {detail}."
        if emit:
            emit(f"\nPROTEUS > {msg}")
        _log({"t": time.time(), "voice": "action", "action": action,
              "detail": detail, "msg": msg})
        return msg

    return maybe_surprise, report_action


def watch(interval=3.0, emit=print, min_interval=30.0, voices=("surprise", "action")):
    _load()
    maybe_surprise, report_action = _make_speaker(emit)
    tr_pos = jr_pos = 0
    print(f"PROTEUS agent_talk: watching surprise+journal (voices: {voices}) "
          f"[Ctrl-C to stop] ...")
    while True:
        try:
            # --- surprise stream ---
            if "surprise" in voices and TRACE.exists():
                lines = TRACE.read_text().splitlines()
                for ln in lines[tr_pos:]:
                    try:
                        rec = json.loads(ln)
                    except Exception:
                        continue
                    s = float(rec.get("surprise", 0) or 0)
                    if s > 0.0:
                        maybe_surprise(s, min_interval=min_interval)
                tr_pos = len(lines)
            # --- verified-action stream (report on what it just did) ---
            if "action" in voices and JOURNAL.exists():
                lines = JOURNAL.read_text().splitlines()
                for ln in lines[jr_pos:]:
                    try:
                        rec = json.loads(ln)
                    except Exception:
                        continue
                    if rec.get("event") == "verified_action" and rec.get("verified") == 1:
                        report_action(rec.get("detail", ""),
                                      rec.get("action", ""),
                                      min_interval=min_interval)
                jr_pos = len(lines)
        except KeyboardInterrupt:
            print("\nPROTEUS agent_talk stopped.")
            return
        time.sleep(interval)


def once(emit=print, surprise=None, action=None):
    _load()
    maybe_surprise, report_action = _make_speaker(emit)
    if surprise is not None:
        maybe_surprise(surprise, min_interval=0.0)
    if action is not None:
        report_action(action, action, min_interval=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="watch the streams (default)")
    ap.add_argument("--once", action="store_true", help="one event, then exit")
    ap.add_argument("--surprise", type=float, help="surprise value for --once")
    ap.add_argument("--action", type=str, help="an action detail string for --once")
    ap.add_argument("--voices", default="surprise,action",
                    help="comma list: surprise,action (default both)")
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--min-interval", type=float, default=30.0)
    ap.add_argument("--emit", choices=["print", "none"], default="print")
    args = ap.parse_args()
    emit = print if args.emit == "print" else None
    if args.once:
        once(emit=emit, surprise=args.surprise, action=args.action)
    else:
        watch(interval=args.interval, emit=emit, min_interval=args.min_interval,
              voices=tuple(v.strip() for v in args.voices.split(",") if v.strip()))


if __name__ == "__main__":
    main()
