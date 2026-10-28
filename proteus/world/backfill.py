#!/usr/bin/env python3
"""backfill_latents.py — Reconstruct JEPA latents for historical journal data.

The journal has 6.3k verified_action entries but (before the autonomous.py
patch) did NOT store the obs vector or latent. This script replays each
session's action trajectory to reconstruct a minimal faithful world state at
every step — every file the agent touched starts in root, moves are replayed
exactly — so the 12-dim obs (and thus the macro-64 latent) is temporally
consistent with the real action that was taken.

Outputs:
  data/latent_pairs.jsonl     (latent, action, desc, obs)  -> bridge training
  data/latent_memory.jsonl    (z, action, z_next, metadata) -> PAM buffer

NOTE: latents for historical data are RECONSTRUCTIONS (exact file counts of
untouched files are unknowable). Live data from autonomous.py now logs the
exact obs+latent directly — those are ground truth.

Usage:
  python -m proteus.world.backfill [--limit N] [--summary]
"""
from proteus.paths import ROOT
import argparse
import json
import pathlib
import sys

import numpy as np

JOURNAL = ROOT / "data" / "autonomous_journal.jsonl"
PAIRS = ROOT / "data" / "latent_pairs.jsonl"
PROTECT = ("KEEP", "IMPORTANT", "DO_NOT_DELETE")


def parse_action(action):
    """'move notes0.txt->txt' -> ('notes0.txt', 'txt'). 'delete_pdf' -> None."""
    if not action:
        return None
    if action.startswith("move "):
        body = action[5:]
        if "->" in body:
            name, target = body.split("->", 1)
            return (name.strip(), target.strip())
    return None


def ext_of(name):
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def is_protected(name):
    return any(name.startswith(p) for p in PROTECT)


def replay_session(entries):
    """Replay a session's verified actions to reconstruct obs per step.
    entries: list of journal dicts (chronological). Returns list of
    (obs_before, action, desc, t) or [] if no reconstructable actions."""
    # 1. gather the file universe touched by this session (in order)
    files = []  # (name, ext)
    for e in entries:
        pa = parse_action(e.get("action"))
        if pa:
            name, target = pa
            if not any(f[0] == name for f in files):
                files.append((name, ext_of(name)))
    if not files:
        return []
    # 2. minimal faithful world: every touched file starts in root
    root = {name: True for name, _ in files}   # name -> in root
    dirs = {"txt": {}, "jpg": {}, "pdf": {}}   # name -> dir
    out = []
    for e in entries:
        pa = parse_action(e.get("action"))
        if not pa:
            continue
        name, target = pa
        # obs BEFORE the action
        obs = _obs_vec(root, dirs)
        out.append({"obs": obs, "action": e.get("action"),
                    "desc": e.get("detail"), "t": e.get("t")})
        # apply the action
        if name in root:
            del root[name]
            if target in dirs:
                dirs[target][name] = True
    return out


def _obs_vec(root, dirs):
    root_txt = sum(1 for n in root if ext_of(n) == "txt")
    root_jpg = sum(1 for n in root if ext_of(n) == "jpg")
    root_pdf = sum(1 for n in root if ext_of(n) == "pdf")
    txt = len(dirs["txt"]); jpg = len(dirs["jpg"]); pdf = len(dirs["pdf"])
    keep_in_root = 1 if any(is_protected(n) for n in root) else 0
    keep_preserved = 1
    junk_in_root = 1 if any(ext_of(n) == "pdf" and not is_protected(n) for n in root) else 0
    total = len(root) + txt + jpg + pdf
    junk_deleted = 1 if (root_pdf == 0 and pdf == 0) else 0
    return [float(x) for x in (root_txt, root_jpg, root_pdf, txt, jpg, pdf,
                               len(root), keep_in_root, keep_preserved,
                               junk_in_root, total, junk_deleted)]


def build(limit=None):
    from proteus.world.world_latent import WorldLatent
    from proteus.world.latent_memory import LatentMemory

    wl = WorldLatent()
    mem = LatentMemory(wl=wl)

    # group verified_actions by session, chronological
    sessions = {}
    for line in JOURNAL.read_text().splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") == "verified_action":
            sessions.setdefault(e.get("session"), []).append(e)

    print(f"sessions with verified actions: {len(sessions)}")
    n_pairs = 0
    n_done = 0
    with open(PAIRS, "w") as f:
        for sess, entries in sessions.items():
            traj = replay_session(entries)
            if not traj:
                continue
            # embed obs vectors in batch (fast)
            obs_mat = np.asarray([s["obs"] for s in traj], dtype=np.float32)
            latents = wl.embed_batch(obs_mat)  # (T, 64)
            for i, s in enumerate(traj):
                row = {"session": sess, "t": s["t"], "obs": s["obs"],
                       "latent": latents[i].tolist(), "action": s["action"],
                       "desc": s["desc"]}
                f.write(json.dumps(row) + "\n")
                n_pairs += 1
                # PAM row: (z_k, a_k, z_{k+1}) — pass precomputed latents
                if i + 1 < len(traj):
                    mem.append(s["obs"], s["action"], traj[i + 1]["obs"],
                               metadata={"session": sess, "t": s["t"],
                                         "desc": s["desc"]},
                               z=latents[i], z_next=latents[i + 1])
            n_done += 1
            if limit and n_done >= limit:
                break
    print(f"pairs written: {n_pairs}  -> {PAIRS}")
    mem.flush()
    print(f"PAM buffer rows: {len(mem)}  -> {mem.path}")
    return n_pairs


def summary():
    n = 0
    kinds = {}
    if PAIRS.exists():
        for line in PAIRS.read_text().splitlines():
            try:
                r = json.loads(line)
                n += 1
                a = (r.get("action") or "").split("->")[-1]
                kinds[a] = kinds.get(a, 0) + 1
            except Exception:
                continue
    print(f"latent_pairs.jsonl: {n} rows")
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
    else:
        build(limit=args.limit or None)