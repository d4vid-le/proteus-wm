#!/usr/bin/env python3
"""transition_pairs.py — Empirical inverse-dynamics transition dataset.

The bridge experiment (redirection 2026-09-04): instead of simulating a future
latent from the target action (circular, 4-step smear), build the dataset from
REAL empirical transitions, exactly matching the reflex's live tuple shape:

    Channel 1: z_t        (whitened latent BEFORE the action — empirical)
    Channel 2: z_{t+1}    (whitened latent AFTER  the action — empirical)
    Channel 3: e_surprise (label-free surprise scalar)

e_surprise = || z_{t+1} - z_hat ||^2 where z_hat = predict_macro_gated(z_t,
A_fixed). A_fixed is a CONSTANT reference action sequence (all move_txt),
independent of the sample, so e measures "how far the world actually moved
from the model's canonical expectation" WITHOUT ever feeding the target action
into the inputs. Zero label leakage: the bridge must deduce the action purely
from the empirical delta (z_t -> z_{t+1}) + a scalar surprise magnitude.

Target to predict: a_i (the action that caused the transition).

All latents are stored in POWER-WHITENED coordinates (same transform /
data/whiten.npz as latent_memory, jepa_llm_bridge, jepa_explain) so every
consumer reads one consistent space.

Outputs:
  data/transition_pairs_clean.jsonl  (whitened z_t, z_next, surprise, action)
  data/transition_stats.json         (surprise mean/std for standardization)

Usage:
  python -m proteus.world.transition_pairs [--limit N] [--summary]
"""
from proteus.paths import ROOT
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

from proteus.world.backfill import replay_session           # noqa: E402
from proteus.bridge.llm_bridge import load_whiten               # noqa: E402  (numpy only)
from proteus.world.le_gates import predict_macro_gated              # noqa: E402
from proteus.world.world_latent import WorldLatent                  # noqa: E402

JOURNAL = ROOT / "data" / "autonomous_journal.jsonl"
OUT = ROOT / "data" / "transition_pairs_clean.jsonl"
STATS = ROOT / "data" / "transition_stats.json"

ACT_FIXED = [0, 0, 0, 0]   # canonical reference action (move_txt x4), label-free

# real_env.ACTIONS = [move_txt, move_jpg, move_pdf, delete_pdf]
_EXT2ACT = {"txt": 0, "jpg": 1, "pdf": 2}


def _ext_to_act(target):
    """Mirror autonomous._ext_to_act (txt=0, jpg=1, pdf=2, else misc=3)."""
    return _EXT2ACT.get(target, 3)


def _norm(z):
    return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


def build(limit=None, gamma_std=True):
    wl = WorldLatent()
    whiten = load_whiten()

    sessions = {}
    for line in JOURNAL.read_text().splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") == "verified_action":
            sessions.setdefault(e.get("session"), []).append(e)

    print(f"sessions with verified actions: {len(sessions)}")

    rows = []
    n_sess = 0
    for sess, entries in sessions.items():
        traj = replay_session(entries)
        # consecutive empirical transitions: (obs_t, action_i, obs_{t+1})
        for i in range(len(traj) - 1):
            s_t = traj[i]
            s_n = traj[i + 1]
            z_t_raw = _norm(wl.embed(s_t["obs"]))
            z_n_raw = _norm(wl.embed(s_n["obs"]))
            # label-free surprise: deviation from the FIXED canonical rollout
            A = torch.as_tensor([ACT_FIXED], dtype=torch.int64)
            z_hat = predict_macro_gated(
                wl.model, torch.as_tensor(z_t_raw).unsqueeze(0), A)
            z_hat = _norm(z_hat.detach().cpu().numpy())
            surprise = float(np.mean((z_hat - z_n_raw) ** 2))
            rows.append({
                "session": sess, "t": s_t["t"], "obs": s_t["obs"],
                "latent": whiten(z_t_raw).tolist(),
                "latent_next": whiten(z_n_raw).tolist(),
                "surprise": surprise,
                "action": s_t["action"], "desc": s_t["desc"],
            })
        n_sess += 1
        if limit and n_sess >= limit:
            break

    print(f"raw transitions: {len(rows)}")

    # ---- dedup (same rule as latent_whiten: action + rounded z_t + z_next) ----
    seen = set()
    uniq = []
    for r in rows:
        k = (r["action"],
             tuple(np.round(r["latent"][:8], 3)),
             tuple(np.round(r["latent_next"][:8], 3)))
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    rows = uniq
    print(f"after dedup: {len(rows)}")

    # ---- standardize surprise (z-score) for a well-scaled bridge input ----
    sur = np.asarray([r["surprise"] for r in rows], dtype=np.float32)
    mu, sd = float(sur.mean()), float(sur.std()) + 1e-8
    for r, s in zip(rows, sur):
        r["surprise_std"] = float((s - mu) / sd)

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(STATS, "w") as f:
        json.dump({"mean": mu, "std": sd, "n": len(rows)}, f)

    from collections import Counter
    targets = Counter()
    for r in rows:
        a = r["action"] or ""
        targets[a.split("->")[-1] if "->" in a else a] += 1
    print(f"transitions -> {OUT} ({len(rows)} rows)")
    print(f"surprise stats -> {STATS}: mean={mu:.4f} std={sd:.4f}")
    print("target distribution:", dict(targets))
    return len(rows)


def summary():
    import glob
    for p in (OUT, STATS):
        pp = pathlib.Path(p)
        if pp.exists():
            print(f"{pp.name}: {pp.stat().st_size} bytes")
    if pathlib.Path(STATS).exists():
        print(json.loads(pathlib.Path(STATS).read_text()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
    else:
        build(limit=args.limit or None)