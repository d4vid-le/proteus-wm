#!/usr/bin/env python3
"""explain.py — PROTEUS grounded explanation (the deliverable).

The full JEPA->LLM bridge in one CLI:
  1. Live world obs (12-dim) -> macro-64 JEPA latent (WorldLatent)
  2. Power-whitened (gamma=0.5) -> the space the bridge was trained on
  3. JEPAToQwenBridge projects the whitened latent -> 2048-dim soft-prefix
  4. Prefix prepended to Qwen's input_embeds, THEN Qwen generates an
     explanation. The projected world state steers the language.

Optional PAM context: --why "move notes0.txt" retrieves the k nearest past
latent neighborhoods from data/latent_memory.jsonl (the Inward JEPA) and asks
Qwen to reconcile the current state against what it did before in a similar
state.

Usage:
  python -m proteus.bridge.explain --obs '1,2,0,3,4,0,10,0,1,1,10,0'
  python -m proteus.bridge.explain --why 'move notes0.txt->txt'
  python -m proteus.bridge.explain --obs '...' --question 'what changed in the root?'
  python -m proteus.bridge.explain --self-test
"""
from proteus.paths import ROOT
import argparse
import json
import pathlib
import sys

import numpy as np

MEM = ROOT / "data" / "latent_memory.jsonl"

# cached singletons
_bridge = None
_model = None
_tok = None
_wl = None


def _load():
    global _bridge, _model, _tok, _wl
    if _bridge is not None:
        return
    from proteus.bridge.llm_bridge import load_bridge, load_qwen, load_whiten, PROMPT
    from proteus.world.world_latent import WorldLatent
    _bridge = load_bridge()
    _model, _tok = load_qwen()
    _wl = WorldLatent()
    return


def latent_of(obs):
    """obs (12-dim) -> whitened 64-dim latent (the bridge's input space)."""
    _load()
    from proteus.bridge.llm_bridge import load_whiten
    z = _wl.embed(obs)                      # raw macro-64
    return load_whiten()(z)                 # gamma=0.5 whitened


def memory_context(query_z, k=3):
    """Inward JEPA: k nearest past (z, action, desc) neighborhoods.

    query_z is ALREADY power-whitened (latent_of output). retrieve() defaults
    to space='whitened' so no double-whitening — the coordinate contract in
    latent_memory.py guarantees buffer storage is uniformly whitened."""
    from proteus.world.latent_memory import LatentMemory
    mem = LatentMemory(path=MEM)
    hits = mem.retrieve(query_z, k=k, space="whitened")
    lines = []
    for h in hits:
        a = h.get("action", "?")
        d = (h.get("metadata") or {}).get("desc", "")
        lines.append(f"- dist={h['dist']:.3f} action={a} detail={d}")
    return lines


def explain(obs, question=None, why=None, k=3, max_new=80):
    _load()
    from proteus.bridge.llm_bridge import answer
    z = latent_of(obs)
    if question is None:
        question = "Describe the world state and what action was taken."
    if why:
        ctx = memory_context(z, k=k)
        ctx_txt = "\n".join(ctx) if ctx else "(no similar past states)"
        question = (f"{question} Previously in a similar state I did:\n"
                    f"{ctx_txt}\nExplain why.")
    return answer(_model, _tok, _bridge, z, question, with_prefix=True,
                  max_new=max_new)


def self_test():
    # a couple of representative obs vectors (loose files in root)
    states = [
        [1, 2, 0, 0, 0, 0, 3, 0, 1, 1, 3, 0],   # 1 txt + 2 jpg loose in root
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0],   # a KEEP pdf in root
    ]
    for i, obs in enumerate(states):
        z = latent_of(obs)
        print(f"state{i}: latent dim={z.shape} norm={np.linalg.norm(z):.3f}")
        print("  context:", memory_context(z, k=2))
        print("  explain:", explain(obs, max_new=50)[:200].replace("\n", " "))
    print("SELF-TEST PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", help="12-dim obs vector (comma-separated)")
    ap.add_argument("--question", default=None, help="question for Qwen")
    ap.add_argument("--why", default=None,
                    help="optional: retrieve PAM context and explain why")
    ap.add_argument("--k", type=int, default=3, help="memory neighbors")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    elif args.obs:
        obs = [float(v) for v in args.obs.split(",")]
        out = explain(obs, question=args.question, why=args.why, k=args.k,
                      max_new=args.max_new)
        print(out)
    else:
        ap.print_help()