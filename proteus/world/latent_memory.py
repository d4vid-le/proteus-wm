#!/usr/bin/env python3
"""latent_memory.py — The Inward JEPA: predictive associative memory (PAM).

Standard RAG asks "what looks like this observation?" (cosine over text). The
Inward JEPA asks the causal question instead: "what action previously reduced
error when I was in a neighborhood like this?" Memory lives in the JEPA latent
space (macro-64), and retrieval is L2-distance in that space — temporal/causal
neighborhoods, not surface similarity.

Buffer M = {(s_k, a_k, s_{k+1}, metadata_k)}, persisted as JSONL.
  - append(obs, action, next_obs, metadata): computes latents via WorldLatent
    (or accepts precomputed latent vectors to avoid recompute).
  - retrieve(obs_or_latent, k=3): L2-nearest past latents + their metadata.
  - surprise_retrieve(): used by the reflex — given the surprise state, pull
    the k past tuples nearest to it so Qwen can see "what I did last time in
    a state like this."

COORDINATE CONTRACT (fixed 2026-09-04):
  Storage is UNIFORMLY POWER-WHITENED. The whiten transform (data/whiten.npz,
  fit on the deduped train latents) is applied at every write boundary, so a
  caller passing a RAW latent and a caller passing a WHITENED latent both end
  up stored in the SAME coordinate system, and L2 retrieval in the buffer is
  always Mahalanobis-equivalent in raw space.
    - append(z=..., z_next=..., space='raw')      # raw latents are whitened
    - append(z=..., z_next=..., space='whitened') # already whitened, as-is
    - retrieve(query, space='whitened')           # query is whitened
    - retrieve(query, space='raw')                # query is raw (whitened)
    - obs vectors (12-dim) are ALWAYS treated as raw: embed + whiten.

Usage:
  python -m proteus.world.latent_memory --self-test
"""
from proteus.paths import ROOT
import argparse
import json
import pathlib
import sys

import numpy as np

OUT = ROOT / "data" / "latent_memory.jsonl"
WHITEN = ROOT / "data" / "whiten.npz"

_whiten_state = {"t": None}


def _load_whiten():
    """Lazy-load the power-whitening transform. Returns (mean, W)."""
    if _whiten_state["t"] is None:
        d = np.load(WHITEN)
        _whiten_state["t"] = (d["mean"], d["W"])
    return _whiten_state["t"]


def whiten(z):
    """Apply the canonical power-whitening (same transform as latent_whiten.py
    and jepa_llm_bridge.load_whiten). z: (..., 64) raw, L2-normalized ->
    (..., 64) whitened, L2-normalized. A single (64,) vector returns (64,)."""
    z = np.asarray(z, dtype=np.float32)
    one_d = z.ndim == 1
    if one_d:
        z = z[None, :]
    mean, W = _load_whiten()
    Zw = (z - mean) @ W
    Zw = Zw / (np.linalg.norm(Zw, axis=-1, keepdims=True) + 1e-8)
    return Zw[0] if one_d else Zw


class LatentMemory:
    def __init__(self, path=OUT, wl=None, max_entries=20000, save_every=64):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.wl = wl  # WorldLatent instance (lazy-imported to avoid torch cost)
        self.max_entries = max_entries
        self.save_every = save_every
        self._dirty_since_save = 0
        self._rows = []
        self._load()

    # ---- persistence ----
    def _load(self):
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            try:
                r = json.loads(line)
                if "z" in r:
                    self._rows.append(r)
            except Exception:
                continue

    def _save(self):
        with open(self.path, "w") as f:
            for r in self._rows[-self.max_entries:]:
                f.write(json.dumps(r) + "\n")

    def flush(self):
        """Persist accumulated rows (call after a batch of appends)."""
        self._save()

    # ---- latent helper ----
    def _z(self, obs_or_z, space="whitened"):
        """Accept either a precomputed 64-dim latent or a 12-dim obs vector.

        space='raw'      -> 64-dim vector is a RAW latent: whiten it first.
        space='whitened' -> 64-dim vector is ALREADY whitened: normalize only.
        A 12-dim obs vector is always raw: embed (WorldLatent) then whiten.
        Every code path returns a WHITENED 64-dim latent, so the buffer and
        the query are always in the same coordinate system."""
        arr = np.asarray(obs_or_z, dtype=np.float32)
        if arr.ndim == 1 and arr.shape[0] == 64:
            z = arr / (np.linalg.norm(arr) + 1e-8)
            return whiten(z) if space == "raw" else z
        # else: it's a 12-dim obs vector (always raw -> embed + whiten)
        if self.wl is None:
            from proteus.world.world_latent import WorldLatent
            self.wl = WorldLatent()
        return whiten(self.wl.embed(arr))

    # ---- API ----
    def append(self, obs, action, next_obs=None, metadata=None,
               z=None, z_next=None, space="raw"):
        """Add one transition to the buffer. obs can be a 12-dim obs vector or
        a precomputed 64-dim latent (pass z=... to skip embedding).

        space='raw' (default): z/z_next are RAW latents -> whitened on store.
        space='whitened': z/z_next are already whitened -> stored as-is.
        Storage is ALWAYS whitened (see module coordinate contract)."""
        z = self._z(obs) if z is None else self._z(z, space=space)
        z_next = None
        if next_obs is not None or z_next is not None:
            z_next = (self._z(next_obs) if z_next is None
                      else self._z(z_next, space=space))
        self._rows.append({
            "z": z.tolist(),
            "action": action,
            "z_next": z_next.tolist() if z_next is not None else None,
            "metadata": metadata or {},
        })
        if len(self._rows) > self.max_entries:
            self._rows = self._rows[-self.max_entries:]
        self._dirty_since_save += 1
        if self._dirty_since_save >= self.save_every:
            self._save()
            self._dirty_since_save = 0

    def retrieve(self, obs_or_z, k=3, space="whitened"):
        """L2-nearest past latents to the query. Returns rows sorted by
        distance (each with a 'dist' field).

        space='whitened' (default): the query is already whitened (e.g. it
        came from jepa_explain.latent_of). space='raw': the query is a raw
        latent (or 12-dim obs) and is whitened here. All comparisons happen
        in the same whitened space the buffer is stored in."""
        if not self._rows:
            return []
        q = self._z(obs_or_z, space=space)
        Z = np.asarray([r["z"] for r in self._rows], dtype=np.float32)
        d = np.linalg.norm(Z - q, axis=1)
        idx = np.argsort(d)[:k]
        out = []
        for i in idx:
            r = dict(self._rows[i])
            r["dist"] = float(d[i])
            out.append(r)
        return out

    def __len__(self):
        return len(self._rows)


def self_test():
    wl = None
    mem = LatentMemory(path=ROOT / "data" / "_test_latent_mem.jsonl")
    # 3 distinct world states (obs vectors)
    states = [
        ([1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0], "move_txt"),
        ([0, 1, 0, 1, 0, 0, 2, 0, 1, 0, 2, 0], "move_jpg"),
        ([0, 0, 1, 0, 1, 0, 3, 1, 1, 0, 3, 0], "delete_pdf"),
    ]
    for obs, a in states:
        mem.append(obs, a)
    # query near state 0 -> should retrieve the move_txt row (smallest dist)
    res = mem.retrieve(states[0][0], k=3)
    assert res[0]["action"] == "move_txt", res[0]["action"]
    assert res[0]["dist"] < res[2]["dist"]
    # near state 2 -> delete_pdf closest
    res2 = mem.retrieve(states[2][0], k=3)
    assert res2[0]["action"] == "delete_pdf", res2[0]["action"]
    print(f"buffer size={len(mem)}")
    print("retrieve near state0: dists=", [round(r['dist'],4) for r in res],
          "actions=", [r['action'] for r in res])
    print("retrieve near state2: dists=", [round(r['dist'],4) for r in res2],
          "actions=", [r['action'] for r in res2])
    # cleanup
    import os
    for f in pathlib.Path(ROOT / "data").glob("_test_latent_mem.jsonl*"):
        os.remove(f)
    print("SELF-TEST PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        ap.print_help()