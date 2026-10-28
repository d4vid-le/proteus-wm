#!/usr/bin/env python3
"""latent_whiten.py — Fix the cone collapse: dedup + whiten the latent space.

Diagnosis (confirmed on real data):
  - pairwise cosine 0.94 (healthy: -0.10..+0.40)  -> narrow cone
  - participation ratio 1.11 of 64 dims           -> dimensional collapse
  - dist-0.0 duplicate rows in the PAM buffer      -> memory smothering

Causes:
  1. The file-world obs are low-diversity (12 count dims), so the macro
     encoder genuinely lives in a low-entropy cone.
  2. Reconstruction artifact: backfill replays each session in isolation
     (world starts empty), so different sessions land on identical obs.

Fixes applied here:
  1. Dedup by (action, rounded-latent) -> no identical transitions in train
     set or PAM buffer (no smothering / no over-indexing).
  2. PCA-whitening (fit on the deduped train latents, top-k components):
     z_white = U_k^T (z - mean) / sqrt(s_k^2 + eps), re-L2-normalized.
     This spreads the cone to fill the sphere, equalizes per-dim variance, and
     is a fixed linear map (same transform at train and inference), so L2
     retrieval in whitened space = Mahalanobis in raw space (still causal).

Outputs:
  data/latent_pairs_clean.jsonl   deduped, whitened pairs (bridge train set)
  data/latent_memory.jsonl        rewritten with whitened z (PAM)
  data/whiten.npz                 (mean, components, scale) for inference

Usage:
  python3 latent_whiten.py [--k 16] [--summary]
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
import sys

import numpy as np

PAIRS = ROOT / "data" / "latent_pairs.jsonl"
PAIRS_CLEAN = ROOT / "data" / "latent_pairs_clean.jsonl"
MEM = ROOT / "data" / "latent_memory.jsonl"
WHITEN = ROOT / "data" / "whiten.npz"


def dedup(rows, key_fn=None, tol=1e-3):
    """Remove rows whose (action, rounded-latent) key repeats. First wins."""
    key_fn = key_fn or (lambda r: (r["action"],
                                   tuple(np.round(r["latent"][:8], 3))))
    seen = set()
    out = []
    for r in rows:
        k = key_fn(r)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def fit_whiten(Z, k=16, eps=1e-4, gamma=1.0):
    """Power (fractional) whitening transform. Z: (N, D) already L2-normalized.
    Returns (mean, W) where z_white = (z - mean) @ W.T, D->k.

    gamma is the whitening exponent:
      gamma=0.0  -> raw PCA (count-dominant cone preserved; subtle action
                    signals smothered)
      gamma=1.0  -> full whitening (variance-equalized; action signals clear,
                    but the count axis is squashed into isotropic noise)
      0<gamma<1  -> Pareto middle: softly deflate the dominant cone so subtle
                    axes get room to breathe WITHOUT flattening the bulk count
                    signal into background noise. (Mahalanobis noise trap fix.)
    """
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = Zc.T @ Zc / max(Zc.shape[0] - 1, 1)
    # robust eigendecomposition of the (small) D x D covariance
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    keep = eigvals[:k]
    U = eigvecs[:, :k]                              # (D, k)
    # sigma^(-gamma) = eigval^(-gamma/2)  (eigval = sigma^2)
    scale = np.maximum(keep, eps) ** (-gamma / 2.0)
    W = (U * scale)                                 # (D, k)
    mean = Z.mean(axis=0, keepdims=True)
    return mean, W


def apply_whiten(Z, mean, W):
    """z_white = (z - mean) @ W, then L2-normalized. Z: (N, D) -> (N, k)."""
    Zw = (Z - mean) @ W
    Zw = Zw / (np.linalg.norm(Zw, axis=1, keepdims=True) + 1e-8)
    return Zw


def run(k=64, gamma=1.0):
    if not PAIRS.exists():
        print("no latent_pairs.jsonl — run backfill_latents.py first")
        return
    rows = [json.loads(l) for l in PAIRS.read_text().splitlines()]

    # --- 1. dedup ---
    n_raw = len(rows)
    rows = dedup(rows)
    print(f"dedup: {n_raw} -> {len(rows)} (removed {n_raw - len(rows)})")

    # --- 2. whiten on train latents ---
    Z = np.asarray([r["latent"] for r in rows], dtype=np.float32)
    mean, W = fit_whiten(Z, k=k, gamma=gamma)
    Zw = apply_whiten(Z, mean, W)
    print(f"whiten: D={Z.shape[1]} -> k={Zw.shape[1]}  gamma={gamma}")

    # --- 3. write clean pairs ---
    with open(PAIRS_CLEAN, "w") as f:
        for r, z in zip(rows, Zw):
            r2 = dict(r)
            r2["latent"] = z.tolist()
            r2["latent_raw"] = r["latent"]  # keep raw for reference
            f.write(json.dumps(r2) + "\n")
    print(f"clean pairs -> {PAIRS_CLEAN} ({len(rows)} rows)")

    # --- 4. rewrite PAM memory with whitened z (deduped too) ---
    if MEM.exists():
        mem = [json.loads(l) for l in MEM.read_text().splitlines()]
        n_mem = len(mem)
        # dedup by (action, rounded z)
        mem = dedup(mem, key_fn=lambda r: (r["action"],
                                           tuple(np.round(r["z"][:8], 3))))
        ZM = np.asarray([r["z"] for r in mem], dtype=np.float32)
        Zm = apply_whiten(ZM, mean, W)
        with open(MEM, "w") as f:
            for r, z in zip(mem, Zm):
                r2 = dict(r)
                r2["z"] = z.tolist()
                r2["z_next"] = (apply_whiten(
                    np.asarray([r["z_next"]], dtype=np.float32), mean, W)[0].tolist()
                    if r.get("z_next") is not None else None)
                f.write(json.dumps(r2) + "\n")
        print(f"memory: {n_mem} -> {len(mem)} rows, z whitened -> {MEM}")
    else:
        # rebuild the PAM buffer from the clean pairs (z, action, desc)
        n_mem = 0
        with open(MEM, "w") as f:
            for r in rows:
                f.write(json.dumps({
                    "z": r["latent"],
                    "action": r.get("action"),
                    "z_next": None,
                    "metadata": {"desc": r.get("desc"),
                                 "session": r.get("session"),
                                 "t": r.get("t")},
                }) + "\n")
                n_mem += 1
        print(f"memory rebuilt from clean pairs: {n_mem} rows -> {MEM}")

    # --- 5. save transform ---
    np.savez_compressed(WHITEN, mean=mean, W=W, k=k)
    print(f"whiten transform -> {WHITEN}")

    # --- post-diagnostic ---
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Zw), min(100, len(Zw)), replace=False)
    cos = Zw[idx] @ Zw[idx].T
    np.fill_diagonal(cos, np.nan)
    u, s, vt = np.linalg.svd(Zw, full_matrices=False)
    pr = (s**2).sum()**2 / (s**4).sum()
    print(f"\npost-whiten: mean cos={np.nanmean(cos):.4f} "
          f"| participation ratio={pr:.2f} of {Zw.shape[1]} "
          f"| per-dim std mean={Zw.std(axis=0).mean():.4f}")


def summary():
    import glob
    for p in (PAIRS, PAIRS_CLEAN, MEM, WHITEN):
        pp = pathlib.Path(p)
        if pp.exists():
            print(f"{pp.name}: {pp.stat().st_size} bytes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="whitening exponent: 0=raw PCA, 1=full whiten, ~0.4-0.5 Pareto middle")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
    else:
        run(k=args.k, gamma=args.gamma)