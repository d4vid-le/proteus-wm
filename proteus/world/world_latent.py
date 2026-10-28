#!/usr/bin/env python3
"""world_latent.py — Canonical JEPA world-latent source (macro-64).

The locked decision: the JEPA state injected into Qwen is the **macro-64
latent** from HJEPA_K4.enc_macro — the space where the transition dynamics
live (WM(s,a) -> s_pred, macro_mse=0.0194). NOT raw ViT-768.

The obs contract is the 12-dim RealEnv file-world vector:
  [root_txt, root_jpg, root_pdf, txt, jpg, pdf, root_total,
   keep_in_root, keep_preserved, junk_in_root, total, junk_deleted]

Loads the SAME checkpoint autonomous.py uses (gated continual-learning model
preferred, else cleanup_hjepa_real.pt), so the latent space matches the model
that has been doing the organizing.

Usage:
  python -m proteus.world.world_latent --obs '1,2,0,3,4,0,10,0,1,1,10,0'   # print latent
  python -m proteus.world.world_latent --self-test
"""
from proteus.paths import ROOT
import argparse
import pathlib
import sys

import numpy as np
import torch

OBS_DIM = 12
ACT_DIM = 4

GATED_CKPT = ROOT / "checkpoints" / "autonomous_gated.pt"
BASE_CKPT = ROOT / "checkpoints" / "cleanup_hjepa_real.pt"


def load_hjepa(ckpt=None, device="cpu"):
    """Load the H-JEPA world model. Prefers the gated continual-learning
    checkpoint (mirrors autonomous.py's load_model), else the base real-env
    checkpoint. The gated checkpoint needs the Proteus Gates wrapper around
    pred_macro before load_state_dict."""
    from proteus.world.hjepa_torch import HJEPA_K4
    from proteus.world.gates import build_gated
    model = HJEPA_K4(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=128,
                     z_micro=32, z_macro=64, k=4, pos_dim=OBS_DIM)
    if ckpt is not None:
        model.load_state_dict(torch.load(pathlib.Path(ckpt), map_location="cpu",
                                         weights_only=True))
    elif GATED_CKPT.exists():
        build_gated(model)
        model.load_state_dict(torch.load(GATED_CKPT, map_location="cpu", weights_only=True))
        ckpt = GATED_CKPT
    elif BASE_CKPT.exists():
        model.load_state_dict(torch.load(BASE_CKPT, map_location="cpu", weights_only=True))
        ckpt = BASE_CKPT
    model.eval()
    return model, pathlib.Path(ckpt)


class WorldLatent:
    """Thin wrapper: 12-dim obs -> macro-64 JEPA latent (cached model)."""

    def __init__(self, model=None, ckpt=None, device="cpu"):
        if model is None:
            model, ckpt = load_hjepa(ckpt, device=device)
        self.model = model
        self.ckpt = str(ckpt)
        self.device = device
        if device != "cpu":
            self.model.to(device)

    @torch.no_grad()
    def embed(self, obs):
        """obs: 12-dim array/list -> (64,) macro latent (L2-normalized)."""
        if isinstance(obs, (list, tuple)):
            obs = np.asarray(obs, dtype=np.float32)
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        z = self.model.enc_macro(x)[0]
        z = z / z.norm(dim=-1, keepdim=True)
        return z.cpu().numpy().astype(np.float32)

    def embed_batch(self, obs):
        """obs: (N,12) -> (N,64) L2-normalized macro latents."""
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        z = self.model.enc_macro(x)
        z = z / z.norm(dim=-1, keepdim=True)
        return z.detach().cpu().numpy().astype(np.float32)


def self_test():
    wl = WorldLatent()
    print(f"checkpoint: {wl.ckpt}")
    vec = [1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 3.0, 0.0, 1.0, 1.0, 10.0, 0.0]
    z = wl.embed(vec)
    assert z.shape == (64,), z.shape
    n = float(np.linalg.norm(z))
    assert abs(n - 1.0) < 1e-3, n
    # nearby obs -> nearby latent (sanity: the manifold is smooth)
    vec2 = list(vec); vec2[0] += 1  # one more txt in root
    z2 = wl.embed(vec2)
    cos = float(np.dot(z, z2) / (np.linalg.norm(z) * np.linalg.norm(z2)))
    print(f"z shape={z.shape} norm={n:.4f}  nearby-obs cosine={cos:.4f}")
    print("SELF-TEST PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", help="comma-separated 12-dim obs vector")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    elif args.obs:
        wl = WorldLatent()
        z = wl.embed([float(v) for v in args.obs.split(",")])
        print("latent:", np.round(z, 4).tolist())
    else:
        ap.print_help()