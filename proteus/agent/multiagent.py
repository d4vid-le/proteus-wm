#!/usr/bin/env python3
"""multiagent.py — Phase 4: multi-agent latent communication (the breakthrough claim).

Binding = mutual predictability through a shared belief state:
  "from my view of the world, I can reconstruct yours."

Two agents observe PARTIAL views of the same CleanupDesktop world:
  Agent A sees the "what's where" view   (root + folder file counts, dims 0..5)
  Agent B sees the "status/safety" view  (messy, keep flags, junk, totals, dims 6..11)
Neither sees the other's view.

They communicate by exchanging their COMPRESSED latent z (32 floats), not text.
The coordinator decodes the FULL world state from the two latents and makes a
safety decision (delete pdf vs move pdf) that requires BOTH views:
  - root_pdf > 0     (A's view: there IS a pdf)
  - keep_in_root==0  (B's view: it's safe to delete)
A alone cannot know keep_in_root; B alone cannot know root_pdf. Latent exchange
binds the two partial worlds into one decision.

Metrics:
  binding_keep_acc : from z_A alone, reconstruct B's keep_in_root (cross-agent)
  bind_mse         : cross-view reconstruction error
  latent_comm      : coordinator from latents never loses KEEP
  a_alone_greedy   : agent A WITHOUT comm loses KEEP (the deadly baseline)
  text_comm        : coordinator from text handoff (works, but costs bits)
  bits_latent      : bytes to send the latent
  bits_text        : bytes to send the text view
"""
from proteus.paths import ROOT
import argparse
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import proteus.envs.cleanup_env as cleanup_env
from proteus.world.hjepa_torch import MLP, VICReg

OBS_DIM = cleanup_env.OBS_DIM
# partial views
VIEW_A_DIM = 6  # dims 0..5   (root txt/jpg/pdf + folder txt/jpg/pdf)
VIEW_B_DIM = 6  # dims 6..11  (messy, keep_in_root, keep_preserved, junk, total, junk_deleted)


def make_env():
    return cleanup_env.CleanupDesktop()


def split_views(vec):
    """vec: (12,) full obs -> (view_A (6,), view_B (6,))."""
    return vec[:VIEW_A_DIM].copy(), vec[VIEW_B_DIM:].copy()


class MultiAgentJEPA(nn.Module):
    """Two partial-view encoders + cross-agent binding decoders + fusion head.

    enc_A : view_A -> z   enc_B : view_B -> z        (each agent's belief)
    dec_B_from_A : z_A -> view_B    (A's latent reconstructs B's world = BINDING)
    dec_A_from_B : z_B -> view_A    (symmetric)
    fusion       : [z_A, z_B] -> full obs (coordinated reconstruction for the task)
    VICReg on both latents to prevent collapse.
    """

    def __init__(self, view_dim=6, latent_dim=32, hidden=128, full_obs_dim=OBS_DIM):
        super().__init__()
        self.view_dim = view_dim
        self.latent_dim = latent_dim
        self.full_obs_dim = full_obs_dim
        self.enc_A = MLP([view_dim, hidden, latent_dim])
        self.enc_B = MLP([view_dim, hidden, latent_dim])
        self.dec_B_from_A = MLP([latent_dim, hidden, view_dim])
        self.dec_A_from_B = MLP([latent_dim, hidden, view_dim])
        self.fusion = MLP([2 * latent_dim, hidden, full_obs_dim])
        self.vicreg = VICReg(gamma=1.0, mu=0.004)

    def encode_A(self, vA):
        return self.enc_A(vA)

    def encode_B(self, vB):
        return self.enc_B(vB)

    def forward(self, vA, vB):
        zA = self.enc_A(vA)
        zB = self.enc_B(vB)
        # binding: reconstruct the OTHER's view from this agent's latent
        rec_B_from_A = self.dec_B_from_A(zA)
        rec_A_from_B = self.dec_A_from_B(zB)
        # fusion: full world from both latents
        full = self.fusion(torch.cat([zA, zB], dim=-1))
        loss_bind = torch.mean((rec_B_from_A - vB) ** 2) + torch.mean((rec_A_from_B - vA) ** 2)
        loss_full = torch.mean((full - torch.cat([vA, vB], dim=-1)) ** 2)
        loss_vic = self.vicreg(zA) + self.vicreg(zB)
        total = loss_bind + loss_full + loss_vic
        return {
            "total": total, "bind": loss_bind, "full": loss_full, "vic": loss_vic,
            "zA": zA, "zB": zB, "rec_B_from_A": rec_B_from_A, "rec_A_from_B": rec_A_from_B,
            "full_pred": full,
        }

    @torch.no_grad()
    def coordinator_decision(self, vA, vB):
        """Coordinator receives both latents, decodes full obs, decides.
        Returns (action, predicted_full_obs)."""
        self.eval()
        zA = self.enc_A(torch.as_tensor(vA, dtype=torch.float32).unsqueeze(0))
        zB = self.enc_B(torch.as_tensor(vB, dtype=torch.float32).unsqueeze(0))
        full = self.fusion(torch.cat([zA, zB], dim=-1)).detach().cpu().numpy()[0]
        root_pdf = full[2]
        keep_in_root = full[7]
        # safe to delete iff there's a pdf AND it's not the KEEP one
        if root_pdf > 0.5 and keep_in_root < 0.5:
            return 3, full  # delete_pdf
        return 2, full  # move_pdf (safe default)

    @torch.no_grad()
    def agent_A_alone(self, vA):
        """No communication: A must guess. It sees a pdf; greedy assumes it's junk
        and deletes. This is the deadly baseline."""
        if vA[2] > 0.5:  # root pdf present
            return 3  # delete -> loses KEEP if it's the KEEP pdf
        return 2

    @torch.no_grad()
    def binding_keep_from_A(self, vA):
        """From A's latent alone, predict B's keep_in_root (dim 1 of view_B)."""
        zA = self.enc_A(torch.as_tensor(vA, dtype=torch.float32).unsqueeze(0))
        rec_B = self.dec_B_from_A(zA).detach().cpu().numpy()[0]
        return rec_B[1]  # view_B dim1 = keep_in_root


def collect_views(n=2000, seed=0):
    """Collect (full_obs) states across the world (random actions for variety)."""
    rng = random.Random(seed)
    states = []
    for _ in range(n):
        env = make_env()
        _, vec = env.reset(seed=rng.randint(0, 100000))
        states.append(vec.copy())
        for _ in range(rng.randint(0, 4)):
            a = rng.randint(0, cleanup_env.ACT_DIM - 1)
            _, vec, _, done, _ = env.step(a)
            states.append(vec.copy())
            if done:
                break
    return np.array(states, dtype=np.float32)


def train(model, states, epochs=400, batch=128, lr=3e-4, log_every=50):
    vA_all = torch.as_tensor(states[:, :VIEW_A_DIM], dtype=torch.float32)
    vB_all = torch.as_tensor(states[:, VIEW_B_DIM:], dtype=torch.float32)
    n = len(states)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    hist = {"total": [], "bind": [], "full": [], "vic": []}
    for epoch in range(epochs):
        idx = rng.choice(n, batch, replace=False)
        vA, vB = vA_all[idx], vB_all[idx]
        opt.zero_grad()
        losses = model(vA, vB)
        losses["total"].backward()
        opt.step()
        for k in hist:
            hist[k].append(float(losses[k].item()))
        if log_every and (epoch + 1) % log_every == 0:
            print(f"epoch {epoch+1:4d} bind {hist['bind'][-1]:.4f} full {hist['full'][-1]:.4f} "
                  f"vic {hist['vic'][-1]:.4f} total {hist['total'][-1]:.4f}")
    return hist


def eval_binding(model, states, batch=512):
    idx = np.random.default_rng(1).choice(len(states), min(batch, len(states)), replace=False)
    vA = states[idx][:, :VIEW_A_DIM]
    vB = states[idx][:, VIEW_B_DIM:]
    keep_true = vB[:, 1]
    keep_pred = np.array([model.binding_keep_from_A(v) for v in vA])
    acc = float(np.mean((keep_pred > 0.5) == (keep_true > 0.5)))
    mse = float(np.mean((keep_pred - keep_true) ** 2))
    return {"binding_keep_acc": acc, "bind_mse": mse}


def eval_task(model, trials=60):
    """Cooperative task: coordinator must never lose KEEP. Baselines:
    latent comm (both latents), A-alone greedy (no comm), text comm (ground truth)."""
    model.eval()
    lat_ok, aalone_ok, text_ok = 0, 0, 0
    keep_lost_lat, keep_lost_aalone, keep_lost_text = 0, 0, 0
    trials_run = 0
    for t in range(trials):
        env = make_env()
        _, vec = env.reset(seed=t * 13 + 5)
        trials_run += 1
        vA, vB = split_views(vec)

        # --- latent comm ---
        env2 = make_env(); _, _ = env2.reset(seed=t * 13 + 5)
        action, full_pred = model.coordinator_decision(vA, vB)
        _, _, r, done, info = env2.step(action)
        if not info.get("keep_lost"):
            lat_ok += 1
        else:
            keep_lost_lat += 1

        # --- A alone (no comm) ---
        env3 = make_env(); _, _ = env3.reset(seed=t * 13 + 5)
        action = model.agent_A_alone(vA)
        _, _, r, done, info = env3.step(action)
        if not info.get("keep_lost"):
            aalone_ok += 1
        else:
            keep_lost_aalone += 1

        # --- text comm (ground truth view_B as text) ---
        env4 = make_env(); _, _ = env4.reset(seed=t * 13 + 5)
        keep_in_root = vB[1]  # B's view, ground truth
        action = 3 if (vA[2] > 0.5 and keep_in_root < 0.5) else 2
        _, _, r, done, info = env4.step(action)
        if not info.get("keep_lost"):
            text_ok += 1
        else:
            keep_lost_text += 1

    n = max(trials_run, 1)
    # text handoff size: B renders its view
    env = make_env(); _, vec = env.reset(seed=5)
    vB_text = cleanup_env.ACTIONS and vec[6:12].tolist()
    text_bytes = len(repr(vB_text))
    latent_bytes = model.latent_dim * 4  # fp32
    return {
        "latent_comm_safe": lat_ok / n,
        "latent_comm_keep_lost": keep_lost_lat / n,
        "a_alone_safe": aalone_ok / n,
        "a_alone_keep_lost": keep_lost_aalone / n,
        "text_comm_safe": text_ok / n,
        "text_comm_keep_lost": keep_lost_text / n,
        "bits_latent_bytes": latent_bytes,
        "bits_text_bytes": text_bytes,
        "compression_ratio": text_bytes / latent_bytes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=2500)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--no-train", action="store_true")
    args = ap.parse_args()

    ckpt = pathlib.Path(args.ckpt or ROOT / "checkpoints" / "multiagent.pt")
    ckpt.parent.mkdir(exist_ok=True)

    torch.manual_seed(0)
    model = MultiAgentJEPA(view_dim=VIEW_A_DIM, latent_dim=args.latent,
                           hidden=args.hidden, full_obs_dim=OBS_DIM)

    print("=" * 64)
    print("MULTI-AGENT LATENT COMMUNICATION (Phase 4)")
    print("=" * 64)
    print(f"\n[collect] {args.states} world states")
    t0 = time.time()
    states = collect_views(n=args.states)
    print(f"  {len(states)} states in {time.time()-t0:.1f}s")

    if not args.no_train:
        print(f"\n[train] {args.epochs} epochs, latent {args.latent}")
        t0 = time.time()
        hist = train(model, states, epochs=args.epochs, log_every=args.epochs // 10)
        print(f"  train wall {time.time()-t0:.1f}s | final total {hist['total'][-1]:.4f}")
        torch.save(model.state_dict(), ckpt)
        print(f"  saved {ckpt}")
    else:
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        print(f"[load] {ckpt}")

    print("\n[test 1/2] cross-agent binding")
    b = eval_binding(model, states)
    for k_, v in b.items():
        print(f"  {k_}: {v:.3f}")

    print("\n[test 2/2] cooperative safety task")
    res = eval_task(model, trials=60)
    for k_, v in res.items():
        print(f"  {k_}: {v:.3f}")

    print("\n" + "=" * 64)
    print("DONE")
    print("=" * 64)


if __name__ == "__main__":
    main()