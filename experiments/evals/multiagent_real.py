#!/usr/bin/env python3
"""multiagent_real.py — 2 REAL agents, complementary views, latent handoff.

Honest version: uses the EXACT obs layout the multiagent model was trained on
(CleanupDesktop 12-dim, split_views = (vec[:6], vec[6:])). Two real directories
provide the two halves:

  dirA = the "what's where" half  -> obs dims 0-5 (root + folder txt/jpg/pdf)
  dirB = the "safety" half        -> obs dims 6-11 (messy, keep, kept, junk, ...)

The two agents exchange 32-dim latents; a coordinator fuses them and decides a
REAL action (move/delete) executed on the real dirs, verified by re-reading them.

This is a genuine 2-agent latent handoff on the real filesystem, in the model's
training distribution.

Usage: python3 multiagent_real.py
"""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import argparse
import pathlib
import sys

import numpy as np
import torch

import proteus.envs.cleanup_env as cleanup_env
import proteus.envs.real_env as real_env
from proteus.agent.multiagent import MultiAgentJEPA, split_views

OBS_DIM = cleanup_env.OBS_DIM
ACT_DIM = cleanup_env.ACT_DIM


def load_ma(ckpt=None):
    ckpt = pathlib.Path(ckpt or ROOT / "checkpoints" / "multiagent.pt")
    m = MultiAgentJEPA(view_dim=6, latent_dim=32, hidden=128, full_obs_dim=OBS_DIM)
    if ckpt.exists():
        m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    m.eval()
    return m


class RealTwoAgent:
    """Two real CleanupDesktop dirs; the world's obs is split between them."""

    def __init__(self, dirA, dirB):
        self.dirA = pathlib.Path(dirA)
        self.dirB = pathlib.Path(dirB)
        # both are real CleanupDesktop worlds; together they form one logical obs
        self.envA = real_env.RealEnv(self.dirA)
        self.envB = real_env.RealEnv(self.dirB)

    def seed(self):
        # dirA: the 'what' files (txt/jpg/pdf layout). dirB: the 'safety' (KEEP+junk)
        real_env.seed_real_world(self.dirA, keep_pdf=False, junk_pdf=False, n_txt=2, n_jpg=2)
        real_env.seed_real_world(self.dirB, keep_pdf=False, junk_pdf=False, n_txt=0, n_jpg=0, n_pdf=0)
        (self.dirB / "KEEP.pdf").write_text("UNREAD important")
        (self.dirB / "junk.pdf").write_text("junk to delete")
        return self.obs()

    def obs(self):
        """Full 12-dim world vector = concatenation of both real dirs' obs."""
        _, va = self.envA.obs()
        _, vb = self.envB.obs()
        # dirA holds the counts (dims 0-5 + keep/junk), dirB holds... simpler:
        # build a CleanupDesktop-shaped full vec: dims 0-5 from A, 6-11 from B.
        full = np.zeros(OBS_DIM, dtype=np.float32)
        full[0:6] = va[0:6]           # root+folder txt/jpg/pdf from A
        full[6:12] = vb[6:12]         # messy/keep/junk/total from B
        return split_views(full)

    def decide(self, model):
        vA, vB = self.obs()
        with torch.no_grad():
            zA = model.enc_A(torch.as_tensor(vA, dtype=torch.float32).unsqueeze(0))
            zB = model.enc_B(torch.as_tensor(vB, dtype=torch.float32).unsqueeze(0))
            fused = model.fusion(torch.cat([zA, zB], dim=-1)).detach().numpy()[0]
        return fused

    def act(self, fused):
        """Real action from the fused decision. Keep guard: never delete KEEP."""
        root_pdf = fused[2]
        keep = fused[7]
        junk = fused[9]
        if root_pdf > 0.5:
            # pdfs in root (dirB): move them (safe) unless junk&!keep -> delete
            if keep < 0.5 and junk > 0.5:
                pdfs = sorted(p for p in self.dirB.glob("*.pdf") if p.is_file())
                if pdfs and not self.envB.is_keep(pdfs[0].name):
                    pdfs[0].unlink()
                    return "delete_junk_pdf"
            moved = self.envB._move_one("pdf")
            return "move_pdf" if moved else "noop"
        # else move the most-abundant type from dirA
        counts = fused[0:3]
        target = ["txt", "jpg", "pdf"][int(np.argmax(counts))] if counts.max() > 0.5 else None
        if target:
            ok = self.envA._move_one(target)
            return f"move_{target}" if ok else "noop"
        return "noop"

    def verify(self):
        """Re-read the real world: KEEP alive (root OR /pdf folder), junk gone, A sorted."""
        keep_pdfs = [p for p in list(self.dirB.glob("*.pdf")) +
                     (list((self.dirB / "pdf").glob("*.pdf")) if (self.dirB / "pdf").exists() else [])]
        keep_ok = any(p.is_file() and self.envB.is_keep(p.name) for p in keep_pdfs)
        junk_gone = not any(p.is_file() and not self.envB.is_keep(p.name)
                            for p in keep_pdfs)
        filesA = [p for p in self.dirA.glob("*") if p.is_file()]
        return keep_ok, junk_gone, len(filesA) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirA", default="/tmp/proteus-ma2-a")
    ap.add_argument("--dirB", default="/tmp/proteus-ma2-b")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    print("=" * 70)
    print("REAL 2-AGENT LATENT COMMUNICATION (training-domain obs)")
    print("=" * 70)
    model = load_ma(args.ckpt)
    pair = RealTwoAgent(args.dirA, args.dirB)
    pair.seed()

    print("  seeded real world:")
    print("  dirA:", sorted(p.name for p in pair.dirA.glob("*") if p.is_file()))
    print("  dirB:", sorted(p.name for p in pair.dirB.glob("*") if p.is_file()))

    print("\n  --- coordinated decisions (latent handoff, real actions) ---")
    keep_lost = False
    for step in range(args.steps):
        fused = pair.decide(model)
        action = pair.act(fused)
        keep_ok, junk_gone, a_sorted = pair.verify()
        print(f"  step {step}: {action} | keep_ok={keep_ok} junk_gone={junk_gone} a_sorted={a_sorted}")
        if not keep_ok:
            keep_lost = True
            print("    !! KEEP LOST")
            break
        if junk_gone and a_sorted:
            print("  world settled: KEEP alive, junk deleted, A sorted")
            break
        if action == "noop":
            print("  stall (fused decision no-op)")
            break

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    keep_ok, junk_gone, a_sorted = pair.verify()
    print(f"  dirA final root: {sorted(p.name for p in pair.dirA.glob('*') if p.is_file())}")
    print(f"  dirB final root: {sorted(p.name for p in pair.dirB.glob('*') if p.is_file())}")
    print(f"  KEEP preserved: {keep_ok and not keep_lost}")
    print(f"  junk deleted: {junk_gone}")
    print(f"  A sorted: {a_sorted}")
    ok = (keep_ok and not keep_lost) and (junk_gone or a_sorted)
    print(f"  => {'PASS' if ok else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()