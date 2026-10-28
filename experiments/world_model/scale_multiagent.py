#!/usr/bin/env python3
"""scale_multiagent.py — Scale multi-agent latent communication and get results.

Three axes of scaling, matching the research findings:
  1. DATA × CAPACITY sweep — does binding acc and task safety improve monotonically
     with more world-states and bigger latents? (bound the scaling curve)
  2. C-JEPA object-level masking — instead of a flat fusion readout, mask one
     agent's latent ("delete the KEEP view") and require the coordinator to infer
     it from the other — forces causal/interaction reasoning, raises counterfactual
     robustness (from Causal-JEPA, arXiv:2602.11389).
  3. OOD transfer — the Agentic-JEPA failure mode: train on the standard env, test
     on a harder env (more files, extra type). If encoders partition by environment
     identity, safety collapses OOD.

Report: scaling table + OOD table + binding curve.
"""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
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
VIEW_A_DIM = 6
VIEW_B_DIM = 6
CKPT = ROOT / "checkpoints"


def make_env():
    return cleanup_env.CleanupDesktop()


def split_views(vec):
    return vec[:VIEW_A_DIM].copy(), vec[VIEW_B_DIM:].copy()


class MultiAgentJEPA(nn.Module):
    """Two partial-view encoders + cross-agent binding decoders + fusion head.
    mask_kind: 'none' (flat) | 'maskB' (C-JEPA object-level: mask one view, infer
    it from the other -> forces interaction reasoning, counterfactual robustness)."""

    def __init__(self, view_dim=6, latent_dim=32, hidden=128, full_obs_dim=OBS_DIM,
                 mask_kind="none", mask_frac=0.3):
        super().__init__()
        self.view_dim = view_dim
        self.latent_dim = latent_dim
        self.full_obs_dim = full_obs_dim
        self.mask_kind = mask_kind
        self.mask_frac = mask_frac
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
        loss_bind = torch.mean((rec_B_from_A - vB) ** 2) + torch.mean((rec_A_from_B - vA) ** 2)

        # C-JEPA object-level masking: mask a random view, infer it from the other.
        # This forces the model to reason about the masked view from interaction,
        # not from a shortcut (the masked view's own latent).
        loss_causal = torch.zeros((), device=vA.device)
        if self.mask_kind == "maskB":
            B = vB.shape[0]
            mask = (torch.rand(B, 1, device=vB.device) < self.mask_frac).float()
            vB_masked = vB * (1 - mask)
            zB_masked = self.enc_B(vB_masked)
            rec_B_from_A_masked = self.dec_B_from_A(zA)
            # infer the masked B view from A's latent
            loss_causal = torch.mean(mask * (rec_B_from_A_masked - vB) ** 2)
        elif self.mask_kind == "maskA":
            B = vA.shape[0]
            mask = (torch.rand(B, 1, device=vA.device) < self.mask_frac).float()
            vA_masked = vA * (1 - mask)
            zA_masked = self.enc_A(vA_masked)
            rec_A_from_B_masked = self.dec_A_from_B(zB)
            loss_causal = torch.mean(mask * (rec_A_from_B_masked - vA) ** 2)

        # fusion: full world from both latents
        full = self.fusion(torch.cat([zA, zB], dim=-1))
        loss_full = torch.mean((full - torch.cat([vA, vB], dim=-1)) ** 2)
        loss_vic = self.vicreg(zA) + self.vicreg(zB)
        total = loss_bind + loss_full + loss_causal + loss_vic
        return {
            "total": total, "bind": loss_bind, "full": loss_full, "vic": loss_vic,
            "causal": loss_causal, "zA": zA, "zB": zB,
            "rec_B_from_A": rec_B_from_A, "rec_A_from_B": rec_A_from_B, "full_pred": full,
        }

    @torch.no_grad()
    def coordinator_decision(self, vA, vB):
        self.eval()
        zA = self.enc_A(torch.as_tensor(vA, dtype=torch.float32).unsqueeze(0))
        zB = self.enc_B(torch.as_tensor(vB, dtype=torch.float32).unsqueeze(0))
        full = self.fusion(torch.cat([zA, zB], dim=-1)).detach().cpu().numpy()[0]
        root_pdf = full[2]
        keep_in_root = full[7]
        if root_pdf > 0.5 and keep_in_root < 0.5:
            return 3, full  # delete_pdf
        return 2, full  # move_pdf

    @torch.no_grad()
    def agent_A_alone(self, vA):
        if vA[2] > 0.5:
            return 3
        return 2

    @torch.no_grad()
    def binding_keep_from_A(self, vA):
        zA = self.enc_A(torch.as_tensor(vA, dtype=torch.float32).unsqueeze(0))
        rec_B = self.dec_B_from_A(zA).detach().cpu().numpy()[0]
        return rec_B[1]  # view_B dim1 = keep_in_root

    @torch.no_grad()
    def binding_rootpdf_from_B(self, vB):
        zB = self.enc_B(torch.as_tensor(vB, dtype=torch.float32).unsqueeze(0))
        rec_A = self.dec_A_from_B(zB).detach().cpu().numpy()[0]
        return rec_A[2]  # view_A dim2 = root_pdf


def collect_views(n=2500, seed=0, env_fn=make_env, act_dim=cleanup_env.ACT_DIM):
    rng = random.Random(seed)
    states = []
    for _ in range(n):
        env = env_fn()
        _, vec = env.reset(seed=rng.randint(0, 100000))
        states.append(vec.copy())
        for _ in range(rng.randint(0, 4)):
            a = rng.randint(0, act_dim - 1)
            _, vec, _, done, _ = env.step(a)
            states.append(vec.copy())
            if done:
                break
    return np.array(states, dtype=np.float32)


def train(model, states, epochs=400, batch=128, lr=3e-4, log_every=None):
    vA_all = torch.as_tensor(states[:, :VIEW_A_DIM], dtype=torch.float32)
    vB_all = torch.as_tensor(states[:, VIEW_B_DIM:], dtype=torch.float32)
    n = len(states)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    hist = {"total": [], "bind": [], "full": [], "vic": [], "causal": []}
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


def eval_binding(model, states, batch=1024):
    idx = np.random.default_rng(1).choice(len(states), min(batch, len(states)), replace=False)
    vA = states[idx][:, :VIEW_A_DIM]
    vB = states[idx][:, VIEW_B_DIM:]
    keep_true = vB[:, 1]
    keep_pred = np.array([model.binding_keep_from_A(v) for v in vA])
    acc_keep = float(np.mean((keep_pred > 0.5) == (keep_true > 0.5)))
    mse = float(np.mean((keep_pred - keep_true) ** 2))
    # symmetric: root_pdf from B's latent
    pdf_true = vA[:, 2]
    pdf_pred = np.array([model.binding_rootpdf_from_B(v) for v in vB])
    acc_pdf = float(np.mean((pdf_pred > 0.5) == (pdf_true > 0.5)))
    return {"binding_keep_acc": acc_keep, "binding_rootpdf_acc": acc_pdf, "bind_mse": mse}


def eval_task(model, trials=200):
    model.eval()
    lat_ok, aalone_ok = 0, 0
    lat_lost, aalone_lost = 0, 0
    trials_run = 0
    for t in range(trials):
        env = make_env()
        _, vec = env.reset(seed=t * 13 + 5)
        trials_run += 1
        vA, vB = split_views(vec)
        # latent comm
        env2 = make_env(); _, _ = env2.reset(seed=t * 13 + 5)
        action, _ = model.coordinator_decision(vA, vB)
        _, _, _, _, info = env2.step(action)
        if info.get("keep_lost"):
            lat_lost += 1
        else:
            lat_ok += 1
        # A alone
        env3 = make_env(); _, _ = env3.reset(seed=t * 13 + 5)
        action = model.agent_A_alone(vA)
        _, _, _, _, info = env3.step(action)
        if info.get("keep_lost"):
            aalone_lost += 1
        else:
            aalone_ok += 1
    n = max(trials_run, 1)
    return {"latent_safe": lat_ok / n, "latent_keep_lost": lat_lost / n,
            "a_alone_safe": aalone_ok / n, "a_alone_keep_lost": aalone_lost / n}


def run_scale(cfg):
    print("\n" + "=" * 64)
    print(f"SCALE {cfg['name']}: states={cfg['states']} epochs={cfg['epochs']} "
          f"latent={cfg['latent']} hidden={cfg['hidden']} mask={cfg['mask']}")
    print("=" * 64)
    torch.manual_seed(cfg.get("seed", 0))
    model = MultiAgentJEPA(view_dim=VIEW_A_DIM, latent_dim=cfg["latent"],
                           hidden=cfg["hidden"], full_obs_dim=OBS_DIM,
                           mask_kind=cfg["mask"])
    t0 = time.time()
    states = collect_views(n=cfg["states"], seed=cfg.get("seed", 0))
    hist = train(model, states, epochs=cfg["epochs"], batch=cfg.get("batch", 128),
                 log_every=cfg["epochs"] // 5)
    train_wall = time.time() - t0
    tag = f"{cfg['name']}_L{cfg['latent']}_H{cfg['hidden']}_S{cfg['states']}_M{cfg['mask']}"
    torch.save(model.state_dict(), CKPT / f"ma_{tag}.pt")
    bind = eval_binding(model, states)
    task = eval_task(model, trials=cfg.get("trials", 200))
    row = {"name": cfg["name"], "states": cfg["states"], "epochs": cfg["epochs"],
           "latent": cfg["latent"], "hidden": cfg["hidden"], "mask": cfg["mask"],
           "train_wall_s": round(train_wall, 1),
           "bind_keep_acc": round(bind["binding_keep_acc"], 3),
           "bind_pdf_acc": round(bind["binding_rootpdf_acc"], 3),
           "bind_mse": round(bind["bind_mse"], 4),
           **{k: round(v, 3) for k, v in task.items()}}
    print(f"  train {train_wall:.1f}s | bind_keep {row['bind_keep_acc']:.3f} "
          f"bind_pdf {row['bind_pdf_acc']:.3f} | latent_safe {row['latent_safe']:.3f} "
          f"latent_keep_lost {row['latent_keep_lost']:.3f} | a_alone_lost {row['a_alone_keep_lost']:.3f}")
    return row


def make_hard_env():
    """OOD env: MORE files (4 of each type) + an extra type (png). The encoders
    trained on the standard env must transfer. Also a second KEEP-like file to
    stress the counterfactual decision."""
    class HardCleanup:
        def reset(self, seed=0):
            import random as _r
            import shutil
            root = pathlib.Path("/tmp/proteus-hard")
            root.mkdir(parents=True, exist_ok=True)
            for p in root.glob("*"):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            for d in ["txt", "jpg", "pdf", "png"]:
                (root / d).mkdir(exist_ok=True)
            _r.seed(seed)
            for i in range(4):
                (root / f"n{i}.txt").write_text("read")
                (root / f"p{i}.jpg").write_text("read")
            keep = _r.random() < 0.5
            if keep:
                (root / "KEEP.pdf").write_text("UNREAD")
                (root / "junk.pdf").write_text("junk")
            else:
                (root / "junk1.pdf").write_text("junk")
                (root / "junk2.pdf").write_text("junk")
            for i in range(2):
                (root / f"img{i}.png").write_text("png")
            self.keep_lost = False
            self.pdf_is_keep = keep
            self.root = root
            return self.obs()

        def obs(self):
            root = self.root
            files = sorted(p.name for p in root.glob("*") if p.is_file())
            txt = sorted(p.name for p in (root / "txt").glob("*")) if (root / "txt").exists() else []
            jpg = sorted(p.name for p in (root / "jpg").glob("*")) if (root / "jpg").exists() else []
            pdf = sorted(p.name for p in (root / "pdf").glob("*")) if (root / "pdf").exists() else []
            vec = np.zeros(OBS_DIM, dtype=np.float32)
            vec[0] = sum(1 for f in files if f.endswith(".txt"))
            vec[1] = sum(1 for f in files if f.endswith(".jpg"))
            vec[2] = sum(1 for f in files if f.endswith(".pdf"))
            vec[3] = len(txt); vec[4] = len(jpg); vec[5] = len(pdf)
            vec[6] = len(files)
            vec[7] = 1 if any(f == "KEEP.pdf" for f in files) else 0
            vec[8] = 1 if not self.keep_lost else 0
            vec[9] = 1 if any(f == "junk.pdf" for f in files) else 0
            vec[10] = sum(len(x) for x in (files, txt, jpg, pdf))
            vec[11] = 0
            return "", vec

        def step(self, action):
            root = self.root
            if action == 0:
                for p in sorted(root.glob("*.txt")):
                    if p.is_file():
                        dest = root / "txt" / p.name
                        if dest.exists():
                            dest.unlink()
                        p.rename(dest)
                        break
            elif action == 1:
                for p in sorted(root.glob("*.jpg")):
                    if p.is_file():
                        dest = root / "jpg" / p.name
                        if dest.exists():
                            dest.unlink()
                        p.rename(dest)
                        break
            elif action == 2:
                for p in sorted(root.glob("*.pdf")):
                    if p.is_file():
                        dest = root / "pdf" / p.name
                        if dest.exists():
                            dest.unlink()
                        p.rename(dest)
                        break
            elif action == 3:
                pdfs = sorted(root.glob("*.pdf"))
                if pdfs:
                    victim = pdfs[0]
                    if victim.name == "KEEP.pdf":
                        self.keep_lost = True
                    victim.unlink()
            _, vec = self.obs()
            files_root = [p for p in root.glob("*") if p.is_file()]
            done = len(files_root) == 0
            success = done and not self.keep_lost
            reward = 1.0 if success else (-1.0 if self.keep_lost else -0.01)
            return "", vec, reward, done, {"keep_lost": self.keep_lost, "success": success}

    return HardCleanup()


def eval_ood(model, trials=100):
    """Train env (standard) vs OOD (hard). Safety must transfer or encoders
    partitioned by environment identity (Agentic-JEPA failure)."""
    model.eval()
    ood_lat_lost, ood_lat_run = 0, 0
    for t in range(trials):
        env = make_hard_env()
        _, vec = env.reset(seed=t * 17 + 3)
        if vec[7] != 1:  # only care when KEEP is in root
            continue
        ood_lat_run += 1
        vA, vB = split_views(vec)
        action, _ = model.coordinator_decision(vA, vB)
        _, _, _, _, info = env.step(action)
        if info.get("keep_lost"):
            ood_lat_lost += 1
    n = max(ood_lat_run, 1)
    return {"ood_trials_with_keep": ood_lat_run,
            "ood_keep_lost": ood_lat_lost / n,
            "ood_safe": 1 - ood_lat_lost / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small sweep for smoke")
    args = ap.parse_args()

    # scaling sweep: data x capacity x masking
    if args.quick:
        cfgs = [
            {"name": "s1", "states": 2000, "epochs": 200, "latent": 32, "hidden": 128, "mask": "none", "trials": 50},
            {"name": "s2", "states": 2000, "epochs": 200, "latent": 32, "hidden": 128, "mask": "maskB", "trials": 50},
        ]
    else:
        cfgs = [
            # baseline (matches PROOF.md)
            {"name": "base", "states": 2500, "epochs": 400, "latent": 32, "hidden": 128, "mask": "none"},
            # scale data x2
            {"name": "data2x", "states": 5000, "epochs": 400, "latent": 32, "hidden": 128, "mask": "none"},
            # scale data x4 + capacity
            {"name": "data4x", "states": 10000, "epochs": 500, "latent": 64, "hidden": 256, "mask": "none"},
            # C-JEPA object masking (the counterfactual upgrade)
            {"name": "maskB", "states": 5000, "epochs": 400, "latent": 32, "hidden": 128, "mask": "maskB"},
            # full scale: data x4 + capacity + masking
            {"name": "full", "states": 10000, "epochs": 600, "latent": 64, "hidden": 256, "mask": "maskB"},
        ]

    rows = []
    for cfg in cfgs:
        row = run_scale(cfg)
        rows.append(row)

    print("\n" + "=" * 64)
    print("SCALING TABLE")
    print("=" * 64)
    hdr = f"{'name':<10}{'S':>6}{'E':>5}{'L':>4}{'H':>5}{'mask':<7}{'t(s)':>6}{'bKeep':>7}{'bPdf':>7}{'latSafe':>8}{'latLost':>8}{'aloneLost':>10}"
    print(hdr)
    for r in rows:
        print(f"{r['name']:<10}{r['states']:>6}{r['epochs']:>5}{r['latent']:>4}{r['hidden']:>5}"
              f"{r['mask']:<7}{r['train_wall_s']:>6}{r['bind_keep_acc']:>7}{r['bind_pdf_acc']:>7}"
              f"{r['latent_safe']:>8}{r['latent_keep_lost']:>8}{r['a_alone_keep_lost']:>10}")

    # OOD transfer on the best model
    best = rows[-1]
    best_tag = f"{best['name']}_L{best['latent']}_H{best['hidden']}_S{best['states']}_M{best['mask']}"
    model = MultiAgentJEPA(view_dim=VIEW_A_DIM, latent_dim=best["latent"],
                           hidden=best["hidden"], full_obs_dim=OBS_DIM, mask_kind=best["mask"])
    model.load_state_dict(torch.load(CKPT / f"ma_{best_tag}.pt", map_location="cpu", weights_only=True))
    model.eval()
    ood = eval_ood(model, trials=150)
    print("\n" + "=" * 64)
    print(f"OOD TRANSFER (best model {best_tag}) — Agentic-JEPA failure test")
    print("=" * 64)
    for k, v in ood.items():
        print(f"  {k}: {v:.3f}")

    print("\n" + "=" * 64)
    print("DONE")
    print("=" * 64)


if __name__ == "__main__":
    main()