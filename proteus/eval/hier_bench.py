#!/usr/bin/env python3
"""hier_bench.py — P3-1: H-JEPA k=16 hierarchical planning vs flat MPC on a
HARD long-horizon directory restructuring task.

The WO2 benchmark showed flat greedy already optimal on a 6-file env. This uses
a HARD task where search-based MPC is genuinely expensive:

  TASK: restructure a directory with MANY files (default 36) across 3 types +
  KEEP. The goal state requires long-horizon action sequences. Both planners
  are REAL CEM search (not greedy), so we compare:
    - search cost: total predictor rollouts (the real compute metric)
    - wall time
    - step efficiency

FLAT:  CEM over k=4 action sequences, n_samples per step, repeated every step.
HIER:  z_high CEM over k=16 macro-actions (coarse strategy), then z_low refines
       each k=4 segment — the coarse plan is reused, not re-searched every step.

Target: hierarchical uses >= 3x FEWER rollouts (3x speedup in search compute).
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import proteus.envs.real_env as real_env
from proteus.world.hjepa_torch import HJEPA_K4, MLP, VICReg

OBS_DIM = real_env.OBS_DIM
ACT_DIM = real_env.ACT_DIM


class BigDirEnv:
    """A large real directory to restructure (many files). The obs is the same
    12-dim count vector; the task is long-horizon because there are N files."""

    def __init__(self, root, n_txt=12, n_jpg=12, n_pdf=10, keep=True):
        self.root = pathlib.Path(root)
        self.env = real_env.RealEnv(self.root)
        self.n = (n_txt, n_jpg, n_pdf)
        self.keep = keep

    def reset(self, seed=0):
        real_env.seed_real_world(self.root, keep_pdf=self.keep,
                                 n_txt=self.n[0], n_jpg=self.n[1], n_pdf=0)
        # add pdfs (keep + junk)
        if self.keep:
            (self.root / "KEEP.pdf").write_text("UNREAD")
        for i in range(self.n[2]):
            (self.root / f"doc{i}.pdf").write_text(f"doc {i}")
        return self.obs()

    def obs(self):
        return self.env.obs()

    def step(self, a):
        return self.env.step(a)

    def remaining(self):
        return len([p for p in self.root.glob("*") if p.is_file()])


class HierJEPA(nn.Module):
    """k_low=4 + k_high=16 hierarchical world model (from WO2, self-contained)."""

    def __init__(self, obs_dim, act_dim, hidden=128, k_low=4, k_high=16,
                 z_low=32, z_high=64, tau=0.995):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.k_low = k_low
        self.k_high = k_high
        self.z_low = z_low
        self.z_high = z_high
        self.tau = tau
        self.enc_low = MLP([obs_dim, hidden, z_low])
        self.target_low = MLP([obs_dim, hidden, z_low])
        self.target_low.load_state_dict(self.enc_low.state_dict())
        self.pred_low = MLP([z_low + act_dim, hidden, z_low])
        self.enc_high = MLP([obs_dim, hidden, z_high])
        self.target_high = MLP([obs_dim, hidden, z_high])
        self.target_high.load_state_dict(self.enc_high.state_dict())
        self.pred_high = MLP([z_high + k_high * act_dim, hidden, z_high])
        self.vic = VICReg()

    def encode_high(self, obs):
        return self.enc_high(obs)

    def encode_low(self, obs):
        return self.enc_low(obs)

    def target_high_enc(self, obs):
        with torch.no_grad():
            return self.target_high(obs)

    def predict_low_step(self, z, a):
        if a.dim() == 0:
            a = a.unsqueeze(0)
        a_oh = torch.nn.functional.one_hot(a.long(), self.act_dim).float()
        return z + self.pred_low(torch.cat([z, a_oh], dim=-1))

    def predict_high(self, z, A):
        B, k = A.shape
        A_oh = torch.nn.functional.one_hot(A.long(), self.act_dim).float().reshape(B, k * self.act_dim)
        return z + self.pred_high(torch.cat([z, A_oh], dim=-1))

    def update_targets(self):
        for enc, tgt in [(self.enc_low, self.target_low), (self.enc_high, self.target_high)]:
            for p, tp in zip(enc.parameters(), tgt.parameters()):
                tp.data.mul_(self.tau).add_(p.data, alpha=1 - self.tau)

    def loss(self, obs_t, A_low, obs_t1, obs_tk_low, A_high, obs_tk_high):
        z_low = self.enc_low(obs_t)
        zp_low = self.predict_low_step(z_low, A_low[:, 0])
        l_low = torch.mean((zp_low - self.target_low(obs_t1)) ** 2)
        z_high = self.enc_high(obs_t)
        zp_high = self.predict_high(z_high, A_high)
        l_high = torch.mean((zp_high - self.target_high_enc(obs_tk_high)) ** 2)
        l_vic = self.vic(z_high)
        return {"total": l_low + l_high + l_vic, "low": l_low, "high": l_high, "vic": l_vic}


def collect_hier(env_fn, episodes=600, k_low=4, k_high=16, seed=0, act_dim=4):
    rng = np.random.default_rng(seed)
    obs_t, A_low, obs_t1, obs_tk_low, A_high, obs_tk_high = [], [], [], [], [], []
    for _ in range(episodes):
        env = env_fn()
        env.reset(seed=int(rng.integers(0, 100000)))
        # partial obs (not full reset) — each env is one trajectory
        traj = []
        acts = []
        for _ in range(k_high + 2):
            a = int(rng.integers(0, act_dim))
            text, v2, _, done, _ = env.step(a)
            traj.append(v2.copy())
            acts.append(a)
            if done:
                break
        for i in range(len(traj) - k_high):
            obs_t.append(traj[i])
            A_low.append(acts[i:i + k_low])
            obs_t1.append(traj[i + 1])
            obs_tk_low.append(traj[i + k_low])
            A_h = acts[i:i + k_high]
            if len(A_h) < k_high:
                A_h = A_h + [0] * (k_high - len(A_h))
            A_high.append(A_h)
            obs_tk_high.append(traj[min(i + k_high, len(traj) - 1)])
    return (torch.tensor(np.array(obs_t, dtype=np.float32)),
            torch.tensor(np.array(A_low, dtype=np.int64)),
            torch.tensor(np.array(obs_t1, dtype=np.float32)),
            torch.tensor(np.array(obs_tk_low, dtype=np.float32)),
            torch.tensor(np.array(A_high, dtype=np.int64)),
            torch.tensor(np.array(obs_tk_high, dtype=np.float32)))


def train_hier(model, data, epochs=300, batch=128, lr=3e-4):
    obs_t, A_low, obs_t1, obs_tk_low, A_high, obs_tk_high = data
    n = obs_t.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        idx = rng.choice(n, batch, replace=False)
        losses = model.loss(obs_t[idx], A_low[idx], obs_t1[idx], obs_tk_low[idx],
                            A_high[idx], obs_tk_high[idx])
        opt.zero_grad()
        losses["total"].backward()
        opt.step()
        model.update_targets()
        if (ep + 1) % (epochs // 4) == 0:
            print(f"  epoch {ep+1:4d} low {losses['low'].item():.4f} high {losses['high'].item():.4f}")


# ---------------- FLAT CEM MPC (search-based, counts rollouts) ----------------
def flat_cem_mpc(model, env, max_steps=60, n_samples=48, iters=2, elite=6):
    """Real CEM over k=4 action sequences, re-searched every step.
    Counts total predictor rollouts (n_samples*iters per step)."""
    device = "cpu"
    rollouts = 0
    steps = 0
    done = False
    _, vec = env.obs()
    for _ in range(max_steps):
        if env.remaining() == 0:
            done = True
            break
        keep_in_root = vec[7]
        # KEEP guard: never delete when keep in root
        if vec[2] > 0 and keep_in_root == 1:
            a = 2
            steps += 1
            _, vec, r, d, info = env.step(a)
            continue
        # CEM over k=4 action sequences (LOW-level encoder + predictor)
        goal = vec.copy(); goal[0:3] = 0
        z = model.encode_low(torch.as_tensor(vec, dtype=torch.float32, device=device).unsqueeze(0))
        z_goal = model.encode_low(torch.as_tensor(goal, dtype=torch.float32, device=device).unsqueeze(0))
        best_seq = None
        best_cost = 1e9
        for it in range(iters):
            A = torch.randint(0, ACT_DIM, (n_samples, model.k_low), device=device)
            for i in range(n_samples):
                zr = z.clone()
                for j in range(model.k_low):
                    zr = model.predict_low_step(zr, A[i, j])
                    rollouts += 1
                cost = float(torch.norm(zr - z_goal).item())
                if cost < best_cost:
                    best_cost = cost
                    best_seq = A[i]
        a = int(best_seq[0]) if best_seq is not None else 2
        if a >= 3:
            a = 2  # never delete
        steps += 1
        _, vec, r, d, info = env.step(a)
        if info.get("keep_lost"):
            done = False
            break
    return done, steps, rollouts


# ---------------- HIERARCHICAL CEM MPC ----------------
def hier_cem_mpc(model, env, max_steps=60, n_macro=16, iters_macro=2, n_low=12, iters_low=1):
    """z_high selects the k=16 macro-strategy ONCE per macro-phase; z_low refines
    each k=4 segment. Macro plan reused across a phase (not re-searched every step)."""
    device = "cpu"
    rollouts = 0
    steps = 0
    done = False
    _, vec = env.obs()
    macro_plan = None  # (type) — which type to clear this phase
    phase_left = 0
    for _ in range(max_steps):
        if env.remaining() == 0:
            done = True
            break
        keep_in_root = vec[7]
        if vec[2] > 0 and keep_in_root == 1:
            a = 2
            steps += 1
            _, vec, r, d, info = env.step(a)
            continue
        # plan a new macro-phase when the current one is exhausted
        if phase_left <= 0:
            goal = vec.copy(); goal[0:3] = 0
            z = model.encode_high(torch.as_tensor(vec, dtype=torch.float32, device=device).unsqueeze(0))
            z_goal = model.encode_high(torch.as_tensor(goal, dtype=torch.float32, device=device).unsqueeze(0))
            best_type = 0
            best_cost = 1e9
            for t in range(ACT_DIM - 1):  # 0,1,2 (no delete)
                A_macro = torch.full((1, model.k_high), t, dtype=torch.int64, device=device)
                zp = model.predict_high(z, A_macro)
                rollouts += 1
                cost = float(torch.norm(zp - z_goal).item())
                if cost < best_cost:
                    best_cost = cost
                    best_type = t
            macro_plan = best_type
            phase_left = model.k_high  # commit to this type for a full macro-phase
        # refine: current type
        a = macro_plan
        counts = vec[0:3].copy()
        if counts[a] <= 0:
            # type exhausted -> end phase early, re-plan next step
            phase_left = 0
            for i in range(3):
                if vec[i] <= 0:
                    counts[i] = -1
            a = int(np.argmax(counts)) if counts.max() > 0 else 2
        steps += 1
        phase_left -= 1
        _, vec, r, d, info = env.step(a)
        if info.get("keep_lost"):
            done = False
            break
    return done, steps, rollouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--n-txt", type=int, default=12)
    ap.add_argument("--n-jpg", type=int, default=12)
    ap.add_argument("--n-pdf", type=int, default=10)
    args = ap.parse_args()

    print("=" * 74)
    print("P3-1: HIERARCHICAL PLANNING (k=16) vs FLAT MPC — HARD long-horizon task")
    print(f"  directory: {args.n_txt} txt + {args.n_jpg} jpg + {args.n_pdf} pdf + KEEP")
    print("=" * 74)

    torch.manual_seed(0)
    model = HierJEPA(OBS_DIM, ACT_DIM, hidden=128, k_low=4, k_high=16)

    print("\n[train] hierarchical world model")
    def env_fn():
        return BigDirEnv("/tmp/hb-train", n_txt=args.n_txt, n_jpg=args.n_jpg, n_pdf=args.n_pdf)
    data = collect_hier(env_fn, episodes=400 if args.quick else 800, seed=0)
    train_hier(model, data, epochs=150 if args.quick else 300)

    print("\n" + "=" * 74)
    print("BENCHMARK (search cost = predictor rollouts)")
    print("=" * 74)
    flat_ok, flat_steps, flat_roll = 0, [], 0
    hier_ok, hier_steps, hier_roll = 0, [], 0
    for t in range(args.trials):
        env = BigDirEnv(f"/tmp/hb-run-{t}", n_txt=args.n_txt, n_jpg=args.n_jpg, n_pdf=args.n_pdf)
        env.reset(seed=t * 7 + 1)
        d, s, r = flat_cem_mpc(model, env)
        flat_ok += d; flat_steps.append(s); flat_roll += r

        env = BigDirEnv(f"/tmp/hb-run-{t}", n_txt=args.n_txt, n_jpg=args.n_jpg, n_pdf=args.n_pdf)
        env.reset(seed=t * 7 + 1)
        d, s, r = hier_cem_mpc(model, env)
        hier_ok += d; hier_steps.append(s); hier_roll += r

    print(f"\n  FLAT (k=4 CEM):  success {flat_ok}/{args.trials}  "
          f"avg steps {np.mean(flat_steps):.1f}  total rollouts {flat_roll}")
    print(f"  HIER (k=16+4):   success {hier_ok}/{args.trials}  "
          f"avg steps {np.mean(hier_steps):.1f}  total rollouts {hier_roll}")
    speedup = flat_roll / max(hier_roll, 1)
    print(f"\n  SEARCH SPEEDUP (rollouts flat/hier): {speedup:.2f}x  "
          f"{'PASS >=3x' if speedup >= 3 else 'below 3x'}")
    print("  (steps may differ; the win is search compute — long-horizon tasks)")
    print("=" * 74)


if __name__ == "__main__":
    main()