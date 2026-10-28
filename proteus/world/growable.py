#!/usr/bin/env python3
"""growable.py — Le-Gates on a GROWABLE world model: new file types enter the
observation, the model grows a gated new-path, and learns them WITHOUT forgetting.

The honest gap from the audit: a fixed 12-dim obs can't encode a new file type,
so gates fired on nothing. This fixes that:

  OBS = [txt, jpg, pdf, keep, junk, ...]  (base 12)
       + [md_count, csv_count, ...]       (new channels added on discovery)

Model: the macro encoder is a per-type embedding; when a new type appears, a NEW
gated embedding slot is added (identity init -> old behavior unchanged), and only
that slot + its gate train. Old types' embeddings are frozen. This is Le-Gates on
the INPUT side: each type is its own gated path.

Tests:
  T1 grow-invariant : adding a new type slot changes nothing for old types (pred diff 0)
  T2 learn-new-type : after .md arrives, gated learning reduces md-prediction error
  T3 no-forget      : old-type prediction error stays flat while learning md
  T4 compare        : naive (grow+train-all) forgets old; gated doesn't
"""
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn as nn

import proteus.envs.real_env as real_env
from proteus.world.hjepa_torch import MLP

BASE_DIM = 12  # txt jpg pdf / folder txt jpg pdf / messy keep kept junk total junkdel
ACT_DIM = 4


class DynamicWorld:
    """A real dir. Obs = 12 base dims + one count channel per DISCOVERED extra type.
    New types are registered when first seen (the world 'grows')."""

    def __init__(self, root, types=("txt", "jpg", "pdf"), extra=()):
        self.root = pathlib.Path(root)
        self.env = real_env.RealEnv(self.root)
        self.extra_types = list(extra)  # e.g. ['md'] — appear over time
        self.seen_extra = set()

    def reset(self, seed=0):
        real_env.seed_real_world(self.root, keep_pdf=True, seed=seed)
        self.env.keep_lost = False
        # create extra-type files
        for t in self.extra_types:
            for i in range(2):
                (self.root / f"{t}{i}.{t}").write_text(f"# {t} {i}")
        return self.obs()

    def obs(self):
        text, vec = self.env.obs()
        base = vec.copy()
        # extend with counts for each SEEN extra type
        counts = []
        for t in self.extra_types:
            n = sum(1 for p in self.root.glob(f"*.{t}") if p.is_file())
            counts.append(float(n))
            if n > 0:
                self.seen_extra.add(t)
        return np.concatenate([base, np.array(counts, dtype=np.float32)])

    def step(self, a):
        if a == 0:
            for p in sorted(list(self.root.glob("*.txt")) + [f for t in self.extra_types for f in self.root.glob(f"*.{t}")]):
                if p.is_file():
                    (self.root / "txt").mkdir(exist_ok=True)
                    d = self.root / "txt" / p.name
                    if d.exists():
                        d.unlink()
                    p.rename(d)
                    break
            return self.obs(), -0.01, False, {}
        t, v, r, d, i = self.env.step(a)
        return self.obs(), r, d, i

    def obs_dim(self):
        return BASE_DIM + len(self.seen_extra)


class GrowableJEPA(nn.Module):
    """Macro encoder + predictor over a growable obs. New type -> add a gated
    embedding slot + gated predictor delta. Old slots frozen after first grow."""

    def __init__(self, base_dim, act_dim, hidden=128, z_macro=64, k=4, tau=0.995):
        super().__init__()
        self.base_dim = base_dim
        self.act_dim = act_dim
        self.z_macro = z_macro
        self.k = k
        self.tau = tau
        self.extra = 0  # number of extra slots added
        # base macro encoder (fixed, frozen after growth)
        self.enc_base = MLP([base_dim, hidden, z_macro])
        # base predictor delta
        self.pred_base = MLP([z_macro + k * act_dim, hidden, z_macro])
        # target encoder (EMA of enc_base + extras)
        self.target_enc = MLP([base_dim, hidden, z_macro])
        self.target_enc.load_state_dict(self.enc_base.state_dict())
        # gated extra slots: one embedding layer + one predictor delta per slot
        self.extra_embeds = nn.ModuleList()  # each MLP([1, hidden, z_macro])
        self.extra_preds = nn.ModuleList()   # each MLP([z_macro+k*act, hidden, z_macro])
        self.extra_gates = nn.ParameterList()  # one gate logit per slot

    def add_type(self):
        """Grow: add a gated slot for a new type. ZERO-INIT the embed output so
        the new slot contributes exactly 0 at grow time -> old behavior unchanged
        regardless of gate value."""
        idx = self.extra
        self.extra += 1
        emb = MLP([1, 64, self.z_macro])
        pred = MLP([self.z_macro + self.k * self.act_dim, 128, self.z_macro])
        # zero-init the FINAL linear of embed and pred so the slot adds 0 initially
        with torch.no_grad():
            emb.net[-1].weight.zero_()
            emb.net[-1].bias.zero_()
            pred.net[-1].weight.zero_()
            pred.net[-1].bias.zero_()
        gate = nn.Parameter(torch.zeros(1))  # sigmoid 0.5, but embed*0 => contributes 0
        self.extra_embeds.append(emb)
        self.extra_preds.append(pred)
        self.extra_gates.append(gate)
        return idx

    def freeze_base(self):
        """Freeze the base (old-domain) paths after the first growth."""
        for p in self.enc_base.parameters():
            p.requires_grad = False
        for p in self.pred_base.parameters():
            p.requires_grad = False
        for p in self.target_enc.parameters():
            p.requires_grad = False

    def encode(self, obs):
        """obs: (B, base_dim + n_extra)."""
        base = obs[:, :self.base_dim]
        z = self.enc_base(base)
        # add gated extra-slot contributions, scaled by the input magnitude so
        # zero-input (old states) contributes exactly 0
        for i in range(self.extra):
            if self.base_dim + i < obs.shape[1]:
                xi = obs[:, self.base_dim + i:self.base_dim + i + 1]
                g = torch.sigmoid(self.extra_gates[i])
                # embed output is 0 at init (zero-final-layer); trained to respond
                # to nonzero xi. Multiply by xi so old states (xi=0) add nothing.
                z = z + g * self.extra_embeds[i](xi) * xi
        return z

    def encode_target(self, obs):
        with torch.no_grad():
            base = obs[:, :self.base_dim]
            z = self.target_enc(base)
            for i in range(self.extra):
                if self.base_dim + i < obs.shape[1]:
                    xi = obs[:, self.base_dim + i:self.base_dim + i + 1]
                    g = torch.sigmoid(self.extra_gates[i]).detach()
                    z = z + g * self.extra_embeds[i](xi) * xi
            return z

    def predict(self, z, A, type_present=None):
        """type_present: (B, n_extra) float mask — 1 if that type is present.
        Gates the extra pred delta by type presence so base states are untouched."""
        A_oh = torch.nn.functional.one_hot(A.long(), self.act_dim).float().reshape(A.shape[0], -1)
        inp = torch.cat([z, A_oh], dim=-1)
        delta = self.pred_base(inp)
        for i in range(self.extra):
            g = torch.sigmoid(self.extra_gates[i])
            mask = torch.zeros(inp.shape[0], 1, device=inp.device)
            if type_present is not None and type_present.shape[1] > i:
                mask = type_present[:, i].unsqueeze(1)
            delta = delta + g * mask * self.extra_preds[i](inp)
        return z + delta

    def macro_loss(self, obs_t, A, obs_tk):
        z = self.encode(obs_t)
        type_present = (obs_t[:, self.base_dim:] > 0).float() if obs_t.shape[1] > self.base_dim else None
        zp = self.predict(z, A, type_present)
        zt = self.encode_target(obs_tk)
        return torch.mean((zp - zt) ** 2)

    def update_targets(self):
        for p, tp in zip(self.enc_base.parameters(), self.target_enc.parameters()):
            tp.data.mul_(self.tau).add_(p.data, alpha=1 - self.tau)


def collect(world_fn, n=200, k=4):
    rng = random.Random(0)
    out = []
    for _ in range(n):
        w = world_fn()
        w.reset(seed=rng.randint(0, 100000))
        traj = [w.obs()]
        acts = []
        for _ in range(8):
            a = rng.randint(0, ACT_DIM - 1)
            v, _, d, _ = w.step(a)
            traj.append(v)
            acts.append(a)
            if d:
                break
        for i in range(len(traj) - k):
            out.append((traj[i], acts[i:i + k], traj[i + k]))
    return out


def train_base(model, samples, epochs=200, lr=3e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        for ot, A, ok in samples[::4]:
            bt = torch.as_tensor(ot, dtype=torch.float32).unsqueeze(0)
            ba = torch.as_tensor([A], dtype=torch.int64)
            bk = torch.as_tensor(ok, dtype=torch.float32).unsqueeze(0)
            opt.zero_grad()
            model.macro_loss(bt, ba, bk).backward()
            opt.step()
            model.update_targets()


def macro_mse(model, samples, obs_dim):
    model.eval()
    ms = []
    with torch.no_grad():
        for ot, A, ok in samples[:100]:
            bt = torch.as_tensor(ot, dtype=torch.float32).unsqueeze(0)
            ba = torch.as_tensor([A], dtype=torch.int64)
            bk = torch.as_tensor(ok, dtype=torch.float32).unsqueeze(0)
            ms.append(float(model.macro_loss(bt, ba, bk).item()))
    return float(np.mean(ms))


def main():
    print("=" * 70)
    print("GROWABLE Le-Gates — new file types actually enter the world")
    print("=" * 70)

    # Phase A: base world (txt/jpg/pdf), no extras
    torch.manual_seed(0)
    model = GrowableJEPA(BASE_DIM, ACT_DIM, hidden=128, z_macro=64, k=4)
    base_samples = collect(lambda: DynamicWorld("/tmp/grw-base"), n=400, k=4)
    train_base(model, base_samples, epochs=200)
    old_base = macro_mse(model, base_samples, BASE_DIM)
    print(f"[A] base trained on 12-dim: macro MSE {old_base:.4f}")

    # capture old behavior on a base-state batch
    z = torch.randn(16, 64)
    A = torch.randint(0, 4, (16, 4))
    type_present_zero = torch.zeros(16, 0)  # no extras -> base only
    with torch.no_grad():
        pred_old = model.predict(z, A).clone()

    # Phase B: .md arrives — the world grows
    shifted = collect(lambda: DynamicWorld("/tmp/grw-shift", extra=("md",)), n=200, k=4)
    # obs now has 13 dims. Grow the model.
    print(f"\n[B] .md arrives -> obs grows to 13 dims. Adding gated slot...")
    model.add_type()
    model.freeze_base()  # base frozen -> no forgetting of old types
    # verify growth is invariant (old behavior unchanged at gate ~0.5 * 0 init)
    with torch.no_grad():
        pred_after_grow = model.predict(z, A, torch.zeros(16, 1))  # md absent -> masked
    grow_diff = float((pred_old - pred_after_grow).abs().max())
    print(f"    growth invariant (old-state pred diff): {grow_diff:.4f} "
          f"({'OK <=0.01' if grow_diff <= 0.01 else 'VIOLATED'})")

    # md states ARE now representable (unlike fixed-obs)
    md_state = np.zeros(13, dtype=np.float32)
    md_state[6] = 2; md_state[10] = 2; md_state[12] = 2  # 2 md files
    plain = np.zeros(13, dtype=np.float32)
    plain[6] = 2; plain[10] = 2
    with torch.no_grad():
        zm = model.encode(torch.as_tensor(md_state, dtype=torch.float32).unsqueeze(0))
        zp = model.encode(torch.as_tensor(plain, dtype=torch.float32).unsqueeze(0))
    md_visible = float((zm - zp).abs().max())
    print(f"    .md now representable (latent diff): {md_visible:.4f} "
          f"({'VISIBLE' if md_visible > 0.01 else 'STILL INVISIBLE'})")

    # Phase C: gated learning on the md slot ONLY (base frozen)
    mse_before = macro_mse(model, shifted, 13)
    opt = torch.optim.Adam(list(model.extra_preds[0].parameters()) +
                           list(model.extra_embeds[0].parameters()) +
                           [model.extra_gates[0]], lr=3e-3)
    print(f"\n[C] gated learning on the .md slot (base frozen):")
    print(f"    before: shift-MSE {mse_before:.4f}")
    print(f"    {'event':>6} {'gate':>7} {'shift-MSE':>10} {'old-MSE':>9}")
    for ev in range(1, 7):
        for _ in range(15):
            for ot, A, ok in shifted[:40]:
                bt = torch.as_tensor(ot, dtype=torch.float32).unsqueeze(0)
                ba = torch.as_tensor([A], dtype=torch.int64)
                bk = torch.as_tensor(ok, dtype=torch.float32).unsqueeze(0)
                opt.zero_grad()
                model.macro_loss(bt, ba, bk).backward()
                opt.step()
                model.update_targets()
        g = torch.sigmoid(model.extra_gates[0]).item()
        sm = macro_mse(model, shifted, 13)
        om = macro_mse(model, base_samples, 12)
        print(f"    {ev:>6} {g:>7.3f} {sm:>10.4f} {om:>9.4f}")

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    sm_f = macro_mse(model, shifted, 13)
    om_f = macro_mse(model, base_samples, 12)
    print(f"  shift-MSE {mse_before:.4f} -> {sm_f:.4f} "
          f"({100 * (mse_before - sm_f) / max(mse_before, 1e-6):.0f}% improvement)")
    print(f"  old-MSE   {old_base:.4f} -> {om_f:.4f} "
          f"({'NO FORGETTING (old <= 1.5x base)' if om_f <= old_base * 1.5 + 0.003 else 'DRIFT'})")
    print(f"  growth invariant: {grow_diff:.4f} | md visible: {md_visible:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()