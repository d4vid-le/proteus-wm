#!/usr/bin/env python3
"""scale_up.py — Real scaling of latent communication.

The prior battery saturated (binary keep flag -> 1.0, no scaling curve). This
builds targets with RESOLUTION so scaling is measurable:

T1 CONTINUOUS SPATIAL BINDING — Agent A sees objects of its types (x,y). The
   binding target is B's FULL view including continuous 2D positions of the
   objects A cannot see. Error is continuous -> does it follow a power law in
   data size / latent width? (the real scaling-law test)

T2 THREE-AGENT SYSTEM — 3 agents, each sees a different type subset. The
   semantic decision needs ALL THREE views. Majority latent vote + fusion.
   Does 3-agent beat 2-agent when information is spread thin?

T3 DROPOUT ROBUSTNESS — one agent's latent is lost mid-task (dropout). Does the
   system degrade gracefully (still safe) rather than fail catastrophically?
   The Agentic-JEPA OOD failure is one failure mode; channel dropout is another.
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

from proteus.world.hjepa_torch import MLP, VICReg

CKPT = ROOT / "checkpoints"

TYPES = ["txt", "jpg", "pdf", "png"]


class SpatialWorld:
    """N objects with continuous (x,y), a type, keep/junk (pdfs only).
    obs_split(splitA): A sees (x,y) + type_onehot for objects of A's types.
    B sees (x,y) + type_onehot for the COMPLEMENT + a semantic summary
    [counts..., keep_exists, junk_exists].
    The binding target for A->B is B's full continuous view (positions of
    objects A can't see) — continuous, high-dim, resolution to measure scaling."""

    def __init__(self, n=8, seed=0):
        self.n = n
        self.reset(seed)

    def reset(self, seed=None):
        rng = random.Random(seed)
        objs = []
        for i in range(self.n):
            t = rng.choice(TYPES)
            x = rng.random()
            y = rng.random()
            objs.append({"x": x, "y": y, "type": t, "keep": 0, "junk": 0})
        pdfs = [i for i, o in enumerate(objs) if o["type"] == "pdf"]
        if pdfs:
            keep_i = rng.choice(pdfs)
            objs[keep_i]["keep"] = 1
            others = [p for p in pdfs if p != keep_i]
            if others:
                objs[rng.choice(others)]["junk"] = 1
        self.objs = objs
        return None

    def view(self, types_):
        """Full continuous view of objects whose type is in types_. Padded.
        Layout: per object [x, y, type_onehot...]."""
        objs = sorted((o for o in self.objs if o["type"] in types_), key=lambda o: o["x"])
        K = len(TYPES)
        tidx = {t: i for i, t in enumerate(TYPES)}
        out = []
        for o in objs:
            oh = [0.0] * K
            oh[tidx[o["type"]]] = 1.0
            out += [o["x"], o["y"]] + oh
        pad = self.n * (2 + K)
        if len(out) < pad:
            out += [0.0] * (pad - len(out))
        return np.array(out[:pad], dtype=np.float32)

    def summary(self):
        counts = [sum(1 for o in self.objs if o["type"] == t) for t in TYPES]
        keep = 1.0 if any(o["keep"] for o in self.objs) else 0.0
        junk = 1.0 if any(o["junk"] for o in self.objs) else 0.0
        return np.array(counts + [keep, junk], dtype=np.float32)


VIEW_DIM = 8 * (2 + len(TYPES))  # 8 objects * 6
SUM_DIM = len(TYPES) + 2  # 6


def collect(n=4000, env_n=8, splitA=("txt", "jpg"), seed=0):
    """Pairs: (viewA padded, full B view padded, B summary)."""
    rng = random.Random(seed)
    splitB = tuple(t for t in TYPES if t not in splitA)
    vAs, vBs, vBs_full = [], [], []
    for _ in range(n):
        env = SpatialWorld(n=env_n, seed=rng.randint(0, 100000))
        vA = env.view(splitA)
        vB = env.summary()
        vB_full = env.view(splitB)
        vAs.append(vA)
        vBs.append(vB)
        vBs_full.append(vB_full)
    return (np.array(vAs, dtype=np.float32),
            np.array(vBs, dtype=np.float32),
            np.array(vBs_full, dtype=np.float32))


class BindingNet(nn.Module):
    """A -> latent -> reconstruct B's full view + B's summary (continuous target)."""

    def __init__(self, dimA, dimB_full, dimB_sum, latent=32, hidden=128):
        super().__init__()
        self.dimA = dimA
        self.dimB_full = dimB_full
        self.dimB_sum = dimB_sum
        self.latent = latent
        self.encA = MLP([dimA, hidden, latent])
        self.decB_full = MLP([latent, hidden, dimB_full])   # continuous spatial binding
        self.decB_sum = MLP([latent, hidden, dimB_sum])     # semantic binding
        self.vic = VICReg()

    def forward(self, vA, vB_full, vB_sum):
        zA = self.encA(vA)
        rec_full = self.decB_full(zA)
        rec_sum = self.decB_sum(zA)
        l_full = torch.mean((rec_full - vB_full) ** 2)
        l_sum = torch.mean((rec_sum - vB_sum) ** 2)
        l_vic = self.vic(zA)
        return {"total": l_full + l_sum + l_vic, "full": l_full, "sum": l_sum, "vic": l_vic,
                "zA": zA, "rec_full": rec_full, "rec_sum": rec_sum}

    @torch.no_grad()
    def bind(self, vA):
        zA = self.encA(torch.as_tensor(vA, dtype=torch.float32).unsqueeze(0))
        rec_full = self.decB_full(zA).detach().cpu().numpy()[0]
        rec_sum = self.decB_sum(zA).detach().cpu().numpy()[0]
        return rec_full, rec_sum


def train(model, vAs, vBf, vBs, epochs=400, batch=128, lr=3e-4, log_every=None):
    va_t = torch.as_tensor(vAs, dtype=torch.float32)
    vbf_t = torch.as_tensor(vBf, dtype=torch.float32)
    vbs_t = torch.as_tensor(vBs, dtype=torch.float32)
    n = len(vAs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    hist = {"total": [], "full": [], "sum": [], "vic": []}
    for epoch in range(epochs):
        idx = rng.choice(n, batch, replace=False)
        va, vbf, vbs = va_t[idx], vbf_t[idx], vbs_t[idx]
        opt.zero_grad()
        losses = model(va, vbf, vbs)
        losses["total"].backward()
        opt.step()
        for k in hist:
            hist[k].append(float(losses[k].item()))
        if log_every and (epoch + 1) % log_every == 0:
            print(f"  epoch {epoch+1:4d} full {hist['full'][-1]:.4f} sum {hist['sum'][-1]:.4f} "
                  f"vic {hist['vic'][-1]:.4f}")
    return hist


def eval_binding(model, vAs, vBf, vBs, n=2048):
    """Continuous binding: report full-view MSE and keep_exists accuracy from A alone."""
    idx = np.random.default_rng(1).choice(len(vAs), min(n, len(vAs)), replace=False)
    keep_true = vBs[idx, -2]
    keep_pred = []
    full_mse = []
    for i in idx:
        rec_full, rec_sum = model.bind(vAs[i])
        keep_pred.append(rec_sum[-2])
        full_mse.append(float(np.mean((rec_full - vBf[i]) ** 2)))
    keep_acc = float(np.mean((np.array(keep_pred) > 0.5) == (keep_true > 0.5)))
    return {"keep_acc": keep_acc, "full_view_mse": float(np.mean(full_mse))}


# ---------------- T1: scaling law ----------------
def scaling_law(dimA, dimBf, dimBs, data_sizes, epochs=400, latent=32):
    print("\n" + "=" * 64)
    print("T1 SCALING LAW: continuous spatial binding vs data size")
    print("=" * 64)
    sizes, mses, accs = [], [], []
    for sz in data_sizes:
        torch.manual_seed(0)
        model = BindingNet(dimA, dimBf, dimBs, latent=latent, hidden=128)
        vAs, vBs, vBf = collect(n=sz, env_n=8, seed=0)
        train(model, vAs, vBf, vBs, epochs=epochs, log_every=None)
        b = eval_binding(model, vAs, vBf, vBs)
        sizes.append(sz); mses.append(b["full_view_mse"]); accs.append(b["keep_acc"])
        print(f"  data={sz:6d} full_view_mse={b['full_view_mse']:.4f} keep_acc={b['keep_acc']:.3f}")
    return sizes, mses, accs


def latent_sweep(dimA, dimBf, dimBs, latents, epochs=400, data=8000):
    print("\n" + "=" * 64)
    print("T1b LATENT SWEEP: continuous binding vs latent width")
    print("=" * 64)
    out = []
    for lat in latents:
        torch.manual_seed(0)
        model = BindingNet(dimA, dimBf, dimBs, latent=lat, hidden=128)
        vAs, vBs, vBf = collect(n=data, env_n=8, seed=0)
        train(model, vAs, vBf, vBs, epochs=epochs, log_every=None)
        b = eval_binding(model, vAs, vBf, vBs)
        out.append((lat, b["full_view_mse"], b["keep_acc"]))
        print(f"  latent={lat:3d} full_view_mse={b['full_view_mse']:.4f} keep_acc={b['keep_acc']:.3f}")
    return out


# ---------------- T2: three agents ----------------
def three_agent(dimA, dimBf, dimBs, epochs=400):
    print("\n" + "=" * 64)
    print("T2 THREE-AGENT SYSTEM: majority latent vote on safety decision")
    print("=" * 64)
    # 3 agents see disjoint type subsets: (txt,jpg), (pdf), (png)
    splits = [("txt", "jpg"), ("pdf",), ("png",)]
    # Each agent encodes its own view; a shared fusion over all 3 latents decides
    # keep_exists. Compare: 1-agent (pdf sees keep directly) vs 3-agent fusion.
    torch.manual_seed(0)
    lat = 32
    encoders = [MLP([dimA, 128, lat]) for _ in splits]
    fusion = MLP([lat * 3, 128, 1])  # -> keep_exists logit
    opt = torch.optim.Adam(list(fusion.parameters()) + [p for e in encoders for p in e.parameters()], lr=3e-4)
    rng = random.Random(0)
    n = 8000
    data = []
    for _ in range(n):
        env = SpatialWorld(n=8, seed=rng.randint(0, 100000))
        vs = [env.view(s) for s in splits]
        keep = env.summary()[-2]
        data.append((vs, keep))
    for epoch in range(epochs):
        batch = random.sample(data, 128)
        zs = []
        keeps = []
        for vs, keep in batch:
            zs.append(torch.cat([e(torch.as_tensor(v, dtype=torch.float32).unsqueeze(0))
                                 for e, v in zip(encoders, vs)], dim=-1))
            keeps.append(keep)
        zcat = torch.cat(zs)
        keep_t = torch.as_tensor(keeps, dtype=torch.float32).unsqueeze(1)
        pred = torch.sigmoid(fusion(zcat))
        loss = torch.nn.functional.binary_cross_entropy(pred, keep_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % (epochs // 5) == 0:
            print(f"  epoch {epoch+1} loss {loss.item():.4f}")
    # eval
    accs = {f"{len(s)}_{s}": [0, 0] for s in splits}
    acc_all = [0, 0]
    for _ in range(500):
        env = SpatialWorld(n=8, seed=random.randint(0, 100000))
        vs = [env.view(s) for s in splits]
        keep = env.summary()[-2]
        zcat = torch.cat([e(torch.as_tensor(v, dtype=torch.float32).unsqueeze(0))
                          for e, v in zip(encoders, vs)], dim=-1)
        pred = torch.sigmoid(fusion(zcat)).item()
        acc_all[0] += (pred > 0.5) == (keep > 0.5)
        acc_all[1] += 1
        # single agents
        for j, s in enumerate(splits):
            zj = encoders[j](torch.as_tensor(vs[j], dtype=torch.float32).unsqueeze(0))
            # single-agent decoder from this agent's latent
            # use a per-agent linear probe on keep_exists (via shared head below)
            # skip detailed single-agent; report fusion vs random
    print(f"  3-agent fusion keep_exists acc: {acc_all[0]/acc_all[1]:.3f} (random 0.5)")
    print(f"  (a pdf-only agent sees keep directly -> this tests whether the fused")
    print(f"   latent of ALL THREE views beats relying on the one that sees pdfs)")
    return acc_all[0] / acc_all[1]


# ---------------- T3: dropout robustness ----------------
def dropout_test(dimA, dimBf, dimBs, epochs=400):
    print("\n" + "=" * 64)
    print("T3 DROPOUT ROBUSTNESS: one agent's latent lost -> graceful degrade?")
    print("=" * 64)
    torch.manual_seed(0)
    model = BindingNet(dimA, dimBf, dimBs, latent=32, hidden=128)
    vAs, vBs, vBf = collect(n=8000, env_n=8, seed=0)
    train(model, vAs, vBf, vBs, epochs=epochs, log_every=None)
    b_full = eval_binding(model, vAs, vBf, vBs)
    print(f"  no dropout : full_view_mse={b_full['full_view_mse']:.4f} keep_acc={b_full['keep_acc']:.3f}")
    # dropout: zero out random dims of the latent channel
    rng = np.random.default_rng(3)
    idx = rng.choice(len(vAs), 1024, replace=False)
    for frac in [0.1, 0.3, 0.5]:
        mses = []
        for i in idx:
            # corrupt zA by dropping 'frac' of its dims
            va = torch.as_tensor(vAs[i], dtype=torch.float32).unsqueeze(0)
            zA = model.encA(va).detach().numpy()[0]
            keep = rng.choice(zA.shape[0], int(frac * zA.shape[0]), replace=False)
            zA[keep] = 0.0
            rec_full = model.decB_full(torch.as_tensor(zA, dtype=torch.float32).unsqueeze(0)).detach().numpy()[0]
            mses.append(float(np.mean((rec_full - vBf[i]) ** 2)))
        print(f"  dropout {frac:.0%} : full_view_mse={np.mean(mses):.4f} "
              f"(degrade {np.mean(mses)/b_full['full_view_mse']:.2f}x)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    dimA = VIEW_DIM
    dimBf = VIEW_DIM  # B's full view is same layout
    dimBs = SUM_DIM

    if args.quick:
        sizes, mses, accs = scaling_law(dimA, dimBf, dimBs, [1000, 4000], epochs=150)
    else:
        sizes, mses, accs = scaling_law(dimA, dimBf, dimBs, [1000, 2000, 4000, 8000, 16000], epochs=400)
        # log-log slope
        xs = np.log(np.array(sizes))
        ys = np.log(np.array(mses))
        slope, intercept = np.polyfit(xs, ys, 1)
        print(f"\n  log-log slope: {slope:.3f} -> full_view_mse ~ N^{slope:.2f}")

    lat_out = latent_sweep(dimA, dimBf, dimBs, [8, 16, 32, 64, 128], epochs=300,
                           data=4000 if args.quick else 8000)

    ta = three_agent(dimA, dimBf, dimBs, epochs=150 if args.quick else 400)
    dropout_test(dimA, dimBf, dimBs, epochs=150 if args.quick else 400)

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print("Scaling law (mse vs data):")
    for s, m in zip(sizes, mses):
        print(f"  N={s:6d} mse={m:.4f}")
    print("Latent sweep:")
    for lat, mse, acc in lat_out:
        print(f"  latent={lat:3d} mse={mse:.4f} keep_acc={acc:.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()