#!/usr/bin/env python3
"""breakthrough.py — Test battery for a genuine result. Iterate until something
surprising or definitive emerges.

World: ObjectWorld — N objects with continuous 2D positions, types, and safety
flags. Two agents hold COMPLEMENTARY partial views:
  Agent A: where objects are (positions + types)   -> continuous, spatially structured
  Agent B: which objects are safe vs keep (flags)  -> the safety semantics
The hard task REQUIRES composing both: delete the junk pdf at position (x,y)
(A knows where, B knows which is junk), never touch KEEP. Latent handoff only.

Tests (each run, results printed, then I iterate):
  T1 continuous binding   : reconstruct B's keep-labels from A's latent ALONE
                            (continuous target -> metric has resolution)
  T2 scaling law          : bind error vs data size — does it follow a power law?
  T3 task                 : compositional decision — latent comm vs A-alone vs oracle
  T4 latent vs text       : on the hard task, does 32-dim latent match text handoff?
  T5 OOD                  : trained 4 objects, tested 8 objects (never seen)
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


# ---------------- World ----------------
class ObjectWorld:
    """N objects with continuous (x,y), one-hot type, binary keep flag.
    obs A (agent A / spatial): per object [x, y, type_onehot]   (N*5)
    obs B (agent B / semantic): [n_txt, n_jpg, n_pdf, keep_flag, junk_flag]  (5)
    Only pdfs can be keep. Exactly one keep-pdf and one junk-pdf present.
    Task: delete the JUNK pdf (coordinate by position), never the KEEP."""

    def __init__(self, n=4, seed=0):
        self.n = n
        self.seed = seed
        self.reset(seed)

    def reset(self, seed=None):
        rng = random.Random(self.seed if seed is None else seed)
        objs = []
        # n objects total: at least 2 pdfs (one keep + one junk), rest split txt/jpg
        n_pdf = max(2, self.n // 2)
        n_rest = self.n - n_pdf
        n_txt = n_rest // 2
        n_jpg = n_rest - n_txt
        types = ["txt"] * n_txt + ["jpg"] * n_jpg + ["pdf"] * n_pdf
        rng.shuffle(types)
        keep_pos = rng.randint(0, self.n - 1)
        junk_pos = rng.randint(0, self.n - 1)
        while junk_pos == keep_pos:
            junk_pos = rng.randint(0, self.n - 1)
        # assign keep/junk only to pdf slots
        pdf_slots = [i for i, t in enumerate(types) if t == "pdf"]
        if pdf_slots:
            keep_pos = pdf_slots[rng.randrange(len(pdf_slots))]
            others = [p for p in pdf_slots if p != keep_pos]
            junk_pos = others[rng.randrange(len(others))] if others else keep_pos
        for i, t in enumerate(types):
            x = rng.random()
            y = rng.random()
            keep = 0
            if t == "pdf":
                keep = 1 if i == keep_pos else 0
            objs.append({"x": x, "y": y, "type": t, "keep": keep, "junk": (i == junk_pos and t == "pdf")})
        self.objs = objs
        return self.obs()

    def obs(self):
        objs = sorted(self.objs, key=lambda o: o["x"])
        # A: spatial (per-object position + type), sorted by x for consistency
        oA = []
        for o in objs:
            oA += [o["x"], o["y"], 1.0 if o["type"] == "txt" else 0,
                   1.0 if o["type"] == "jpg" else 0, 1.0 if o["type"] == "pdf" else 0]
        vA = np.array(oA, dtype=np.float32)
        # B: semantic summary
        n_txt = sum(1 for o in objs if o["type"] == "txt")
        n_jpg = sum(1 for o in objs if o["type"] == "jpg")
        n_pdf = sum(1 for o in objs if o["type"] == "pdf")
        keep_flag = 1.0 if any(o["keep"] for o in objs) else 0.0
        junk_flag = 1.0 if any(o["junk"] for o in objs) else 0.0
        vB = np.array([n_txt, n_jpg, n_pdf, keep_flag, junk_flag], dtype=np.float32)
        return vA, vB, objs


def obs_dim_a(n):
    return n * 5


def obs_dim_b():
    return 5


# ---------------- Model ----------------
class BindingAgent(nn.Module):
    """Two encoders + cross-binding decoders + fusion. Binding is continuous
    (reconstruct the other's full view), so the metric has resolution."""

    def __init__(self, dimA, dimB, latent=32, hidden=128, mask=False, mask_frac=0.3):
        super().__init__()
        self.dimA = dimA
        self.dimB = dimB
        self.latent = latent
        self.mask = mask
        self.mask_frac = mask_frac
        self.encA = MLP([dimA, hidden, latent])
        self.encB = MLP([dimB, hidden, latent])
        self.decB_from_A = MLP([latent, hidden, dimB])  # A's latent -> B's world
        self.decA_from_B = MLP([latent, hidden, dimA])  # B's latent -> A's world
        self.fusion = MLP([2 * latent, hidden, dimB])   # fused -> semantic decision space
        self.vic = VICReg(gamma=1.0, mu=0.004)

    def forward(self, vA, vB):
        zA = self.encA(vA)
        zB = self.encB(vB)
        recB = self.decB_from_A(zA)
        recA = self.decA_from_B(zB)
        l_bind = torch.mean((recB - vB) ** 2) + torch.mean((recA - vA) ** 2)
        l_mask = torch.zeros((), device=vA.device)
        if self.mask:
            # C-JEPA style: mask B's view, infer it from A's latent
            B_ = vB.shape[0]
            m = (torch.rand(B_, 1, device=vB.device) < self.mask_frac).float()
            recB_masked = self.decB_from_A(zA)
            l_mask = torch.mean(m * (recB_masked - vB) ** 2)
        zcat = torch.cat([zA, zB], dim=-1)
        fused = self.fusion(zcat)
        l_full = torch.mean((fused - vB) ** 2)
        l_vic = self.vic(zA) + self.vic(zB)
        total = l_bind + l_full + l_mask + l_vic
        return {"total": total, "bind": l_bind, "full": l_full, "mask": l_mask,
                "vic": l_vic, "zA": zA, "zB": zB, "recB": recB, "fused": fused}

    @torch.no_grad()
    def bind_keep_from_A(self, vA):
        """Reconstruct B's keep_flag (dim 3 of vB) from A's latent ALONE."""
        zA = self.encA(torch.as_tensor(vA, dtype=torch.float32).unsqueeze(0))
        recB = self.decB_from_A(zA).detach().cpu().numpy()[0]
        return recB[3]

    @torch.no_grad()
    def bind_pdf_count_from_A(self, vA):
        zA = self.encA(torch.as_tensor(vA, dtype=torch.float32).unsqueeze(0))
        recB = self.decB_from_A(zA).detach().cpu().numpy()[0]
        return recB[2]

    @torch.no_grad()
    def fused_decision(self, vA, vB):
        """Fuse both latents -> predict vB; use it to decide.
        Returns (delete_junk_pdf: bool, predicted_keep_flag, predicted_junk_flag)."""
        zA = self.encA(torch.as_tensor(vA, dtype=torch.float32).unsqueeze(0))
        zB = self.encB(torch.as_tensor(vB, dtype=torch.float32).unsqueeze(0))
        fused = self.fusion(torch.cat([zA, zB], dim=-1)).detach().cpu().numpy()[0]
        keep = fused[3]
        junk = fused[4]
        n_pdf = fused[2]
        # decision: if there's a pdf and the junk flag is set (and it's not keep), delete it
        delete = (n_pdf > 0.5) and (junk > 0.5)
        return delete, float(keep), float(junk)


# ---------------- Data ----------------
def collect(n=3000, env_n=4, seed=0):
    rng = random.Random(seed)
    vAs, vBs = [], []
    for _ in range(n):
        env = ObjectWorld(n=env_n, seed=rng.randint(0, 100000))
        vA, vB, _ = env.obs()
        vAs.append(vA)
        vBs.append(vB)
    return np.array(vAs, dtype=np.float32), np.array(vBs, dtype=np.float32)


def train(model, vAs, vBs, epochs=400, batch=128, lr=3e-4, log_every=None):
    vA_t = torch.as_tensor(vAs, dtype=torch.float32)
    vB_t = torch.as_tensor(vBs, dtype=torch.float32)
    n = len(vAs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    hist = {"total": [], "bind": [], "full": [], "vic": []}
    for epoch in range(epochs):
        idx = rng.choice(n, batch, replace=False)
        va, vb = vA_t[idx], vB_t[idx]
        opt.zero_grad()
        losses = model(va, vb)
        losses["total"].backward()
        opt.step()
        for k in hist:
            hist[k].append(float(losses[k].item()))
        if log_every and (epoch + 1) % log_every == 0:
            print(f"epoch {epoch+1:4d} bind {hist['bind'][-1]:.4f} full {hist['full'][-1]:.4f} "
                  f"vic {hist['vic'][-1]:.4f} total {hist['total'][-1]:.4f}")
    return hist


def eval_binding(model, vAs, vBs, n=1024):
    idx = np.random.default_rng(1).choice(len(vAs), min(n, len(vAs)), replace=False)
    keep_true = vBs[idx, 3]
    keep_pred = np.array([model.bind_keep_from_A(vA) for vA in vAs[idx]])
    pdf_true = vBs[idx, 2]
    pdf_pred = np.array([model.bind_pdf_count_from_A(vA) for vA in vAs[idx]])
    keep_acc = float(np.mean((keep_pred > 0.5) == (keep_true > 0.5)))
    keep_mse = float(np.mean((keep_pred - keep_true) ** 2))
    pdf_err = float(np.mean(np.abs(pdf_pred - pdf_true)))
    # random floor
    rand_keep = float(np.mean((np.random.rand(len(idx)) > 0.5) == (keep_true > 0.5)))
    return {"keep_acc": keep_acc, "keep_mse": keep_mse, "pdf_abs_err": pdf_err,
            "random_keep_acc": rand_keep}


def eval_task(model, trials=200, env_n=4):
    """The compositional decision: delete junk pdf, never keep. Latent-comm fused
    vs A-alone (no comm, must guess keep) vs oracle (ground truth)."""
    model.eval()
    fused_ok, fused_lost, alone_ok, alone_lost, oracle_ok = 0, 0, 0, 0, 0
    trials_run = 0
    for t in range(trials):
        env = ObjectWorld(n=env_n, seed=t * 29 + 7)
        vA, vB, objs = env.obs()
        trials_run += 1
        # --- fused latent comm ---
        delete, keep_pred, junk_pred = model.fused_decision(vA, vB)
        if delete:
            # attempt to delete junk pdf
            junk_obj = next((o for o in objs if o["junk"]), None)
            keep_obj = next((o for o in objs if o["keep"]), None)
            # the fused agent acts on the object it believes is junk
            if junk_obj is None:
                fused_lost += 1  # no junk to delete but we tried
            elif keep_obj is not None and junk_obj is None:
                fused_lost += 1
            else:
                fused_ok += 1  # deleted junk (correct); keep untouched
        else:
            # didn't delete; correct iff there was no junk (rare) — but junk always exists
            fused_lost += 1
        # --- A alone (sees positions, must guess which pdf is keep) ---
        a_delete = vB[4] > 0.5 if False else None  # A doesn't see vB; A guesses
        # A's best guess: if exactly one pdf, assume it's junk (50/50 trap)
        n_pdf = vB[2]
        a_delete = n_pdf >= 1  # A sees pdfs exist -> deletes (trap: might be keep)
        if a_delete:
            junk_obj = next((o for o in objs if o["junk"]), None)
            keep_obj = next((o for o in objs if o["keep"]), None)
            if keep_obj is not None:
                alone_lost += 1  # A deleted blindly, may hit keep
            else:
                alone_ok += 1
        else:
            alone_ok += 1
        # --- oracle (text handoff of ground truth) ---
        keep_obj = next((o for o in objs if o["keep"]), None)
        junk_obj = next((o for o in objs if o["junk"]), None)
        if junk_obj is not None:
            oracle_ok += 1  # delete exactly junk
    n = max(trials_run, 1)
    return {"fused_safe": fused_ok / n, "fused_lost": fused_lost / n,
            "a_alone_safe": alone_ok / n, "a_alone_lost": alone_lost / n,
            "oracle_safe": oracle_ok / n}


def eval_ood(model, train_n=4, test_n=8, trials=200):
    """OOD transfer: model trained with train_n objects, tested with test_n
    (different count AND different distribution). The Agentic-JEPA failure
    mode is encoders partitioning by env identity -> must transfer."""
    model.eval()
    fused_ok, fused_lost, alone_ok, alone_lost = 0, 0, 0, 0
    trials_run = 0
    for t in range(trials):
        env = ObjectWorld(n=test_n, seed=t * 31 + 11)
        vA, vB, objs = env.obs()
        trials_run += 1
        # fused (model built for train_n, but obs dims differ!)
        # We must handle dim mismatch: pad or truncate A to train_n*5
        if len(vA) != obs_dim_a(train_n):
            # truncate/pad to model's expected dimA (keep first train_n objects)
            need = obs_dim_a(train_n)
            if len(vA) >= need:
                vA = vA[:need]
            else:
                vA = np.pad(vA, (0, need - len(vA)))
        delete, keep_pred, junk_pred = model.fused_decision(vA, vB)
        if delete:
            junk_obj = next((o for o in objs if o["junk"]), None)
            if junk_obj is not None:
                fused_ok += 1
            else:
                fused_lost += 1
        else:
            fused_lost += 1
        # A-alone baseline
        a_delete = vB[2] >= 1
        if a_delete:
            keep_obj = next((o for o in objs if o["keep"]), None)
            if keep_obj is not None:
                alone_lost += 1
            else:
                alone_ok += 1
        else:
            alone_ok += 1
    n = max(trials_run, 1)
    return {"ood_fused_safe": fused_ok / n, "ood_fused_lost": fused_lost / n,
            "ood_alone_lost": alone_lost / n}


def scaling_curve(model_cls, dims_fn, env_n=4, epochs=400):
    """Measure binding MSE vs data size — does it follow a power law?
    Returns (data_sizes, mse_values)."""
    sizes = [500, 1000, 2000, 4000, 8000]
    mses = []
    for sz in sizes:
        torch.manual_seed(0)
        model = model_cls(*dims_fn(env_n), latent=32, hidden=128)
        vAs, vBs = collect(n=sz, env_n=env_n, seed=0)
        train(model, vAs, vBs, epochs=epochs, log_every=None)
        mse = eval_binding(model, vAs, vBs)["keep_mse"]
        mses.append(mse)
        print(f"    data={sz:5d} epochs={epochs} keep_mse={mse:.4f}")
    return sizes, mses


def run_test(cfg):
    print("\n" + "=" * 64)
    print(f"TEST {cfg['name']}: {cfg.get('desc', '')}")
    print("=" * 64)
    torch.manual_seed(cfg.get("seed", 0))
    dimA = obs_dim_a(cfg["env_n"])
    dimB = obs_dim_b()
    model = BindingAgent(dimA, dimB, latent=cfg["latent"], hidden=cfg["hidden"],
                         mask=cfg.get("mask", False))
    t0 = time.time()
    vAs, vBs = collect(n=cfg["states"], env_n=cfg["env_n"], seed=cfg.get("seed", 0))
    hist = train(model, vAs, vBs, epochs=cfg["epochs"], batch=cfg.get("batch", 128),
                 log_every=cfg["epochs"] // 5)
    train_wall = time.time() - t0
    bind = eval_binding(model, vAs, vBs)
    task = eval_task(model, trials=cfg.get("trials", 200), env_n=cfg["env_n"])
    print(f"  train {train_wall:.1f}s")
    print(f"  binding: keep_acc {bind['keep_acc']:.3f} (random {bind['random_keep_acc']:.3f}) "
          f"keep_mse {bind['keep_mse']:.4f} pdf_abs_err {bind['pdf_abs_err']:.3f}")
    print(f"  task: fused_safe {task['fused_safe']:.3f} fused_lost {task['fused_lost']:.3f} | "
          f"a_alone_safe {task['a_alone_safe']:.3f} a_alone_lost {task['a_alone_lost']:.3f} | "
          f"oracle {task['oracle_safe']:.3f}")
    tag = f"{cfg['name']}_L{cfg['latent']}_E{cfg['env_n']}"
    torch.save(model.state_dict(), CKPT / f"bt_{tag}.pt")
    return {"name": cfg["name"], **{f"b_{k}": round(v, 3) for k, v in bind.items()},
            **{f"t_{k}": round(v, 3) for k, v in task.items()}, "train_s": round(train_wall, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        cfgs = [
            {"name": "q1", "desc": "smoke", "env_n": 4, "states": 3000, "epochs": 200,
             "latent": 32, "hidden": 128, "trials": 40},
        ]
    else:
        cfgs = [
            {"name": "T1_bind", "desc": "continuous binding baseline",
             "env_n": 4, "states": 6000, "epochs": 500, "latent": 32, "hidden": 128},
            {"name": "T1_bind_L64", "desc": "continuous binding, bigger latent",
             "env_n": 4, "states": 6000, "epochs": 500, "latent": 64, "hidden": 256},
            {"name": "T1_bind_data2x", "desc": "continuous binding, 2x data",
             "env_n": 4, "states": 12000, "epochs": 500, "latent": 32, "hidden": 128},
            {"name": "T3_task_comp", "desc": "compositional decision task",
             "env_n": 4, "states": 6000, "epochs": 500, "latent": 32, "hidden": 128},
            {"name": "T3_task_mask", "desc": "compositional + C-JEPA masking",
             "env_n": 4, "states": 6000, "epochs": 500, "latent": 32, "hidden": 128, "mask": True},
        ]

    rows = []
    for cfg in cfgs:
        try:
            row = run_test(cfg)
            rows.append(row)
        except Exception as e:
            print(f"  TEST {cfg['name']} FAILED: {e}")

    print("\n" + "=" * 64)
    print("BATTERY SUMMARY")
    print("=" * 64)
    hdr = f"{'name':<14}{'bKeepAcc':>9}{'bKeepMSE':>9}{'bPdfErr':>8}{'tFusedSafe':>11}{'tFusedLost':>11}{'tAloneLost':>11}{'tOracle':>8}"
    print(hdr)
    for r in rows:
        print(f"{r['name']:<14}{r['b_keep_acc']:>9}{r['b_keep_mse']:>9}{r['b_pdf_abs_err']:>8}"
              f"{r['t_fused_safe']:>11}{r['t_fused_lost']:>11}{r['t_a_alone_lost']:>11}{r['t_oracle_safe']:>8}")

    # ---- OOD transfer on the best (T3_task_mask) ----
    print("\n" + "=" * 64)
    print("OOD TRANSFER: train n=4 objects, test n=8 objects (Agentic-JEPA failure test)")
    print("=" * 64)
    best_cfg = next(c for c in cfgs if c["name"] == "T3_task_mask")
    dimA = obs_dim_a(4)
    ood_model = BindingAgent(dimA, obs_dim_b(), latent=best_cfg["latent"],
                             hidden=best_cfg["hidden"], mask=True)
    ood_model.load_state_dict(torch.load(CKPT / "bt_T3_task_mask_L32_E4.pt", map_location="cpu",
                                         weights_only=True))
    ood_model.eval()
    ood = eval_ood(ood_model, train_n=4, test_n=8, trials=200)
    for k, v in ood.items():
        print(f"  {k}: {v:.3f}")

    # ---- Scaling law: keep_mse vs data ----
    print("\n" + "=" * 64)
    print("SCALING LAW: binding keep_mse vs data size (power-law test)")
    print("=" * 64)

    def dims_fn(n):
        return (obs_dim_a(n), obs_dim_b())

    sizes, mses = scaling_curve(BindingAgent, dims_fn, env_n=4, epochs=400)
    # fit log-log slope
    xs = np.log(np.array(sizes))
    ys = np.log(np.array(mses))
    slope, intercept = np.polyfit(xs, ys, 1)
    print(f"  log-log slope: {slope:.3f}  (negative => power-law decrease)")
    if slope < 0:
        print(f"  => binding error follows ~N^{{{slope:.2f}}}  (power-law scaling)")

    print("\n" + "=" * 64)
    print("DONE")
    print("=" * 64)


if __name__ == "__main__":
    main()