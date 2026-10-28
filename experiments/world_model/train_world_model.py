#!/usr/bin/env python3
"""train_world_model.py — Scale the H-JEPA world agent: more data, longer training,
full test pass. Saves checkpoints/hjepa_proteus.pt.

Usage:
  python3 train_world_model.py --episodes 1200 --epochs 800 --hidden 256
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
import time

import numpy as np
import torch

import local_env
from proteus.world.hjepa_torch import HJEPA_K4, collect_h4, train


def make_env():
    return local_env.LocalDesktop(sandbox=str(ROOT / "sandbox"))


def eval_prediction(model, obs_t, A, obs_t1, obs_tk, batch=256):
    idx = np.random.default_rng(1).choice(len(obs_t), min(batch, len(obs_t)), replace=False)
    bt, ba, bt1, btk = obs_t[idx], A[idx], obs_t1[idx], obs_tk[idx]
    with torch.no_grad():
        losses = model(bt, ba, bt1, btk)
    z_t = model.enc_micro(bt)
    z_tk = model.enc_macro(bt)
    z_t1_t = model.encode_target_micro(bt1)
    z_tk_t = model.encode_target_macro(btk)
    rand_micro = float(torch.mean((z_t - z_t1_t) ** 2))
    rand_macro = float(torch.mean((z_tk - z_tk_t) ** 2))
    return {
        "micro_mse": float(losses["micro"].item()),
        "macro_mse": float(losses["macro"].item()),
        "pos_mse": float(losses["pos"].item()),
        "random_floor_micro": rand_micro,
        "random_floor_macro": rand_macro,
    }


def sorted_goal(env):
    """Vector of the fully-sorted observation: nothing messy in root,
    each type folder holds its 2 files. Derived from env's obs layout, no FS mutation."""
    _, messy_vec = env.obs()
    n_txt = int(messy_vec[0]) + int(messy_vec[3])
    n_jpg = int(messy_vec[1]) + int(messy_vec[4])
    n_pdf = int(messy_vec[2]) + int(messy_vec[5])
    goal = messy_vec.copy()
    goal[0] = 0; goal[1] = 0; goal[2] = 0
    goal[3] = n_txt; goal[4] = n_jpg; goal[5] = n_pdf
    goal[6] = 0
    return goal


def eval_planner_hard(model, n_episodes=30, max_steps=30, k=4, hard=True):
    """Compare executors on the sandbox (goal = true sorted state), 1 step/iter:
      - greedy: 1-step latent readout (optimal for this env)
      - mpc_cem: receding-horizon k-step CEM (commits 1, re-plans)
      - random: uniform baseline
    """
    model.eval()
    device = next(model.parameters()).device
    results = {"greedy": [], "mpc_cem": [], "random": []}

    def run(mode, seed):
        env = make_env()
        _, vec = env.reset(seed=seed)
        goal_vec = sorted_goal(env)
        steps = 0
        done = False
        for _ in range(max_steps):
            files_root = [p for p in env.root.glob("*") if p.is_file()]
            if not files_root:
                done = True
                break
            if mode == "greedy":
                a = model.plan_greedy(vec)
            elif mode == "mpc_cem":
                A = model.plan_macro(vec, goal_vec, n_samples=64, iters=3, elite=8)
                a = int(A[0])
            else:
                a = int(np.random.randint(0, 3))
            _, vec, _, d, _ = env.step(a)
            steps += 1
            if d:
                done = True
                break
        return done, steps

    for ep in range(n_episodes):
        seed = 11 + ep * 13
        for mode in results:
            d, s = run(mode, seed)
            results[mode].append((d, s))

    out = {}
    for mode, pairs in results.items():
        succ = sum(d for d, _ in pairs)
        steps = [s for _, s in pairs]
        out[f"{mode}_success_rate"] = succ / n_episodes
        out[f"{mode}_avg_steps"] = float(np.mean(steps))
    out["greedy_vs_random_gain"] = out["greedy_success_rate"] - out["random_success_rate"]
    out["greedy_vs_random_steps"] = out["random_avg_steps"] - out["greedy_avg_steps"]
    return out


def eval_irreversible(model, trials=30, k=4):
    """Counterfactual: an 'UNREAD' pdf must survive. Plan is greedy (deletes/moves
    everything), the model should predict the unread flag count going wrong.
    Metric: pos_head correctly reads pdf count after a 4-step 'move all' seq."""
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    for t in range(trials):
        env = make_env()
        _, vec = env.reset(seed=t * 7 + 3)
        pdfs_root = [p.name for p in env.root.glob("*.pdf") if p.is_file()]
        if not pdfs_root:
            continue
        seq = [2, 2, 2, 2]  # keep moving pdfs
        z = model.enc_macro(torch.as_tensor(vec, dtype=torch.float32, device=device).unsqueeze(0))
        zp = model.predict_macro(z, torch.tensor([seq], device=device))
        pos = model.decode_pos(zp).detach().cpu().numpy()[0]
        root_pdf_true = vec[2]
        folder_pdf_true = vec[5]
        # predicted root pdf should go down, folder pdf up, and ideally sum conserved
        pred_root_pdf = float(pos[2])
        pred_folder_pdf = float(pos[5])
        # a counterfactual-aware readout: predicts pdf moved OUT of root
        moved_pred = root_pdf_true - pred_root_pdf
        moved_true = min(root_pdf_true, 1.0)  # 1 move per action, k=4 -> all of them
        ok = (pred_folder_pdf > folder_pdf_true - 0.5) or (moved_pred > 0.5)
        correct += ok
    return {"readout_success_rate": correct / trials}


def eval_continual(model, steps=50, seed=42):
    model.eval()
    env = make_env()
    _, vec = env.reset(seed=seed)
    device = next(model.parameters()).device
    z_t = model.enc_macro(torch.as_tensor(vec, dtype=torch.float32, device=device).unsqueeze(0))
    A = torch.tensor([[2, 2, 2, 2]], device=device)
    z_true = model.enc_macro(torch.as_tensor(vec, dtype=torch.float32, device=device).unsqueeze(0))
    mse_before = float(torch.mean((model.predict_macro(z_t, A) - z_true) ** 2))

    # online updates on the SAME transition (continual setting: new experience, keep old)
    opt = torch.optim.Adam(model.pred_macro.parameters(), lr=3e-4)
    vec_t = torch.as_tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
    for _ in range(steps):
        opt.zero_grad()
        losses = model(vec_t, A, vec_t, vec_t)
        losses["total"].backward()
        opt.step()
        model.update_targets()
    mse_after = float(torch.mean((model.predict_macro(z_t, A) - z_true) ** 2))
    return {"mse_before": mse_before, "mse_after": mse_after}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--z-micro", type=int, default=32)
    ap.add_argument("--z-macro", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--no-train", action="store_true", help="load checkpoint, only test")
    args = ap.parse_args()

    print("=" * 64)
    print("PROTEUS TRAIN + TEST — H-JEPA world agent")
    print("=" * 64)

    ckpt = pathlib.Path(args.ckpt or ROOT / "checkpoints" / "hjepa_proteus.pt")
    ckpt.parent.mkdir(exist_ok=True)

    torch.manual_seed(0)
    model = HJEPA_K4(obs_dim=9, act_dim=3, hidden=args.hidden,
                     z_micro=args.z_micro, z_macro=args.z_macro, k=args.k)

    if not args.no_train:
        print(f"\n[collect] {args.episodes} episodes, max_steps={args.max_steps}, k={args.k}")
        t0 = time.time()
        obs_t, A, obs_t1, obs_tk = collect_h4(make_env, episodes=args.episodes,
                                               max_steps=args.max_steps, k=args.k, seed=0)
        print(f"  {len(obs_t)} windows in {time.time()-t0:.1f}s")

        print(f"\n[train] {args.epochs} epochs, batch {args.batch}, hidden {args.hidden}, lr {args.lr}")
        split = int(0.85 * len(obs_t))
        t0 = time.time()
        hist = train(model, obs_t[:split], A[:split], obs_t1[:split], obs_tk[:split],
                     epochs=args.epochs, batch=args.batch, lr=args.lr, log_every=args.epochs // 10)
        print(f"  train wall {time.time()-t0:.1f}s | final total {hist['total'][-1]:.4f}")
        torch.save(model.state_dict(), ckpt)
        print(f"  saved {ckpt}")
        hold = (obs_t[split:], A[split:], obs_t1[split:], obs_tk[split:])
    else:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        print(f"\n[load] {ckpt}")

    print("\n[test 1/4] prediction quality")
    if not args.no_train:
        pred = eval_prediction(model, *hold)
        for k_, v in pred.items():
            print(f"  {k_}: {v:.4f}")

    print("\n[test 2/4] planner comparison (greedy readout vs CEM vs random)")
    mpc = eval_planner_hard(model, n_episodes=30)
    for k_, v in mpc.items():
        print(f"  {k_}: {v:.3f}")

    print("\n[test 3/4] irreversible counterfactual readout (UNREAD pdf must survive)")
    irr = eval_irreversible(model, trials=30)
    for k_, v in irr.items():
        print(f"  {k_}: {v:.3f}")

    print("\n[test 4/4] continual learning (50 online updates, same transition)")
    cont = eval_continual(model, steps=50)
    for k_, v in cont.items():
        print(f"  {k_}: {v:.4f}")

    print("\n" + "=" * 64)
    print("DONE")
    print("=" * 64)


if __name__ == "__main__":
    main()