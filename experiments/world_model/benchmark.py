"""benchmark.py - Train H-JEPA and run baseline evals."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import pathlib
import sys
import time
import numpy as np
import torch

from local_env import LocalDesktop
from proteus.world.hjepa_torch import HJEPA_K4, collect_h4, train


def make_env():
    return LocalDesktop(sandbox=str(ROOT / "sandbox"))


def eval_prediction(model, obs_t, A, obs_t1, obs_tk, batch=128):
    idx = np.random.default_rng(1).choice(len(obs_t), min(batch, len(obs_t)), replace=False)
    bt, ba, bt1, btk = obs_t[idx], A[idx], obs_t1[idx], obs_tk[idx]
    with torch.no_grad():
        losses = model(bt, ba, bt1, btk)
    z_t = model.enc_micro(bt)
    z_tk = model.enc_macro(bt)
    z_t1_target = model.encode_target_micro(bt1)
    z_tk_target = model.encode_target_macro(btk)
    rand_micro = float(torch.mean((z_t - z_t1_target) ** 2))
    rand_macro = float(torch.mean((z_tk - z_tk_target) ** 2))
    return {
        "micro_mse": float(losses["micro"].item()),
        "macro_mse": float(losses["macro"].item()),
        "pos_mse": float(losses["pos"].item()),
        "random_floor_micro": rand_micro,
        "random_floor_macro": rand_macro,
    }


def eval_mpc(model, n_episodes=20, max_steps=25):
    model.eval()
    envg = make_env()
    envg.reset(seed=999)
    for d in ["txt", "jpg", "pdf"]:
        (envg.root / d).mkdir(exist_ok=True)
    for f in ["notes.txt", "todo.txt", "photo1.jpg", "photo2.jpg", "paper.pdf", "doc.pdf"]:
        ext = f.rsplit(".", 1)[-1]
        (envg.root / f).rename(envg.root / ext / f)
    _, goal_vec = envg.obs()
    device = next(model.parameters()).device
    goal_z = model.enc_macro(torch.as_tensor(goal_vec, dtype=torch.float32, device=device).unsqueeze(0))

    mp_success, mp_steps, rand_success, rand_steps = 0, [], 0, []
    for ep in range(n_episodes):
        env = make_env()
        _, vec = env.reset(seed=ep * 7 + 1)
        steps = 0
        done = False
        for _ in range(max_steps):
            files_root = [p for p in env.root.glob("*") if p.is_file()]
            if not files_root:
                break
            A = model.plan_macro(vec, goal_vec, n_samples=48, iters=2, elite=8)
            for a in A:
                _, vec, _, done, _ = env.step(int(a))
                steps += 1
                if done:
                    break
            if done:
                break
        if done:
            mp_success += 1
        mp_steps.append(steps)

        env = make_env()
        _, vec = env.reset(seed=ep * 7 + 1)
        steps = 0
        done = False
        for _ in range(max_steps):
            files_root = [p for p in env.root.glob("*") if p.is_file()]
            if not files_root:
                break
            a = int(np.random.randint(0, 3))
            _, vec, _, done, _ = env.step(a)
            steps += 1
            if done:
                break
        if done:
            rand_success += 1
        rand_steps.append(steps)
    return {
        "mpc_success_rate": mp_success / n_episodes,
        "mpc_avg_steps": float(np.mean(mp_steps) if mp_steps else 0),
        "random_success_rate": rand_success / n_episodes,
        "random_avg_steps": float(np.mean(rand_steps) if rand_steps else 0),
    }


def eval_counterfactual(model, trials=20):
    model.eval()
    device = next(model.parameters()).device
    jepa_rejects = 0
    for t in range(trials):
        env = make_env()
        _, vec = env.reset(seed=t * 13 + 5)
        pdfs = sorted(p.name for p in env.root.glob("*.pdf") if p.is_file())
        if not pdfs:
            continue
        unread = pdfs[0]
        (env.root / unread).write_text("UNREAD important research")
        seq = [2, 2, 2, 2]  # move pdf (action 2); repeated no-ops once no pdf remains
        z_t = model.enc_macro(torch.as_tensor(vec, dtype=torch.float32, device=device).unsqueeze(0))
        z_pred = model.predict_macro(z_t, torch.tensor([seq], device=device))
        pos_pred = model.decode_pos(z_pred).detach().cpu().numpy()[0]
        total_pdf_before = vec[2] + vec[5]
        pred_pdf_after = float(pos_pred[2]) + float(pos_pred[5])
        if pred_pdf_after < total_pdf_before - 0.5:
            jepa_rejects += 1
    return {"jepa_reject_rate": jepa_rejects / trials}


def eval_continual(model, seed=42):
    model.eval()
    env = make_env()
    _, vec = env.reset(seed=seed)
    vec[2] = 0  # no pdf in root
    device = next(model.parameters()).device
    z_t = model.enc_macro(torch.as_tensor(vec, dtype=torch.float32, device=device).unsqueeze(0))
    A_fail = torch.tensor([[2, 2, 2, 2]], device=device)
    z_true = model.enc_macro(torch.as_tensor(vec, dtype=torch.float32, device=device).unsqueeze(0))
    z_pred_before = model.predict_macro(z_t, A_fail)
    mse_before = float(torch.mean((z_pred_before - z_true) ** 2))

    opt = torch.optim.Adam(model.pred_macro.parameters(), lr=3e-4)
    for _ in range(5):
        opt.zero_grad()
        losses = model(torch.as_tensor(vec, dtype=torch.float32, device=device).unsqueeze(0),
                        A_fail,
                        torch.as_tensor(vec, dtype=torch.float32, device=device).unsqueeze(0),
                        torch.as_tensor(vec, dtype=torch.float32, device=device).unsqueeze(0))
        losses["total"].backward()
        opt.step()
        model.update_targets()
    z_pred_after = model.predict_macro(z_t, A_fail)
    mse_after = float(torch.mean((z_pred_after - z_true) ** 2))
    return {"mse_before": mse_before, "mse_after": mse_after}


def eval_text():
    import proteus.bridge.qwen_client as qwen_client
    gen = qwen_client.qwen_generate("what is pdf?")[:80]
    emb = qwen_client.qwen_embed("grep -rn error /var/log")
    return {"qwen_generate": gen, "embed_dim": emb.shape[0]}


def main():
    print("=" * 60)
    print("PROTEUS BENCHMARK - H-JEPA 2-level temporal world model")
    print("=" * 60)

    print("\n[1/5] Collecting trajectories...")
    t0 = time.time()
    obs_t, A, obs_t1, obs_tk = collect_h4(make_env, episodes=400, max_steps=20, k=4, seed=0)
    print(f"  {len(obs_t)} windows in {time.time() - t0:.1f}s")

    print("\n[2/5] Training H-JEPA...")
    torch.manual_seed(0)
    model = HJEPA_K4(obs_dim=9, act_dim=3, hidden=128, k=4)
    split = int(0.8 * len(obs_t))
    hist = train(model, obs_t[:split], A[:split], obs_t1[:split], obs_tk[:split],
                  epochs=300, batch=64, lr=3e-4, log_every=50)
    ckpt = ROOT / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt / "hjepa_k4.pt")
    print(f"  saved checkpoint {ckpt / 'hjepa_k4.pt'}")

    print("\n[3/5] Prediction quality...")
    pred = eval_prediction(model, obs_t[split:], A[split:], obs_t1[split:], obs_tk[split:])
    for k, v in pred.items():
        print(f"  {k}: {v:.4f}")

    print("\n[4/5] MPC planning vs random...")
    mpc = eval_mpc(model, n_episodes=20)
    for k, v in mpc.items():
        print(f"  {k}: {v}")

    print("\n[5/5] Counterfactual + continual + text...")
    cf = eval_counterfactual(model, trials=20)
    print(f"  counterfactual reject rate: {cf['jepa_reject_rate']:.2f}")
    cont = eval_continual(model)
    print(f"  continual macro MSE: {cont['mse_before']:.4f} -> {cont['mse_after']:.4f}")
    tx = eval_text()
    print(f"  text: qwen='{tx['qwen_generate']}'")
    print(f"         embed dim={tx['embed_dim']}")

    print("\n" + "=" * 60)
    print("DONE - baseline established")
    print("=" * 60)


if __name__ == "__main__":
    main()