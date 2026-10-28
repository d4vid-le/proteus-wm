#!/usr/bin/env python3
"""continual_demo.py — SHOW continual learning live on the autonomous agent.

The honest, robust demonstration. Latent MSE on a shifted distribution is noisy;
the GROUND TRUTH is task performance. So:

  1. TRAIN a base H-JEPA on the standard world (txt/jpg/pdf + keep).
  2. SHIFT: a new file type `.md` arrives (never seen). The agent must now route
     `.md` files too.
  3. Measure TASK SUCCESS before CL: run the autonomous loop in the shifted world,
     count how many non-keep files get correctly routed.
  4. CONTINUAL LEARN from verified shifted experiences (anchored replay).
  5. Measure TASK SUCCESS after CL. If it improves, the agent genuinely learned
     the new distribution WITHOUT forgetting the old (measured on the old world).

The result: task-level learning curve + no-forgetting check, both on real FS.
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

import numpy as np
import torch

import proteus.envs.real_env as real_env
from proteus.world.hjepa_torch import HJEPA_K4, collect_h4, train as hj_train

OBS_DIM = real_env.OBS_DIM
ACT_DIM = real_env.ACT_DIM
CKPT = ROOT / "checkpoints"

VALID_MOVE = {".txt": "txt", ".md": "txt", ".jpg": "jpg", ".jpeg": "jpg",
              ".png": "jpg", ".pdf": "pdf"}


class ShiftedWorld:
    """A real directory with an extra .md type the base model never saw."""

    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.env = real_env.RealEnv(self.root)

    def reset(self, seed=0):
        import shutil
        root = self.root
        for p in root.glob("*"):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        for d in ["txt", "jpg", "pdf", "md"]:
            (root / d).mkdir(exist_ok=True)
        for i in range(2):
            (root / f"note{i}.txt").write_text(f"n{i}")
            (root / f"photo{i}.jpg").write_text(f"p{i}")
        (root / "doc.md").write_text("# md doc")
        (root / "notes.md").write_text("# md notes")
        (root / "KEEP.pdf").write_text("UNREAD")
        (root / "junk.pdf").write_text("junk")
        return self.obs()

    def obs(self):
        text, vec = self.env.obs()
        root = sorted(p.name for p in self.root.glob("*") if p.is_file())
        md_root = sum(1 for f in root if f.endswith(".md"))
        # .md gets a DISTINCT channel: reuse dim 9 (junk) as a learnable new
        # signal — the base model treats dim9 as 'junk pdf present', but here it
        # means 'md present'. The model must LEARN the new semantics of dim9
        # under the shifted distribution (this is the genuine, learnable shift).
        vec = vec.copy()
        vec[9] = 1.0 if md_root > 0 else 0.0
        vec[0] += md_root  # md also routes to txt, but dim9 flags them distinctly
        vec[6] = len(root)
        vec[10] = len(root) + sum(len(list(d.glob("*"))) for d in
                                  [self.root / x for x in ("txt", "jpg", "pdf")])
        return text, vec

    def step(self, action):
        if action == 0:
            # move txt OR md (txt-like)
            for p in sorted(list(self.root.glob("*.txt")) + list(self.root.glob("*.md"))):
                if p.is_file():
                    dest = self.root / "txt" / p.name
                    if dest.exists():
                        dest.unlink()
                    p.rename(dest)
                    break
            text, vec = self.obs()
            return text, vec, -0.01, False, {}
        return self.env.step(action)


def route_one(model, root, env):
    """Use the MODEL's latent readout to decide which loose file to move.
    The model's pos_head predicts root counts; we move the type it thinks is
    most abundant. This makes CL matter: if the model learned the shifted world,
    its decisions improve."""
    files = [p for p in root.glob("*") if p.is_file() and not env.is_keep(p.name)]
    if not files:
        return False, None, "none"
    _, vec = env.obs()
    # model latent decision
    z = model.enc_macro(torch.as_tensor(vec, dtype=torch.float32).unsqueeze(0))
    pos = model.decode_pos(z).detach().numpy()[0]
    pred_counts = pos[0:3].copy()  # txt, jpg, pdf
    # + md folded into txt (the shifted type); + misc via total
    # pick the argmax among types that actually have a file present (grounded gate)
    present = {"txt": vec[0], "jpg": vec[1], "pdf": vec[2]}
    ext2idx = {"txt": 0, "jpg": 1, "pdf": 2}
    best = None
    for ext in ("txt", "jpg", "pdf"):
        if present[ext] > 0 and pred_counts[ext2idx[ext]] >= 0:
            if best is None or pred_counts[ext2idx[ext]] > pred_counts[ext2idx[best]]:
                best = ext
    if best is None:
        return False, None, "nothing-predicted"
    target = best
    # move a file of that type (prefer the first loose one)
    for p in sorted(files):
        if p.suffix.lower() in (".txt", ".md") and target == "txt":
            (root / target).mkdir(exist_ok=True)
            if (root / target / p.name).exists():
                (root / target / p.name).unlink()
            p.rename(root / target / p.name)
            return True, p.name, target
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and target == "jpg":
            (root / target).mkdir(exist_ok=True)
            if (root / target / p.name).exists():
                (root / target / p.name).unlink()
            p.rename(root / target / p.name)
            return True, p.name, target
        if p.suffix.lower() == ".pdf" and target == "pdf":
            (root / target).mkdir(exist_ok=True)
            if (root / target / p.name).exists():
                (root / target / p.name).unlink()
            p.rename(root / target / p.name)
            return True, p.name, target
    return False, None, "no-match"


def task_success(model, trials=30, shifted=True):
    """Ground-truth: the MODEL decides each move (via latent readout), the world
    executes it. Success = all actionable files routed AND keep survived."""
    successes = 0
    for t in range(trials):
        root = pathlib.Path(f"/tmp/proteus-task-{t}")
        if shifted:
            w = ShiftedWorld(root)
            w.reset(seed=t)
            env = w.env
            root = w.root
        else:
            real_env.seed_real_world(root, keep_pdf=True, seed=t)
            env = real_env.RealEnv(root)
        # count actionable files
        actionable = [p for p in root.glob("*") if p.is_file() and not env.is_keep(p.name)]
        n_actionable = len(actionable)
        routed = 0
        for _ in range(30):
            moved, _, _ = route_one(model, root, env)
            if not moved:
                break
            routed += 1
        # AFTER routing: KEEP must SURVIVE (still present somewhere, never deleted)
        keep_survived = any(
            p.is_file() and env.is_keep(p.name)
            for p in list(root.glob("*")) + [f for d in ("txt", "jpg", "pdf") if (root / d).exists() for f in (root / d).glob("*")])
        if routed == n_actionable and keep_survived:
            successes += 1
    return successes / trials


def eval_old_world(model, trials=30):
    return task_success(model, trials=trials, shifted=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=4)
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()

    print("=" * 70)
    print("CONTINUAL LEARNING — LIVE DEMONSTRATION (task-level)")
    print("=" * 70)

    print("\nPhase 1: train base H-JEPA on the STANDARD world")
    torch.manual_seed(0)
    model = HJEPA_K4(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=128,
                     z_micro=32, z_macro=64, k=4, pos_dim=OBS_DIM)
    obs_t, A, obs_t1, obs_tk = collect_h4(lambda: real_env.RealEnv(
        pathlib.Path("/tmp/proteus-cl-base")), episodes=1200, max_steps=12, k=4,
        seed=0, act_dim=ACT_DIM)
    split = int(0.8 * len(obs_t))
    hist = hj_train(model, obs_t[:split], A[:split], obs_t1[:split], obs_tk[:split],
                    epochs=400, batch=128, log_every=None)
    print(f"  base trained: total loss {hist['total'][-1]:.4f}")

    print("\nPhase 2: baseline task success (OLD world + SHIFTED world, no CL)")
    old0 = eval_old_world(model, trials=args.trials)
    new0 = task_success(model, trials=args.trials, shifted=True)
    print(f"  OLD world   task success: {old0:.2f}")
    print(f"  SHIFTED world task success: {new0:.2f}")

    print("\nPhase 3: continual learning on verified SHIFTED experiences (anchored replay)")
    print(f"{'event':>6} {'shift-task':>11} {'old-task':>9} {'Δshift':>8} {'Δold':>7}")
    print("-" * 46)
    shift_succ = [new0]
    old_succ = [old0]
    for ev in range(1, args.events + 1):
        # collect verified shifted experiences (simulate the daemon's buffer)
        root = pathlib.Path(f"/tmp/proteus-cl-buf-{ev}")
        w = ShiftedWorld(root)
        w.reset(seed=ev * 7)
        # run a cleanup, log the (obs,A,obs_tk) from it
        env = w.env
        traj = [w.obs()[1]]
        acts = []
        for _ in range(12):
            a = 0 if any(p.suffix in (".txt", ".md") for p in root.glob("*") if p.is_file()) else 1
            _, v2, _, done, _ = w.step(a)
            traj.append(v2)
            acts.append(a)
            if done or len([p for p in root.glob("*") if p.is_file() and not env.is_keep(p.name)]) == 0:
                break
        # build verified (obs_t, A, obs_t1, obs_tk) windows
        tuples = []
        for i in range(len(traj) - 4):
            tuples.append((traj[i], acts[i:i + 4], traj[i + 1], traj[i + 4]))
        # anchored replay: mix with old-world anchors (collected from OLD dist)
        if ev == 1:
            anchors = []
            for s in range(8):
                r = pathlib.Path(f"/tmp/proteus-anchor-{s}")
                real_env.seed_real_world(r, keep_pdf=True, seed=s)
                envr = real_env.RealEnv(r)
                tr = [envr.obs()[1]]
                ar = []
                for _ in range(6):
                    aa = random.randint(0, 3)
                    _, vv, _, dd, _ = envr.step(aa)
                    tr.append(vv); ar.append(aa)
                    if dd:
                        break
                for i in range(max(len(tr) - 4, 0)):
                    anchors.append((tr[i], ar[i:i + 4], tr[i + 1], tr[i + 4]))
        # CL update: train the DECISION path (enc_macro + pos_head) so the
        # readout adapts to the shifted world, PLUS pred_macro + VICReg to keep
        # the latent healthy. This is what fixes task-level routing.
        opt = torch.optim.Adam([
            *model.pred_macro.parameters(),
            *model.enc_macro.parameters(),
            *model.pos_head.parameters(),
        ], lr=3e-4)
        replay = tuples + (anchors[:20] if ev == 1 else [])
        for _ in range(40):
            for ot, AA, ot1, otk in replay:
                bt = torch.as_tensor(ot, dtype=torch.float32).unsqueeze(0)
                ba = torch.as_tensor([AA], dtype=torch.int64)
                b1 = torch.as_tensor(ot1, dtype=torch.float32).unsqueeze(0)
                bk = torch.as_tensor(otk, dtype=torch.float32).unsqueeze(0)
                opt.zero_grad()
                losses = model(bt, ba, b1, bk)
                # pos_head supervised on the TRUE current obs counts (grounded)
                pos_target = torch.as_tensor(ot, dtype=torch.float32).unsqueeze(0)[:, :12]
                l_pos = torch.mean((model.decode_pos(model.enc_macro(bt)) - pos_target) ** 2)
                (losses["total"] + 2.0 * l_pos).backward()
                opt.step()
                model.update_targets()
        ns = task_success(model, trials=args.trials, shifted=True)
        os_ = eval_old_world(model, trials=args.trials)
        shift_succ.append(ns); old_succ.append(os_)
        print(f"{ev:>6} {ns:>11.2f} {os_:>9.2f} {ns-new0:>+8.2f} {os_-old0:>+7.2f}")
        torch.save(model.state_dict(), CKPT / "continual_live.pt")

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  SHIFTED world task success: {new0:.2f} -> {shift_succ[-1]:.2f} "
          f"({100*(shift_succ[-1]-new0):.0f}pp change)")
    print(f"  OLD world task success:     {old0:.2f} -> {old_succ[-1]:.2f} "
          f"({100*(old_succ[-1]-old0):+.0f}pp -> "
          f"{'NO FORGETTING' if old_succ[-1] >= old0 - 0.05 else 'FORGOT'})")
    print(f"  checkpoint: checkpoints/continual_live.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()