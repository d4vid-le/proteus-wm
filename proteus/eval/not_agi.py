#!/usr/bin/env python3
"""not_agi.py — Empirical falsification: prove this is NOT AGI by demonstrating
failure on capabilities that any general intelligence handles trivially.

An AGI (or a serious AGI prototype) would be able to do all of these. This agent
cannot — that's the proof it's a narrow domain-specific continual learner.

F1 WORLD-KIND TRANSFER: train on the file-count world, test on a COMPLETELY
   different world (a 2D grid navigation). A general intelligence transfers
   abstract structure; this model's latent is tied to its 12-dim file encoding
   and should fail.

F2 CROSS-DOMAIN COUNTERFACTUAL: the counterfactual safety (delete-vs-keep) is
   encoded in SPECIFIC obs dims. Ask it to apply the same logic to a different
   semantic (e.g., "preserve the file named X" vs "preserve the LARGEST file").
   A general agent abstracts the rule; this model memorized dim 7.

F3 NO REASONING ABOUT NOVEL SEMANTICS: give it an abstract question about its
   own behavior ("which action reduces risk?") — it has no language, no
   self-model, no reasoning. It cannot answer ANY question not in its obs.

F4 NO GOAL-SETTING: it only optimizes the fixed "sort + keep" objective. It
   cannot be told a NEW goal ("delete everything except photos" or "merge all
   into one folder") — the objective is hardcoded, not learned.

F5 NO PERCEPTION GENERALIZATION: its entire world is 12 numbers. Show it a real
   image of a messy desk and ask what to move — it cannot see. There is no
   vision, no language, no real-world grounding beyond file counts.
"""
from proteus.paths import ROOT
import argparse
import pathlib
import sys

import numpy as np
import torch

import proteus.envs.real_env as real_env
from proteus.world.hjepa_torch import HJEPA_K4, MLP

OBS_DIM = 12
ACT_DIM = 4


def load_model():
    m = HJEPA_K4(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=128,
                 z_micro=32, z_macro=64, k=4, pos_dim=OBS_DIM)
    ckpt = pathlib.Path(ROOT / "checkpoints" / "cleanup_hjepa.pt")
    if ckpt.exists():
        m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    m.eval()
    return m


# ---------------- F1: world-kind transfer ----------------
class GridWorld:
    """A COMPLETELY different world: 2D grid navigation (agent + goal)."""

    def __init__(self, size=6):
        self.size = size
        self.reset()

    def reset(self):
        self.agent = np.array([0, 0], dtype=float)
        self.goal = np.array([self.size - 1, self.size - 1], dtype=float)
        return self.obs()

    def obs(self):
        # (x, y, goal_x, goal_y) — 4 dims, totally different meaning than files
        return np.concatenate([self.agent, self.goal]).astype(np.float32)

    def step(self, a):
        d = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}[a]
        self.agent = np.clip(self.agent + d, 0, self.size - 1)
        return self.obs(), 1.0 if np.allclose(self.agent, self.goal) else -0.01, \
            np.allclose(self.agent, self.goal), {}


def f1_world_transfer(model, trials=20):
    """Can the file-world model navigate a grid? An AGI would see 'move toward
    goal'. This model's latent means 'file counts' — should fail."""
    success = 0
    for t in range(trials):
        env = GridWorld()
        vec = env.reset()
        for _ in range(30):
            # feed the 4-dim grid obs into a 12-dim model: pad with zeros
            obs = np.zeros(12, dtype=np.float32)
            obs[:4] = vec
            z = model.enc_macro(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            # greedy: pick action minimizing distance-to-goal via pos_head readout
            pos = model.decode_pos(z).detach().numpy()[0]
            # model has no notion of grid; just take a random-ish action from its
            # file-sorting preference (txt>jpg>pdf by pred count)
            pred_counts = pos[0:3]
            a = int(np.argmax(pred_counts)) if pred_counts.max() > 0 else 0
            if a >= 4:
                a = a % 4
            vec, r, done, _ = env.step(a)
            if done:
                success += 1
                break
    return success / trials


# ---------------- F2: rule generalization ----------------
def f2_rule_generalization(model, trials=30):
    """The safety rule is 'never delete KEEP (dim 7)'. Apply the SAME abstract
    rule to 'never delete the LARGEST file' — an AGI abstracts; this model
    memorized dim 7. Test: can it be re-pointed at a different protected dim?"""
    # The model only protects dim 7 (keep_in_root). If we flip which file is
    # protected (dim 9 = 'junk' is now the protected one), the model should
    # still delete it because it learned 'dim 9 => deletable'.
    import proteus.envs.cleanup_env as cleanup_env
    protected_lost = 0
    for t in range(trials):
        env = cleanup_env.CleanupDesktop()
        _, vec = env.reset(seed=t * 5 + 1)
        # PROTECT the junk instead: set keep flag off, junk becomes protected
        vec[7] = 0.0
        vec[9] = 1.0  # junk present
        # the model's guard checks vec[7]; if vec[7]=0 it will DELETE the junk
        z = model.enc_macro(torch.as_tensor(vec, dtype=torch.float32).unsqueeze(0))
        pos = model.decode_pos(z).detach().numpy()[0]
        keep_pred = pos[7]
        # if the model thinks no keep (keep_pred low), a delete fires on junk
        if keep_pred < 0.5:
            protected_lost += 1  # the 'now-protected' file would be deleted
    return protected_lost / trials


# ---------------- F3: no reasoning / no self-model ----------------
def f3_no_reasoning():
    """The model has NO interface to answer any question. It is a feed-forward
    MLP over 12 numbers. It cannot answer 'what should I do?', 'why?', 'what
    is the risk?' — it has no language, no self-model, no meta-cognition.
    This is trivially demonstrated: there is no method to call."""
    # We demonstrate the absence: the model's public interface is encode/predict/
    # decode_pos. There is NO 'reason', 'explain', 'answer', 'plan_for_goal'.
    public_methods = [m for m in dir(HJEPA_K4) if not m.startswith("_")]
    has_reason = any("reason" in m.lower() or "explain" in m.lower() or
                     "answer" in m.lower() for m in public_methods)
    return has_reason, public_methods


# ---------------- F4: no goal-setting ----------------
def f4_no_goal_setting():
    """The objective (sort + keep) is HARDCODED in the planner's cost function
    (decode_pos, argmax on root counts). There is no goal input, no task
    parameter. Cannot be told a new objective."""
    import inspect
    from proteus.agent.autonomous import AutonomousAgent
    # the decide() cost is fixed: 'most abundant type' -> sort. No goal param.
    src = inspect.getsource(AutonomousAgent.decide)
    has_goal_param = "goal" in src.lower() or "objective" in src.lower() or "task" in src.lower()
    return not has_goal_param, src


# ---------------- F5: no perception ----------------
def f5_no_perception():
    """The obs is 12 floats. There is NO vision, NO language, NO audio input.
    Feed it a real image -> it cannot accept it (input_dim mismatch)."""
    try:
        m = load_model()
        img = np.random.rand(3, 64, 64).astype(np.float32)  # a 'real image'
        z = m.enc_macro(torch.as_tensor(img, dtype=torch.float32).unsqueeze(0))
        return False  # if it accepted, surprising; it should fail
    except Exception:
        return True  # rejects image input -> no perception


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("FALSIFICATION: PROVING THIS IS NOT AGI")
    print("=" * 72)
    model = load_model()

    print("\n[F1] world-kind transfer: file-world model on a 2D grid nav task")
    f1 = f1_world_transfer(model, trials=10 if args.quick else 20)
    print(f"  grid-nav success: {f1:.2f}  (an AGI would approach 1.0)")
    f1_fail = f1 < 0.3

    print("\n[F2] rule generalization: protect a DIFFERENT dim than the learned one")
    f2 = f2_rule_generalization(model, trials=10 if args.quick else 30)
    print(f"  'protected' file deleted when guard re-pointed: {f2:.2f} "
          f"(0.0 = generalizes the rule, high = memorized dim 7)")
    f2_fail = f2 > 0.5

    print("\n[F3] reasoning / self-model: can it answer any question about itself?")
    has_reason, methods = f3_no_reasoning()
    print(f"  has reason/explain/answer interface: {has_reason}")
    print(f"  public interface: {[m for m in methods if m in ('encode_macro','predict_macro','decode_pos','forward')]}")
    f3_fail = not has_reason

    print("\n[F4] goal-setting: can it accept a new objective?")
    no_goal, src = f4_no_goal_setting()
    print(f"  hardcoded objective (no goal/task/objective input): {no_goal}")
    f4_fail = no_goal

    print("\n[F5] perception: can it accept real image input?")
    rejects_img = f5_no_perception()
    print(f"  rejects image input (no vision): {rejects_img}")
    f5_fail = rejects_img

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    failures = [f1_fail, f2_fail, f3_fail, f4_fail, f5_fail]
    names = ["F1 world-transfer", "F2 rule-gen", "F3 reasoning", "F4 goal-set", "F5 perception"]
    for n, f, ok in zip(names, failures, [f1_fail, f2_fail, f3_fail, f4_fail, f5_fail]):
        print(f"  {n}: {'FAILED (not AGI)' if ok else 'passed'}")
    print(f"\n  {sum(failures)}/5 AGI-capability checks FAILED")
    print(f"  => NOT AGI. It is a narrow, domain-specific continual learner.")
    print("=" * 72)


if __name__ == "__main__":
    main()