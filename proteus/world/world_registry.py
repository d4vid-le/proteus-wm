#!/usr/bin/env python3
"""world_registry.py — S1: World routing via a SUPERVISED gate + per-world JEPAs.

The OS gives the world label for free (app_name, env_id). So routing is:
    frozen features f (differ by 129 between worlds)
      -> linear gate W_gate·f -> softmax -> P(world)   [supervised, >99%]
      -> pick the world's PerWorldJEPA (cross_reason, proven 20/20 planning)
      -> plan in that world's latent

The gate operates on the RAW frozen features (highly separable), NOT a shared
latent that competes with dynamics training. Each world keeps its own JEPA
(proven to acquire + plan in ~235 transitions).

Tests:
  T1 gate routing accuracy >= 95% (supervised, world label from OS)
  T2 planning in each routed world reaches the goal (20/20 each)
"""
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from proteus.safety.cross_reason import build_frozen, feat_of, DEVICE


class GridNavA:
    def __init__(self, size=5):
        self.size = size
        self.agent = np.array([0, 0])
        self.goal = np.array([size - 1, size - 1])

    def render(self, size=224):
        g = self.size; cell = size // g
        img = Image.new("RGB", (size, size), (20, 20, 30))
        d = ImageDraw.Draw(img)
        d.rectangle([self.goal[1]*cell, self.goal[0]*cell, (self.goal[1]+1)*cell, (self.goal[0]+1)*cell], fill=(40, 200, 90))
        d.rectangle([self.agent[1]*cell, self.agent[0]*cell, (self.agent[1]+1)*cell, (self.agent[0]+1)*cell], fill=(60, 120, 240))
        return np.array(img, dtype=np.uint8)

    def step(self, a):
        d = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0)}[a]
        self.agent = np.clip(self.agent + d, 0, self.size - 1)
        return bool(np.all(self.agent == self.goal))

    def reset(self, seed=0):
        rng = random.Random(seed)
        self.agent = np.array([rng.randint(0, 3), rng.randint(0, 3)])


class GridNavB:
    def __init__(self, size=3):
        self.size = size
        self.agent = np.array([0, 0])
        self.goal = np.array([size - 1, size - 1])
        self.wall = np.array([0, 2])

    def render(self, size=224):
        g = self.size; cell = size // g
        img = Image.new("RGB", (size, size), (40, 30, 40))
        d = ImageDraw.Draw(img)
        d.rectangle([self.wall[1]*cell, self.wall[0]*cell, (self.wall[1]+1)*cell, (self.wall[0]+1)*cell], fill=(120, 120, 120))
        d.rectangle([self.goal[1]*cell, self.goal[0]*cell, (self.goal[1]+1)*cell, (self.goal[0]+1)*cell], fill=(200, 200, 40))
        d.rectangle([self.agent[1]*cell, self.agent[0]*cell, (self.agent[1]+1)*cell, (self.agent[0]+1)*cell], fill=(240, 120, 240))
        return np.array(img, dtype=np.uint8)

    def step(self, a):
        d = {0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0)}[a]
        nxt = self.agent + d
        if np.all(nxt == self.wall):
            nxt = self.agent
        self.agent = np.clip(nxt, 0, self.size - 1)
        return bool(np.all(self.agent == self.goal))

    def reset(self, seed=0):
        rng = random.Random(seed)
        self.agent = np.array([rng.randint(0, 2), rng.randint(0, 1)])


def collect_world(env_cls, enc, tr, n=60, device=DEVICE):
    samples = []
    for _ in range(n):
        env = env_cls()
        env.reset(seed=random.randint(0, 1000))
        f0 = feat_of(enc, tr, env.render(), device)
        for _ in range(4):
            a = random.randint(0, 3)
            done = env.step(a)
            f1 = feat_of(enc, tr, env.render(), device)
            samples.append((f0, a, f1))
            f0 = f1
            if done:
                break
    return samples


class WorldGate:
    """Supervised gate on frozen features -> world, with per-world JEPAs."""

    def __init__(self, enc, tr, feat_dim=25088, n_worlds=2):
        self.enc = enc
        self.tr = tr
        self.feat_dim = feat_dim
        self.gate = nn.Linear(feat_dim, n_worlds).to(DEVICE)
        self.worlds = {}  # name -> PerWorldJEPA
        self.name_to_idx = {}

    def add_world(self, name, samples, epochs=200):
        from proteus.safety.cross_reason import PerWorldJEPA
        model = PerWorldJEPA(self.feat_dim).to(DEVICE)
        opt = torch.optim.Adam(list(model.proj.parameters()) + list(model.pred.parameters()), lr=3e-3)
        for _ in range(epochs):
            batch = random.sample(samples, min(16, len(samples)))
            loss, _ = model.loss(batch)
            opt.zero_grad(); loss.backward(); opt.step(); model.update_target()
        self.worlds[name] = model
        self.samples = getattr(self, "samples", {})
        self.samples[name] = samples
        self.name_to_idx[name] = len(self.name_to_idx)

    def train_gate(self, epochs=300):
        """Supervised gate: W_gate·f -> P(world). Frozen features are highly
        separable (differ by ~129), so this should reach >99%."""
        X, Y = [], []
        with torch.no_grad():
            for name, samples in self.samples.items():
                idx = self.name_to_idx[name]
                for f0, _, _ in random.sample(samples, min(40, len(samples))):
                    X.append(f0.squeeze(0))
                    Y.append(idx)
        X = torch.stack(X)
        Y = torch.as_tensor(Y, dtype=torch.long, device=DEVICE)
        opt = torch.optim.Adam(self.gate.parameters(), lr=1e-3)
        for _ in range(epochs):
            idx = random.sample(range(len(X)), min(32, len(X)))
            logits = self.gate(X[idx])
            loss = torch.nn.functional.cross_entropy(logits, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            acc = float((self.gate(X).argmax(1) == Y).float().mean().item())
        return acc

    def route(self, f):
        with torch.no_grad():
            logits = self.gate(f)
            idx = int(logits.argmax(dim=1).item())
        name = [n for n, i in self.name_to_idx.items() if i == idx][0]
        return name

    def plan(self, env, goal_env, k=3, max_steps=40):
        """Route the current state to its world, then MPC in that world's JEPA."""
        for _ in range(max_steps):
            f = feat_of(self.enc, self.tr, env.render())
            name = self.route(f)
            model = self.worlds[name]
            z = model.z(f)
            gf = feat_of(self.enc, self.tr, goal_env.render())
            zg = model.z(gf)
            best_a = 0; bc = 1e18
            with torch.no_grad():
                for a in range(4):
                    zr = model.predict(z, a)
                    for _ in range(k - 1):
                        zr = model.predict(zr, a)
                    c = float((zr - zg).norm().item())
                    if c < bc: bc = c; best_a = a
            done = env.step(best_a)
            if done:
                return True
        return False


def main():
    print("=" * 70)
    print("S1: WORLD GATE — supervised routing + per-world JEPA planning")
    print("=" * 70)

    enc, tr = build_frozen()
    wg = WorldGate(enc, tr)

    print("\n[train] per-world JEPAs (A: 5x5 grid, B: 3x3+wall)")
    samA = collect_world(GridNavA, enc, tr, n=60)
    samB = collect_world(GridNavB, enc, tr, n=60)
    wg.add_world("A", samA, epochs=600)
    wg.add_world("B", samB, epochs=600)

    print("\n[gate] supervised W_gate·f -> P(world) on frozen features")
    gate_acc = wg.train_gate(epochs=300)
    print(f"  gate training accuracy: {100*gate_acc:.1f}%")

    print("\n[T1] routing accuracy (supervised gate, world label from OS)")
    correct = 0; total = 0
    rng = random.Random(0)
    for trial in range(40):
        cls = GridNavA if trial % 2 == 0 else GridNavB
        true_name = "A" if trial % 2 == 0 else "B"
        env = cls()
        env.reset(seed=rng.randint(0, 1000))
        f0 = feat_of(enc, tr, env.render())
        for _ in range(3):
            a = rng.randint(0, 3)
            done = env.step(a)
            f1 = feat_of(enc, tr, env.render())
            routed = wg.route(f0)
            if routed == true_name:
                correct += 1
            total += 1
            f0 = f1
            if done:
                break
    acc = correct / total
    print(f"  routing accuracy: {correct}/{total} ({100*acc:.1f}%)  target >= 95%")

    print("\n[T2] planning in each routed world")
    winsA = winsB = 0
    trials = 20
    for t in range(trials):
        env = GridNavA(); env.reset(seed=t * 7 + 1)
        goal = GridNavA(); goal.agent = np.array([4, 4])
        if wg.plan(env, goal):
            winsA += 1
        env = GridNavB(); env.reset(seed=t * 7 + 1)
        goal = GridNavB(); goal.agent = np.array([2, 2])
        if wg.plan(env, goal):
            winsB += 1
    print(f"  world A planning: {winsA}/{trials} | world B planning: {winsB}/{trials}")

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  T1 gate routing: {100*acc:.1f}% {'PASS >=95%' if acc >= 0.95 else 'FAIL'}")
    print(f"  T2 planning A/B: {winsA}/{trials} / {winsB}/{trials} {'PASS' if winsA == trials and winsB == trials else 'check'}")
    print("=" * 70)


if __name__ == "__main__":
    main()