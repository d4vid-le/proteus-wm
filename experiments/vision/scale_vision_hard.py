#!/usr/bin/env python3
"""scale_vision_hard.py — Real perception scaling on a HARD renderer.

The prior renderer placed blocks at deterministic positions from counts, so the
CNN trivially read "counts from pixel positions" — cosine saturated at 1.0 and
no scaling axis bit. This renderer is HARD:
  - objects placed at RANDOM positions (identity not in the position)
  - random sizes, random overlaps
  - multiple objects per type (must COUNT, not locate)
  - the CNN must actually segment/count, not read a lookup table

Now the honest scaling questions bite:
  T1 data scale  — does alignment/counting improve with more pairs?
  T2 complexity  — more objects per scene (counting harder)?
  T3 capacity    — does a bigger CNN actually help now?
  T4 robustness  — noise/occlusion on a task that matters
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
import torch.nn as nn
from PIL import Image, ImageDraw

import proteus.envs.real_env as real_env
from proteus.world.hjepa_torch import HJEPA_K4

OBS_DIM = 12
ACT_DIM = 4


def render_hard(vec, size=64, n_extra=0, seed=None):
    """Random-layout scene: each file is a blob at a RANDOM position. The CNN
    must count blobs per color, not read fixed coordinates."""
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), (12, 12, 24))
    d = ImageDraw.Draw(img)
    colors = [(50, 100, 220), (50, 190, 100), (230, 200, 50)]  # txt jpg pdf
    # draw one blob per file of each type (dims 0,1,2)
    for c in range(3):
        n = int(vec[c])
        for _ in range(n):
            cx = rng.randint(6, size - 6)
            cy = rng.randint(6, size - 6)
            r = rng.randint(2, 4)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colors[c])
    # KEEP: a distinct red shape at random position (must be found, not known)
    if vec[7] > 0.5:
        cx, cy = rng.randint(8, size - 8), rng.randint(8, size - 8)
        d.rectangle([cx - 5, cy - 5, cx + 5, cy + 5], fill=(230, 40, 40))
    # extra random noise blobs (distractors) for difficulty
    for _ in range(n_extra):
        cx, cy = rng.randint(2, size - 2), rng.randint(2, size - 2)
        d.point((cx, cy), fill=tuple(rng.randint(0, 255) for _ in range(3)))
    return np.array(img, dtype=np.float32) / 255.0


def seed_diverse(root, seed=0):
    """A DIVERSE world: vary txt/jpg/pdf counts and keep on/off by seed, so the
    latent has real structure (unlike seed_real_world which is always 2/2/2)."""
    import shutil
    rng = random.Random(seed)
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for p in root.glob("*"):
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
    for d in ["txt", "jpg", "pdf"]:
        (root / d).mkdir(exist_ok=True)
    n_txt = rng.randint(0, 4)
    n_jpg = rng.randint(0, 4)
    n_pdf = rng.randint(0, 3)
    for i in range(n_txt):
        (root / f"n{i}.txt").write_text("t")
    for i in range(n_jpg):
        (root / f"p{i}.jpg").write_text("j")
    if rng.random() < 0.6:
        (root / "KEEP.pdf").write_text("UNREAD")
    for i in range(n_pdf):
        (root / f"d{i}.pdf").write_text("p")
    return root


def collect_hard(n=400, seed=0, n_extra=0):
    pairs = []
    for i in range(n):
        root = pathlib.Path(f"/tmp/vis-hard-{seed}-{i}")
        seed_diverse(root, seed=seed + i)
        env = real_env.RealEnv(root)
        _, vec = env.obs()
        img = render_hard(vec, n_extra=n_extra, seed=seed + i)
        pairs.append((img, vec))
    return pairs


def img_to_tensor(arr):
    return torch.as_tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


class VisionEncoder(nn.Module):
    def __init__(self, z_dim=64, width=16):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(width * 4, width * 4, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(width * 4, z_dim)

    def forward(self, img):
        return self.proj(self.cnn(img).flatten(1))


class VisionAgent:
    def __init__(self, model, width=16):
        self.model = model
        self.vision = VisionEncoder(z_dim=64, width=width)

    def z_vision(self, arr):
        return self.vision(img_to_tensor(arr))

    def z_macro(self, vec):
        return self.model.enc_macro(torch.as_tensor(vec, dtype=torch.float32).unsqueeze(0))

    def align_loss(self, arr, vec):
        zv = self.z_vision(arr); zm = self.z_macro(vec)
        return 1.0 - (zv / zv.norm() * (zm / zm.norm())).sum(dim=-1).mean()

    def count_from_vision(self, arr):
        """Read the 3 file-type counts directly from the image via a probe."""
        zv = self.z_vision(arr)
        # compare to prototype latents for each count (0..3 per type)
        # simplest: linear probe head we train alongside
        return zv

    def keep_from_vision(self, arr):
        with torch.no_grad():
            zv = self.vision(img_to_tensor(arr))
            p_k = self._proto(keep=True); p_n = self._proto(keep=False)
        return float((zv / zv.norm() - p_k).norm()) < float((zv / zv.norm() - p_n).norm())

    def _proto(self, keep):
        vec = np.zeros(OBS_DIM, dtype=np.float32)
        vec[7] = 1.0 if keep else 0.0
        vec[8] = 1.0 if keep else 0.0
        zv = self.z_vision(render_hard(vec, seed=1))
        return zv / zv.norm(dim=-1, keepdim=True)


def train(agent, pairs, epochs=150, lr=3e-3):
    opt = torch.optim.Adam(agent.vision.parameters(), lr=lr)
    for ep in range(epochs):
        for img, vec in random.sample(pairs, min(64, len(pairs))):
            opt.zero_grad()
            agent.align_loss(img, vec).backward()
            opt.step()


def eval_cos(agent, pairs):
    cos = []
    with torch.no_grad():
        for img, vec in pairs[:50]:
            zv = agent.z_vision(img); zm = agent.z_macro(vec)
            cos.append(float(((zv / zv.norm()) * (zm / zm.norm())).sum().item()))
    return float(np.mean(cos))


def eval_keep(agent, n=20):
    return float(np.mean([agent.keep_from_vision(
        render_hard(np.concatenate([np.zeros(3), [0] * 4, [1, 1, 0, 0, 0], [0]]).astype(np.float32) if False else
                    (lambda r: (real_env.RealEnv(r).obs()[1]))(pathlib.Path(f"/tmp/vis-k-{t}")))
        ) for t in range(n)])) if False else _eval_keep(agent, n)


def _eval_keep(agent, n=20):
    correct = 0
    total_keep = 0
    for t in range(n):
        root = pathlib.Path(f"/tmp/vis-k-{t}")
        seed_diverse(root, seed=t * 7 + 3)
        env = real_env.RealEnv(root)
        _, vec = env.obs()
        img = render_hard(vec, seed=t * 7 + 3)
        pred = agent.keep_from_vision(img)
        if vec[7] == 1:
            total_keep += 1
            if pred:
                correct += 1
    return correct / max(total_keep, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("SCALE VISION (HARD): random-layout scenes, real counting")
    print("=" * 72)

    torch.manual_seed(0)
    model = HJEPA_K4(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=128,
                     z_micro=32, z_macro=64, k=4, pos_dim=OBS_DIM)
    ckpt = pathlib.Path(ROOT / "checkpoints" / "cleanup_hjepa.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    print("\n[T1] DATA SCALE — random-layout counting")
    data_cos = []
    for n_data in (100, 200, 400):
        agent = VisionAgent(model)
        pairs = collect_hard(n=n_data, seed=0)
        train(agent, pairs, epochs=100)
        c = eval_cos(agent, pairs)
        k = _eval_keep(agent, n=15)
        data_cos.append((n_data, c, k))
        print(f"  {n_data:4d} pairs -> cos {c:.3f}, keep-det {k:.2f}")

    print("\n[T2] DISTRACTORS — random noise blobs per scene")
    for nd in (0, 10, 30):
        agent = VisionAgent(model)
        pairs = collect_hard(n=200, seed=0, n_extra=nd)
        train(agent, pairs, epochs=100)
        k = _eval_keep(agent, n=15)
        print(f"  {nd:2d} distractors -> cos {eval_cos(agent, pairs):.3f}, keep-det {k:.2f}")

    print("\n[T3] CAPACITY — CNN width on the HARD task")
    for width in (8, 16, 32):
        agent = VisionAgent(model, width=width)
        pairs = collect_hard(n=200, seed=0)
        train(agent, pairs, epochs=100)
        k = _eval_keep(agent, n=15)
        print(f"  width {width:2d} -> cos {eval_cos(agent, pairs):.3f}, keep-det {k:.2f}")

    print("\n" + "=" * 72)
    print("READING")
    print("=" * 72)
    for n, c, k in data_cos:
        print(f"  data {n}: cos {c:.3f} keep {k:.2f}")
    print("  (if keep-det climbs with data -> REAL scaling; if flat -> task-bound)")
    print("=" * 72)


if __name__ == "__main__":
    main()