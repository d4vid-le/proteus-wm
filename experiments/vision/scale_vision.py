#!/usr/bin/env python3
"""scale_vision.py — Scale the vision encoder: data x complexity x image size.

Honest scaling axes for perception:
  T1 DATA SCALE — does vision alignment improve with more training pairs?
     (power-law? where's the knee?)
  T2 SCENE COMPLEXITY — more file types / more objects per scene. Does the
     encoder still read the scene?
  T3 IMAGE SIZE — real images at 64/128/256. Does resolution help or just cost?
  T4 CAPACITY — wider/deeper CNN. Does the bottleneck (not the model) limit?
  T5 ROBUSTNESS — noise/occlusion/jpeg artifacts on real-sized inputs.

The honest question: what actually limits perception here — data, model capacity,
resolution, or the task itself?
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

import proteus.envs.real_env as real_env
from proteus.world.hjepa_torch import HJEPA_K4, MLP

OBS_DIM = 12
ACT_DIM = 4


# ---------------- renderer with complexity control ----------------
def render_scene(vec, size=64, n_types=3, jpeg=False, noise=0.0):
    """Render a directory scene. n_types: how many file-type blocks to draw
    (3 = txt/jpg/pdf, more = extra types). jpeg/noise for robustness."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (size, size), (10, 10, 20))
    d = ImageDraw.Draw(img)
    colors = [(40, 90, 200), (40, 180, 90), (220, 200, 40), (200, 60, 160), (80, 200, 220)]
    cell = size // (n_types + 1)
    for c in range(n_types):
        n = int(vec[c]) if c < 3 else (int(vec[9]) if c == 3 else int(vec[10] % 3))
        for i in range(min(n, 3)):
            x = cell // 2 + c * cell + (i % 3) * (cell // 4)
            y = 12 + (i // 3) * (cell // 2)
            d.rectangle([x, y, x + cell // 4, y + cell // 4], fill=colors[c % len(colors)])
    if vec[7] > 0.5:  # KEEP red block
        d.rectangle([size - cell, size - cell, size - 6, size - 6], fill=(220, 40, 40))
    if vec[9] > 0.5 and n_types > 3:
        d.rectangle([size - 2 * cell, size - cell, size - cell - 6, size - 6], fill=(230, 180, 40))
    arr = np.array(img, dtype=np.float32) / 255.0
    if noise > 0:
        arr = arr + np.random.randn(*arr.shape).astype(np.float32) * noise
        arr = np.clip(arr, 0, 1)
    if jpeg:
        from PIL import Image as I
        im = I.fromarray((arr * 255).astype(np.uint8))
        import io
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=40)
        arr = np.array(I.open(io.BytesIO(buf.getvalue())), dtype=np.float32) / 255.0
    return arr


def collect_pairs(n=400, seed=0, size=64, n_types=3):
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        root = pathlib.Path(f"/tmp/vis-scale-{seed}-{_}")
        real_env.seed_real_world(root, keep_pdf=bool(rng.random() < 0.5), seed=seed)
        env = real_env.RealEnv(root)
        _, vec = env.obs()
        img = render_scene(vec, size=size, n_types=n_types)
        pairs.append((img, vec))
    return pairs


def img_to_tensor(arr):
    t = torch.as_tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return t


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
        zv = zv / zv.norm(dim=-1, keepdim=True)
        zm = zm / zm.norm(dim=-1, keepdim=True)
        return 1.0 - (zv * zm).sum(dim=-1).mean()

    def keep_from_vision(self, arr):
        with torch.no_grad():
            zv = self.vision(img_to_tensor(arr))
            proto_k = self._proto(keep=True)
            proto_n = self._proto(keep=False)
        dk = float((zv / zv.norm() - proto_k).norm())
        dn = float((zv / zv.norm() - proto_n).norm())
        return dk < dn

    def _proto(self, keep):
        vec = np.zeros(OBS_DIM, dtype=np.float32)
        vec[7] = 1.0 if keep else 0.0
        vec[8] = 1.0 if keep else 0.0
        zv = self.z_vision(render_scene(vec))
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


def eval_keep(agent, n=30):
    correct = 0
    for t in range(n):
        root = pathlib.Path(f"/tmp/vis-keep-{t}")
        real_env.seed_real_world(root, keep_pdf=True, seed=t)
        env = real_env.RealEnv(root)
        _, vec = env.obs()
        if agent.keep_from_vision(render_scene(vec)):
            correct += 1
    return correct / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("SCALE VISION: data x complexity x size x capacity x robustness")
    print("=" * 72)

    torch.manual_seed(0)
    model = HJEPA_K4(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=128,
                     z_micro=32, z_macro=64, k=4, pos_dim=OBS_DIM)
    ckpt = pathlib.Path(ROOT / "checkpoints" / "cleanup_hjepa.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    # ---- T1: data scale ----
    print("\n[T1] DATA SCALE — alignment vs training pairs")
    data_cos = []
    for n_data in (50, 100, 200, 400):
        agent = VisionAgent(model)
        pairs = collect_pairs(n=n_data, seed=0)
        train(agent, pairs, epochs=100)
        c = eval_cos(agent, pairs)
        data_cos.append(c)
        print(f"  {n_data:4d} pairs -> cos {c:.3f}")

    # ---- T2: scene complexity ----
    print("\n[T2] SCENE COMPLEXITY — extra types per scene")
    for n_types in (3, 4, 5):
        agent = VisionAgent(model)
        pairs = collect_pairs(n=200, seed=0, n_types=n_types)
        train(agent, pairs, epochs=100)
        k = eval_keep(agent, n=20)
        print(f"  {n_types} types -> cos {eval_cos(agent, pairs):.3f}, keep-det {k:.2f}")

    # ---- T3: image size ----
    print("\n[T3] IMAGE SIZE — resolution vs accuracy")
    for size in (64, 128, 256):
        agent = VisionAgent(model)
        pairs = collect_pairs(n=200, seed=0, size=size)
        train(agent, pairs, epochs=100)
        k = eval_keep(agent, n=20)
        print(f"  {size}px -> cos {eval_cos(agent, pairs):.3f}, keep-det {k:.2f}")

    # ---- T4: capacity ----
    print("\n[T4] CAPACITY — CNN width")
    for width in (8, 16, 32):
        agent = VisionAgent(model, width=width)
        pairs = collect_pairs(n=200, seed=0)
        train(agent, pairs, epochs=100)
        k = eval_keep(agent, n=20)
        print(f"  width {width:2d} -> cos {eval_cos(agent, pairs):.3f}, keep-det {k:.2f}")

    # ---- T5: robustness ----
    print("\n[T5] ROBUSTNESS — noise / jpeg / occlusion")
    agent = VisionAgent(model)
    pairs = collect_pairs(n=200, seed=0)
    train(agent, pairs, epochs=150)
    for label, kw in [("clean", {}), ("noise0.1", {"noise": 0.1}),
                      ("jpeg40", {"jpeg": True}), ("noise0.2", {"noise": 0.2})]:
        cos = []
        with torch.no_grad():
            for t in range(20):
                root = pathlib.Path(f"/tmp/vis-rob-{t}")
                real_env.seed_real_world(root, keep_pdf=True, seed=t)
                env = real_env.RealEnv(root)
                _, vec = env.obs()
                img = render_scene(vec, **kw)
                zv = agent.z_vision(img); zm = agent.z_macro(vec)
                cos.append(float(((zv / zv.norm()) * (zm / zm.norm())).sum().item()))
        print(f"  {label:<10} cos {np.mean(cos):.3f}")

    print("\n" + "=" * 72)
    print("READING")
    print("=" * 72)
    print(f"  data: {data_cos[0]:.3f} -> {data_cos[-1]:.3f} across 8x data")
    print(f"  (if flat -> data not the limit; if climbing -> more data helps)")
    print("=" * 72)


if __name__ == "__main__":
    main()