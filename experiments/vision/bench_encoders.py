#!/usr/bin/env python3
"""bench_encoders.py — Benchmark frozen vision encoders for the JEPA way.

Compare latency (MPS) + alignment quality across encoders to pick the best
speed/accuracy tradeoff for the frozen-ViT -> latent path.

Encoders: ViT-B/16, ViT-B/32, ResNet18, ResNet34, MobileNetV3-Small/Large,
EfficientNet-B0, ConvNeXt-Tiny, RegNet-Y-400mf.

For each:
  - latency: image -> z_vis on MPS (ms)
  - alignment: train small proj, measure cosine to z_macro
  - KEEP detection from the latent
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
from PIL import Image

import proteus.envs.real_env as real_env
from proteus.world.hjepa_torch import HJEPA_K4

OBS_DIM = 12
ACT_DIM = 4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

ENC_DEFS = {
    "vit_b_16": {"name": "vit_b_16", "weights": "ViT_B_16_Weights.IMAGENET1K_V1", "feat": 768, "img": 224, "head": "heads"},
    "vit_b_32": {"name": "vit_b_32", "weights": "ViT_B_32_Weights.IMAGENET1K_V1", "feat": 768, "img": 224, "head": "heads"},
    "resnet18": {"name": "resnet18", "weights": "ResNet18_Weights.IMAGENET1K_V1", "feat": 512, "img": 224, "head": "fc"},
    "resnet34": {"name": "resnet34", "weights": "ResNet34_Weights.IMAGENET1K_V1", "feat": 512, "img": 224, "head": "fc"},
    "mobilenet_v3_small": {"name": "mobilenet_v3_small", "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1", "feat": 576, "img": 224, "head": "classifier"},
    "mobilenet_v3_large": {"name": "mobilenet_v3_large", "weights": "MobileNet_V3_Large_Weights.IMAGENET1K_V1", "feat": 960, "img": 224, "head": "classifier"},
    "efficientnet_b0": {"name": "efficientnet_b0", "weights": "EfficientNet_B0_Weights.IMAGENET1K_V1", "feat": 1280, "img": 224, "head": "classifier"},
    "convnext_tiny": {"name": "convnext_tiny", "weights": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1", "feat": 768, "img": 224, "head": "classifier"},
    "regnet_y_400mf": {"name": "regnet_y_400mf", "weights": "RegNet_Y_400MF_Weights.IMAGENET1K_V1", "feat": 440, "img": 224, "head": "fc"},
}


def build_enc(name, device):
    import torchvision.models as M
    d = ENC_DEFS[name]
    getter = getattr(M, d["name"])
    enum_name, wname = d["weights"].split(".")
    enum = getattr(M, enum_name)
    we = getattr(enum, wname)
    model = getter(weights=we)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    # strip the classifier to expose the feature vector
    if d["head"] == "heads":
        model.heads = nn.Identity()      # ViT -> (1,768) CLS feature
    elif d["head"] == "fc":
        model.fc = nn.Identity()
    elif d["head"] == "classifier":
        model.classifier = nn.Identity()
    model.to(device)
    tr = we.transforms()
    return model, tr, d["feat"]


def render_scene(vec, size=224):
    from PIL import ImageDraw
    img = Image.new("RGB", (size, size), (12, 12, 24))
    d = ImageDraw.Draw(img)
    colors = [(50, 100, 220), (50, 190, 100), (230, 200, 50)]
    cell = size // 5
    for c in range(3):
        n = int(vec[c])
        for i in range(min(n, 4)):
            x = cell // 2 + c * cell + (i % 4) * (cell // 5)
            y = 12 + (i // 4) * (cell // 2)
            d.rectangle([x, y, x + cell // 5, y + cell // 5], fill=colors[c])
    if vec[7] > 0.5:
        d.rectangle([size - cell, size - cell, size - 6, size - 6], fill=(230, 40, 40))
    return np.array(img, dtype=np.uint8)


def seed_diverse(root, seed=0):
    import shutil
    rng = random.Random(seed)
    root = pathlib.Path(root); root.mkdir(parents=True, exist_ok=True)
    for p in root.glob("*"):
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
    for d in ["txt", "jpg", "pdf"]:
        (root / d).mkdir(exist_ok=True)
    for x in range(rng.randint(0, 4)):
        (root / f"n{x}.txt").write_text("t")
    for x in range(rng.randint(0, 4)):
        (root / f"p{x}.jpg").write_text("j")
    if rng.random() < 0.6:
        (root / "KEEP.pdf").write_text("UNREAD")
    for x in range(rng.randint(0, 3)):
        (root / f"d{x}.pdf").write_text("p")


def collect_pairs(n=150, seed=0):
    pairs = []
    for i in range(n):
        root = pathlib.Path(f"/tmp/be-{seed}-{i}")
        seed_diverse(root, seed=seed + i)
        env = real_env.RealEnv(root)
        _, vec = env.obs()
        pairs.append((render_scene(vec), vec))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encs", default="vit_b_16,mobilenet_v3_small,resnet18,efficientnet_b0,convnext_tiny")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("BENCH FROZEN ENCODERS (JEPA way): latency vs alignment")
    print(f"  device: {DEVICE}")
    print("=" * 74)

    torch.manual_seed(0)
    model = HJEPA_K4(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=128,
                     z_micro=32, z_macro=64, k=4, pos_dim=OBS_DIM)
    ckpt = pathlib.Path(ROOT / "checkpoints" / "cleanup_hjepa.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval(); model.to(DEVICE)

    train_pairs = collect_pairs(n=120 if args.quick else 200, seed=0)
    test_pairs = collect_pairs(n=60, seed=99)

    results = []
    for name in args.encs.split(","):
        name = name.strip()
        if name not in ENC_DEFS:
            print(f"  skip unknown {name}")
            continue
        try:
            enc, tr, feat = build_enc(name, DEVICE)
        except Exception as e:
            print(f"  {name}: build FAILED {str(e)[:60]}")
            continue
        proj = nn.Linear(feat, 64).to(DEVICE)

        # latency
        x = tr(Image.fromarray(train_pairs[0][0])).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            f0 = enc(x)
        f0 = f0.flatten(1) if f0.dim() > 2 else f0
        print(f"    {name}: feature shape {tuple(f0.shape)} (expected ({feat},))")
        n_rep = 20
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_rep):
                enc(x)
        ms = (time.perf_counter() - t0) / n_rep * 1000

        # train projection
        opt = torch.optim.Adam(proj.parameters(), lr=3e-3)
        for ep in range(40 if args.quick else 80):
            for img, vec in random.sample(train_pairs, min(32, len(train_pairs))):
                xi = tr(Image.fromarray(img)).unsqueeze(0).to(DEVICE)
                zt = model.enc_macro(torch.as_tensor(vec, dtype=torch.float32, device=DEVICE).unsqueeze(0))
                with torch.no_grad():
                    f = enc(xi)
                z = proj(f); z = z / z.norm(dim=-1, keepdim=True)
                zt = zt / zt.norm(dim=-1, keepdim=True)
                opt.zero_grad()
                (1 - (z * zt).sum(-1).mean()).backward()
                opt.step()

        # alignment
        cos = []
        with torch.no_grad():
            for img, vec in test_pairs[:30]:
                xi = tr(Image.fromarray(img)).unsqueeze(0).to(DEVICE)
                z = proj(enc(xi)); z = z / z.norm(dim=-1, keepdim=True)
                zt = model.enc_macro(torch.as_tensor(vec, dtype=torch.float32, device=DEVICE).unsqueeze(0))
                zt = zt / zt.norm(dim=-1, keepdim=True)
                cos.append(float((z * zt).sum().item()))
        mc = float(np.mean(cos))
        results.append((name, ms, mc, feat))
        print(f"  {name:<20} {ms:6.2f}ms  cos {mc:.3f}  feat {feat}")

    print("\n" + "=" * 74)
    print("RANKING (speed-accuracy tradeoff)")
    print("=" * 74)
    for name, ms, mc, feat in sorted(results, key=lambda r: r[2] / max(r[1], 0.001), reverse=True):
        print(f"  {name:<20} cos/ms {mc/max(ms,0.001):.3f}  ({ms:.1f}ms, cos {mc:.3f})")
    print("=" * 74)


if __name__ == "__main__":
    main()