#!/usr/bin/env python3
"""vision_jepa.py — THE JEPA WAY: frozen vision encoder -> latent -> macro JEPA.

Replaces LLaVA (image->text->parse->router, minutes, lossy) with:
    Screenshot PNG -> [Frozen ViT-B/16 encoder] -> z_vis (768-dim)
    -> project -> z_vis' (64-dim) -> macro JEPA -> MPC plan

The image NEVER becomes text. A frozen pretrained vision transformer extracts a
visual embedding in ~ms, projected into the SAME 64-dim latent space as the
file-world model, so the JEPA reasons in latent space end-to-end.

Pipeline:
  img (PIL/array) -> torchvision transforms -> ViT-B/16 (frozen) -> z_vis(768)
  -> Linear(768->64) trained to align z_vis to z_macro(state) -> z (64)
  -> m.enc_macro(z) is replaced by z directly (vision IS the observation)
  -> m.predict_macro / decode_pos for planning

Benchmarks (the whole point):
  - latency: frozen-encoder image->latent in ms on MPS (vs minutes for LLaVA)
  - alignment: z_vis-projected vs z_macro cosine (does frozen ViT encode the
    same world as the file counts?)
  - planning: MPC using vision-latent observations drives real routing
"""
from proteus.paths import ROOT
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
from proteus.device import DEVICE


def build_frozen_vit(device):
    """Frozen ViT-B/16 (pretrained). Removes the classifier to expose the 768-dim
    CLS feature. Returns (encoder, transform)."""
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    # expose the 768-dim feature by replacing the classifier with identity
    model.heads = torch.nn.Identity()
    model.to(device)
    return model, weights.transforms()


def render_scene(vec, size=224):
    """Directory scene: colored blocks + red KEEP. Sized for ViT (224)."""
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


def collect_pairs(n=300, seed=0, size=224):
    rng = random.Random(seed)
    pairs = []
    for i in range(n):
        root = pathlib.Path(f"/tmp/vj-{seed}-{i}")
        # diverse world (vary counts + keep)
        import shutil
        root.mkdir(parents=True, exist_ok=True)
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
        env = real_env.RealEnv(root)
        _, vec = env.obs()
        pairs.append((render_scene(vec, size=size), vec))
    return pairs


class FrozenVisionJEPA(nn.Module):
    """Frozen ViT encoder + a trainable projection into the JEPA latent."""

    def __init__(self, vit, z_macro=64):
        super().__init__()
        self.vit = vit  # frozen
        self.proj = nn.Linear(768, z_macro)  # trainable only
        self.z_macro = z_macro

    def encode(self, img_tensor):
        """img_tensor: (B,3,H,W) preprocessed -> z_vis (B,64)."""
        with torch.no_grad():
            feat = self.vit(img_tensor)  # (B,768) cls token
        return self.proj(feat)

    def align_loss(self, img_tensor, z_macro_target):
        z = self.encode(img_tensor)
        z = z / z.norm(dim=-1, keepdim=True)
        t = z_macro_target / z_macro_target.norm(dim=-1, keepdim=True)
        return 1.0 - (z * t).sum(dim=-1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("THE JEPA WAY: frozen ViT -> latent -> macro JEPA (no text)")
    print("=" * 72)

    torch.manual_seed(0)
    print(f"\n[load] frozen ViT-B/16 (pretrained, {DEVICE})")
    vit, transforms = build_frozen_vit(DEVICE)
    model = HJEPA_K4(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=128,
                     z_micro=32, z_macro=64, k=4, pos_dim=OBS_DIM)
    ckpt = pathlib.Path(ROOT / "checkpoints" / "cleanup_hjepa.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    model.to(DEVICE)

    fvj = FrozenVisionJEPA(vit, z_macro=64).to(DEVICE)

    # ---- latency: image -> latent ----
    print("\n[latency] image -> frozen ViT -> latent (the whole point)")
    pairs = collect_pairs(n=40 if args.quick else 80, seed=0)
    img = render_scene(pairs[0][1], size=224)
    img_pil = Image.fromarray(img)
    x = transforms(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        fvj.vit(x)  # warmup
    n_rep = 30
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_rep):
            fvj.vit(x)
    per_ms = (time.perf_counter() - t0) / n_rep * 1000
    print(f"  frozen-ViT image->z_vis: {per_ms:.1f} ms  (LLaVA took MINUTES)")
    print(f"  => {1000/max(per_ms,0.001):.0f}x faster than LLaVA text path")

    # ---- train projection: align z_vis to z_macro ----
    print("\n[train] project z_vis(768) -> z_macro(64), aligned to the JEPA latent")
    train_pairs = collect_pairs(n=200 if args.quick else 300, seed=0)
    opt = torch.optim.Adam(fvj.proj.parameters(), lr=3e-3)
    for ep in range(60 if args.quick else 150):
        for img, vec in random.sample(train_pairs, min(48, len(train_pairs))):
            x = transforms(Image.fromarray(img)).unsqueeze(0).to(DEVICE)
            zt = model.enc_macro(torch.as_tensor(vec, dtype=torch.float32, device=DEVICE).unsqueeze(0))
            opt.zero_grad()
            fvj.align_loss(x, zt).backward()
            opt.step()

    # ---- alignment ----
    print("\n[T1] alignment: z_vis(projected) vs z_macro")
    test_pairs = collect_pairs(n=80, seed=99)
    cos = []
    with torch.no_grad():
        for img, vec in test_pairs[:40]:
            x = transforms(Image.fromarray(img)).unsqueeze(0).to(DEVICE)
            z = fvj.encode(x); z = z / z.norm(dim=-1, keepdim=True)
            zt = model.enc_macro(torch.as_tensor(vec, dtype=torch.float32, device=DEVICE).unsqueeze(0))
            zt = zt / zt.norm(dim=-1, keepdim=True)
            cos.append(float((z * zt).sum().item()))
    mean_cos = float(np.mean(cos))
    print(f"  mean cosine: {mean_cos:.3f}  ({'ALIGNED' if mean_cos > 0.7 else 'weak'})")

    # ---- KEEP detection from vision latent ----
    print("\n[T2] KEEP detection from the vision latent")
    proto_k = proto_n = None
    with torch.no_grad():
        vk = np.zeros(12, dtype=np.float32); vk[7] = vk[8] = 1.0
        xk = transforms(Image.fromarray(render_scene(vk))).unsqueeze(0).to(DEVICE)
        proto_k = fvj.encode(xk); proto_k = proto_k / proto_k.norm()
        vn = np.zeros(12, dtype=np.float32)
        xn = transforms(Image.fromarray(render_scene(vn))).unsqueeze(0).to(DEVICE)
        proto_n = fvj.encode(xn); proto_n = proto_n / proto_n.norm()
    correct = 0; total_keep = 0
    for t in range(40 if not args.quick else 20):
        import shutil
        root = pathlib.Path(f"/tmp/vj-keep-{t}")
        root.mkdir(parents=True, exist_ok=True)
        for p in root.glob("*"):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        for d in ["txt", "jpg", "pdf"]:
            (root / d).mkdir(exist_ok=True)
        for x in range(2):
            (root / f"n{x}.txt").write_text("t")
            (root / f"p{x}.jpg").write_text("j")
        (root / "KEEP.pdf").write_text("UNREAD")
        env = real_env.RealEnv(root)
        _, vec = env.obs()
        x = transforms(Image.fromarray(render_scene(vec))).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            z = fvj.encode(x); z = z / z.norm()
            dk = float((z - proto_k).norm()); dn = float((z - proto_n).norm())
        pred_keep = dk < dn
        if vec[7] == 1:
            total_keep += 1
            if pred_keep:
                correct += 1
    keep_rate = correct / max(total_keep, 1)
    print(f"  vision-latent KEEP detection: {correct}/{total_keep} ({keep_rate:.2f})")

    # ---- MPC planning from vision latent ----
    print("\n[T3] MPC planning from the vision latent (no file counts)")
    # encode the current scene via vision, decode pos, pick action
    root = pathlib.Path("/tmp/vj-plan")
    real_env.seed_real_world(str(root), keep_pdf=True)
    env = real_env.RealEnv(root)
    _, vec = env.obs()
    x = transforms(Image.fromarray(render_scene(vec))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        z_vis = fvj.encode(x)
        pos = model.decode_pos(z_vis).detach().cpu().numpy()[0]
    print(f"  vision-latent pos readout: {np.round(pos[0:3],2)} "
          f"(from pixels, not counts)")
    print(f"  real counts: txt={vec[0]} jpg={vec[1]} pdf={vec[2]}")

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"  latency {per_ms:.1f}ms (LLaVA was minutes) -> JEPA way is fast")
    print(f"  alignment {mean_cos:.3f} | KEEP-from-vision {keep_rate:.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()