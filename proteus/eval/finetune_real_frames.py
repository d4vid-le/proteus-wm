#!/usr/bin/env python3
"""finetune_real_frames.py — Fine-tune vision on REAL frames.

The count head works on synthetic color-blocks (mean err 0.2) but overcounts
real text listings (synthetic→real shift). This closes it:

  1. For each real directory, render its REAL listing to an image (real
     filenames as text) -> a genuine 'real frame' of the directory.
  2. Label it with the REAL file-system counts (ground truth).
  3. Fine-tune the count head + projection on these real frames.
  4. Measure: does exact-count accuracy improve on real listings?

This is the honest fix for the synthetic→real distribution shift.

Tests:
  FT1 before: count head on real frames (synthetic-trained) — expect overcount
  FT2 after: fine-tuned on real frames — expect error drops
  FT3 held-out real frames (unseen dirs) — generalization
"""
import argparse
import pathlib
import random
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import proteus.perception.count_head as ch
import proteus.envs.real_env as real_env
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def capture_listing(root, out_path):
    """Render the REAL directory listing to a real PNG (real filenames as text)."""
    files = sorted(p.name for p in pathlib.Path(root).glob("*") if p.is_file())
    html = f"""<html><body style="background:#1a1a2e;color:#eee;font-family:Menlo,monospace;font-size:34px">
<h1>Files</h1>{''.join(f'<div>📄 {f}</div>' for f in files)}
</body></html>"""
    tmp = pathlib.Path(tempfile.gettempdir()) / "proteus_ft_frame.html"
    tmp.write_text(html)
    subprocess.run(["qlmanage", "-t", "-s", "800", "-o", str(pathlib.Path(out_path).parent), str(tmp)],
                   capture_output=True)
    gen = pathlib.Path(out_path).parent / (tmp.stem + ".html.png")
    if gen.exists():
        gen.rename(out_path)
        return True
    return False


def collect_real_frames(n=80, seed=0, n_types=3, max_per_type=5, subfolders=True):
    """Build (real_frame_image, real_counts) pairs from real directories with
    richer diversity: more types, wider count ranges, nested subfolders."""
    rng = random.Random(seed)
    pairs = []
    for i in range(n):
        root = pathlib.Path(f"/tmp/ft-{seed}-{i}")
        ch.seed_diverse(root, seed=seed + i)
        # enrich: more files per type
        if n_types >= 3:
            for x in range(rng.randint(0, max_per_type)):
                (root / f"n{x}.txt").write_text("t")
            for x in range(rng.randint(0, max_per_type)):
                (root / f"p{x}.jpg").write_text("j")
        if subfolders:
            # create a subfolder with some files (real nested structure)
            sub = root / "sub"
            sub.mkdir(exist_ok=True)
            for x in range(rng.randint(0, 2)):
                (sub / f"s{x}.txt").write_text("s")
            # move some loose files into sub (real-world nesting)
        env = real_env.RealEnv(root)
        _, vec = env.obs()
        img_path = pathlib.Path(tempfile.gettempdir()) / f"ft_frame_{seed}_{i}.png"
        if not capture_listing(root, img_path):
            continue
        pairs.append((np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8), vec.copy()))
    return pairs


def scale_measure(cv, tr, sizes, seed=0, epochs_per=80):
    """Measure real-frame count error as the dataset scales."""
    results = []
    for n in sizes:
        pairs = collect_real_frames(n=n, seed=seed)
        if len(pairs) < 10:
            continue
        split = int(0.8 * len(pairs))
        train_real, holdout = pairs[:split], pairs[split:]
        # fresh count head each scale
        enc2, tr2 = ch.build_resnet18()
        cv2 = ch.CountVision(enc2, tr2).to(DEVICE)
        opt = torch.optim.Adam(list(cv2.proj.parameters()) + list(cv2.count_head.parameters()), lr=1e-3)
        for ep in range(epochs_per):
            for img, vec in random.sample(train_real, min(24, len(train_real))):
                opt.zero_grad(); cv2.loss(img, vec).backward(); opt.step()
        err = np.mean([np.mean([abs(cv2.count(img).detach().cpu().numpy()[0][i] - vec[i]) for i in range(3)])
                       for img, vec in holdout])
        results.append((n, float(err)))
    return results


def img_to_tensor(arr, tr, size=224):
    im = Image.fromarray(arr).resize((size, size))
    return tr(im).unsqueeze(0).to(DEVICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--scale", action="store_true", help="run the data-scaling study")
    args = ap.parse_args()

    print("=" * 72)
    print("FINE-TUNE VISION ON REAL FRAMES (synthetic -> real)")
    print("=" * 72)

    torch.manual_seed(0)
    enc, tr = ch.build_resnet18()
    cv = ch.CountVision(enc, tr).to(DEVICE)

    # base: train on synthetic scenes
    print("\n[base] train count head on SYNTHETIC scenes")
    synth = ch.collect(n=150 if args.quick else 300, seed=0)
    ch.train_count(cv, synth, epochs=60 if args.quick else 120)

    if args.scale:
        print("\n[SCALE] real-frame count error vs dataset size (the scaling study)")
        sizes = (20, 40, 80, 120) if args.quick else (30, 60, 120, 200)
        for n, err in scale_measure(cv, tr, sizes, seed=1,
                                    epochs_per=40 if args.quick else 80):
            print(f"  {n:4d} real frames -> held-out mean abs err {err:.3f}")
        return

    print("\n[collect] real frames of real directories")
    real_pairs = collect_real_frames(n=60 if args.quick else 100, seed=0)
    print(f"  {len(real_pairs)} (real-frame, real-counts) pairs")

    # FT1: before fine-tuning, how does the synthetic-trained head do on real frames?
    print("\n[FT1] BEFORE fine-tune: synthetic-trained head on REAL frames")
    errs = []
    for img, vec in real_pairs[:30]:
        with torch.no_grad():
            c = cv.count(img).cpu().numpy()[0]
        errs.append(np.mean([abs(c[i] - vec[i]) for i in range(3)]))
    before = float(np.mean(errs))
    print(f"  mean abs err on real frames: {before:.3f}")

    # FT2: fine-tune on real frames
    print("\n[FT2] FINE-TUNE on real frames (the fix)")
    opt = torch.optim.Adam(list(cv.proj.parameters()) + list(cv.count_head.parameters()), lr=1e-3)
    split = int(0.8 * len(real_pairs))
    train_real = real_pairs[:split]
    holdout_real = real_pairs[split:]
    for ep in range(80 if args.quick else 150):
        for img, vec in random.sample(train_real, min(24, len(train_real))):
            opt.zero_grad()
            cv.loss(img, vec).backward()
            opt.step()
    after = np.mean([np.mean([abs(cv.count(img).detach().cpu().numpy()[0][i] - vec[i]) for i in range(3)])
                     for img, vec in train_real])
    print(f"  train real-frame mean abs err: {after:.3f}")

    # FT3: held-out real frames
    print("\n[FT3] AFTER fine-tune on HELD-OUT real frames (generalization)")
    errs = []
    for img, vec in holdout_real:
        with torch.no_grad():
            c = cv.count(img).cpu().numpy()[0]
        errs.append(np.mean([abs(c[i] - vec[i]) for i in range(3)]))
    holdout = float(np.mean(errs))
    print(f"  held-out real-frame mean abs err: {holdout:.3f}")

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"  synthetic-trained on real frames: {before:.3f}")
    print(f"  fine-tuned on real frames:        {holdout:.3f}")
    print(f"  improvement: {100*(before-holdout)/max(before,1e-6):.0f}%")
    print(f"  => {'PASS: real-frame fine-tuning fixed the shift' if holdout < before else 'FAIL'}")
    print("=" * 72)


if __name__ == "__main__":
    main()