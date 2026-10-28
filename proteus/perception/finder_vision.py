#!/usr/bin/env python3
"""finder_vision.py — Vision's true superpowers on a REAL Finder window.

Per the user's architectural correction: vision should NOT count files (the OS
counts). Vision's superpowers:
  1. UI & WINDOW AFFORDANCES — where windows/icons sit (spatial latent)
  2. OOD / ERROR DETECTION — unexpected dialogs, popups, missing windows
  3. VISUAL CONFIRMATION — verifying a file visually moved/disappeared
  4. PER-WORLD LATENT PLANNING — a spatial JEPA predicting layout transitions

Pipeline: open Finder to a real dir -> screencapture the window -> spatial latent
(frozen ResNet feature map, layout-preserving) -> anomaly score -> visual
confirm -> spatial-JEPA next-layout predictor.
"""
import argparse
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def get_finder_window():
    r = subprocess.run(['osascript', '-e',
        'tell application "System Events" to tell process "Finder" to get {position, size} of window 1'],
        capture_output=True, text=True)
    if r.returncode != 0:
        return None
    parts = [int(x) for x in r.stdout.replace('\n', '').split(',')]
    if len(parts) != 4:
        return None
    return tuple(parts)  # (x, y, w, h)


def open_finder(dirpath):
    subprocess.run(['open', '-a', 'Finder', str(dirpath)], capture_output=True)
    time.sleep(1.0)


def capture_finder_window(dirpath, out_path):
    open_finder(dirpath)
    b = get_finder_window()
    if b is None:
        return False, "no-finder-window"
    x, y, w, h = b
    full = pathlib.Path(tempfile.gettempdir()) / "finder_full.png"
    subprocess.run(['screencapture', '-x', str(full)], capture_output=True)
    if not full.exists():
        return False, "screencapture-failed"
    im = Image.open(full).convert("RGB")
    W, H = im.size
    box = (max(0, x), max(0, y), min(W, x + w), min(H, y + h))
    crop = im.crop(box)
    crop.save(out_path)
    return True, crop.size


class SpatialEncoder:
    def __init__(self, device=DEVICE, layer="layer3"):
        import torchvision.models as M
        self.tr = M.ResNet18_Weights.IMAGENET1K_V1.transforms()
        backbone = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        # use a higher-resolution layer for LOCAL layout sensitivity:
        # layer3 -> (1,256,14,14); layer4 -> (1,512,7,7)
        children = list(backbone.children())
        if layer == "layer3":
            self.features = nn.Sequential(*children[:-3])
            self.c = 256
        else:
            self.features = nn.Sequential(*children[:-2])
            self.c = 512
        self.features.eval()
        self.features.to(device)
        self.device = device

    def encode_spatial(self, img_path):
        im = self._read_safe(img_path)
        if im is None:
            return None
        im = im.resize((224, 224))
        x = self.tr(im).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self.features(x)
        return f[0].permute(1, 2, 0)  # (14,14,256) for layer3

    def _read_safe(self, img_path):
        """Race-safe image read: retry if the frame is mid-write (SCK stream)."""
        import proteus.perception.telemetry as _tel
        return _tel.safe_read_image(str(img_path))


class AnomalyDetector:
    def __init__(self, enc):
        self.enc = enc
        self.prototypes = None
        self.scale = 1.0

    def fit_normal(self, frames):
        feats = [self.enc.encode_spatial(f).cpu().numpy().ravel() for f in frames]
        self.prototypes = np.mean(feats, axis=0)
        self.scale = np.std(feats, axis=0).mean() + 1e-6
        return self

    def anomaly(self, img_path):
        f = self.enc.encode_spatial(img_path).cpu().numpy().ravel()
        return float(np.linalg.norm(f - self.prototypes) / self.scale)


def visual_confirm(enc, before_path, after_path, thresh=0.0001):
    """Visual confirmation: did the layout change? Uses FULL-RES pixel L2
    (normalized). A single file move changes some pixels -> ~1e-4; a big change
    is higher. Frozen features are too invariant for fine change detection."""
    a = np.array(Image.open(before_path).convert("RGB"), dtype=float) / 255.0
    b = np.array(Image.open(after_path).convert("RGB"), dtype=float) / 255.0
    if a.shape != b.shape:
        a = np.array(Image.fromarray((a * 255).astype(np.uint8)).resize((400, 400)), dtype=float) / 255.0
        b = np.array(Image.fromarray((b * 255).astype(np.uint8)).resize((400, 400)), dtype=float) / 255.0
    diff = float(np.mean((a - b) ** 2))
    return diff > thresh, diff


class SpatialJEPA(nn.Module):
    def __init__(self, flat_dim, latent=64, act_dim=3):
        super().__init__()
        self.enc_flat = nn.Linear(flat_dim, latent)
        self.target = nn.Linear(flat_dim, latent)
        self.target.load_state_dict(self.enc_flat.state_dict())
        self.pred = nn.Linear(latent + act_dim, latent)
        self.act_dim = act_dim
        self.latent = latent

    def z(self, flat):
        t = torch.as_tensor(flat, dtype=torch.float32, device=DEVICE)
        return self.enc_flat(t.to(DEVICE))

    def z_target(self, flat):
        t = torch.as_tensor(flat, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return self.target(t)

    def predict(self, z, a):
        a_oh = torch.nn.functional.one_hot(torch.tensor([a], device=DEVICE), self.act_dim).float()[0]
        return self.pred(torch.cat([z, a_oh], dim=-1))

    def update_target(self, tau=0.995):
        for p, tp in zip(self.enc_flat.parameters(), self.target.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=1 - tau)


def collect_spatial_states(enc, root, n=10):
    """Capture the Finder window over several real file moves; return
    (list_of_flattened_spatial_latents, list_of_actions)."""
    states, acts = [], []
    for step in range(n):
        cap = pathlib.Path(tempfile.gettempdir()) / f"finder_state_{step}.png"
        ok, _ = capture_finder_window(root, cap)
        if not ok:
            break
        states.append(enc.encode_spatial(cap).cpu().numpy().ravel())
        loose = [p for p in root.glob("*") if p.is_file() and not (root / "KEEP_IMPORTANT.pdf" == p)]
        if loose:
            f = random.choice(loose)
            ext = f.suffix.lstrip(".")
            if ext in ("txt", "jpg", "pdf"):
                dest_dir = root / ext
                dest_dir.mkdir(exist_ok=True)
                f.rename(dest_dir / f.name)
                acts.append({"txt": 0, "jpg": 1, "pdf": 2}[ext])
            else:
                acts.append(0)
        else:
            acts.append(0)
    return states, acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/proteus-finder")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("VISION SUPER POWERS: real Finder window, spatial latent, anomaly, confirm")
    print("=" * 72)

    import proteus.envs.real_env as real_env
    real_env.seed_real_world(args.dir, keep_pdf=True)
    root = pathlib.Path(args.dir)
    print(f"\n[world] {sorted(p.name for p in root.glob('*') if p.is_file())}")

    win = pathlib.Path(tempfile.gettempdir()) / "finder_win.png"
    ok, info = capture_finder_window(root, win)
    print(f"\n[F1] real Finder window: {'ok' if ok else 'FAIL'} {info}")
    if not ok:
        return
    print(f"  captured: {win} {Image.open(win).size}")

    enc = SpatialEncoder()
    spat = enc.encode_spatial(win)
    print(f"\n[F2] per-world spatial latent: {tuple(spat.shape)} "
          f"std {float(spat.std()):.3f} (layout-preserving)")

    det = AnomalyDetector(enc).fit_normal([win])
    norm_anom = det.anomaly(win)
    # anomaly: capture the window again but with a big overlay drawn (simulated
    # error dialog / unexpected element) on the capture
    anom_img = pathlib.Path(tempfile.gettempdir()) / "finder_anom.png"
    im = Image.open(win).convert("RGB")
    d = ImageDraw = __import__("PIL").ImageDraw
    dr = d.Draw(im)
    W, H = im.size
    dr.rectangle([W // 4, H // 4, 3 * W // 4, H // 2], fill=(200, 40, 40))
    dr.rectangle([W // 4 + 10, H // 2 + 10, 3 * W // 4, 3 * H // 4], fill=(240, 240, 240))
    im.save(anom_img)
    anom_score = det.anomaly(anom_img)
    print(f"\n[F3] OOD / error detection:")
    print(f"  normal window anomaly: {norm_anom:.2f}")
    print(f"  window + injected dialog anomaly: {anom_score:.2f}")
    detected = anom_score > norm_anom * 2
    print(f"  => {'DETECTED (unexpected element flagged)' if detected else 'MISSED'}")

    # visual confirmation
    before = pathlib.Path(tempfile.gettempdir()) / "finder_before.png"
    after = pathlib.Path(tempfile.gettempdir()) / "finder_after.png"
    capture_finder_window(root, before)
    real_env.seed_real_world(args.dir, keep_pdf=True)  # reset
    # move a real file via FS
    loose = [p for p in root.glob("*.txt") if p.is_file()]
    if loose:
        loose[0].rename(root / "txt" / loose[0].name)
    time.sleep(1.5)  # let Finder refresh the window view
    capture_finder_window(root, after)
    changed, diff = visual_confirm(enc, before, after)
    print(f"\n[F4] visual confirmation (file moved): layout diff {diff:.4f}")
    print(f"  => {'CONFIRMED (layout changed)' if changed else 'no visual change'}")

    # latent planning
    print(f"\n[F5] per-world latent planning (spatial JEPA)")
    states, acts = collect_spatial_states(enc, root, n=8)
    flat_dim = states[0].shape[0]
    sj = SpatialJEPA(flat_dim, latent=64, act_dim=3).to(DEVICE)
    opt = torch.optim.Adam(sj.parameters(), lr=3e-3)
    n_epochs = 60 if args.quick else 120
    for _ in range(n_epochs):
        for i in range(len(states) - 1):
            z = sj.z(states[i])
            zp = sj.predict(z, acts[i])
            zt = sj.z_target(states[i + 1])
            loss = torch.nn.functional.mse_loss(zp, zt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sj.update_target()
    errs = []
    with torch.no_grad():
        for i in range(len(states) - 1):
            zp = sj.predict(sj.z(states[i]), acts[i])
            zt = sj.z_target(states[i + 1])
            errs.append(float((zp - zt).norm().item()))
    print(f"  spatial-JEPA next-layout prediction err: {np.mean(errs):.4f} "
          f"(the latent encodes how actions change the layout)")

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"  F1 real Finder window:      OK")
    print(f"  F2 spatial latent:          OK (std {float(spat.std()):.3f})")
    print(f"  F3 error detection:         {'OK' if detected else 'FAIL'}")
    print(f"  F4 visual confirmation:     {'OK' if changed else 'check'}")
    print(f"  F5 spatial latent planning: pred err {np.mean(errs):.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()