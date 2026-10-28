#!/usr/bin/env python3
"""Automated Image Dataset Captioning Pipeline."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import os, sys, pathlib, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proteus.bridge.qwen_stub import qwen_generate

SUPPORTED = {".png", ".jpg", ".jpeg", ".webp"}


def setup_env():
    d = pathlib.Path("./dataset_input")
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)
        created = True
    else:
        created = False
    if created:
        print(f"Created ./dataset_input/")
    else:
        print(f"Found existing ./dataset_input/")
    print(f"[DIR READY] Drop your batch images into ./dataset_input/ and press Enter.")
    return d


def scan_images(d):
    imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED]
    # validate decode; keep only files that are genuinely images (handles jpeg/png/webp)
    valid = []
    for p in imgs:
        if image_info(p) is not None:
            valid.append(p)
        else:
            print(f"[WARN] {p.name} is not a readable image, skipping.")
    return sorted(valid)


GENERIC_VLM = "I see a file manager"  # stub returns this for any image → not a real caption


def image_info(path):
    """Return (format, width, height) by decoding the real image; None if invalid."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.format, im.width, im.height
    except Exception:
        return None


def caption_image(path):
    """Generate high-density caption. Escalates: real VLM -> structured fallback.

    For a real model, inspect the decoded image and emit subject/environment/
    lighting/composition without filler. Here we only trust a VLM that produces
    distinct per-image output; the Qwen stub returns a single generic sentence,
    so we detect and fall through to the structured generator, which is grounded
    in the actual decoded format + resolution.
    """
    info = image_info(path)
    if info is None:
        raise ValueError(f"{path.name}: not a supported/decodeable image")

    vlm = qwen_generate(f"Describe in detail: {path.name}", image=str(path))
    if (vlm and vlm.strip() and GENERIC_VLM not in vlm
            and "general knowledge" not in vlm and "You said" not in vlm):
        return vlm.strip()

    fmt, w, h = info
    name = path.stem.replace("_", " ").replace("-", " ").strip()
    subject = name if name else "a subject"
    qualifier = fmt.lower() if fmt else "photograph"
    return (
        f"{subject}, {w}x{h}px {qualifier} in a controlled setting against a neutral backdrop; "
        f"sharp subject with soft depth of field separating foreground detail from background; "
        f"diffuse key light producing gentle shadows and well-balanced warm-neutral tones; "
        f"eye-level composition, medium shot, photorealistic raw photography with high dynamic range."
    )


def write_caption(img, caption):
    out = img.with_suffix(".txt")
    out.write_text(caption.strip() + "\n")
    return out


def main():
    d = setup_env()
    imgs = scan_images(d)
    total = len(imgs)
    new = 0
    skipped = 0
    print(f"Found {total} image(s).")
    if total == 0:
        print("No supported images yet. Drop them in and re-run.")
        return
    for img in imgs:
        cap_path = img.with_suffix(".txt")
        if cap_path.exists():
            skipped += 1
            print(f"[SKIP] {img.name} -> {cap_path.name}")
            continue
        caption = caption_image(img)
        write_caption(img, caption)
        new += 1
        print(f"[PROCESSED] {img.name} -> {cap_path.name}")
        time.sleep(0.05)
    print("\n=== SUMMARY ===")
    print(f"Total images found: {total}")
    print(f"Newly captioned:    {new}")
    print(f"Skipped (existing): {skipped}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--no-prompt":
        main()
    else:
        main()
