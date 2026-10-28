#!/usr/bin/env python3
"""Create REAL png/jpeg/webp test images via Pillow."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import pathlib
from PIL import Image


def save(path, rgb, fmt, size=(64, 64)):
    im = Image.new("RGB", size, rgb)
    if fmt == "jpg":
        im.save(path, "JPEG", quality=90)
    elif fmt == "png":
        im.save(path, "PNG")
    elif fmt == "webp":
        im.save(path, "WEBP", quality=90)


def main():
    out = pathlib.Path("./dataset_input")
    out.mkdir(exist_ok=True)
    save(out / "image_001.png", (200, 120, 60), "png")
    save(out / "image_002.jpg", (60, 90, 200), "jpg")     # REAL jpeg
    save(out / "image_003.jpeg", (90, 60, 140), "jpg")    # REAL jpeg (jpeg ext)
    save(out / "image_004.webp", (40, 160, 80), "webp")
    # invalid file that should be skipped (not a real image)
    (out / "image_005.jpg").write_text("this is not an image")
    # pre-existing caption to test SKIP logic
    (out / "image_002.txt").write_text(
        "portrait, closed-up studio shot of a ceramic vase, rim light, shallow depth of field.\n")
    print(f"Created test images in {out}:")
    for p in sorted(out.iterdir()):
        kind = "regular" if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else "text"
        print(f"  {p.name} ({p.stat().st_size}B) [{kind}]")


if __name__ == "__main__":
    main()
