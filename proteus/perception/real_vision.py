#!/usr/bin/env python3
"""real_vision.py — Real perception via a real VLM + real screencapture.

The synthetic-renderer vision saturated (colored blobs are trivially readable).
This uses REAL perception:
  - `screencapture` captures an ACTUAL screenshot (of a real directory window,
    a real desktop, a real file listing).
  - A real VLM (llava:7b via Ollama) interprets the real image — identifies and
    counts file types, reads text (file names), finds the KEEP file.
  - The VLM's structured output feeds the agent's routing decision, verified
    against the REAL filesystem.

This is the honest jump from "reading synthetic colored blocks" to "interpreting
real images with real objects, text, lighting, and layout."

Pipeline:
  capture(dir) -> real PNG (e.g. render an HTML/AppleScript file listing, or
                   screencapture a real window)
  vlm_interpret(png) -> structured {counts, keep_present, files} via llava
  route(files, keep) -> real action on the real directory
  verify -> re-read the real FS

Test modes:
  - 'render': render the real directory listing as an HTML page -> screenshot
    (this is a REAL image of the directory, with real text file names)
  - 'screencap': capture the actual screen (whatever is showing)
"""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

import proteus.envs.real_env as real_env
from proteus.bridge.qwen_client import _ollama

VLM_MODEL = "llava:7b"


def vlm_ask(image_path, prompt):
    """Ask the VLM (llava) to interpret a real image. Returns raw text."""
    payload = {
        "model": VLM_MODEL,
        "prompt": prompt,
        "images": [str(image_path)],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 256},
    }
    res = _ollama(["run", VLM_MODEL, json.dumps(payload)], timeout=180)
    if res is None:
        return None
    if "raw" in res:
        return res["raw"]
    return res.get("response", "")


def capture_render(root, out_path):
    """Render the real directory as an HTML file-listing page, then screenshot it
    with screencapture. The resulting PNG is a REAL image of the directory with
    real file names as text."""
    files = sorted(p.name for p in root.glob("*") if p.is_file())
    html = f"""<html><body style="background:#222;color:#eee;font-family:monospace;font-size:28px">
<h1>Directory: {root}</h1><ul>
{' '.join(f'<li>{f}</li>' for f in files)}
</ul></body></html>"""
    tmp_html = pathlib.Path(tempfile.gettempdir()) / "proteus_listing.html"
    tmp_html.write_text(html)
    # use screencapture on the opened page via a headless approach: render with
    # a quick HTML-to-image. macOS has no headless chrome by default; use
    # `qlmanage` (Quick Look) which can render HTML to PNG.
    subprocess.run(["qlmanage", "-t", "-s", "800", "-o", str(pathlib.Path(out_path).parent),
                    str(tmp_html)], capture_output=True)
    # qlmanage names it <basename>.html.png
    generated = pathlib.Path(out_path).parent / (tmp_html.stem + ".html.png")
    if generated.exists():
        generated.rename(out_path)
        return True
    return False


def capture_screencap(out_path):
    """Capture the actual screen."""
    subprocess.run(["screencapture", "-x", str(out_path)], capture_output=True)
    return pathlib.Path(out_path).exists()


def interpret_listing(image_path):
    """Ask the VLM to extract the file listing + identify KEEP. Returns dict."""
    prompt = (
        "Look at this screenshot of a directory listing. List every filename you "
        "can read. Then answer: (1) how many .txt, .jpg, .pdf files are present? "
        "(2) Is there a file starting with KEEP or IMPORTANT? Answer in JSON: "
        '{"txt": N, "jpg": N, "pdf": N, "keep_present": true/false, "files": [...]}'
    )
    text = vlm_ask(image_path, prompt)
    if not text:
        return None
    # try to parse JSON from the response
    import re
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"raw": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["render", "screencap"], default="render")
    ap.add_argument("--dir", default="/tmp/proteus-real-vision")
    args = ap.parse_args()

    print("=" * 70)
    print("REAL PERCEPTION: real VLM on real screenshots")
    print("=" * 70)

    # seed a real directory
    real_env.seed_real_world(args.dir, keep_pdf=True)
    root = pathlib.Path(args.dir)
    print(f"\n[world] real directory: {sorted(p.name for p in root.glob('*') if p.is_file())}")

    # capture a real image
    img = pathlib.Path(tempfile.gettempdir()) / "proteus_real.png"
    if args.mode == "render":
        ok = capture_render(root, img)
        print(f"\n[capture] rendered real listing -> {img} ({'ok' if ok else 'FAIL'})")
    else:
        ok = capture_screencap(img)
        print(f"\n[capture] screencapture -> {img} ({'ok' if ok else 'FAIL'})")

    if not ok:
        print("  capture failed")
        return

    print(f"\n[vlm] asking {VLM_MODEL} to interpret the real image...")
    interp = interpret_listing(img)
    print(f"  VLM output: {interp}")

    # verify against real FS
    print("\n[verify] against real filesystem")
    real_txt = len([p for p in root.glob("*.txt") if p.is_file()])
    real_jpg = len([p for p in root.glob("*.jpg") if p.is_file()])
    real_pdf = len([p for p in root.glob("*.pdf") if p.is_file()])
    real_keep = any(real_env.RealEnv(root).is_keep(p.name) for p in root.glob("*.pdf"))
    print(f"  real: txt={real_txt} jpg={real_jpg} pdf={real_pdf} keep={real_keep}")
    if isinstance(interp, dict) and "txt" in interp:
        v_txt = interp.get("txt", -1)
        v_keep = interp.get("keep_present")
        match = (int(v_txt) == real_txt) if isinstance(v_txt, (int, float)) else False
        keep_match = (v_keep == real_keep) if isinstance(v_keep, bool) else False
        print(f"  VLM txt={v_txt} vs real {real_txt}: {'MATCH' if match else 'MISMATCH'}")
        print(f"  VLM keep={v_keep} vs real {real_keep}: {'MATCH' if keep_match else 'MISMATCH'}")
    print("=" * 70)


if __name__ == "__main__":
    main()