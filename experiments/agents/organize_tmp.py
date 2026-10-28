#!/usr/bin/env python3
"""organize_tmp.py — Organize /private/tmp into a real structure.

Group the mixed files into folders WITHOUT deleting anything:
  candidates/  -> cand_*.jpg + cand_*.json (paired seed candidates)
  refs/        -> ref_*.b64, ref_*.txt, ref_*.png, base_b64.txt, blue_b64.txt,
                  rank_clean_*.b64, test_img.b64, triposr_view_0.b64, rf*.b64
  json/        -> *.json configs/results (bench, think, vision, unsloth, flux)
  logs/        -> *.log (energy-server, httpd, unsloth, lora, pulseview, ollama)
  regions/     -> *region*.txt, summary*.txt, imgs.txt, full_region.txt, s6_region.txt
  proteus-session-debris/ -> our test files (debug-task, dbg2, rv2, trace-final,
                  cap_test.png, cli_t1.png, screen_test.png, reflex_daemon*.log,
                  cl-* / clx, audit.log, dav2_test.npy)

NEVER touched: system files (KMP, crash samples, RustDesk, sockets, Keystone,
__pycache__, outputs copy, test01 copy, triposr-src).

Safety: --dry-run by default (reports the grouping). --execute moves files into
the folders. Nothing is deleted — only organized.
"""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import argparse
import pathlib
import shutil

GROUPS = {
    "candidates": ["cand_"],
    "refs": ["ref_", "base_b64", "blue_b64", "rank_clean", "test_img",
             "triposr_view", "rf", "rf96"],
    "json": [".json"],
    "logs": [".log"],
    "regions": ["region", "summary", "imgs.txt", "full_region", "s6_region"],
}
SYS_NAMES = {"RustDesk-501", "__pycache__", "com.google.Keystone", "powerlog",
             "outputs copy", "test01 copy", "triposr-src"}
SYS_PREFIXES = ("__KMP", "Python_", "ai_unsloth_studio_si", "com_woro_simplevoice_si")
SESSION_NAMES = {"debug-task", "debug-task2", "dbg2", "rv2", "trace-final",
                 "cap_test.png", "cli_t1.png", "screen_test.png",
                 "reflex_daemon.log", "reflex_daemon2.log", "audit.log",
                 "dav2_test.npy"}
SESSION_PREFIXES = ("cl-", "clx", "cl-check", "cl-check2")


def classify(name, path):
    """Return the folder a file/dir belongs to, or None (system/leave)."""
    if name in SYS_NAMES or name.startswith(SYS_PREFIXES):
        return None
    if name in SESSION_NAMES or name.startswith(SESSION_PREFIXES):
        return "proteus-session-debris"
    for folder, pats in GROUPS.items():
        for p in pats:
            if name.startswith(p) or (p == ".json" and name.endswith(p)):
                return folder
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/private/tmp")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.dir)
    items = sorted(root.iterdir())

    # map folder -> list of items
    plan = {}
    for item in items:
        folder = classify(item.name, item)
        if folder is not None:
            plan.setdefault(folder, []).append(item)

    print("=" * 66)
    print(f"TMP ORGANIZER — {root}")
    print("=" * 66)
    for folder, items in sorted(plan.items()):
        print(f"\n[{folder}/]  ({len(items)} items)")
        for it in items[:8]:
            kind = "dir" if it.is_dir() else "file"
            print(f"  {kind:4} {it.name}")
        if len(items) > 8:
            print(f"  ... +{len(items) - 8} more")

    moved = 0
    if args.execute:
        print("\n[organize] moving into folders:")
        for folder, items in plan.items():
            dest = root / folder
            dest.mkdir(exist_ok=True)
            for it in items:
                target = dest / it.name
                if target.exists():
                    target = dest / (it.stem + "_dup" + it.suffix)
                shutil.move(str(it), str(target))
                moved += 1
                print(f"  {it.name} -> {folder}/")
        print(f"  moved {moved} items")
    else:
        total = sum(len(v) for v in plan.values())
        print(f"\n[organize] dry-run: {total} items would be grouped "
              f"(pass --execute to move). Nothing deleted.")

    print("\n" + "=" * 66)
    print("RESULT")
    print("=" * 66)
    print(f"  grouped into {len(plan)} folders {'(EXECUTED)' if args.execute else '(DRY-RUN)'}")
    print("  system + project dirs (outputs copy, triposr-src, KMP) untouched")
    print("=" * 66)


if __name__ == "__main__":
    main()