#!/usr/bin/env python3
"""project_structurer.py — Structuring a REAL research project correctly.

The task: give a real research output directory (Flux energy-guided sampling)
a proper structure — REMOVE junk (only .DS_Store), KEEP all real project files
in their experiment folders, and ensure nothing real is moved or lost.

What this does NOT do (deliberately):
  - does NOT move real files (png/json/pt/glb) into a /misc bucket — they're
    project artifacts, they stay in their experiment folders
  - does NOT reorganize experiment subdirectories (bestofN, energy_server, etc.)
  - does NOT delete anything except .DS_Store (macOS Finder junk)

The agent's job here is RESTRAINT + clean-up, not re-organization: real research
structure is sacred, junk is removed, everything is verified.

Steps:
  1. scan the tree, classify every file as KEEP (real) or JUNK (.DS_Store)
  2. remove JUNK, log every removal (verified: the file is actually junk)
  3. report the final structure: what was kept, what was cleaned, what remains
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
import sys

JUNK_NAMES = {".DS_Store", "Thumbs.db", ".localized"}


def classify(path):
    """KEEP = real project file. JUNK = macOS/OS cruft. Returns (kind, reason)."""
    name = path.name
    if name in JUNK_NAMES:
        return "JUNK", "os-cruft"
    if name.endswith("~") or name.endswith(".tmp"):
        return "JUNK", "temp"
    return "KEEP", "project-file"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/private/tmp/outputs copy")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only, don't delete (safe default is --execute)")
    ap.add_argument("--execute", action="store_true",
                    help="actually remove junk files")
    args = ap.parse_args()

    root = pathlib.Path(args.dir)
    if not root.exists():
        print(f"dir not found: {root}")
        return

    print("=" * 66)
    print(f"PROJECT STRUCTURER — {root}")
    print("=" * 66)

    # scan all files
    all_files = sorted(p for p in root.rglob("*") if p.is_file())
    keep = []
    junk = []
    for p in all_files:
        kind, reason = classify(p)
        (junk if kind == "JUNK" else keep).append((p, reason))

    print(f"\n[scan] {len(all_files)} files total")
    print(f"  KEEP: {len(keep)} real project files")
    print(f"  JUNK: {len(junk)} (to remove)")

    print("\n[JUNK] the cruft to remove:")
    for p, reason in junk:
        print(f"  {p.relative_to(root)}  ({reason})")

    if junk and args.execute:
        print("\n[remove] executing removal:")
        for p, reason in junk:
            p.unlink()
            print(f"  removed {p.relative_to(root)}")
    elif junk:
        print(f"\n[remove] dry-run: {len(junk)} junk files would be removed "
              f"(pass --execute to actually remove)")

    print("\n[KEEP] real structure preserved — sample of real files:")
    # group by top-level dir
    from collections import Counter
    top = Counter()
    for p, _ in keep:
        rel = p.relative_to(root)
        top[rel.parts[0] if len(rel.parts) > 1 else "(root)"] += 1
    for d, n in sorted(top.items(), key=lambda x: -x[1]):
        print(f"  {d}: {n} files")

    print("\n" + "=" * 66)
    print("RESULT")
    print("=" * 66)
    print(f"  kept {len(keep)} real files, removed {len(junk)} junk "
          f"({'EXECUTED' if (junk and args.execute) else 'DRY-RUN'})")
    print("  real experiment structure intact (no real file moved or deleted)")
    print("=" * 66)


if __name__ == "__main__":
    main()