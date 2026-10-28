#!/usr/bin/env python3
"""clean_tmp.py — Clean proteus's OWN debris from /private/tmp (safely).

/private/tmp is a shared system temp dir (3843 items, 300MB). Most of the mess
is OUR OWN test debris from building proteus: dozens of proteus-* test dirs,
*.log files, sck_* capture tests, proteus_* temp files.

This cleans ONLY that — identifiable by prefix — and NEVER touches:
  - system/process files (__KMP_REGISTERED_LIB_*, *.sample.txt from crashes)
  - the real project dirs (outputs copy, real_dataset*, etc.)
  - anything we didn't create

Safety:
  - dry-run by default: report what would be removed
  - --execute to actually remove
  - patterns are narrow (proteus, sck_, proteus_, our specific logs)
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

# what we own: proteus test dirs, our logs, our SCK/test files, and ALL the
# per-test data dirs created by the eval/training scripts (ch-, vis-, vj-, cw-,
# be-, ft-, diver-, etc.)
OWNED_PREFIXES = ("proteus-", "sck_", "proteus_", "vc-", "ch-", "vis-", "vj-",
                  "cw-", "be-", "ft-", "diver", "hb-", "cl-", "grw-", "hb-",
                  "leg-", "leg2-", "gv-", "gc-", "gr-", "h2h-", "ev-", "cur-",
                  "diag-", "visk-", "rv2-", "grw-", "scale-", "hard-", "test-",
                  "vj-", "lt-", "vis-hard-", "vis-scale-", "vr-", "dbg-",
                  "ll-", "clv", "le-gate-", "f4-", "diag", "v-", "ll-")
# exact session-debris names
OWNED_EXACT_NAMES = {
    "rv2", "v-base", "v-md", "v-py", "trace-final", "reflex_daemon.log",
    "reflex_daemon2.log", "screen_test.png", "cap_test.png", "cli_t1.png",
    "debug-task", "debug-task2", "le-gate-base.pt",
}
OWNED_EXACT = {
    "proteus-anchor", "proteus-audit", "proteus-auto", "proteus-chaos",
    "proteus-cl", "proteus-cleanup", "proteus-daemon", "proteus-debug",
    "proteus-eval", "proteus-final", "proteus-finder", "proteus-fixed",
    "proteus-flux", "proteus-hard", "proteus-ma", "proteus-pytest",
    "proteus-real", "proteus-realframes", "proteus-rv", "proteus-task",
    "proteus-test", "proteus-vc", "proteus-vision",
}
# our specific files
OWNED_FILES = {
    "proteus_listing.html", "proteus_listing.html.png", "proteus_shell.pipe",
    "proteus_vision_stream.png", "proteus_continual-base.pt", "sck_capture",
    "sck_stream", "proteus_v2", "proteus",
}
LOG_SUFFIX = ".log"


def is_ours(path):
    """Is this path something proteus created (safe to remove)?"""
    name = path.name
    # our log files
    if name.startswith("proteus-") and name.endswith(LOG_SUFFIX):
        return True
    # our test-data dirs: any name starting with our prefixes
    for pre in OWNED_PREFIXES:
        if name.startswith(pre):
            return True
    if name == "proteus":
        return True
    if name.startswith("vc-"):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/private/tmp")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only (default; safe)")
    ap.add_argument("--execute", action="store_true",
                    help="actually remove our debris")
    args = ap.parse_args()

    root = pathlib.Path(args.dir)
    # only top-level items (don't recurse into real project dirs)
    items = sorted(root.iterdir())
    ours = [p for p in items if is_ours(p)]
    theirs = [p for p in items if not is_ours(p)]

    print("=" * 66)
    print(f"TMP CLEANER — {root}")
    print("=" * 66)
    print(f"\n[scan] {len(items)} top-level items")
    print(f"  proteus debris (ours, safe): {len(ours)}")
    print(f"  other (system/real, untouched): {len(theirs)}")

    print("\n[would remove] our proteus debris:")
    for p in ours[:50]:
        kind = "dir" if p.is_dir() else "file"
        print(f"  [{kind}] {p.name}")
    if len(ours) > 50:
        print(f"  ... and {len(ours) - 50} more")

    print("\n[keep] system + real files (NOT touched):")
    for p in theirs[:20]:
        print(f"  {p.name}")
    if len(theirs) > 20:
        print(f"  ... and {len(theirs) - 20} more (incl. KMP libs, crash samples, real dirs)")

    if ours and args.execute:
        print(f"\n[remove] executing: {len(ours)} items")
        removed = 0
        for p in ours:
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                removed += 1
            except Exception as e:
                print(f"  FAILED {p.name}: {str(e)[:50]}")
        print(f"  removed {removed}/{len(ours)}")
    else:
        print(f"\n[remove] dry-run: {len(ours)} items would be removed "
              f"(pass --execute)")

    print("\n" + "=" * 66)
    print("RESULT")
    print("=" * 66)
    print(f"  {len(ours)} proteus debris items {'REMOVED' if (ours and args.execute) else 'IDENTIFIED'}")
    print(f"  system/process files + real project dirs untouched")
    print("=" * 66)


if __name__ == "__main__":
    main()