#!/usr/bin/env python3
"""prune_data.py — Bounded retention: keep proteus's data dir from building up.

Caps / rotates the growing STREAM and LOG files so the data dir stays lean and
reads stay fast, and clears build caches. It NEVER touches ground truth or the
trained-on clean sets:

  KEPT (untouched): autonomous_journal.jsonl, latent_pairs.jsonl*,
  latent_pairs_clean.jsonl, transition_pairs_clean.jsonl, whiten.npz, owner.key,
  checkpoints/, mycelium.jsonl (rebuilt only with --rebuild-mycelium).

Usage:
  python3 prune_data.py                    # safe defaults: cap streams/logs + clear caches
  python3 prune_data.py --dry-run          # show what would change, change nothing
  python3 prune_data.py --rebuild-mycelium 8   # also rebuild the graph, keep last 8 revs/source
  python3 prune_data.py --cap-journal 20000    # also cap the journal to N lines
"""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import argparse
import json
import pathlib
import shutil
import sys

DATA = ROOT / "data"

# stream/log files -> keep the LAST N lines (order used by the writers)
STREAM_CAPS = {
    "telemetry.jsonl": 4000,
    "interactions.jsonl": 20000,
    "latent_memory.jsonl": 6000,
    "eris_trace.jsonl": 500,
    "eris_memory.jsonl": 500,
    "cleanup_tuples.jsonl": 5000,
}
LOG_CAP = 500  # lines to keep in *.log files
# files that are ground truth / trained-on / config: never touched
NEVER = {"autonomous_journal.jsonl", "latent_pairs.jsonl", "latent_pairs_clean.jsonl",
         "transition_pairs_clean.jsonl", "transition_stats.json", "whiten.npz",
         "owner.key", "autonomous_state.json", "com.proteus.reflex.plist"}


def size_mb(p):
    return p.stat().st_size / 1e6 if p.exists() else 0.0


def dir_size_mb(p):
    """Recursive size of a directory (for the 'data/ now' readout)."""
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def keep_last_lines(path, n, dry):
    """Truncate a JSONL/log to its last n lines. Returns (before, after) MB."""
    before = size_mb(path)
    lines = path.read_text().splitlines() if path.exists() else []
    if len(lines) <= n or n == 0:
        return before, before
    if dry:
        return before, size_mb_est_from_lines(lines[-n:])
    path.write_text("\n".join(lines[-n:]) + ("\n" if lines[-n:] else ""))
    return before, size_mb(path)


def size_mb_est_from_lines(lines):
    return sum(len(l.encode()) + 1 for l in lines) / 1e6


def clear_caches(dry):
    freed = 0.0
    for pattern in ("__pycache__", ".pytest_cache"):
        for p in ROOT.rglob(pattern):
            if p.is_dir():
                freed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                if not dry:
                    shutil.rmtree(p, ignore_errors=True)
    for p in ROOT.rglob("*.pyc"):
        if p.is_file():
            freed += p.stat().st_size
            if not dry:
                try:
                    p.unlink()
                except OSError:
                    pass
    for p in ROOT.rglob(".DS_Store"):
        if p.is_file():
            freed += p.stat().st_size
            if not dry:
                try:
                    p.unlink()
                except OSError:
                    pass
    return freed / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild-mycelium", type=int, metavar="N",
                    help="rebuild mycelium.jsonl, keep last N revisions per source")
    ap.add_argument("--cap-journal", type=int, metavar="N",
                    help="also cap autonomous_journal.jsonl to N lines")
    ap.add_argument("--log-cap", type=int, default=LOG_CAP)
    args = ap.parse_args()

    print("=" * 64)
    print("PROTEUS PRUNE  (dry-run)" if args.dry_run else "PROTEUS PRUNE")
    print("=" * 64)
    total_before = 0.0
    report = []

    # 1) streams
    for name, cap in STREAM_CAPS.items():
        p = DATA / name
        if not p.exists():
            continue
        b, a = keep_last_lines(p, cap, args.dry_run)
        total_before += b
        report.append((name, b, a, f"cap {cap}"))

    # 2) optional journal cap
    if args.cap_journal:
        p = DATA / "autonomous_journal.jsonl"
        b, a = keep_last_lines(p, args.cap_journal, args.dry_run)
        total_before += b
        report.append((p.name, b, a, f"cap {args.cap_journal}"))

    # 3) logs
    for p in DATA.glob("*.log"):
        b, a = keep_last_lines(p, args.log_cap, args.dry_run)
        report.append((p.name, b, a, f"log cap {args.log_cap}"))

    # 4) caches
    cf = clear_caches(args.dry_run)
    total_before += cf
    report.append(("caches (__pycache__/pyc/.DS_Store)", cf, 0.0, "cleared"))

    print(f"{'file':34} {'before MB':>10} {'after MB':>10}")
    for name, b, a, note in report:
        print(f"{name:34} {b:>10.2f} {a:>10.2f}   {note}")
    print(f"{'TOTAL touched':34} {sum(r[1] for r in report):>10.2f} "
          f"{sum(r[2] for r in report):>10.2f}")

    # 5) optional mycelium rebuild (derived graph -> lean)
    if args.rebuild_mycelium:
        b = size_mb(DATA / "mycelium.jsonl")
        if not args.dry_run:
            import proteus.agent.mycelium as mycelium
            mycelium.build(max_revisions=args.rebuild_mycelium)
        a = size_mb(DATA / "mycelium.jsonl")
        print(f"{'mycelium.jsonl (rebuild)':34} {b:>10.2f} {a:>10.2f}   "
              f"last {args.rebuild_mycelium} revs/source (source history kept in journal+PAM)")

    total_after = sum(r[2] for r in report)
    if not args.dry_run:
        print(f"\n  data/ now: {dir_size_mb(DATA):.2f} MB")
    else:
        print("\n  (dry-run: nothing changed)")
    print("  KEPT: ground truth + clean bridge sets + checkpoints + owner.key")


if __name__ == "__main__":
    main()
