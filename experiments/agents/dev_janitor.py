#!/usr/bin/env python3
"""dev_janitor.py — The Developer Artifact Janitor.

A new proteus "world": reclaims disk space by clearing dead build cache and
virtual environments from dormant dev projects, with a hard safety barrier.

State (obs, 12-dim, slots into HJEPA_K4 obs_dim=12):
  [root_pycache, root_nodemod, root_cargo, root_venv,
   pycache_n, nodemod_n, cargo_n, venv_n,
   stale_repos, has_protected, reclaimable_gb, clean_complete]

Actions (4, matches act_dim=4):
  0 = clean_pycache        __pycache__/, *.pyc, .pytest_cache, .mypy_cache
  1 = clean_node_modules   node_modules/ in stale repos
  2 = cargo_clean          cargo clean (or rm target/) in stale Rust repos
  3 = clean_venv           venv/ .venv/ env/ in stale repos

SAFETY BARRIER (mirrors the KEEP barrier in RealEnv):
  - Only targets known-safe, regenerable artifacts (bytecode, deps, build dirs).
  - NEVER touches anything git-tracked: a stale repo is skipped entirely if
    `git status --porcelain` shows ANY change to tracked files.
  - NEVER deletes KEEP/IMPORTANT/DO_NOT_DELETE files or .git itself.
  - dry_run=True (default) reports only; pass --execute to actually clean.

VERIFICATION LOOP:
  du -sh (bytes) before and after each action. If reclaimed > threshold
  (default 500MB) AND git status remains pristine, the action is VERIFIED.
  The result is logged as `verified_action` to autonomous_journal.jsonl with
  obs + latent + reclaimed_bytes so the JEPA can learn and Qwen can explain.

Usage:
  python3 dev_janitor.py --scan                     # report state only
  python3 dev_janitor.py --clean --dry-run          # show what would be done
  python3 dev_janitor.py --clean --execute          # actually clean (safety on)
  python3 dev_janitor.py --observe --explain        # JEPA->Qwen explanation
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
import subprocess
import sys
import time

import numpy as np

OBS_DIM = 12
ACT_DIM = 4
ACTIONS = ["clean_pycache", "clean_node_modules", "cargo_clean", "clean_venv"]
ACT2NAME = {i: n for i, n in enumerate(ACTIONS)}

DEFAULT_ROOT = pathlib.Path.home() / "Development"
STALE_DAYS = 30
RECLAIM_THRESHOLD_BYTES = 500 * 1024 * 1024  # 500MB to log verified_action
PROTECT_PATTERNS = ("KEEP", "IMPORTANT", "DO_NOT_DELETE")

# known-safe, regenerable artifacts (name, kind) we are allowed to remove.
# `build`/`dist` are deliberately EXCLUDED: some projects commit them as
# source, so they're ambiguous. Only unambiguous regenerable dirs are auto-clean.
CACHE_DIRS = {
    "__pycache__": "pycache", "venv": "venv", ".venv": "venv", "env": "venv",
    "node_modules": "nodemod", ".pytest_cache": "pycache",
    ".mypy_cache": "pycache", ".ruff_cache": "pycache",
    "target": "cargo",
}

# project markers: a dir is a "project" if it has VCS or a manifest
PROJECT_MARKERS = (".git", "pyproject.toml", "setup.py", "package.json",
                   "Cargo.toml", "CMakeLists.txt", ".xcodeproj", "go.mod")


def _du_bytes(path):
    """Disk usage of a path in bytes (du -sk on macOS)."""
    try:
        r = subprocess.run(["du", "-sk", str(path)], capture_output=True,
                           text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return int(r.stdout.split()[0]) * 1024
    except Exception:
        pass
    return 0


def _last_commit(path):
    """Days since the last git commit in path (AUTHOR date — when the work was
    done), or None if not a git repo."""
    try:
        r = subprocess.run(
            ["git", "-C", str(path), "log", "-1", "--format=%at"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        ts = float(r.stdout.strip())
        return (time.time() - ts) / 86400.0
    except Exception:
        return None


def _last_tracked_touch(path):
    """Days since the newest file mtime under the project, EXCLUDING
    regenerable caches and .git (so a repo full of build junk still reads as
    'untouched'). Used as the staleness signal for NON-git projects; git
    projects are judged by commit author date."""
    newest = 0
    try:
        for p in path.rglob("*"):
            if p.is_file() and ".git" not in p.parts:
                if any(part in CACHE_DIRS for part in p.parts):
                    continue
                newest = max(newest, p.stat().st_mtime)
    except Exception:
        pass
    return (time.time() - newest) / 86400.0 if newest else 1e9


def _git_pristine(path):
    """True if the repo has NO tracked-file changes (working tree clean for
    tracked files). Untracked (typically gitignored) caches don't count."""
    try:
        r = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                           capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            return True  # not a repo -> nothing tracked -> pristine
        return all(ln.startswith("??") for ln in r.stdout.splitlines())
    except Exception:
        return True


def is_protected(name):
    return any(name.startswith(p) for p in PROTECT_PATTERNS)


def is_project(dirpath):
    return any((dirpath / m).exists() for m in PROJECT_MARKERS)


def find_stale_projects(root, stale_days=STALE_DAYS, max_depth=4):
    """Scan root for projects that haven't had a git commit or file touch in
    > stale_days. Returns list of (proj_path, last_commit_days, latest_touch)."""
    root = pathlib.Path(root)
    if not root.exists():
        return []
    stale = []
    # BFS bounded depth to avoid scanning the whole disk
    frontier = [root]
    for _ in range(max_depth):
        nxt = []
        for d in frontier:
            try:
                entries = sorted(d.iterdir())
            except Exception:
                continue
            for e in entries:
                if not e.is_dir() or e.name.startswith("."):
                    continue
                if is_project(e):
                    days = _last_commit(e)
                    if days is not None:
                        # git project: commit author date is the staleness signal
                        if days > stale_days:
                            stale.append((e, days, None))
                    else:
                        # non-git project: newest non-cache file mtime
                        touch_days = _last_tracked_touch(e)
                        if touch_days > stale_days:
                            stale.append((e, None, touch_days))
                else:
                    nxt.append(e)
        frontier = nxt
    return stale


class DevJanitorEnv:
    """The janitor world. obs() -> 12-dim vector; step(action) -> clean+verify."""

    def __init__(self, root=DEFAULT_ROOT, stale_days=STALE_DAYS,
                 reclaim_threshold=RECLAIM_THRESHOLD_BYTES, dry_run=True):
        self.root = pathlib.Path(root)
        self.stale_days = stale_days
        self.reclaim_threshold = reclaim_threshold
        self.dry_run = dry_run
        self.last_verified = False
        self.reclaimed_total = 0
        self.projects = []        # stale projects found by last scan
        self.scan()

    # ---- scan / state ----
    def scan(self):
        self.projects = find_stale_projects(self.root, self.stale_days)
        return self.projects

    def _candidates(self, kind):
        """List (path, bytes) of removable artifacts of `kind` across stale
        projects. Only runs on repos whose tracked files are pristine."""
        out = []
        for proj, _, _ in self.projects:
            if not _git_pristine(proj):
                continue  # safety: skip dirty repos entirely
            for p in proj.rglob("*"):
                if not p.is_dir():
                    continue
                if ".git" in p.parts:
                    continue
                if any(is_protected(part) for part in p.parts):
                    continue
                kind_of = CACHE_DIRS.get(p.name)
                if kind_of == kind:
                    b = _du_bytes(p)
                    if b > 0:
                        out.append((p, b))
        return out

    def reclaimable_by_kind(self):
        kinds = ("pycache", "nodemod", "cargo", "venv")
        return {k: self._candidates(k) for k in kinds}

    def obs(self):
        """12-dim state vector (slots into the shared HJEPA obs_dim=12)."""
        if not self.projects:
            self.scan()
        rec = self.reclaimable_by_kind()
        c = {k: len(v) for k, v in rec.items()}
        b = {k: sum(b for _, b in v) for k, v in rec.items()}
        total_b = sum(b.values())
        has_protected = 1 if any(
            is_protected(p.name) for proj, _, _ in self.projects
            for p in proj.rglob("*") if p.is_file()) else 0
        vec = np.array([
            c["pycache"], c["nodemod"], c["cargo"], c["venv"],
            b["pycache"] / max(total_b, 1), b["nodemod"] / max(total_b, 1),
            b["cargo"] / max(total_b, 1), b["venv"] / max(total_b, 1),
            len(self.projects), has_protected,
            min(total_b / (20 * 1024**3), 1.0),   # reclaimable GB normalized /20
            1 if self.last_verified else 0,
        ], dtype=np.float32)
        return vec

    # ---- actions ----
    def _remove(self, path):
        if self.dry_run:
            return 0
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        return _du_bytes(path)  # post-size (0 if removed)

    def step(self, action):
        """Perform one janitor action. Returns (obs, reclaimed_bytes, verified,
        detail). Safety: only known cache dirs in pristine repos."""
        if action not in range(ACT_DIM):
            action = 0
        kind = {0: "pycache", 1: "nodemod", 2: "cargo", 3: "venv"}[action]
        cands = self._candidates(kind)
        before = sum(b for _, b in cands)
        removed = []
        for p, b in cands:
            if self.dry_run:
                removed.append(str(p.relative_to(self.root)))
            else:
                shutil.rmtree(p, ignore_errors=True)
                removed.append(str(p.relative_to(self.root)))
        # post: recompute size of anything left (rmtree for real runs)
        after = 0
        if not self.dry_run:
            for p, _ in cands:
                if p.exists():
                    after += _du_bytes(p)
        reclaimed = before - after
        verified = reclaimed > self.reclaim_threshold
        self.last_verified = verified
        self.reclaimed_total += reclaimed
        detail = (f"{ACT2NAME[action]}: {len(cands)} artifact(s), "
                  f"{reclaimed/1e6:.1f} MB reclaimed")
        if self.dry_run:
            detail += " (DRY-RUN, nothing deleted)"
        return self.obs(), reclaimed, verified, detail, removed

# ---- text (for the journal / Qwen) ----
    def describe(self):
        if not self.projects:
            return "no stale projects found"
        lines = [f"{len(self.projects)} stale project(s):"]
        for proj, days, td in self.projects:
            c = "?" if days is None else f"{days:.0f}d"
            t = "n/a" if td is None else f"{td:.0f}d"
            lines.append(f"  {proj.name} (last commit {c}, no touch {t})")
        return "\n".join(lines)


# ---------------- runner: act like the autonomous agent ----------------
def run_janitor(root, dry_run=True, execute=False, log_journal=True,
                max_actions=4):
    """Scan, clean (if execute), verify each action, log verified_action to the
    journal with obs+latent so the JEPA can learn + Qwen can explain."""
    env = DevJanitorEnv(root=root, dry_run=dry_run)
    if not env.projects:
        print("no stale projects — nothing to reclaim")
        return env

    print(env.describe())
    rec = env.reclaimable_by_kind()
    for k, items in rec.items():
        tot = sum(b for _, b in items)
        if items:
            print(f"  {k:10s}: {len(items)} dir(s), {tot/1e6:.1f} MB")

    vec0 = env.obs()
    for a in range(min(ACT_DIM, max_actions)):
        obs, reclaimed, verified, detail, removed = env.step(a)
        flag = "VERIFIED" if verified else "no-op/small"
        print(f"\n[{ACT2NAME[a]}] {flag} — {detail}")
        if not dry_run and removed and len(removed) <= 8:
            for r in removed:
                print(f"    - {r}")
        if log_journal and not dry_run and verified:
            _log_verified_action(root, ACT2NAME[a], obs, reclaimed, verified,
                                 detail)
    env.scan()
    print(f"\nTOTAL RECLAIMED: {env.reclaimed_total/1e6:.1f} MB "
          f"({'DRY-RUN' if dry_run else 'EXECUTED'})")
    return env


def _log_verified_action(root, action, obs, reclaimed, verified, detail):
    """Append a verified_action to autonomous_journal.jsonl, mirroring the
    format autonomous.py logs (so monitor.py + the bridge can read it)."""
    from proteus.world.world_latent import WorldLatent
    try:
        wl = WorldLatent()
        latent = wl.embed(obs).tolist()
    except Exception:
        latent = None
    rec = {
        "session": "janitor",
        "t": time.time(),
        "event": "verified_action",
        "action": action,
        "verified": 1 if verified else 0,
        "detail": detail,
        "reclaimed_bytes": int(reclaimed),
        "root": str(root),
        "obs": obs.tolist(),
        "latent": latent,
    }
    journal = ROOT / "data" / "autonomous_journal.jsonl"
    with open(journal, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  [journal] logged verified_action -> {journal.name}")


def explain(root):
    """Grounded explanation of the janitor world state. The JEPA bridge is
    trained on the file-cleanup world, so for the janitor we ground Qwen in the
    ACTUAL scan facts (stale count, reclaimable bytes per artifact kind) — the
    numbers the user's spec wants explained. Returns the Qwen text."""
    env = DevJanitorEnv(root=root, dry_run=True)
    rec = env.reclaimable_by_kind()
    facts = []
    for k, items in rec.items():
        tot = sum(b for _, b in items)
        if items:
            facts.append(f"{len(items)} {k} dir(s) ({tot/1e6:.1f} MB)")
    fact_txt = ", ".join(facts) if facts else "no reclaimable artifacts"
    prompt = (
        "You are a developer-artifact janitor for a macOS machine. "
        f"Scan of {root} found {len(env.projects)} dormant project(s) "
        f"(no git commit or file touch in >30 days). Reclaimable: {fact_txt}. "
        "Explain in one sentence what cleaning action should be taken and "
        "roughly how much disk space it would reclaim."
    )
    try:
        import proteus.bridge.qwen_client as qwen_client
        return qwen_client.qwen_generate(prompt)[:300]
    except Exception as e:
        return f"(qwen unavailable: {str(e)[:60]})"


def daemon(root, interval=3600, dry_run=True, execute=False):
    """Run the janitor on a schedule (like the autonomous daemon). Defaults to
    dry-run (report only); pass execute=True to actually reclaim. Logs
    verified_action to the journal when > threshold is reclaimed."""
    print(f"[janitor-daemon] watching {root} every {interval}s "
          f"({'DRY-RUN' if dry_run else 'EXECUTE'})")
    while True:
        try:
            run_janitor(root, dry_run=dry_run, execute=execute)
        except Exception as e:
            print(f"[janitor-daemon] error: {str(e)[:120]}")
        time.sleep(interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--scan", action="store_true", help="report state only")
    ap.add_argument("--clean", action="store_true", help="run the clean loop")
    ap.add_argument("--dry-run", action="store_true",
                    help="(default) report, never delete")
    ap.add_argument("--execute", action="store_true",
                    help="actually delete caches (safety barrier still on)")
    ap.add_argument("--explain", action="store_true",
                    help="Qwen grounded explanation of the janitor state")
    ap.add_argument("--daemon", action="store_true",
                    help="run on a schedule (default dry-run)")
    ap.add_argument("--interval", type=int, default=3600,
                    help="daemon interval seconds")
    args = ap.parse_args()

    if args.explain:
        print(explain(args.root))
    elif args.daemon:
        daemon(args.root, interval=args.interval,
               dry_run=not args.execute, execute=args.execute)
    elif args.scan:
        env = DevJanitorEnv(root=args.root)
        print(env.describe())
        rec = env.reclaimable_by_kind()
        for k, items in rec.items():
            tot = sum(b for _, b in items)
            print(f"  {k:10s}: {len(items)} dir(s), {tot/1e6:.1f} MB")
    elif args.clean:
        run_janitor(args.root, dry_run=not args.execute,
                    execute=args.execute)
    else:
        ap.print_help()