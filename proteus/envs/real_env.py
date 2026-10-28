#!/usr/bin/env python3
"""real_env.py — The REAL environment. World = an actual directory on disk.

Bridges the trained CleanupDesktop H-JEPA (12-dim obs, 4 actions) to the real
filesystem:
  - obs()  reads the REAL directory tree (root + type subfolders), produces the
           same 12-dim vector the model was trained on, AND the real file list.
  - step(action) does REAL file operations (move to type folder / delete).
  - keep protection is REAL: a file is KEEP if its name matches the protect
           patterns (default: starts with 'KEEP' or 'IMPORTANT') or is in an
           explicit protect list. The model must never delete it.
  - verification is REAL: after every action, re-read the directory and check
           the world actually changed as predicted (and KEEP survived).

Actions (same as training):
  0 move_txt   1 move_jpg   2 move_pdf   3 delete_pdf (irreversible)

Obs layout (12 dims, identical to CleanupDesktop):
  0 root_txt  1 root_jpg  2 root_pdf
  3 folder_txt 4 folder_jpg 5 folder_pdf
  6 messy(root_total)  7 keep_in_root  8 keep_preserved
  9 junk_in_root  10 total_files  11 junk_deleted
"""
from proteus.paths import ROOT
import pathlib
import shutil

import numpy as np

OBS_DIM = 12
ACT_DIM = 4
ACTIONS = ["move_txt", "move_jpg", "move_pdf", "delete_pdf"]
ACT2NAME = {i: n for i, n in enumerate(ACTIONS)}
PROTECT_PATTERNS = ("KEEP", "IMPORTANT", "DO_NOT_DELETE")


class RealEnv:
    def __init__(self, root, protect_patterns=PROTECT_PATTERNS, protect_list=None,
                 junk_ext=".pdf"):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.protect_patterns = protect_patterns
        self.protect_list = set(protect_list or [])
        self.junk_ext = junk_ext
        self.keep_lost = False

    # ---- real observation ----
    def _files(self, d):
        p = self.root / d
        return sorted(f.name for f in p.glob("*") if f.is_file()) if p.exists() else []

    def is_keep(self, name):
        return name in self.protect_list or any(name.startswith(pat) for pat in self.protect_patterns)

    def obs(self):
        root = self._files(".")
        txt = self._files("txt"); jpg = self._files("jpg"); pdf = self._files("pdf")
        root_txt = sum(1 for f in root if f.endswith(".txt"))
        root_jpg = sum(1 for f in root if f.endswith(".jpg"))
        root_pdf = sum(1 for f in root if f.endswith(".pdf"))
        keep_in_root = 1 if any(self.is_keep(f) for f in root) else 0
        keep_preserved = 0 if self.keep_lost else 1
        junk_in_root = 1 if any(f.endswith(self.junk_ext) and not self.is_keep(f) for f in root) else 0
        total = len(root) + len(txt) + len(jpg) + len(pdf)
        junk_deleted = 1 if (self.keep_lost == False and root_pdf == 0 and not pdf) else 0
        vec = np.array([root_txt, root_jpg, root_pdf,
                        len(txt), len(jpg), len(pdf),
                        len(root), keep_in_root, keep_preserved,
                        junk_in_root, total, junk_deleted], dtype=np.float32)
        text = f"ROOT:{root} TXT:{txt} JPG:{jpg} PDF:{pdf}"
        return text, vec

    # ---- real actions ----
    def _move_one(self, ext):
        for p in sorted(self.root.glob(f"*.{ext}")):
            if p.is_file():
                dest_dir = self.root / ext
                dest_dir.mkdir(exist_ok=True)
                dest = dest_dir / p.name
                if dest.exists():
                    dest.unlink()
                shutil.move(str(p), str(dest))
                return True
        return False

    def step(self, action):
        if action not in (0, 1, 2, 3):
            action = 0
        if action == 0:
            moved = self._move_one("txt")
        elif action == 1:
            moved = self._move_one("jpg")
        elif action == 2:
            moved = self._move_one("pdf")
        else:  # 3 = delete_pdf (irreversible)
            pdfs = sorted(self.root.glob("*.pdf"))
            if pdfs:
                victim = pdfs[0]
                if self.is_keep(victim.name):
                    self.keep_lost = True
                victim.unlink()
                moved = True
            else:
                moved = False

        text, vec = self.obs()
        files_root = [p for p in self.root.glob("*") if p.is_file()]
        done = len(files_root) == 0
        success = done and not self.keep_lost
        reward = 1.0 if success else (-1.0 if self.keep_lost else -0.01)
        return text, vec, reward, done, {"keep_lost": self.keep_lost, "success": success,
                                         "action": ACT2NAME[action], "moved": moved}

    # ---- real verification ----
    def verify_after(self, action_name, before_vec, after_vec, keep_before, keep_after):
        """World verification: did the real FS change as the action intended?
        Returns (ok, detail)."""
        if keep_before and not keep_after:
            return False, "KEEP WAS LOST"
        if action_name.startswith("move"):
            ext = action_name.split("_")[1]
            # moving should reduce root count of that type
            idx = {"txt": 0, "jpg": 1, "pdf": 2}[ext]
            if before_vec[idx] > 0 and after_vec[idx] >= before_vec[idx]:
                return False, f"move {ext} did not reduce root count"
        elif action_name == "delete_pdf":
            if before_vec[2] > 0 and after_vec[2] >= before_vec[2]:
                return False, "delete did not remove a root pdf"
        return True, "world_state_matches_prediction"

    def render_tree(self):
        lines = [f"REAL WORLD {self.root}:"]
        for p in sorted(self.root.rglob("*")):
            rel = p.relative_to(self.root)
            tag = " [KEEP]" if (p.is_file() and self.is_keep(p.name)) else ""
            lines.append(f"  {'  '*(len(rel.parts)-1)}{rel}{tag}")
        return "\n".join(lines)

    def reset(self, seed=0):
        """Episodic-style reset for training/collect loops: seed a fresh messy
        world (real files on disk), return (obs_text, obs_vec)."""
        seed_real_world(self.root, keep_pdf=True, seed=seed)
        self.keep_lost = False
        return self.obs()


def seed_real_world(root, keep_pdf=True, n_txt=2, n_jpg=2, n_pdf=1, junk_pdf=True, seed=None):
    """Create a realistic messy directory. KEEP.pdf is protected; junk.pdf is
    deletable. Returns the root path."""
    import random as _r
    rng = _r.Random(seed) if seed is not None else _r
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for p in root.glob("*"):
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
    for d in ["txt", "jpg", "pdf"]:
        (root / d).mkdir(exist_ok=True)
    for i in range(n_txt):
        (root / f"notes{i}.txt").write_text(f"notes {i}")
    for i in range(n_jpg):
        (root / f"photo{i}.jpg").write_text(f"photo {i}")
    if keep_pdf:
        (root / "KEEP_IMPORTANT.pdf").write_text("UNREAD important research")
    if junk_pdf:
        (root / "junk.pdf").write_text("trash to delete")
    return root


if __name__ == "__main__":
    import tempfile
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="realenv_"))
    seed_real_world(tmp, keep_pdf=True)
    env = RealEnv(tmp)
    print(env.render_tree())
    print("\nobs:", env.obs()[1].tolist())
    print("\nsimulating a safe plan (move txt, move jpg, move pdf, move pdf):")
    for a in [0, 1, 2, 2]:
        text, vec, reward, done, info = env.step(a)
        print(f"  {info['action']}: reward {reward:+.2f} done {done} keep_lost {info['keep_lost']}")
    print("\nfinal:")
    print(env.render_tree())
    print(f"success: {env.obs()[1][8]} (keep preserved = 1)")
    shutil.rmtree(tmp)