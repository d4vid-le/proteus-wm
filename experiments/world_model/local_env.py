"""Local computer env - sandboxed file desktop, no cloud, no risk.
World = a folder on YOUR disk. Agent sees file list, acts via move files, verified by after list.
You experience agent like watching GridWorld but with real files."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import os, shutil, pathlib, random, time
import numpy as np

class LocalDesktop:
    """Sandboxed desktop: sandbox"""
    def __init__(self, sandbox="sandbox"):
        self.root = pathlib.Path(sandbox)
        self.root.mkdir(exist_ok=True)
        # goal: Downloads sorted by type
        self.goal_desc = "move .txt to /txt, .jpg to /jpg, .pdf to /pdf"

    def reset(self, seed=0):
        # create messy state: 6 files in root
        for p in self.root.glob("*"):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        for d in ["txt","jpg","pdf"]:
            (self.root/d).mkdir(exist_ok=True)
        # create messy files in root (not sorted)
        files = ["notes.txt","todo.txt","photo1.jpg","photo2.jpg","paper.pdf","doc.pdf"]
        random.seed(seed)
        for f in files:
            (self.root/f).write_text(f"dummy {f}")
        return self.obs()

    def obs(self):
        """Observation = file list as text + as vector for WM."""
        files = sorted([p.name for p in self.root.glob("*") if p.is_file()])
        # also list subfolders
        txt = sorted([p.name for p in (self.root/"txt").glob("*")]) if (self.root/"txt").exists() else []
        jpg = sorted([p.name for p in (self.root/"jpg").glob("*")]) if (self.root/"jpg").exists() else []
        pdf = sorted([p.name for p in (self.root/"pdf").glob("*")]) if (self.root/"pdf").exists() else []
        text = f"ROOT:{files} TXT:{txt} JPG:{jpg} PDF:{pdf}"
        # vector for WM: 6 files * 3 types one-hot = 18 dims, plus 3 folder counts
        # simple: count files in root by ext
        vec = np.zeros(9, dtype=np.float32)
        # root counts
        for f in files:
            if f.endswith(".txt"): vec[0]+=1
            elif f.endswith(".jpg"): vec[1]+=1
            elif f.endswith(".pdf"): vec[2]+=1
        vec[3]=len(txt); vec[4]=len(jpg); vec[5]=len(pdf)
        vec[6]=len(files)  # messy
        # goal is 0 in root, 2 each in folders
        return text, vec

    def step(self, action):
        """Action = dict: {"move": "notes.txt", "to": "txt"} or DSML parsed."""
        # action is string like "move notes.txt to txt" or act id
        # We support act id 0-5 for demo: 0: move first .txt to txt, 1: move .jpg to jpg, 2: move .pdf to pdf, 3: noop
        # For real DSML, parse
        if isinstance(action, int):
            # map int to file type move
            mapping = {0:"txt",1:"jpg",2:"pdf",3:"noop"}
            target = mapping.get(action, "noop")
            if target=="noop":
                pass
            else:
                # move one file of that type from root to folder
                (self.root/target).mkdir(parents=True, exist_ok=True)
                for p in sorted(self.root.glob(f"*.{target}")):
                    if p.is_file():
                        dest=self.root/target/p.name
                        if dest.exists():
                            dest.unlink()
                        shutil.move(str(p), str(dest))
                        break
        elif isinstance(action, dict):
            src = action.get("file"); dst = action.get("to")
            if src and dst:
                src_p = self.root/src
                dst_p = self.root/dst/src
                if src_p.exists():
                    shutil.move(str(src_p), str(dst_p))
        # also support DSML string: parse "notes.txt"
        elif isinstance(action, str) and ".txt" in action or ".jpg" in action or ".pdf" in action:
            # parse filename
            import re
            m=re.search(r'([\w\-]+\.(txt|jpg|pdf))', action)
            if m:
                fname=m.group(1)
                ext=fname.split(".")[-1]
                src=self.root/fname
                if src.exists():
                    shutil.move(str(src), str(self.root/ext/fname))
        text, vec = self.obs()
        # reward: 1 if sorted, -0.01 else
        files_root = [p for p in self.root.glob("*") if p.is_file()]
        done = len(files_root)==0
        reward = 1.0 if done else -0.01
        return text, vec, reward, done, {}

    def render_ascii(self):
        text, _ = self.obs()
        return text

    def render_tree(self):
        out=[]
        out.append(f"SANDBOX {self.root}:")
        for p in sorted(self.root.rglob("*")):
            rel=p.relative_to(self.root)
            prefix="  "*(len(rel.parts)-1)
            out.append(f"{prefix}{rel} {'(dir)' if p.is_dir() else ''}")
        return "\n".join(out)
