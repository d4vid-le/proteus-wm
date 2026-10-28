"""vector_memory.py - Persistent vector memory, coupled with learning.
Simple hashing embedding, no external DB, local."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import json, pathlib, hashlib, numpy as np

class VectorMemory:
    def __init__(self, path="vector_mem.jsonl", dim=32):
        self.path=pathlib.Path(path)
        self.dim=dim
        self.items=[]
        self.load()
    def _embed(self, text):
        # simple hash embedding: 32-dim, deterministic, local, no API
        h=hashlib.sha256(text.encode()).digest()
        # expand to dim
        arr=np.frombuffer(h, dtype=np.uint8).astype(float)
        # repeat to dim
        vec=np.tile(arr, (self.dim//len(arr)+1))[:self.dim]
        # normalize
        vec=(vec-128)/128
        return vec/np.linalg.norm(vec)
    def add(self, text, meta=None):
        vec=self._embed(text)
        rec={"text":text, "vec":vec.tolist(), "meta":meta or {}}
        self.items.append(rec)
        with self.path.open("a") as f:
            f.write(json.dumps(rec)+"\n")
        return rec
    def load(self):
        self.items=[]
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    try:
                        rec=json.loads(line)
                        rec["vec"]=np.array(rec["vec"])
                        self.items.append(rec)
                    except: pass
        # convert vecs
        for rec in self.items:
            if isinstance(rec["vec"], list):
                rec["vec"]=np.array(rec["vec"])
    def search(self, query, k=3):
        qvec=self._embed(query)
        scored=[]
        for rec in self.items:
            sim=float(np.dot(qvec, rec["vec"]))
            scored.append((sim, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for sim,rec in scored[:k]]
    def stats(self):
        return {"count": len(self.items), "dim": self.dim}
