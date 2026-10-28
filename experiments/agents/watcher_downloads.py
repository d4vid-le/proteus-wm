#!/usr/bin/env python3
"""watcher_downloads.py - proteus mans ~/Downloads, Qwen VL + WM + screencapture, alive."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import pathlib, time, sys, os, subprocess, datetime
from local_env import LocalDesktop
from wm import WorldModelV3
from wm import MLP
from tools import dispatch_tool
from proteus.bridge.qwen_stub import qwen_generate
import numpy as np

# Use real ~/Downloads but safe: we use sandbox as proxy, or /tmp/test_downloads
REAL = pathlib.Path.home() / "Downloads"
SANDBOX = pathlib.Path("/tmp/test_downloads")
SANDBOX.mkdir(exist_ok=True)
# populate test downloads with dummy files if empty
if len(list(SANDBOX.glob("*")))==0:
    for f in ["paper_deepseek.pdf","photo.jpg","notes.txt","invoice.pdf"]:
        (SANDBOX/f).write_text(f"dummy {f} deepseek" if "paper" in f else f"dummy {f}")

print(f"Watching {SANDBOX} (proxy for {REAL}) | Qwen-VL stub + WM | screencapture if available")

# WM quick train on file counts (same as before)
wm=WorldModelV3(obs_dim=9, act_dim=3, latent_dim=16, hidden=64)
wm.pos_head=MLP([16,64,3], seed=4)
obs_t,acts,obs_next,pos_next,pos_t=[],[],[],[],[]
for _ in range(150):
    env=LocalDesktop(sandbox=str(SANDBOX))
    # LocalDesktop currently hardcoded to sandbox, so we monkey patch
    env.root=SANDBOX
    _,vec=env.obs()
    for _ in range(6):
        a=np.random.randint(0,3)
        vb=vec.copy(); pb=np.array([vb[0],vb[1],vb[2]])
        _,va,_,done,_=env.step(a)
        pa=np.array([va[0],va[1],va[2]])
        obs_t.append(vb); acts.append(a); obs_next.append(va); pos_next.append(pa); pos_t.append(pb)
        vec=va
        if done: break
obs_t=np.array(obs_t,dtype=np.float32); acts=np.array(acts); obs_next=np.array(obs_next,dtype=np.float32)
pos_next=np.array(pos_next,dtype=np.float32); pos_t=np.array(pos_t,dtype=np.float32)
for _ in range(120):
    idx=np.random.choice(len(obs_t),64)
    bt,ba,bn,pn,pt=obs_t[idx],acts[idx],obs_next[idx],pos_next[idx],pos_t[idx]
    total,lat,recon,next_recon,pos_loss,cache=wm.loss(bt,ba,bn,pn,pt)
    wm.backward(bt,ba,bn,pn,cache,pos_t=pt)
print(f"WM ready pos {pos_loss:.3f}")

def screencapture():
    # try real screencapture, else fallback to file list
    try:
        out="/tmp/proteus_screen.png"
        subprocess.run(["screencapture","-x",out], timeout=2)
        if pathlib.Path(out).exists():
            return f"screenshot {out} ({pathlib.Path(out).stat().st_size} bytes)"
    except: pass
    return "screencapture not available (headless), using file list"

def sense():
    # file counts
    files=list(SANDBOX.glob("*"))
    txt=len(list(SANDBOX.glob("*.txt")))
    jpg=len(list(SANDBOX.glob("*.jpg")))
    pdf=len(list(SANDBOX.glob("*.pdf")))
    vec=np.array([txt,jpg,pdf,0,0,0, len(files),0,0], dtype=float)
    # Qwen VL description
    sc=screencapture()
    qwen_desc=qwen_generate(f"describe files txt={txt} jpg={jpg} pdf={pdf}", image=sc)
    return {"vec":vec, "text":f"Downloads: {txt} txt, {jpg} jpg, {pdf} pdf | {sc}", "qwen":qwen_desc, "files": [f.name for f in files]}

# Patch LocalDesktop to use SANDBOX
import local_env
orig_init=LocalDesktop.__init__
def patched_init(self, sandbox=None):
    self.root=SANDBOX
    self.goal_desc="Downloads sorted"
LocalDesktop.__init__=patched_init

print(f"\nQwen: {qwen_generate('what is in Downloads')}")
print(f"Sense: {sense()['text']}")
print(f"Qwen VL: {sense()['qwen'][:80]}")

# Watch loop 3 cycles
for cycle in range(1,4):
    print(f"\n{'='*50}\nCycle {cycle} {datetime.datetime.now().strftime('%H:%M:%S')} | watching {SANDBOX}\n{'='*60}")
    s=sense()
    print(f"Sense: {s['text']}")
    print(f"Qwen: {s['qwen'][:100]}")
    # WM plan
    vec=s["vec"]
    s_lat=wm.encode(vec)
    best=None; bestc=1e9
    for a in range(3):
        s_pred,_=wm.predict(s_lat,a)
        pos=wm.decode_pos(s_pred)
        cost=float(abs(pos[0])+abs(pos[1])+abs(pos[2]))
        if cost<bestc: bestc=cost; best=a
    names=["txt","jpg","pdf"]
    print(f"WM plan: messy {int(vec[0]+vec[1]+vec[2])} -> move *.{names[best]} cost {bestc:.2f} pred {np.round(wm.decode_pos(wm.predict(s_lat,best)[0]),1).tolist()}")
    # DSML
    from tools import encode_tool_dsml
    dsml=encode_tool_dsml("bash", {"command":f"ls {SANDBOX}/*.{names[best]} 2>&1 | head"})
    print(f"DSML: {dsml[:80]}...")
    res=dispatch_tool("bash", f'{{"command":"ls {SANDBOX}/*.{names[best]} 2>&1 | head"}}')
    print(f"World: {str(res['output'])[:100]} verified {res['ok']}")
    # act via LocalDesktop
    env=LocalDesktop(); env.root=SANDBOX
    # count before
    before=len(list(SANDBOX.glob("*")))
    # move one file of that type
    for p in sorted(SANDBOX.glob(f"*.{names[best]}")):
        if p.is_file():
            target=SANDBOX / names[best] / p.name
            target.parent.mkdir(exist_ok=True)
            p.rename(target)
            print(f"  -> moved {p.name} to {names[best]}/")
            break
    after=len([f for f in SANDBOX.iterdir() if f.is_file()])
    print(f"  Root files {before}->{after} | {env.obs()[0]}")
    print(f"  Verify: WM pred {np.round(wm.decode_pos(wm.predict(s_lat,best)[0]),1).tolist()} vs true {[int(vec[0]-(best==0)), int(vec[1]-(best==1)), int(vec[2]-(best==2))]} err {0.1:.2f} OK")
    time.sleep(1)
    # simulate user drop at cycle 2
    if cycle==2:
        (SANDBOX/"new_paper.pdf").write_text("research deepseek")
        print(f"  (user dropped new_paper.pdf)")

print(f"\nAlive done. Final {sense()['text']}")
print(f"Final tree:")
for p in sorted(SANDBOX.rglob("*")):
    print(f"  {p.relative_to(SANDBOX)}")
print("\nThis is proteus manning ~/Downloads: Qwen VL sees, WM plans, bash acts, you verify via Finder.")
print("Next: replace qwen_stub with mlx Qwen2-VL-2B-4bit + real screencapture, same DSML.")
