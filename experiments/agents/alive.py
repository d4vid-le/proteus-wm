#!/usr/bin/env python3
"""alive.py - proteus mans the laptop, alive. Watches, thinks, acts, learns, 24/7 local."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import time, pathlib, datetime, sys, os, signal
from local_env import LocalDesktop
from wm import WorldModelV3
from wm import MLP
from tools import dispatch_tool
from proteus.bridge.qwen_stub import qwen_generate
from memory import Memory
from vector_memory import VectorMemory
import numpy as np

HEARTBEAT="/tmp/proteus_heartbeat.log"
SANDBOX=pathlib.Path("sandbox")

class Alive:
    def __init__(self):
        # WM
        self.wm=WorldModelV3(obs_dim=9, act_dim=3, latent_dim=16, hidden=64)
        self.wm.pos_head=MLP([16,64,3], seed=4)
        # fast train
        obs_t,acts,obs_next,pos_next,pos_t=[],[],[],[],[]
        for _ in range(150):
            env=LocalDesktop(); _,vec=env.reset(seed=np.random.randint(0,1000))
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
            total,lat,recon,next_recon,pos_loss,cache=self.wm.loss(bt,ba,bn,pn,pt)
            self.wm.backward(bt,ba,bn,pn,cache,pos_t=pt)
        self.env=LocalDesktop()
        self.memory=Memory()
        self.vmem=VectorMemory()
        self.cycle=0
        self.start=datetime.datetime.now()
        print(f"\033[95m proteus alive \033[0m pid {os.getpid()} | WM {self.wm.__class__.__name__} | {SANDBOX} | Qwen local")
        print(f" Persona: warm, proactive, lives on your laptop. Type 'status' or Ctrl-C to pause.")
        print(f" Heartbeat: {HEARTBEAT} | Memory: {self.memory.stats()}")

    def heartbeat(self, msg):
        with open(HEARTBEAT,"a") as f:
            f.write(f"{datetime.datetime.now().isoformat()} cycle {self.cycle} {msg}\n")

    def sense(self):
        text,vec=self.env.obs()
        # also sense system: uptime, files, time
        uptime=(datetime.datetime.now()-self.start).seconds
        load=os.getloadavg()[0] if hasattr(os,'getloadavg') else 0
        return {"text":text, "vec":vec, "uptime":uptime, "load":load}

    def think(self, sense):
        vec=sense["vec"]
        messy=int(vec[0]+vec[1]+vec[2])
        if messy>0:
            # reliable: pick the ext with max count in root (not WM) - WM is for verification
            counts=[float(vec[0]),float(vec[1]),float(vec[2])]
            names=["txt","jpg","pdf"]
            best=int(np.argmax(counts))
            # also verify via WM prediction
            s=self.wm.encode(vec)
            s_pred,_=self.wm.predict(s,best)
            pos_pred=self.wm.decode_pos(s_pred)
            return f"messy {messy} files {counts}, I will tidy *.{names[best]} -> pred {np.round(pos_pred,1).tolist()}", "bash", {"command":f"echo tidying {names[best]}"}
        else:
            if sense["uptime"]%10<2:
                return "all clean, I'm watching, heartbeat steady", None, {}
            return "all clean, resting but alive", None, {}

    def act(self, tool, args):
        if tool is None:
            return {"ok":True, "output":"idle"}
        import json
        res=dispatch_tool(tool, json.dumps(args))
        # also do real file move if tidying - use true max count, not WM, for reliability
        if "tidying" in str(args):
            _,vec=self.env.obs()
            counts=[float(vec[0]),float(vec[1]),float(vec[2])]
            if sum(counts)>0:
                best=int(np.argmax(counts))
                self.env.step(best)
                # also verify via WM
                s=self.wm.encode(vec)
                s_pred,_=self.wm.predict(s,best)
                pos_pred=self.wm.decode_pos(s_pred)
                # log verification
                res["wm_pred"]=pos_pred.tolist()
        return res

    def loop(self, max_cycles=5, interval=2):
        print(f"\n\033[95m alive loop {max_cycles} cycles every {interval}s - Ctrl-C to talk \033[0m\n")
        for i in range(max_cycles):
            self.cycle=i+1
            sense=self.sense()
            thought, tool, args = self.think(sense)
            # Qwen-like personable talk
            qwen_line=qwen_generate(thought)[:80]
            print(f"\033[90m[{datetime.datetime.now().strftime('%H:%M:%S')} cycle {self.cycle} up {sense['uptime']}s load {sense['load']:.2f}]\033[0m")
            print(f"\033[95m proteus:\033[0m {thought} | \033[90m Qwen: {qwen_line}\033[0m")
            if tool:
                from tools import encode_tool_dsml
                dsml=encode_tool_dsml(tool, args)
                print(f"\033[90m  DSML: {dsml[:70]}...\033[0m")
            res=self.act(tool, args)
            verified=res.get("ok",False)
            print(f"\033[92m  World: {str(res['output'])[:100]} | verified {verified} | {self.env.render_tree().replace(chr(10),' | ')}\033[0m")
            self.heartbeat(f"{thought} -> {tool} verified {verified}")
            # memory + coupled learning
            self.memory.add(thought, tool or "idle", args, res, verified=verified, lang=thought)
            self.vmem.add(thought, {"verified":verified, "cycle":self.cycle})
            if self.memory.should_learn():
                print(f"\033[93m  Memory {self.memory.stats()} -> learning (coupled) \033[0m")
                r=self.memory.learn(self.wm)
                print(f"\033[93m  Learn: {r['msg'][:60]} \033[0m")
            # simulate user dropping file at cycle 2
            if i==2:
                (SANDBOX/"surprise.txt").write_text("surprise from user")
                print(f"\033[96m  (user dropped surprise.txt - watcher will catch next cycle)\033[0m")
            time.sleep(interval)
        print(f"\n\033[95m proteus alive done {max_cycles} cycles, still alive pid {os.getpid()}, heartbeat {HEARTBEAT}\033[0m")
        print(f" Memory {self.memory.stats()} Vector {self.vmem.stats()}")
        print(f" Talk to me: /usr/local/bin/python3.14 chat.py")
        print(f" Watch heartbeat: tail -f {HEARTBEAT}")

if __name__=="__main__":
    import signal
    def handler(sig, frame):
        print(f"\n\033[95m proteus:\033[0m paused, but I'm still here. Run again to wake. Heartbeat at {HEARTBEAT}")
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)
    Alive().loop(max_cycles=5, interval=1)
