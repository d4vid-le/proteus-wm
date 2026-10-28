"""watcher.py - Autonomous local agent: watches sandbox, auto-sorts, learns."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import time, pathlib, sys
from local_env import LocalDesktop
from wm import WorldModelV3
from wm import MLP
from tools import dispatch_tool
from proteus.bridge.qwen_stub import qwen_generate
from memory import Memory
import numpy as np

SANDBOX=pathlib.Path("sandbox")

class Watcher:
    def __init__(self):
        # WM quick train
        self.wm=WorldModelV3(obs_dim=9, act_dim=3, latent_dim=16, hidden=64)
        self.wm.pos_head=MLP([16,64,3], seed=4)
        # fast train 150 eps
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
        for _ in range(150):
            idx=np.random.choice(len(obs_t),64)
            bt,ba,bn,pn,pt=obs_t[idx],acts[idx],obs_next[idx],pos_next[idx],pos_t[idx]
            total,lat,recon,next_recon,pos_loss,cache=self.wm.loss(bt,ba,bn,pn,pt)
            self.wm.backward(bt,ba,bn,pn,cache,pos_t=pt)
        self.env=LocalDesktop()
        self.memory=Memory()
        print(f"\033[95m watcher \033[0m WM ready | watching {SANDBOX} | Qwen local")

    def decompose(self, user_task):
        """Qwen decomposes natural language task into tool steps."""
        q=user_task.lower()
        # Qwen general knowledge
        know=qwen_generate(user_task)
        print(f"\033[95m Qwen:\033[0m {know[:120]}")
        # decompose
        if "research" in q and "pdf" in q:
            steps=[("grep",'{"pattern":"research"}'), ("list",'{"path":"sandbox"}')]
        elif "sort" in q or "organize" in q:
            steps=[("stat",'{}')]
            # WM will plan moves
            _,vec=self.env.obs()
            s=self.wm.encode(vec)
            # plan 3 steps
            from itertools import product
            best=None; bestc=1e9
            for seq in product(range(3), repeat=2):
                cur=s
                for a in seq:
                    cur,_=self.wm.predict(cur,a)
                pos=self.wm.decode_pos(cur)
                cost=float(abs(pos[0])+abs(pos[1])+abs(pos[2]))
                if cost<bestc: bestc=cost; best=seq[0]
            names=["txt","jpg","pdf"]
            steps.append(("bash",f'{{"command":"ls sandbox/*.{names[best]} 2>&1"}}'))
        elif "find" in q:
            steps=[("find",'{"pattern":"*.pdf"}'), ("grep",'{"pattern":"deepseek"}')]
        else:
            steps=[("list",'{"path":"sandbox"}'), ("qwen",f'{{"prompt":"{user_task}"}}')]
        return steps

    def run_once(self, user_task="organize research papers"):
        print(f"\n{'='*60}\nTask: {user_task}\n{'='*60}")
        steps=self.decompose(user_task)
        print(f"Plan: {len(steps)} steps via Qwen + WM")
        for i,(tool,args) in enumerate(steps,1):
            # encode DSML
            from tools import encode_tool_dsml
            dsml=encode_tool_dsml(tool, eval(args) if isinstance(args,str) and args.startswith('{') else {"prompt":args} if tool=="qwen" else {})
            print(f"\nStep {i} DSML: {tool} {args[:60]}")
            # predict via WM if file-related
            try:
                _,vec=self.env.obs()
                s=self.wm.encode(vec)
                # simple verification: before vec sum
                before_sum=float(vec[0]+vec[1]+vec[2])
            except: before_sum=0
            res=dispatch_tool(tool, args)
            verified=res["ok"]
            # verify via world_result
            _,vec_after=self.env.obs()
            after_sum=float(vec_after[0]+vec_after[1]+vec_after[2]) if len(vec_after)>=3 else 0
            print(f" World: {str(res['output'])[:150]} | verified {verified} | files {before_sum}->{after_sum}")
            # save to memory (coupled)
            self.memory.add(user_task, tool, args, res, verified=verified, lang=f"step {i} {tool}")
            if self.memory.should_learn():
                print(f" Memory {self.memory.stats()} -> learning (coupled)")
                r=self.memory.learn(self.wm)
                print(f" Learn: {r}")
            # if sort task, also do actual WM-driven moves
            if "sort" in user_task.lower() and i==len(steps):
                # do full sort via WM
                for _ in range(6):
                    _,vec=self.env.obs()
                    if vec[0]==0 and vec[1]==0 and vec[2]==0: break
                    s=self.wm.encode(vec)
                    best=None; bestc=1e9
                    for a in range(3):
                        s_pred,_=self.wm.predict(s,a)
                        pos=self.wm.decode_pos(s_pred)
                        cost=float(abs(pos[0])+abs(pos[1])+abs(pos[2]))
                        if cost<bestc: bestc=cost; best=a
                    self.env.step(best)
                print(f" WM sorted: {self.env.render_tree().replace(chr(10),' | ')}")

    def watch(self, interval=2, max_cycles=3):
        """Autonomous: poll sandbox, auto-act when messy."""
        print(f"\nWatcher polling {SANDBOX} every {interval}s, {max_cycles} cycles...")
        for cycle in range(max_cycles):
            _,vec=self.env.obs()
            messy=float(vec[0]+vec[1]+vec[2])
            print(f"\nCycle {cycle+1}: messy files in root = {int(messy)} {vec[:3].tolist()}")
            if messy>0:
                print(" -> messy detected, auto-sorting via WM + Qwen...")
                self.run_once("sort")
            else:
                print(" -> clean, no action, learning from idle")
            # simulate new file drop
            if cycle==1:
                # user drops new file
                import pathlib
                (SANDBOX/"new_notes.txt").write_text("new")
                print(" (user dropped new_notes.txt)")
            time.sleep(interval)
        print("\nWatcher done. Memory:", self.memory.stats())

if __name__=="__main__":
    w=Watcher()
    # demo decompose
    w.run_once("find all pdfs with deepseek")
    w.run_once("organize research papers")
    w.watch(interval=1, max_cycles=3)
