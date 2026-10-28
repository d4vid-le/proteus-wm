#!/usr/bin/env python3
"""agent_scalable.py - Reliable local agent with bash tool calling + memory + learning coupled."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import sys
import json, re
from tools import TOOL_SCHEMAS, dispatch_tool, encode_tool_dsml, decode_tool_dsml, safe_bash
from memory import Memory
from local_env import LocalDesktop
import numpy as np
from wm import WorldModelV3
from wm import MLP

SYSTEM = """You are proteus, local agent on this Mac. You have tools: bash, list, read, write, stat.
You are reliable: every tool call is DSML, verified by world_result, saved to memory, and memory feeds learning.
You talk like Qwen in terminal - warm, concise, personable. You explain what you will do, then do it via tool, then verify."""

class ScalableAgent:
    def __init__(self):
        # WM for file task (9 obs, 3 acts) - same as before, but now coupled with memory
        self.wm=WorldModelV3(obs_dim=9, act_dim=3, latent_dim=16, hidden=64)
        self.wm.pos_head=MLP([16,64,3], seed=4)
        self._quick_train()
        self.memory=Memory()
        self.env=LocalDesktop()
        print(f"\033[95m proteus scalable \033[0m | tools {len(TOOL_SCHEMAS)} | memory {self.memory.stats()} | WM {self.wm.__class__.__name__}")

    def _quick_train(self, episodes=200):
        obs_t,acts,obs_next,pos_next,pos_t=[],[],[],[],[]
        for _ in range(episodes):
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
        for _ in range(200):
            idx=np.random.choice(len(obs_t),64)
            bt,ba,bn,pn,pt=obs_t[idx],acts[idx],obs_next[idx],pos_next[idx],pos_t[idx]
            total,lat,recon,next_recon,pos_loss,cache=self.wm.loss(bt,ba,bn,pn,pt)
            self.wm.backward(bt,ba,bn,pn,cache,pos_t=pt)

    def think(self, user_query):
        """Simple planner: retrieve memory, decide tool."""
        # 1. retrieve relevant memories
        try:
            mems=self.memory.retrieve(user_query, k=2)
        except: mems=[]
        mem_hint=""
        if mems:
            m0=mems[0]
            mem_hint=f" (memory: last similar '{m0.get('user','')}' -> {m0.get('tool', m0.get('assistant',''))} verified)"
        # 2. decide tool based on query keywords + WM
        q=user_query.lower()
        if "list" in q or "files" in q or "what is in" in q:
            return "list", {"path":"sandbox"}, f"I'll list sandbox for you{mem_hint} ✨"
        elif "read" in q:
            m=re.search(r'read\s+(\S+)', q)
            fname=m.group(1) if m else "notes.txt"
            return "read", {"path":f"sandbox/{fname}"}, f"I'll read {fname}{mem_hint} 📖"
        elif "write" in q or "create" in q:
            m=re.search(r'write\s+(\S+)\s+(.*)', q)
            if m:
                return "write", {"path":f"sandbox/{m.group(1)}","content":m.group(2)}, f"I'll create {m.group(1)}{mem_hint} ✏️"
            else:
                return "write", {"path":"sandbox/hello.txt","content":"hello from proteus"}, f"I'll create hello.txt{mem_hint}"
        elif "bash" in q or "run" in q or "execute" in q:
            m=re.search(r'bash\s+(.*)', q) or re.search(r'run\s+(.*)', q)
            cmd=m.group(1) if m else "ls -la"
            return "bash", {"command":cmd}, f"I'll run bash: {cmd}{mem_hint} 💻"
        elif "sort" in q or "organize" in q or "tidy" in q:
            # use WM planner: sense then act
            _,vec=self.env.obs()
            s=self.wm.encode(vec)
            a=self._wm_act(s)
            names=["txt","jpg","pdf"]
            return "bash", {"command":f"ls sandbox/*.{names[a]} 2>&1 | head"}, f"I'll tidy by moving *.{names[a]}{mem_hint} 🧹"
        else:
            # general chat - use Qwen persona, but still ground via stat
            _,vec=self.env.obs()
            return None, {}, f"hey! I'm proteus 🌸 sandbox has {int(vec[0])} txt, {int(vec[1])} jpg, {int(vec[2])} pdf. I have tools bash/list/read/write/stat. Try 'list files', 'bash ls', 'sort', or 'read notes.txt' - I'm local and I learn from every verified tool call (memory {self.memory.stats()['verified']} verified)."

    def _wm_act(self, s):
        best=None; bestc=1e9
        for a in range(self.wm.act_dim):
            s_pred,_=self.wm.predict(s,a)
            pos=self.wm.decode_pos(s_pred)
            cost=float(abs(pos[0])+abs(pos[1])+abs(pos[2]))
            if cost<bestc:
                bestc=cost; best=a
        return best

    def step(self, user_query):
        """One turn: think -> DSML -> tool -> world_result -> verify -> memory -> learn (coupled)."""
        tool, args, lang = self.think(user_query)
        print(f"\n\033[95m proteus:\033[0m {lang}")
        if tool is None:
            # pure chat, no tool
            self.memory.add(user_query, "chat", {}, {"output": lang}, verified=False, lang=lang)
            return lang
        # encode DSML
        dsml=encode_tool_dsml(tool, args)
        print(f"\033[90m DSML: {dsml[:100]}...\033[0m")
        # predict via WM (for verification)
        try:
            _,vec=self.env.obs()
            s=self.wm.encode(vec)
            # for bash/stat we can predict file count delta
            if tool in ["bash","list","stat"]:
                # simple verification: after tool, vec should change predictably
                # we use WM to predict next pos if it's a sort action
                if "sort" in user_query.lower() or "organize" in user_query.lower():
                    a=self._wm_act(s)
                    s_pred,_=self.wm.predict(s,a)
                    pos_pred=self.wm.decode_pos(s_pred)
                else:
                    pos_pred=None
            else:
                pos_pred=None
        except: pos_pred=None
        # dispatch tool (real local bash)
        result=dispatch_tool(tool, json.dumps(args))
        verified=result.get("ok", False)
        # additional WM verification for sort
        if pos_pred is not None and tool=="bash" and "sort" in user_query.lower():
            # after tool, get true vec and compare
            _,vec_after=self.env.obs()
            true_pos=np.array([vec_after[0],vec_after[1],vec_after[2]])
            if pos_pred is not None:
                err=np.linalg.norm(pos_pred-true_next) if (true_next:=true_pos) is not None else 1e9
                verified=verified and err<1.0
        print(f"\033[92m World->Agent: {str(result['output'])[:300]} | verified {verified}\033[0m")
        # save to memory (coupled)
        self.memory.add(user_query, tool, args, result, verified=verified, lang=lang)
        # coupled learning: memory growth triggers WM update
        if self.memory.should_learn():
            print(f"\033[93m Memory {self.memory.stats()} -> learning (coupled)...\033[0m")
            res=self.memory.learn(self.wm)
            print(f"\033[93m Learn: {res}\033[0m")
        # also handle sort via env.step for file moves (if bash was ls, we also do actual sort via env)
        if "sort" in user_query.lower() and tool=="bash":
            # actually perform sort via LocalDesktop env for demo
            for _ in range(6):
                _,vec=self.env.obs()
                if vec[0]==0 and vec[1]==0 and vec[2]==0:
                    break
                s=self.wm.encode(vec)
                a=self._wm_act(s)
                self.env.step(a)
            print(f"\033[92m Sorted! Now: {self.env.render_tree().replace(chr(10),' | ')}\033[0m")
        return result["output"]

if __name__=="__main__":
    agent=ScalableAgent()
    print("\nTry: 'list files', 'bash ls -la', 'read notes.txt', 'write hello.txt hi', 'sort', 'learn', 'bye'")
    while True:
        try:
            q=input("\n\033[96m you:\033[0m ")
            if q.strip().lower() in ["bye","exit","quit"]:
                print("bye! memory at", agent.memory.path)
                break
            out=agent.step(q)
            if out and len(str(out))<500 and "list" not in q.lower():
                print(f"\n\033[95m proteus:\033[0m {str(out)[:500]}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            import traceback; traceback.print_exc()
