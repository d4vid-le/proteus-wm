"""Local demo v2 - file task with 3 types, full 6 files sorted."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import numpy as np, sys
from local_env import LocalDesktop
from wm import WorldModelV3, MLP

# collect with 3-dim pos [txt,jpg,pdf] in root
def collect(episodes=300):
    obs_t, acts, obs_next, pos_next, pos_t = [],[],[],[],[]
    for _ in range(episodes):
        env=LocalDesktop(); _,vec=env.reset(seed=np.random.randint(0,1000))
        for _ in range(7):
            a=np.random.randint(0,3)
            text_before, vec_before=env.obs()
            pos_before=np.array([vec_before[0],vec_before[1],vec_before[2]], dtype=float)
            _, vec_after, _, done,_=env.step(a)
            pos_after=np.array([vec_after[0],vec_after[1],vec_after[2]], dtype=float)
            obs_t.append(vec_before); acts.append(a); obs_next.append(vec_after)
            pos_next.append(pos_after); pos_t.append(pos_before)
            if done: break
    return np.array(obs_t,dtype=np.float32), np.array(acts), np.array(obs_next,dtype=np.float32), np.array(pos_next,dtype=np.float32), np.array(pos_t,dtype=np.float32)

np.random.seed(0)
wm=WorldModelV3(obs_dim=9, act_dim=3, latent_dim=16, hidden=64)
# fix pos to 3 dims
wm.pos_head=MLP([16,64,3], seed=4)
# need target not used for pos, but keep
obs_t,acts,obs_next,pos_next,pos_t=collect(300)
print(f"train {obs_t.shape} pos_head 3 dims")
for epoch in range(300):
    idx=np.random.choice(len(obs_t),128)
    bt,ba,bn,pn,pt=obs_t[idx],acts[idx],obs_next[idx],pos_next[idx],pos_t[idx]
    total,lat,recon,next_recon,pos_loss,cache=wm.loss(bt,ba,bn,pn,pt)
    wm.backward(bt,ba,bn,pn,cache,pos_t=pt)
    if epoch%80==0:
        print(f"epoch {epoch} pos {pos_loss:.3f}")

# patch agent to use 3-dim cost
from agent import Agent
orig_act=Agent.act_mpc
def new_act(self,s,horizon=1):
    n_act=getattr(self.wm,'act_dim',3)
    if hasattr(self.wm,'decode_pos'):
        best=None; bestc=1e9
        for a in range(n_act):
            s_pred,_=self.wm.predict(s,a)
            pos=self.wm.decode_pos(s_pred)
            # cost = sum of root files (want 0)
            cost=float(abs(pos[0])+abs(pos[1])+abs(pos[2]))
            if cost<bestc:
                bestc=cost; best=a
        return best
    return orig_act(self,s,horizon)
Agent.act_mpc=new_act

print("\n=== NOW YOU SEE IT - open Finder to sandbox ===")
env=LocalDesktop(); text,vec=env.reset(seed=42)
print(f"START: {text}")
print(env.render_tree())
for step in range(8):
    s=wm.encode(vec)
    lang,a=Agent(wm).talk(vec, mode="language")
    # make lang more readable for file task
    pos=wm.decode_pos(s)
    pos_next_p=wm.decode_pos(wm.predict(s,a)[0])
    print(f"\nStep {step} AGENT SENSE: files in root txt={int(vec[0])} jpg={int(vec[1])} pdf={int(vec[2])}")
    print(f"  Talk(lang): Agent at {np.round(pos,1).tolist()} will move {['txt','jpg','pdf'][a]} to {np.round(pos_next_p,1).tolist()}  <- YOU VERIFY: count files")
    block,act=Agent(wm).talk(vec, mode="action")
    print(f"  Talk(DSML): move {['*.txt','*.jpg','*.pdf'][act]}")
    s_pred,_=wm.predict(s,act)
    _, vec_next, reward, done,_=env.step(act)
    s_true=wm.encode_target(vec_next)
    mse=float(np.mean((s_pred-s_true)**2))
    print(f"  World->Agent: {env.render_tree().replace(chr(10), ' | ')}  mse {mse:.4f} {'OK' if mse<0.05 else 'drift'}")
    if done:
        print(f"\nDONE in {step+1} steps! All 6 files sorted. You saw each talk verified by folder.")
        break
    vec=vec_next
else:
    print("FAIL - not sorted")
