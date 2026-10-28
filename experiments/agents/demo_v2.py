import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import numpy as np
from env import GridWorld
from wm_v2 import WorldModelV2

def collect(episodes=100):
    obs_t, acts, obs_next = [],[],[]
    for _ in range(episodes):
        env=GridWorld(5)
        obs=env.reset()
        for _ in range(20):
            a=np.random.randint(0,4)
            obs2,_,done,_=env.step(a)
            obs_t.append(obs); acts.append(a); obs_next.append(obs2)
            obs=obs2
            if done: obs=env.reset()
    return np.array(obs_t,dtype=np.float32), np.array(acts), np.array(obs_next,dtype=np.float32)

def train():
    wm=WorldModelV2(hidden=128)
    obs_t, acts, obs_next = collect(200)
    print(f"dataset {obs_t.shape}")
    B=128
    for epoch in range(500):
        idx=np.random.choice(len(obs_t),B)
        bt,ba,bn=obs_t[idx],acts[idx],obs_next[idx]
        total,lat,recon,cache=wm.loss(bt,ba,bn)
        wm.backward(bt,ba,bn,cache,lr=3e-4)
        if epoch%50==0:
            print(f"epoch {epoch:3d} total={total:.4f} latent={lat:.4f} recon={recon:.4f} var={cache['s_pred'].std():.3f}")
    bt,ba,bn=obs_t[:256],acts[:256],obs_next[:256]
    total,lat,recon,_=wm.loss(bt,ba,bn)
    print(f"eval total={total:.4f} latent={lat:.4f} recon={recon:.4f}")
    s=wm.encode(bt[0])
    s_pred,_=wm.predict(s,int(ba[0]))
    s_true=wm.encode_target(bn[0])
    print(f"single-step MSE {np.mean((s_pred-s_true)**2):.5f} (good <0.05)")
    return wm

if __name__=="__main__":
    np.random.seed(0)
    wm=train()
    from agent import Agent
    # patch agent to use v2 wm
    env=GridWorld(5)
    agent=Agent(wm)  # Agent expects encode/predict/decode same API, works with v2 (encode_target not needed)
    # monkey patch encode to use v2 encode
    # test rollout
    for ep in range(3):
        obs=env.reset()
        print(f"\nEp {ep} start")
        for t in range(20):
            s=wm.encode(obs)
            a=agent.act_mpc(s)
            obs,_,done,_=env.step(a)
            print(env.render_ascii(),"->",a)
            print("---")
            if done: print("done"); break
