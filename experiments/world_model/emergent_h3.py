import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import numpy as np, sys
from local_env import LocalDesktop
from wm_h3 import WorldModelH3
from itertools import product

def collect_h3(episodes=1000):
    obs_t=[]; acts=[]; obs_t3=[]; pos_t3=[]
    for _ in range(episodes):
        env=LocalDesktop(); _,vec=env.reset(seed=np.random.randint(0,1000))
        traj=[vec]
        acts_traj=[]
        for _ in range(6):
            a=np.random.randint(0,3)
            _,vec_next,_,done,_=env.step(a)
            traj.append(vec_next); acts_traj.append(a)
            if done: break
        # sample h3 windows
        for i in range(len(traj)-3):
            obs_t.append(traj[i])
            acts.append(acts_traj[i:i+3])
            obs_t3.append(traj[i+3])
            # pos is first 3 dims of vec
            pos_t3.append(traj[i+3][:3])
    return np.array(obs_t,dtype=np.float32), np.array(acts), np.array(obs_t3,dtype=np.float32), np.array(pos_t3,dtype=np.float32)

def train(hidden, episodes=1000, epochs=300):
    obs_t,acts,obs_t3,pos_t3=collect_h3(episodes)
    print(f"hidden {hidden} data {len(obs_t)}")
    wm=WorldModelH3(hidden=hidden)
    for epoch in range(epochs):
        idx=np.random.choice(len(obs_t),128)
        bt,ba,bt3,pt3=obs_t[idx],acts[idx],obs_t3[idx],pos_t3[idx]
        total,lat,pos,cache=wm.loss_h3(bt,ba,bt3,pt3)
        wm.backward_h3(pt3,cache)
        if epoch%80==0:
            print(f" epoch {epoch} lat {lat:.4f} pos {pos:.4f}")
    return wm

# quick test with horizon 1 vs 3
for hidden in [32,128,256]:
    print(f"\n=== Train hidden={hidden} h3 ===")
    wm=train(hidden, episodes=800, epochs=250)
    # test 6-step optimal via horizon 3 MPC
    env=LocalDesktop(); _,vec=env.reset(seed=42)
    steps=0
    for _ in range(10):
        s=wm.encode(vec)
        # horizon 3 MPC: score each 3-seq by predicted final pos cost
        best=None; bestc=1e9
        for seq in product(range(3), repeat=3):
            cur=s
            for a in seq:
                cur,_=wm.predict(cur,a)
            pos=wm.decode_pos(cur)
            cost=float(abs(pos[0])+abs(pos[1])+abs(pos[2]))
            if cost<bestc:
                bestc=cost; best=seq[0]
        _,vec,_,done,_=env.step(best)
        steps+=1
        if done: break
    print(f"hidden {hidden} steps {steps} {'6/6 emergent' if steps==6 else 'fail'}")
