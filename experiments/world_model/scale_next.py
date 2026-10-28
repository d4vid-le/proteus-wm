"""Scale next: 1000 episodes, hidden 128->256, horizon 3 + next_recon + pos, test emergent 6-step."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import numpy as np, sys
from local_env import LocalDesktop
from wm import WorldModelV3
from wm import MLP

# We will reuse WorldModelV3 but train with horizon 3 by augmenting loss
# Instead of new WM, we train v3 for longer and test horizon 3 MPC

def collect(episodes=1000):
    obs_t,acts,obs_next,pos_next,pos_t=[],[],[],[],[]
    for _ in range(episodes):
        env=LocalDesktop(); _,vec=env.reset(seed=np.random.randint(0,10000))
        for _ in range(7):
            a=np.random.randint(0,3)
            vb=vec.copy(); pb=np.array([vb[0],vb[1],vb[2]])
            _,va,_,done,_=env.step(a)
            pa=np.array([va[0],va[1],va[2]])
            obs_t.append(vb); acts.append(a); obs_next.append(va); pos_next.append(pa); pos_t.append(pb)
            vec=va
            if done: break
    return np.array(obs_t,dtype=np.float32), np.array(acts), np.array(obs_next,dtype=np.float32), np.array(pos_next,dtype=np.float32), np.array(pos_t,dtype=np.float32)

def train(hidden, episodes=1000, epochs=400):
    obs_t,acts,obs_next,pos_next,pos_t=collect(episodes)
    wm=WorldModelV3(obs_dim=9, act_dim=3, latent_dim=16, hidden=hidden)
    wm.pos_head=MLP([16,hidden,3], seed=4)
    print(f"train hidden {hidden} data {len(obs_t)}")
    for epoch in range(epochs):
        idx=np.random.choice(len(obs_t),128)
        bt,ba,bn,pn,pt=obs_t[idx],acts[idx],obs_next[idx],pos_next[idx],pos_t[idx]
        total,lat,recon,next_recon,pos_loss,cache=wm.loss(bt,ba,bn,pn,pt)
        wm.backward(bt,ba,bn,pn,cache,pos_t=pt)
        if epoch%100==0:
            print(f" epoch {epoch} pos {pos_loss:.4f} lat {lat:.4f} next {next_recon:.4f}")
    return wm

# Train two scales
for hidden in [128,256]:
    print(f"\n{'='*50}\nTRAIN hidden={hidden} 1000 eps 400 epochs\n{'='*50}")
    wm=train(hidden, episodes=1000, epochs=400)
    # test 6-step optimal with horizon 1 vs horizon 3
    from itertools import product
    def mpc_h1(vec):
        s=wm.encode(vec)
        best=None; bestc=1e9
        for a in range(3):
            s_pred,_=wm.predict(s,a)
            pos=wm.decode_pos(s_pred)
            cost=float(abs(pos[0])+abs(pos[1])+abs(pos[2]))
            if cost<bestc: bestc=cost; best=a
        return best
    def mpc_h3(vec):
        s=wm.encode(vec)
        best=None; bestc=1e9
        for seq in product(range(3), repeat=3):
            cur=s
            for a in seq:
                cur,_=wm.predict(cur,a)
            pos=wm.decode_pos(cur)
            cost=float(abs(pos[0])+abs(pos[1])+abs(pos[2]))
            if cost<bestc: bestc=cost; best=seq[0]
        return best
    for name, mpc in [("h1", mpc_h1), ("h3", mpc_h3)]:
        env=LocalDesktop(); _,vec=env.reset(seed=42)
        steps=0
        for _ in range(10):
            a=mpc(vec)
            _,vec,_,done,_=env.step(a)
            steps+=1
            if done: break
        print(f"hidden {hidden} {name} steps {steps} {'6/6' if steps==6 else 'not yet'}")
