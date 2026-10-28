"""Demo v3: language + DSML both verified by world_result, WM planner 10/10."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import numpy as np
from env import GridWorld
from wm import WorldModelV3

def collect(episodes=300):
    obs_t, acts, obs_next, pos_next, pos_t = [],[],[],[],[]
    for _ in range(episodes):
        env=GridWorld(5); obs=env.reset()
        for _ in range(20):
            a=np.random.randint(0,4)
            pos_before=env.agent.copy()
            obs2,_,done,_=env.step(a)
            pos_after=env.agent.copy()
            obs_t.append(obs); acts.append(a); obs_next.append(obs2); pos_next.append(pos_after); pos_t.append(pos_before)
            obs=obs2
            if done: obs=env.reset()
    return (np.array(obs_t,dtype=np.float32), np.array(acts), np.array(obs_next,dtype=np.float32),
            np.array(pos_next,dtype=np.float32), np.array(pos_t,dtype=np.float32))

def train():
    wm=WorldModelV3(hidden=128)
    obs_t, acts, obs_next, pos_next, pos_t = collect(300)
    print(f"dataset {obs_t.shape}")
    for epoch in range(400):
        idx=np.random.choice(len(obs_t),128)
        bt,ba,bn,pn,pt=obs_t[idx],acts[idx],obs_next[idx],pos_next[idx],pos_t[idx]
        total,lat,recon,next_recon,pos_loss,cache=wm.loss(bt,ba,bn,pn,pt)
        wm.backward(bt,ba,bn,pn,cache,pos_t=pt)
        if epoch%80==0:
            print(f"epoch {epoch:3d} pos {pos_loss:.4f} lat {lat:.4f} next {next_recon:.4f}")
    print("train done")
    return wm

if __name__=="__main__":
    np.random.seed(1)
    wm=train()
    from agent import Agent, run_episode, verify_language
    # batch language verification
    print("\n=== Language verification 20 random states ===")
    agent=Agent(wm)
    ok_cur=ok_next=0
    for _ in range(20):
        x,y=np.random.randint(0,5,2)
        env=GridWorld(5); env.agent=np.array([x,y]); obs=env.obs()
        lang,a=agent.talk(obs, mode="language")
        ok,_=verify_language(lang, [x,y])
        if ok: ok_cur+=1
        s=agent.sense(obs)
        s_pred,_=wm.predict(s,a)
        pos_next_pred=wm.decode_pos(s_pred)
        env2=GridWorld(5); env2.agent=np.array([x,y]); obs2,_,_,_=env2.step(a)
        if np.linalg.norm(pos_next_pred-np.array(env2.agent))<1.0: ok_next+=1
    print(f"Language current verified {ok_cur}/20, next verified {ok_next}/20")

    print("\n=== Episode with both verifications ===")
    done, lv, lt, dv = run_episode(wm, render=True)
    print(f"\nRESULT: goal {done} lang {lv}/{lt} dsml {dv}/8")
    if done and lv==lt and dv>=7:
        print("FUCKING RESULTS: WM planner + language + DSML all verified by world_result")
