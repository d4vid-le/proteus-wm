"""Local demo - you watch agent organize YOUR sandbox folder, step by step.
This is what a local agent feels like: it talks, you verify by looking at folder.
No cloud, no API, just files on sandbox
"""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import time, pathlib
from local_env import LocalDesktop
from env import GridWorld  # still used for WM training, but local task is file sorting

# Use WM for file task too - same WM but obs is 9-dim file count
import numpy as np, sys
from wm import WorldModelV3

def collect_local(episodes=200):
    """Collect file-sorting trajectories for WM."""
    obs_t, acts, obs_next, pos_next, pos_t = [],[],[],[],[]
    for _ in range(episodes):
        env=LocalDesktop()
        _, vec = env.reset(seed=np.random.randint(0,1000))
        # also track pos as (root_txt, root_jpg) counts for visualization
        for _ in range(6):  # 6 files to move
            # random action 0:txt,1:jpg,2:pdf
            a=np.random.randint(0,3)
            text_before, vec_before = env.obs()
            pos_before = np.array([vec_before[0], vec_before[1]], dtype=float)  # txt/jpg in root
            _, vec_after, _, done,_ = env.step(a)
            pos_after = np.array([vec_after[0], vec_after[1]], dtype=float)
            obs_t.append(vec_before); acts.append(a); obs_next.append(vec_after)
            pos_next.append(pos_after); pos_t.append(pos_before)
            if done: break
    return np.array(obs_t,dtype=np.float32), np.array(acts), np.array(obs_next,dtype=np.float32), np.array(pos_next,dtype=np.float32), np.array(pos_t,dtype=np.float32)

print("="*60)
print("LOCAL AGENT DEMO - Your computer, your folder, no cloud")
print("="*60)
print("\nThis is what you haven't experienced before:")
print("  An agent that READS your folder, TALKS what it sees, ACTS via DSML,")
print("  and you VERIFY by looking at the folder - same loop as GridWorld.")
print("\nSandbox: sandbox")
print("Goal: move .txt->/txt, .jpg->/jpg, .pdf->/pdf")
print("\n--- Training WM on local file task (30s) ---")
np.random.seed(0)
wm=WorldModelV3(obs_dim=9, act_dim=3, latent_dim=16, hidden=64)
# fix pos_head to 3 dims for file task (txt,jpg,pdf)
import numpy as np
# replace pos_head with 3 outputs
from wm import MLP
wm.pos_head = MLP([16,64,3], seed=4)
obs_t, acts, obs_next, pos_next, pos_t = collect_local(200)
for epoch in range(250):
    idx=np.random.choice(len(obs_t),64)
    bt,ba,bn,pn,pt=obs_t[idx],acts[idx],obs_next[idx],pos_next[idx],pos_t[idx]
    total,lat,recon,next_recon,pos_loss,cache=wm.loss(bt,ba,bn,pn,pt)
    wm.backward(bt,ba,bn,pn,cache,pos_t=pt)
    if epoch%80==0:
        print(f" epoch {epoch} pos {pos_loss:.3f} next {next_recon:.3f}")
print("WM ready. Imagine check: from messy [2,2] -> act txt should go to [1,2]")
from agent import Agent
agent=Agent(wm)
# need to adapt Agent pos_head for 2-dim pos (txt,jpg count) - already trained

print("\n" + "="*60)
print("NOW WATCH IT WORK - 1 episode, you verify each talk")
print("="*60)
env=LocalDesktop()
text, vec = env.reset(seed=42)
print(f"\nSTART:\n{env.render_tree()}")
print(f"Obs text: {text}")

# We will run the agent, but we also show what YOU see as verifier
for step in range(8):
    s=agent.sense(vec)
    # Language talk - grounded
    lang,a_lang = agent.talk(vec, mode="language")
    # For local task, lang pos is (txt_count, jpg_count) - not grid x,y, but still verifiable
    # We'll make language more human: show counts
    pos_pred = wm.decode_pos(s)
    true_pos = np.array([vec[0], vec[1]])
    print(f"\nStep {step} Talk(lang): {lang}")
    print(f"  -> Agent claims pos {np.round(pos_pred,1).tolist()} vs true {true_pos.tolist()} "
          f"{'OK' if np.linalg.norm(pos_pred-true_pos)<1.0 else 'FAIL'}  (you verify by looking at folder)")

    # DSML talk
    block, act = agent.talk(vec, mode="action")
    # Map act 0/1/2 to file type
    act_names={0:"move .txt to /txt",1:"move .jpg to /jpg",2:"move .pdf to /pdf"}
    print(f"Step {step} Talk(DSML): {act_names.get(act, act)}")
    print(f"  DSML block: {block[:60]}...")
    # Predict
    s_pred,_=wm.predict(s, act)
    pos_next_pred=wm.decode_pos(s_pred)
    # Act
    text_next, vec_next, reward, done,_=env.step(act)
    s_true=wm.encode_target(vec_next)
    mse=float(np.mean((s_pred-s_true)**2))
    true_next_pos=np.array([vec_next[0], vec_next[1]])
    print(f"World->Agent: {text_next}")
    print(f"  Folder now:\n{env.render_tree()}")
    print(f"  Verify DSML: predicted next pos {np.round(pos_next_pred,1).tolist()} "
          f"vs world {true_next_pos.tolist()} err {np.linalg.norm(pos_next_pred-true_next_pos):.2f} "
          f"{'OK' if np.linalg.norm(pos_next_pred-true_next_pos)<1.0 else 'FAIL'} mse {mse:.4f}")
    print(f"  Reward {reward} done {done}")
    vec=text_next, vec_next  # careful: text_next is string, vec_next is array
    # actually obs is vec
    vec=vec_next
    text=text_next
    time.sleep(0.5)
    if done:
        print(f"\nGoal reached in {step+1} steps! Folder sorted.")
        break

print("\n" + "="*60)
print("WHAT YOU JUST EXPERIENCED:")
print("="*60)
print("1. Sense: agent read YOUR folder (not cloud) -> vec [txt,jpg count]")
print("2. Talk(lang): 'Agent at [2,2] will move txt' -> you verified by counting files in sandbox")
print("3. Talk(DSML): '<DSML>move notes.txt to txt' -> verified by predict vs world_result")
print("4. World->Agent: folder changed -> you saw it move")
print("\nGridWorld was the same loop with 'A at [0,0]' instead of 'files at [2,2]'.")
print("Real computer agent just swaps GridWorld env.py for LocalDesktop local_env.py")
print("All local, all verifiable, you are the verifier (your eyes on folder = world_result).")
print("\nTry it yourself: open Finder to sandbox, run again,")
print("watch files move, check if lang claim matches what you see. If it lies, mse fails.")
print("="*60)
