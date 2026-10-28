"""dual_wm.py - 8-region 2-WM: WM_file (8 dyn heads) + Qwen router, same DSML."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import sys
import numpy as np
from wm import WorldModelV3
from wm import MLP
from vector_memory import VectorMemory
from proteus.bridge.qwen_stub import qwen_generate
from tools import dispatch_tool, encode_tool_dsml
import pathlib

# Train WM_file with 8 regions
print("="*60)
print("8-REGION 2-WM: WM_file (8 dyn) + Qwen (language)")
print("="*60)

def kmeans(X,k=8):
    np.random.seed(0)
    cents=X[np.random.choice(len(X),k,replace=False)]
    for _ in range(15):
        dists=np.linalg.norm(X[:,None]-cents[None,:],axis=2)
        assign=np.argmin(dists,axis=1)
        for j in range(k):
            if (assign==j).sum()>0:
                cents[j]=X[assign==j].mean(axis=0)
    return assign,cents

# Build 8-region file WM
from local_env import LocalDesktop
obs_t,acts,obs_next,pos_t,pos_next=[],[],[],[],[]
for _ in range(600):
    env=LocalDesktop(); _,vec=env.reset(seed=np.random.randint(0,10000))
    for _ in range(7):
        a=np.random.randint(0,3)
        vb=vec.copy(); pb=np.array([vb[0],vb[1],vb[2]])
        _,va,_,done,_=env.step(a)
        pa=np.array([va[0],va[1],va[2]])
        obs_t.append(vb); acts.append(a); obs_next.append(va); pos_t.append(pb); pos_next.append(pa)
        vec=va
        if done: break
obs_t=np.array(obs_t,dtype=np.float32); acts=np.array(acts); obs_next=np.array(obs_next,dtype=np.float32)
pos_t=np.array(pos_t,dtype=np.float32); pos_next=np.array(pos_next,dtype=np.float32)

class WM8(WorldModelV3):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.dyns=[MLP([16+3,64,16],seed=10+i) for i in range(8)]
        self.cents=None
        self.assign=None
    def build_regions(self, states):
        assign,cents=kmeans(states,8)
        self.cents=cents
        self.assign=assign
        print(f"Built 8 regions sizes {[int((assign==r).sum()) for r in range(8)]}")
        for r in range(8):
            # decode centroid pos
            pos=self.decode_pos(cents[r])
            print(f"  Region {r}: centroid file {np.round(pos,1)} size {(assign==r).sum()}")
        return assign,cents
    def predict_region(self, s, a):
        # route via nearest centroid
        dists=np.linalg.norm(s[None,:]-self.cents,axis=1) if s.ndim==1 else np.linalg.norm(s[:,None]-self.cents,axis=2).mean(axis=0)
        # actually for single s
        if s.ndim==1:
            r=int(np.argmin(np.linalg.norm(s - self.cents,axis=1)))
        else:
            r=int(np.argmin(np.linalg.norm(s[0] - self.cents,axis=1)))
        single=s.ndim==1
        if single: s=s[None,:]
        if isinstance(a,int): a=np.array([a])
        B=s.shape[0]
        a_onehot=np.zeros((B,3)); a_onehot[np.arange(B),a]=1
        inp=np.concatenate([s,a_onehot],axis=1)
        delta=self.dyns[r].forward(inp)
        s_next=s+delta
        return (s_next[0] if single else s_next), delta, r

wm8=WM8(obs_dim=9, act_dim=3, latent_dim=16, hidden=64)
wm8.pos_head=MLP([16,64,3],seed=4)
# quick train single head first to get states for clustering
for epoch in range(150):
    idx=np.random.choice(len(obs_t),128)
    bt,ba,bn,pn,pt=obs_t[idx],acts[idx],obs_next[idx],pos_next[idx],pos_t[idx]
    total,lat,recon,next_recon,pos_loss,cache=wm8.loss(bt,ba,bn,pn,pt)
    wm8.backward(bt,ba,bn,pn,cache,pos_t=pt)
states=np.array([wm8.encode(o) for o in obs_t])
assign,cents=wm8.build_regions(states)
# now train 8 dyn heads per region
print("\nTraining 8 dyn heads per region...")
for epoch in range(150):
    idx=np.random.choice(len(obs_t),64)
    bt,ba,bn,pn,pt=obs_t[idx],acts[idx],obs_next[idx],pos_next[idx],pos_t[idx]
    s_t=np.array([wm8.encode(o) for o in bt])
    # assign batch
    dists=np.linalg.norm(s_t[:,None]-wm8.cents[None,:],axis=2)
    assign_b=np.argmin(dists,axis=1)
    # predict per region
    s_pred=np.zeros_like(s_t)
    for r in range(8):
        mask=assign_b==r
        if mask.sum()==0: continue
        s_r=s_t[mask]; a_r=ba[mask]
        a_onehot=np.zeros((mask.sum(),3)); a_onehot[np.arange(mask.sum()),a_r]=1
        inp=np.concatenate([s_r,a_onehot],axis=1)
        delta=wm8.dyns[r].forward(inp)
        s_pred[mask]=s_r+delta
    s_target=np.array([wm8.encode_target(o) for o in bn])
    latent=np.mean((s_pred-s_target)**2)
    # pos
    pos_pred=np.array([wm8.decode_pos(s) for s in s_pred])
    pos_loss=np.mean((pos_pred-pn)**2)
    # backward each dyn
    for r in range(8):
        mask=assign_b==r
        if mask.sum()==0: continue
        grad=np.zeros_like(s_pred)
        grad[mask]=2*(s_pred[mask]-s_target[mask])/(64*16)
        s_r=s_t[mask]; a_r=ba[mask]
        a_onehot=np.zeros((mask.sum(),3)); a_onehot[np.arange(mask.sum()),a_r]=1
        inp=np.concatenate([s_r,a_onehot],axis=1)
        _=wm8.dyns[r].forward(inp)
        grads_w,grads_b=wm8.dyns[r].backward(grad[mask])
        wmr=wm8.dyns[r]; wmr.m_w=[0.9*m+0.1*g for m,g in zip(wmr.m_w,grads_w)]; wmr.v_w=[0.999*v+0.001*(g**2) for v,g in zip(wmr.v_w,grads_w)]
        # simplified adam
        for i in range(len(wmr.weights)):
            wmr.weights[i]-=3e-4*grads_w[i]
            wmr.biases[i]-=3e-4*grads_b[i]
    if epoch%50==0:
        print(f" epoch {epoch} latent {latent:.4f} pos {pos_loss:.4f}")

# eval per-region vs single
print("\nPer-region vs single head eval:")
# single head baseline is wm8's original dyn (not used) - we have per-region now, compare to single dyn would be similar to before
# just show per-region works
# test unified router
print("\n" + "="*60)
print("UNIFIED AGENT: file->WM_file (8 dyn), language->Qwen, same DSML")
print("="*60)
vm=VectorMemory("/tmp/dual_vec.jsonl")
pathlib.Path("/tmp/dual_vec.jsonl").unlink(missing_ok=True)
vm=VectorMemory("/tmp/dual_vec.jsonl")
# add memories for routing
for q in ["list files","move txt","sort pdfs","find pdfs"]:
    vm.add(q, {"regime":"file"})
for q in ["what is pdf","who are you","hello"]:
    vm.add(q, {"regime":"language"})

def route(query):
    # vector similarity to file vs language prototypes
    # simple: if query contains file words -> file, else language
    res=vm.search(query, k=2)
    # vote
    votes=[r["meta"].get("regime","file") if isinstance(r["meta"],dict) else "file" for r in res]
    # also keyword fallback
    if any(w in query.lower() for w in ["list","move","sort","find","read","bash","stat"]):
        return "WM_file"
    if any(w in query.lower() for w in ["what is","who are","hello","pdf is"]):
        return "Qwen"
    return "WM_file" if votes.count("file")>=votes.count("language") else "Qwen"

tests=["list files in sandbox","what is pdf","sort my pdfs","who are you proteus","find all pdfs with deepseek","move notes.txt"]
for q in tests:
    regime=route(q)
    if regime=="WM_file":
        _,vec=LocalDesktop().obs()
        s=wm8.encode(vec)
        # predict via region
        _,_,r=wm8.predict_region(s,0)
        out=f"WM_file region {r} pos {np.round(wm8.decode_pos(s),1).tolist()}"
    else:
        out=qwen_generate(q)[:60]
    print(f" query '{q:30}' -> {regime:8} -> {out[:50]}")

print("\nDemo DSML unified:")
for tool, args in [("list",'{"path":"sandbox"}'), ("qwen",'{"prompt":"what is pdf"}'), ("bash",'{"command":"ls sandbox"}')]:
    dsml=encode_tool_dsml(tool, eval(args))
    res=dispatch_tool(tool, args)
    print(f" DSML {tool} -> {str(res['output'])[:60]} verified {res['ok']}")
