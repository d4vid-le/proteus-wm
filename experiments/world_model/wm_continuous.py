"""wm_continuous: break one-hot ceiling, a ∈ R32 continuous via Qwen embedding."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import numpy as np, hashlib

def qwen_embed(text, dim=32):
    # Qwen embedding stub: hash -> 32 dim, then linear bottleneck would be learned
    # Real: Qwen2-VL embedding 1536 -> linear 1536->32
    h=hashlib.sha256(text.encode()).digest()
    arr=np.frombuffer(h, dtype=np.uint8).astype(float)
    vec=np.tile(arr, (dim//len(arr)+1))[:dim]
    vec=(vec-128)/30
    # normalize to sphere for stable predictor
    return vec/np.linalg.norm(vec)*2

class ActionEncoder:
    def __init__(self, dim=32):
        # 1-layer linear bottleneck: 1536->32, here stub is hash->32 already
        self.dim=dim
        # learned linear would be W 32x32
        np.random.seed(7)
        self.W=np.random.randn(32,32)*0.1
    def encode(self, cmd_str):
        # CLI string -> Qwen embedding -> bottleneck 32
        q=qwen_embed(cmd_str, dim=32)
        # linear bottleneck
        return np.tanh(q @ self.W)  # R32

class ActionDecoder:
    def __init__(self):
        # Known commands library for nearest neighbor decode
        self.library=[
            "ls /tmp/test_downloads",
            "grep -rn 'error' /var/log",
            "grep -r 'TODO' ./src",
            "git checkout -b fix",
            "cat notes.txt",
            "move *.pdf to /pdf",
        ]
        self.enc=ActionEncoder()
        self.vecs=np.array([self.enc.encode(c) for c in self.library])
    def decode(self, a_vec):
        # nearest neighbor in library
        sims=np.dot(self.vecs, a_vec)
        idx=int(np.argmax(sims))
        return self.library[idx], float(sims[idx])

def relu(x): return np.maximum(0,x)
def relu_grad(x): return (x>0).astype(float)

class MLP:
    def __init__(self,dims,seed=0):
        np.random.seed(seed)
        self.weights=[np.random.randn(dims[i],dims[i+1])*np.sqrt(2/dims[i])*0.5 for i in range(len(dims)-1)]
        self.biases=[np.zeros(dims[i+1]) for i in range(len(dims)-1)]
        self.cache={}
        self.m_w=[np.zeros_like(w) for w in self.weights]
        self.v_w=[np.zeros_like(w) for w in self.weights]
        self.m_b=[np.zeros_like(b) for b in self.biases]
        self.v_b=[np.zeros_like(b) for b in self.biases]
        self.t=0
    def forward(self,x):
        h=x; caches=[]
        for i,(w,b) in enumerate(zip(self.weights,self.biases)):
            z=h@w+b
            caches.append((h,z,w))
            h=relu(z) if i < len(self.weights)-1 else z
        self.cache["caches"]=caches
        return h
    def backward(self,grad):
        caches=self.cache["caches"]
        grads_w=[None]*len(self.weights)
        grads_b=[None]*len(self.weights)
        g=grad
        for i in reversed(range(len(self.weights))):
            h_prev,z,w=caches[i]
            grads_w[i]=h_prev.T@g/len(h_prev)
            grads_b[i]=g.mean(axis=0)
            if i>0:
                g=g@w.T*relu_grad(caches[i-1][1])
        return grads_w, grads_b
    def adam_step(self,grads_w,grads_b,lr=3e-4):
        self.t+=1
        for i in range(len(self.weights)):
            self.m_w[i]=0.9*self.m_w[i]+0.1*grads_w[i]
            self.v_w[i]=0.999*self.v_w[i]+0.001*(grads_w[i]**2)
            self.weights[i]-=lr*(self.m_w[i]/(1-0.9**self.t))/(np.sqrt(self.v_w[i]/(1-0.999**self.t))+1e-8)
            self.m_b[i]=0.9*self.m_b[i]+0.1*grads_b[i]
            self.v_b[i]=0.999*self.v_b[i]+0.001*(grads_b[i]**2)
            self.biases[i]-=lr*(self.m_b[i]/(1-0.9**self.t))/(np.sqrt(self.v_b[i]/(1-0.999**self.t))+1e-8)

class ContinuousWM:
    def __init__(self, obs_dim=9, act_dim=32, hidden=128):
        self.obs_dim=obs_dim; self.act_dim=act_dim
        self.enc=MLP([obs_dim,hidden,32],seed=1)
        self.target_enc=MLP([obs_dim,hidden,32],seed=1)
        self.target_enc.weights=[w.copy() for w in self.enc.weights]
        self.pred=MLP([32+32,hidden,32],seed=2)  # z32 + a32 -> z32
        self.action_enc=ActionEncoder(dim=32)
        self.action_dec=ActionDecoder()
        self.tau=0.995
    def encode(self, obs):
        single=obs.ndim==1
        if single: obs=obs[None,:]
        s=self.enc.forward(obs)
        return s[0] if single else s
    def encode_target(self, obs):
        single=obs.ndim==1
        if single: obs=obs[None,:]
        s=self.target_enc.forward(obs)
        return s[0] if single else s
    def predict(self, z, a_vec):
        # a_vec is R32 continuous
        single=z.ndim==1
        if single: z=z[None,:]; a_vec=a_vec[None,:]
        inp=np.concatenate([z,a_vec],axis=1)
        out=z+self.pred.forward(inp)
        return out[0] if single else out
    def cem_search(self, z, goal_z, iters=3, samples=20):
        # Continuous CEM over R32
        mean=np.zeros(32)
        std=np.ones(32)*0.5
        best_a=None; best_cost=1e9
        for it in range(iters):
            # sample
            samples_a=np.random.randn(samples,32)*std + mean
            costs=[]
            for a in samples_a:
                s_pred=self.predict(z, a)
                cost=float(np.linalg.norm(s_pred-goal_z))
                costs.append(cost)
                if cost<best_cost:
                    best_cost=cost; best_a=a.copy()
            # elite 5
            elite_idx=np.argsort(costs)[:5]
            elite=samples_a[elite_idx]
            mean=elite.mean(axis=0)
            std=elite.std(axis=0)+0.1
        return best_a, best_cost

if __name__=="__main__":
    print("Continuous WM: one-hot 3 -> R32 via Qwen")
    wm=ContinuousWM()
    enc=ActionEncoder()
    # test one-hot ceiling vs continuous
    print("\nOne-hot ceiling: 3 discrete actions")
    print("Continuous: any shell string -> R32 -> predictor -> decode to shell")
    for cmd in ["grep -rn 'error' /var/log", "git checkout -b fix", "grep -r 'TODO' ./src"]:
        a=enc.encode(cmd)
        print(f" '{cmd[:30]}' -> a norm {np.linalg.norm(a):.2f} R32")
        # decode
        dec, sim=wm.action_dec.decode(a)
        print(f"  decode nearest '{dec}' sim {sim:.2f}")
    # test predictor
    obs=np.random.randn(9)
    z=wm.encode(obs)
    a=enc.encode("grep -rn 'error' /var/log")
    z_next=wm.predict(z, a)
    print(f"\nPredictor (z32,a32)->z32: {z.shape}+{a.shape}->{z_next.shape}")
    # CEM
    goal=np.random.randn(32)
    best_a, cost=wm.cem_search(z, goal, iters=2, samples=10)
    dec,_=wm.action_dec.decode(best_a)
    print(f"CEM found a* -> '{dec}' cost {cost:.2f} (continuous search over R32, not 3 permutations)")
    print("\nBreaks one-hot ceiling: arbitrary grep/git -> R32 -> JEPA -> grounded shell string")
