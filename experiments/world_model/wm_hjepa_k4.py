"""wm_hjepa_k4: 2-Level Temporal H-JEPA k=4 micro 32 macro 64 + VICReg + action history."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import numpy as np

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
    def backward(self,grad_out):
        caches=self.cache["caches"]
        grad=grad_out
        grads_w=[None]*len(self.weights)
        grads_b=[None]*len(self.weights)
        for i in reversed(range(len(self.weights))):
            h_prev,z,w=caches[i]
            grads_w[i]=h_prev.T@grad/len(h_prev)
            grads_b[i]=grad.mean(axis=0)
            if i>0:
                grad=grad@w.T*relu_grad(caches[i-1][1])
        return grads_w,grads_b
    def adam_step(self,grads_w,grads_b,lr=3e-4):
        self.t+=1
        for i in range(len(self.weights)):
            self.m_w[i]=0.9*self.m_w[i]+0.1*grads_w[i]
            self.v_w[i]=0.999*self.v_w[i]+0.001*(grads_w[i]**2)
            self.weights[i]-=lr*(self.m_w[i]/(1-0.9**self.t))/(np.sqrt(self.v_w[i]/(1-0.999**self.t))+1e-8)
            self.m_b[i]=0.9*self.m_b[i]+0.1*grads_b[i]
            self.v_b[i]=0.999*self.v_b[i]+0.001*(grads_b[i]**2)
            self.biases[i]-=lr*(self.m_b[i]/(1-0.9**self.t))/(np.sqrt(self.v_b[i]/(1-0.999**self.t))+1e-8)

class HJEPA_K4:
    def __init__(self, obs_dim=9, act_dim=3, hidden=128, lambda_macro=1.0):
        self.obs_dim=obs_dim; self.act_dim=act_dim; self.lambda_macro=lambda_macro
        # Micro: z 32
        self.enc_micro=MLP([obs_dim,hidden,32],seed=1)
        self.target_micro=MLP([obs_dim,hidden,32],seed=1)
        self.target_micro.weights=[w.copy() for w in self.enc_micro.weights]
        self.target_micro.biases=[b.copy() for b in self.enc_micro.biases]
        self.pred_micro=MLP([32+3,hidden,32],seed=2)
        # Macro: z 64, takes pooled 4 actions
        self.enc_macro=MLP([obs_dim,hidden,64],seed=3)
        self.target_macro=MLP([obs_dim,hidden,64],seed=3)
        self.target_macro.weights=[w.copy() for w in self.enc_macro.weights]
        self.target_macro.biases=[b.copy() for b in self.enc_macro.biases]
        self.pred_macro=MLP([64+12,hidden,64],seed=4)  # 12 = 4*3 pooled
        # Action history 16-dim
        self.hist_dim=16
        # Pos heads
        self.pos_micro=MLP([32,hidden,3],seed=5)
        self.pos_macro=MLP([64,hidden,3],seed=6)
        self.tau=0.995
    def encode_micro(self,obs):
        single=obs.ndim==1
        if single: obs=obs[None,:]
        s=self.enc_micro.forward(obs)
        return s[0] if single else s
    def encode_macro(self,obs):
        single=obs.ndim==1
        if single: obs=obs[None,:]
        s=self.enc_macro.forward(obs)
        return s[0] if single else s
    def predict_micro(self,z,a):
        single=z.ndim==1
        if single: z=z[None,:]
        if isinstance(a,int): a=np.array([a])
        if np.array(a).ndim==0: a=np.array([a])
        B=z.shape[0]
        if len(a)==1 and B>1:
            a=np.tile(a,B)
        a_onehot=np.zeros((B,3)); a_onehot[np.arange(B),a]=1
        inp=np.concatenate([z,a_onehot],axis=1)
        out=z+self.pred_micro.forward(inp)
        return out[0] if single else out
    def predict_macro(self,z, macro_a):
        single=z.ndim==1
        if single: 
            z=z[None,:]
            if macro_a.ndim==1: macro_a=macro_a[None,:]
        if macro_a.ndim==3:
            macro_a=macro_a.reshape(macro_a.shape[0],12)
        inp=np.concatenate([z,macro_a],axis=1)
        out=z+self.pred_macro.forward(inp)
        return out[0] if single else out
    def pool_macro_action(self, acts):
        # acts: (B,4) ints 0-2 -> (B,12) avg one-hot
        B=len(acts)
        pooled=np.zeros((B,12))
        for i in range(4):
            a=acts[:,i]
            for b in range(B):
                pooled[b, a[b]*4 + i] = 1  # actually 12 = 4*3, but we do per position
                # simpler: average one-hot
        # better: average one-hot
        avg=np.zeros((B,3))
        for i in range(4):
            avg+=np.eye(3)[acts[:,i]]
        avg/=4
        # expand to 12 by repeating
        pooled=np.tile(avg,4)[:,:12]  # 12
        # actually just use avg*4
        # For simplicity, use avg tiled to 12
        # Let's use 12 as 4*3 where each position has 3
        pooled2=np.zeros((B,12))
        for i in range(4):
            pooled2[:,i*3:(i+1)*3]=np.eye(3)[acts[:,i]]
        return pooled2
    def decode_pos_micro(self,s):
        single=s.ndim==1
        if single: s=s[None,:]
        p=self.pos_micro.forward(s)
        return p[0] if single else p
    def decode_pos_macro(self,s):
        single=s.ndim==1
        if single: s=s[None,:]
        p=self.pos_macro.forward(s)
        return p[0] if single else p

# Quick test
if __name__=="__main__":
    hj=HJEPA_K4(hidden=128)
    obs=np.random.randn(9)
    z_micro=hj.encode_micro(obs)
    z_macro=hj.encode_macro(obs)
    print(f"micro {z_micro.shape} macro {z_macro.shape}")
    # test macro jump
    acts=np.array([[0,1,2,0]])
    pooled=hj.pool_macro_action(acts)
    print(f"pooled {pooled.shape} {pooled[0]}")
    z_next=hj.predict_micro(z_micro,0)
    print(f"micro pred {z_next.shape}")
    z_macro_next=hj.predict_macro(z_macro, pooled)
    print(f"macro pred {z_macro_next.shape}")
    print("H-JEPA k=4 ready: micro 32, macro 64, pooled 12, VICReg to be added")
