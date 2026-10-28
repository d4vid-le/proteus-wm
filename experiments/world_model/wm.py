"""wm_v3: WM with next-obs recon + pos head, no L2 sphere - for planner"""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import numpy as np

def relu(x): return np.maximum(0, x)
def relu_grad(x): return (x>0).astype(float)

class MLP:
    def __init__(self, dims, seed=0):
        np.random.seed(seed)
        self.weights=[]; self.biases=[]
        for i in range(len(dims)-1):
            self.weights.append(np.random.randn(dims[i],dims[i+1])*np.sqrt(2/dims[i])*0.5)
            self.biases.append(np.zeros(dims[i+1]))
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
    def backward(self, grad_out):
        caches=self.cache["caches"]
        grad=grad_out
        grads_w=[None]*len(self.weights)
        grads_b=[None]*len(self.biases)
        for i in reversed(range(len(self.weights))):
            h_prev,z,w=caches[i]
            grads_w[i]=h_prev.T@grad / h_prev.shape[0]
            grads_b[i]=grad.mean(axis=0)
            if i>0:
                grad_h=grad@w.T
                grad=grad_h*relu_grad(caches[i-1][1])
        return grads_w, grads_b
    def adam_step(self, grads_w, grads_b, lr=3e-4):
        self.t+=1
        beta1=0.9; beta2=0.999; eps=1e-8
        for i in range(len(self.weights)):
            self.m_w[i]=beta1*self.m_w[i]+(1-beta1)*grads_w[i]
            self.v_w[i]=beta2*self.v_w[i]+(1-beta2)*(grads_w[i]**2)
            self.weights[i]-= lr*(self.m_w[i]/(1-beta1**self.t))/(np.sqrt(self.v_w[i]/(1-beta2**self.t))+eps)
            self.m_b[i]=beta1*self.m_b[i]+(1-beta1)*grads_b[i]
            self.v_b[i]=beta2*self.v_b[i]+(1-beta2)*(grads_b[i]**2)
            self.biases[i]-= lr*(self.m_b[i]/(1-beta1**self.t))/(np.sqrt(self.v_b[i]/(1-beta2**self.t))+eps)

class WorldModelV3:
    def __init__(self, obs_dim=75, act_dim=4, latent_dim=32, hidden=128):
        self.obs_dim=obs_dim; self.act_dim=act_dim; self.latent_dim=latent_dim
        self.enc=MLP([obs_dim,hidden,latent_dim],seed=1)
        self.target_enc=MLP([obs_dim,hidden,latent_dim],seed=1)
        self.target_enc.weights=[w.copy() for w in self.enc.weights]
        self.target_enc.biases=[b.copy() for b in self.enc.biases]
        self.dyn=MLP([latent_dim+act_dim,hidden,latent_dim],seed=2)
        self.dec=MLP([latent_dim,hidden,obs_dim],seed=3)
        self.pos_head=MLP([latent_dim,hidden,2],seed=4)  # predicts x,y
        self.tau=0.995
        # prototypes for kNN (filled after training)
        self.protos=None

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
    def predict(self,s,act):
        single=s.ndim==1
        if single: s=s[None,:]
        if isinstance(act,int): act=np.array([act])
        if act.ndim==0: act=act[None]
        B=s.shape[0]
        a_onehot=np.zeros((B,self.act_dim)); a_onehot[np.arange(B),act]=1
        inp=np.concatenate([s,a_onehot],axis=1)
        delta=self.dyn.forward(inp)
        s_next=s+delta  # full residual, no 0.5, no norm
        return (s_next[0] if single else s_next), delta
    def decode(self,s):
        single=s.ndim==1
        if single: s=s[None,:]
        o=self.dec.forward(s)
        o=1/(1+np.exp(-o))
        return o[0] if single else o
    def decode_pos(self,s):
        single=s.ndim==1
        if single: s=s[None,:]
        p=self.pos_head.forward(s)  # raw x,y
        return p[0] if single else p

    def loss(self, obs_t, act, obs_next, pos_next=None, pos_t=None):
        # pos_next: (B,2) next pos, pos_t: (B,2) current pos - both for grounded language
        B=obs_t.shape[0]
        s_t=self.enc.forward(obs_t)
        a_onehot=np.zeros((B,self.act_dim)); a_onehot[np.arange(B),act]=1
        inp=np.concatenate([s_t,a_onehot],axis=1)
        delta=self.dyn.forward(inp)
        s_pred=s_t+delta
        s_target=self.target_enc.forward(obs_next)
        latent_loss=np.mean((s_pred-s_target)**2)
        obs_recon=1/(1+np.exp(-self.dec.forward(s_t)))
        recon_loss=np.mean((obs_recon-obs_t)**2)
        obs_next_pred=1/(1+np.exp(-self.dec.forward(s_pred)))
        next_recon_loss=np.mean((obs_next_pred-obs_next)**2)
        # pos head: train on BOTH s_t->pos_t and s_pred->pos_next
        pos_loss=0
        # s_pred -> pos_next
        pos_pred=self.pos_head.forward(s_pred)
        if pos_next is not None:
            pos_loss+=np.mean((pos_pred-pos_next)**2)
        # s_t -> pos_t (ground current language)
        # need to forward s_t through pos_head separately (overwrite cache, so save)
        pos_pred_t=self.pos_head.forward(s_t)
        if pos_t is not None:
            pos_loss+=np.mean((pos_pred_t-pos_t)**2)
            # average
            pos_loss/=2
        # also keep pos_pred_t in cache for backward
        total=latent_loss + 0.5*recon_loss + 0.5*next_recon_loss + 1.0*pos_loss
        cache=dict(s_t=s_t, s_pred=s_pred, s_target=s_target, obs_recon=obs_recon, obs_next_pred=obs_next_pred, pos_pred=pos_pred, pos_pred_t=pos_pred_t if pos_t is not None else None, inp=inp)
        return total, latent_loss, recon_loss, next_recon_loss, pos_loss, cache

    def backward(self, obs_t, act, obs_next, pos_next, cache, lr=3e-4, pos_t=None):
        B=obs_t.shape[0]
        s_pred=cache["s_pred"]; s_target=cache["s_target"]
        grad_s_pred = 2*(s_pred-s_target)/(B*self.latent_dim)
        obs_next_pred=cache["obs_next_pred"]
        grad_next_recon = 2*(obs_next_pred-obs_next)*(obs_next_pred*(1-obs_next_pred))/(B*self.obs_dim)*0.5
        grads_dec_next_w, grads_dec_next_b = self.dec.backward(grad_next_recon)
        z0=self.dec.cache["caches"][0][1]
        grad_hidden = grad_next_recon @ self.dec.weights[1].T
        grad_hidden = grad_hidden * relu_grad(z0)
        grad_s_pred_next = grad_hidden @ self.dec.weights[0].T
        grad_s_pred += grad_s_pred_next
        # pos head grad: s_pred -> pos_next
        pos_pred=cache["pos_pred"]
        grad_pos = 2*(pos_pred-pos_next)/(B*2)*0.5  # 0.5 because pos_loss averaged
        # need to backward pos_head for s_pred (cache currently is s_t forward, so need redo)
        # Forward s_pred again to get correct cache
        _=self.pos_head.forward(cache["s_pred"])
        grads_pos_w_next, grads_pos_b_next = self.pos_head.backward(grad_pos)
        grad_hidden_pos = grad_pos @ self.pos_head.weights[1].T
        grad_hidden_pos = grad_hidden_pos * relu_grad(self.pos_head.cache["caches"][0][1])
        grad_s_pred_pos = grad_hidden_pos @ self.pos_head.weights[0].T
        grad_s_pred += grad_s_pred_pos
        # pos_t grad: s_t -> pos_t (also trains current language)
        grad_s_t_pos = 0
        grads_pos_w_t=None; grads_pos_b_t=None
        if pos_t is not None and cache["pos_pred_t"] is not None:
            pos_pred_t=cache["pos_pred_t"]
            grad_pos_t = 2*(pos_pred_t-pos_t)/(B*2)*0.5
            _=self.pos_head.forward(cache["s_t"])
            grads_pos_w_t, grads_pos_b_t = self.pos_head.backward(grad_pos_t)
            grad_hidden_t = grad_pos_t @ self.pos_head.weights[1].T
            grad_hidden_t = grad_hidden_t * relu_grad(self.pos_head.cache["caches"][0][1])
            grad_s_t_pos = grad_hidden_t @ self.pos_head.weights[0].T
            # average grads for pos_head
            grads_pos_w = [(a+b)/2 for a,b in zip(grads_pos_w_next, grads_pos_w_t)]
            grads_pos_b = [(a+b)/2 for a,b in zip(grads_pos_b_next, grads_pos_b_t)]
        else:
            grads_pos_w, grads_pos_b = grads_pos_w_next, grads_pos_b_next
        # now dyn backward
        grad_delta=grad_s_pred
        grad_s_t_from_pred=grad_s_pred
        grads_dyn_w, grads_dyn_b = self.dyn.backward(grad_delta)
        # grad to s_t via dyn inp
        z0_dyn=self.dyn.cache["caches"][0][1]
        grad_hidden_dyn = grad_delta @ self.dyn.weights[1].T
        grad_hidden_dyn = grad_hidden_dyn * relu_grad(z0_dyn)
        grad_inp = grad_hidden_dyn @ self.dyn.weights[0].T
        grad_s_t_dyn = grad_inp[:,:self.latent_dim]
        total_grad_s_t = grad_s_t_from_pred + grad_s_t_dyn
        if pos_t is not None:
            total_grad_s_t += grad_s_t_pos
        obs_recon=cache["obs_recon"]
        grad_recon = 2*(obs_recon-obs_t)*(obs_recon*(1-obs_recon))/(B*self.obs_dim)*0.5
        s_t=cache["s_t"]
        _=self.dec.forward(s_t)
        grads_dec_w, grads_dec_b = self.dec.backward(grad_recon)
        grads_dec_w = [(a+b)/2 for a,b in zip(grads_dec_w, grads_dec_next_w)]
        grads_dec_b = [(a+b)/2 for a,b in zip(grads_dec_b, grads_dec_next_b)]
        z0_dec2=self.dec.cache["caches"][0][1]
        grad_hidden2 = grad_recon @ self.dec.weights[1].T
        grad_hidden2 = grad_hidden2 * relu_grad(z0_dec2)
        grad_s_t_recon = grad_hidden2 @ self.dec.weights[0].T
        total_grad_s_t += grad_s_t_recon
        grads_enc_w, grads_enc_b = self.enc.backward(total_grad_s_t)
        # steps
        self.dyn.adam_step(grads_dyn_w, grads_dyn_b, lr=lr)
        self.pos_head.adam_step(grads_pos_w, grads_pos_b, lr=lr)
        self.dec.adam_step(grads_dec_w, grads_dec_b, lr=lr)
        self.enc.adam_step(grads_enc_w, grads_enc_b, lr=lr)
        # EMA
        for i in range(len(self.enc.weights)):
            self.target_enc.weights[i]=self.tau*self.target_enc.weights[i]+(1-self.tau)*self.enc.weights[i]
            self.target_enc.biases[i]=self.tau*self.target_enc.biases[i]+(1-self.tau)*self.enc.biases[i]
        return total_grad_s_t

    def build_prototypes(self):
        from env import GridWorld
        self.protos={}
        for x in range(5):
            for y in range(5):
                env=GridWorld(5); env.agent=np.array([x,y]); obs=env.obs()
                s=self.encode_target(obs)
                self.protos[(x,y)]=s
