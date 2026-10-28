"""proteus v2: numpy WM with Adam + EMA target + LayerNorm - no torch needed.
Fixes v1 collapse (MSE 2.0) by: EMA target encoder, Adam, latent norm.
"""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import numpy as np

def relu(x): return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(float)

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps
    def forward(self, x):
        # x: (B,D)
        mean = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta
        self.cache = (x, x_norm, mean, var)
        return out
    def backward(self, grad_out):
        x, x_norm, mean, var = self.cache
        B, D = x.shape
        std_inv = 1/np.sqrt(var + self.eps)
        grad_gamma = (grad_out * x_norm).sum(axis=0)
        grad_beta = grad_out.sum(axis=0)
        grad_norm = grad_out * self.gamma
        grad_var = (grad_norm * (x - mean) * -0.5 * (var + self.eps)**(-1.5)).sum(axis=1, keepdims=True)
        grad_mean = (grad_norm * -std_inv).sum(axis=1, keepdims=True) + grad_var * (-2*(x-mean)).mean(axis=1, keepdims=True)
        grad_x = grad_norm * std_inv + grad_var * 2*(x-mean)/D + grad_mean/D
        return grad_x, grad_gamma, grad_beta

class MLP:
    def __init__(self, dims, seed=0, use_layernorm=False):
        np.random.seed(seed)
        self.weights = []
        self.biases = []
        self.use_ln = use_layernorm
        self.lns = []
        for i in range(len(dims)-1):
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2/dims[i]) * 0.5
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)
            if use_layernorm and i < len(dims)-2:
                self.lns.append(LayerNorm(dims[i+1]))
            else:
                self.lns.append(None)
        self.cache = {}
        # Adam state
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.m_gamma = [np.zeros_like(ln.gamma) if ln else None for ln in self.lns]
        self.v_gamma = [np.zeros_like(ln.gamma) if ln else None for ln in self.lns]
        self.m_beta = [np.zeros_like(ln.beta) if ln else None for ln in self.lns]
        self.v_beta = [np.zeros_like(ln.beta) if ln else None for ln in self.lns]
        self.t = 0

    def forward(self, x):
        h = x
        caches = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w + b
            ln = self.lns[i]
            if ln is not None:
                z = ln.forward(z)
            caches.append((h, z, w, ln))
            if i < len(self.weights)-1:
                h = relu(z)
            else:
                h = z
        self.cache["caches"] = caches
        return h

    def backward(self, grad_out):
        caches = self.cache["caches"]
        grad = grad_out
        grads_w = [None]*len(self.weights)
        grads_b = [None]*len(self.biases)
        grads_gamma = [None]*len(self.lns)
        grads_beta = [None]*len(self.lns)
        for i in reversed(range(len(self.weights))):
            h_prev, z, w, ln = caches[i]
            # if layernorm, grad through it
            if ln is not None:
                # z is after LN, before relu (except last)
                # grad is dL/dz (already relu grad applied from next layer)
                # need backward through LN
                grad, g_gamma, g_beta = ln.backward(grad)
                grads_gamma[i] = g_gamma
                grads_beta[i] = g_beta
            grads_w[i] = h_prev.T @ grad / h_prev.shape[0]
            grads_b[i] = grad.mean(axis=0)
            if i > 0:
                grad_h = grad @ w.T
                _, z_prev, _, ln_prev = caches[i-1]
                # relu grad for previous
                grad = grad_h * relu_grad(z_prev)
        return grads_w, grads_b, grads_gamma, grads_beta

    def adam_step(self, grads_w, grads_b, grads_gamma, grads_beta, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for i in range(len(self.weights)):
            # weights
            self.m_w[i] = beta1*self.m_w[i] + (1-beta1)*grads_w[i]
            self.v_w[i] = beta2*self.v_w[i] + (1-beta2)*(grads_w[i]**2)
            m_hat = self.m_w[i] / (1 - beta1**self.t)
            v_hat = self.v_w[i] / (1 - beta2**self.t)
            self.weights[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # biases
            self.m_b[i] = beta1*self.m_b[i] + (1-beta1)*grads_b[i]
            self.v_b[i] = beta2*self.v_b[i] + (1-beta2)*(grads_b[i]**2)
            m_hat_b = self.m_b[i] / (1 - beta1**self.t)
            v_hat_b = self.v_b[i] / (1 - beta2**self.t)
            self.biases[i] -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
            # layernorm
            if self.lns[i] is not None and grads_gamma[i] is not None:
                self.m_gamma[i] = beta1*self.m_gamma[i] + (1-beta1)*grads_gamma[i]
                self.v_gamma[i] = beta2*self.v_gamma[i] + (1-beta2)*(grads_gamma[i]**2)
                m_hat_g = self.m_gamma[i] / (1 - beta1**self.t)
                v_hat_g = self.v_gamma[i] / (1 - beta2**self.t)
                self.lns[i].gamma -= lr * m_hat_g / (np.sqrt(v_hat_g) + eps)
                self.m_beta[i] = beta1*self.m_beta[i] + (1-beta1)*grads_beta[i]
                self.v_beta[i] = beta2*self.v_beta[i] + (1-beta2)*(grads_beta[i]**2)
                m_hat_b2 = self.m_beta[i] / (1 - beta1**self.t)
                v_hat_b2 = self.v_beta[i] / (1 - beta2**self.t)
                self.lns[i].beta -= lr * m_hat_b2 / (np.sqrt(v_hat_b2) + eps)

    def copy_from(self, other):
        for i in range(len(self.weights)):
            self.weights[i] = other.weights[i].copy()
            self.biases[i] = other.biases[i].copy()
            if self.lns[i] and other.lns[i]:
                self.lns[i].gamma = other.lns[i].gamma.copy()
                self.lns[i].beta = other.lns[i].beta.copy()

class WorldModelV2:
    def __init__(self, obs_dim=75, act_dim=4, latent_dim=32, hidden=128):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.enc = MLP([obs_dim, hidden, latent_dim], seed=1, use_layernorm=True)
        self.target_enc = MLP([obs_dim, hidden, latent_dim], seed=1, use_layernorm=True)
        self.target_enc.copy_from(self.enc)
        self.dyn = MLP([latent_dim + act_dim, hidden, latent_dim], seed=2, use_layernorm=True)
        self.dec = MLP([latent_dim, hidden, obs_dim], seed=3, use_layernorm=False)
        self.tau = 0.995

    def encode(self, obs):
        single = obs.ndim == 1
        if single: obs = obs[None,:]
        s = self.enc.forward(obs)
        # L2 normalize latent (stabilizes)
        s = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-6)
        return s[0] if single else s

    def encode_target(self, obs):
        single = obs.ndim == 1
        if single: obs = obs[None,:]
        s = self.target_enc.forward(obs)
        s = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-6)
        return s[0] if single else s

    def predict(self, s, act):
        single = s.ndim == 1
        if single: s = s[None,:]
        if isinstance(act,int): act=np.array([act])
        if act.ndim==0: act=act[None]
        B=s.shape[0]
        a_onehot=np.zeros((B,self.act_dim)); a_onehot[np.arange(B),act]=1
        inp=np.concatenate([s,a_onehot],axis=1)
        delta=self.dyn.forward(inp)
        s_next = s + 0.5 * delta  # smaller residual
        # renormalize
        s_next = s_next / (np.linalg.norm(s_next,axis=1,keepdims=True)+1e-6)
        return (s_next[0] if single else s_next), delta

    def decode(self, s):
        single=s.ndim==1
        if single: s=s[None,:]
        o=self.dec.forward(s)
        o=1/(1+np.exp(-o))
        return o[0] if single else o

    def loss(self, obs_t, act, obs_next):
        B=obs_t.shape[0]
        s_t=self.enc.forward(obs_t)
        s_t_norm = s_t / (np.linalg.norm(s_t,axis=1,keepdims=True)+1e-6)
        a_onehot=np.zeros((B,self.act_dim)); a_onehot[np.arange(B),act]=1
        inp=np.concatenate([s_t_norm,a_onehot],axis=1)
        delta=self.dyn.forward(inp)
        s_pred = s_t_norm + 0.5*delta
        s_pred = s_pred / (np.linalg.norm(s_pred,axis=1,keepdims=True)+1e-6)
        # target with EMA encoder, no grad
        s_target = self.target_enc.forward(obs_next)
        s_target = s_target / (np.linalg.norm(s_target,axis=1,keepdims=True)+1e-6)
        latent_loss=np.mean((s_pred - s_target)**2)
        # recon
        obs_recon=1/(1+np.exp(-self.dec.forward(s_t_norm)))
        recon_loss=np.mean((obs_recon-obs_t)**2)
        # variance regularization to avoid collapse (VICReg style)
        std = s_pred.std(axis=0).mean()
        var_loss = np.maximum(0, 1 - std)  # encourage std~1
        total = latent_loss + 0.2*recon_loss + 0.01*var_loss
        cache=dict(s_t=s_t, s_t_norm=s_t_norm, s_pred=s_pred, s_target=s_target, obs_recon=obs_recon, inp=inp, delta=delta)
        return total, latent_loss, recon_loss, cache

    def backward(self, obs_t, act, obs_next, cache, lr=3e-4):
        B=obs_t.shape[0]
        s_pred=cache["s_pred"]; s_target=cache["s_target"]; s_t_norm=cache["s_t_norm"]
        # grad latent
        grad_s_pred = 2*(s_pred - s_target)/(B*self.latent_dim)
        # backward through L2 norm: s_pred = (s_t+0.5d)/norm -> approx ignore norm grad for stability, use straight
        grad_delta = grad_s_pred * 0.5
        grad_s_t_from_pred = grad_s_pred  # via s_t part

        grads_dyn_w, grads_dyn_b, grads_dyn_g, grads_dyn_beta = self.dyn.backward(grad_delta)
        # grad to s_t via dyn input
        # compute grad to inp via dyn backward chain (approx via weights)
        # Use stored cache to get grad_inp properly: we already have grad_delta, get grad_hidden
        # Simplified: backprop to inp via first layer
        # Let's reuse MLP backward internal to get grad_inp: we have grad_hidden after first backward step
        # For v2, we estimate via weight
        z0=self.dyn.cache["caches"][0][1]
        # grad_hidden after LN backward
        grad_hidden = grad_delta @ self.dyn.weights[1].T
        # relu
        grad_hidden = grad_hidden * relu_grad(self.dyn.cache["caches"][0][1])  # actually need after LN, approximate
        # LN backward would have been handled, but we approximate
        W0=self.dyn.weights[0]
        grad_inp = grad_hidden @ W0.T
        grad_s_t_from_dyn_input = grad_inp[:,:self.latent_dim]

        total_grad_s_t = grad_s_t_from_pred + grad_s_t_from_dyn_input * 0.5
        # add decoder grad
        obs_recon=cache["obs_recon"]
        grad_recon = 2*(obs_recon - obs_t)*(obs_recon*(1-obs_recon))/(B*self.obs_dim)*0.2
        grads_dec_w, grads_dec_b, grads_dec_g, grads_dec_b2 = self.dec.backward(grad_recon)
        # grad to s_t from dec
        grad_hidden_dec = grad_recon @ self.dec.weights[1].T
        grad_hidden_dec = grad_hidden_dec * relu_grad(self.dec.cache["caches"][0][1])
        grad_s_t_from_dec = grad_hidden_dec @ self.dec.weights[0].T
        total_grad_s_t += grad_s_t_from_dec

        # backward through L2 norm (approx)
        # s_t_norm = s_t / norm, grad -> grad_s_t
        norm = np.linalg.norm(cache["s_t"],axis=1,keepdims=True)+1e-6
        # d norm / d s_t approx
        total_grad_s_t = total_grad_s_t / norm - (s_t_norm * (total_grad_s_t * s_t_norm).sum(axis=1,keepdims=True))/norm

        grads_enc_w, grads_enc_b, grads_enc_g, grads_enc_beta = self.enc.backward(total_grad_s_t)

        self.dyn.adam_step(grads_dyn_w, grads_dyn_b, grads_dyn_g, grads_dyn_beta, lr=lr)
        self.dec.adam_step(grads_dec_w, grads_dec_b, grads_dec_g, grads_dec_b2, lr=lr)
        self.enc.adam_step(grads_enc_w, grads_enc_b, grads_enc_g, grads_enc_beta, lr=lr)
        # EMA update target
        for i in range(len(self.enc.weights)):
            self.target_enc.weights[i] = self.tau*self.target_enc.weights[i] + (1-self.tau)*self.enc.weights[i]
            self.target_enc.biases[i] = self.tau*self.target_enc.biases[i] + (1-self.tau)*self.enc.biases[i]
            if self.enc.lns[i]:
                self.target_enc.lns[i].gamma = self.tau*self.target_enc.lns[i].gamma + (1-self.tau)*self.enc.lns[i].gamma
                self.target_enc.lns[i].beta = self.tau*self.target_enc.lns[i].beta + (1-self.tau)*self.enc.lns[i].beta
        return total_grad_s_t
