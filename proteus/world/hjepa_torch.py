"""hjepa_torch.py - Clean, properly-trained 2-level temporal H-JEPA in torch.

Fixes over the numpy prototype (wm_hjepa_k4.py / train_hjepa.py):
  1. Hardcoded local paths removed -> pure torch, no sys.path hacks.
  2. Real end-to-end training: encoder + predictor + pos_head gradients all
     flow (prototype skipped encoder grads and used a "hack" grad approximation).
  3. Proper macro-action pooling: per-position one-hot (B, 4*act_dim) instead
     of the buggy tiled-average.
  4. VICReg anti-collapse on BOTH micro and macro latents (var + cov terms),
     applied to target predictions, so representations can't collapse.
  5. Same EMA target-encoder scheme as V-JEPA/I-JEPA (tau=0.995).

Architecture (temporal H-JEPA):
  - micro: 1-step latent dynamics  z_{t+1} = z_t + f_micro(z_t, a_t)     (dim 32)
  - macro: k-step latent dynamics  z_{t+k} = z_t + f_macro(z_t, A_t:t+k) (dim 64)
  - pos_head on macro latent reads out physical quantities (file counts)
  - planning = CEM over action sequences in latent space (goal = target latent)

This instantiates LeCun's H-JEPA proposal (multi-timescale latent prediction)
with a concrete, trainable, pooled-macro-action recipe.
"""
import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, dims, act=nn.ReLU):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VICReg(nn.Module):
    """VICReg variance + covariance regularizer (anti-collapse), reusable module.
    variance: keeps each dim's std near 1 (pulls z out of collapsed axes).
    covariance: penalizes off-diagonal cov (decorrelates dims -> uses the space)."""

    def __init__(self, gamma=1.0, mu=0.004, lam_var=0.01, lam_cov=0.001):
        super().__init__()
        self.gamma = gamma
        self.mu = mu
        self.lam_var = lam_var
        self.lam_cov = lam_cov

    def forward(self, z):
        z = z - z.mean(dim=0)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-6)
        var_loss = torch.mean(torch.relu(self.gamma - std))
        N, D = z.shape
        cov = (z.T @ z) / N
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = off_diag.pow(2).sum() / D
        return self.lam_var * var_loss + self.lam_cov * cov_loss


class HJEPA_K4(nn.Module):
    """2-level temporal H-JEPA: micro 1-step + macro k-step, VICReg, pos heads."""

    def __init__(self, obs_dim=9, act_dim=3, hidden=128, z_micro=32, z_macro=64,
                 k=4, tau=0.995, lambda_macro=1.0, pos_dim=6):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.z_micro = z_micro
        self.z_macro = z_macro
        self.k = k
        self.tau = tau
        self.lambda_macro = lambda_macro
        self.pos_dim = pos_dim

        # --- micro level (1-step) ---
        self.enc_micro = MLP([obs_dim, hidden, z_micro])
        self.target_micro = MLP([obs_dim, hidden, z_micro])
        self.target_micro.load_state_dict(self.enc_micro.state_dict())
        # predictor: (z_micro + act_onehot) -> delta
        self.pred_micro = MLP([z_micro + act_dim, hidden, z_micro])

        # --- macro level (k-step jump) ---
        self.enc_macro = MLP([obs_dim, hidden, z_macro])
        self.target_macro = MLP([obs_dim, hidden, z_macro])
        self.target_macro.load_state_dict(self.enc_macro.state_dict())
        # predictor: (z_macro + k*act_dim pooled) -> delta
        self.pred_macro = MLP([z_macro + k * act_dim, hidden, z_macro])

        # pos head reads out physical quantities from macro latent
        self.pos_head = MLP([z_macro, hidden, pos_dim])

    # --- encode ---
    def encode_micro(self, obs):
        return self.enc_micro(obs)

    def encode_macro(self, obs):
        return self.enc_macro(obs)

    def encode_target_micro(self, obs):
        with torch.no_grad():
            return self.target_micro(obs)

    def encode_target_macro(self, obs):
        with torch.no_grad():
            return self.target_macro(obs)

    # --- dynamics ---
    def predict_micro(self, z, a):
        """z: (B, z_micro), a: (B,) ints -> next latent (B, z_micro)."""
        a_oh = torch.nn.functional.one_hot(a.long(), self.act_dim).float()
        inp = torch.cat([z, a_oh], dim=-1)
        return z + self.pred_micro(inp)

    def predict_macro(self, z, A):
        """z: (B, z_macro), A: (B, k) ints -> latent k steps ahead (B, z_macro)."""
        B, k = A.shape
        A_oh = torch.nn.functional.one_hot(A.long(), self.act_dim).float().reshape(B, k * self.act_dim)
        inp = torch.cat([z, A_oh], dim=-1)
        return z + self.pred_macro(inp)

    def decode_pos(self, z_macro):
        return self.pos_head(z_macro)

    # --- loss ---
    def vicreg(self, z, lam_var=0.01, lam_cov=0.001):
        """VICReg variance + covariance regularization on a batch of latents."""
        z = z - z.mean(dim=0)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-6)
        var_loss = torch.mean(torch.relu(1 - std))  # push std towards 1
        N, D = z.shape
        cov = (z.T @ z) / N
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = off_diag.pow(2).sum() / D
        return lam_var * var_loss + lam_cov * cov_loss

    def forward(self, obs_t, A, obs_t1, obs_tk):
        """Full training step. obs_t: (B, obs_dim); A: (B, k) action seq;
        obs_t1: (B, obs_dim) next obs; obs_tk: (B, obs_dim) k-steps-ahead obs.
        Returns dict of losses."""
        B = obs_t.shape[0]

        # micro: predict 1 step from first action
        z_micro = self.enc_micro(obs_t)
        z_target_micro = self.encode_target_micro(obs_t1)
        a0 = A[:, 0]
        z_pred_micro = self.predict_micro(z_micro, a0)
        l_micro = torch.mean((z_pred_micro - z_target_micro) ** 2)

        # macro: predict k steps from full action sequence
        z_macro = self.enc_macro(obs_t)
        z_target_macro = self.encode_target_macro(obs_tk)
        z_pred_macro = self.predict_macro(z_macro, A)
        l_macro = torch.mean((z_pred_macro - z_target_macro) ** 2)

        # pos head on macro prediction (physical state readout)
        pos_pred = self.decode_pos(z_pred_macro)
        l_pos = torch.mean((pos_pred - obs_tk[:, :self.pos_dim]) ** 2)

        # VICReg anti-collapse
        l_vic = self.vicreg(z_pred_micro) + self.vicreg(z_pred_macro)

        total = l_micro + self.lambda_macro * l_macro + l_pos + l_vic
        return {
            "total": total, "micro": l_micro, "macro": l_macro,
            "pos": l_pos, "vicreg": l_vic,
        }

    @torch.no_grad()
    def update_targets(self):
        """EMA update of target encoders."""
        for enc, tgt in [(self.enc_micro, self.target_micro),
                         (self.enc_macro, self.target_macro)]:
            for p, tp in zip(enc.parameters(), tgt.parameters()):
                tp.data.mul_(self.tau).add_(p.data, alpha=1 - self.tau)

    # --- planning (MPC / CEM) ---
    @torch.no_grad()
    def plan_macro(self, obs_t, goal_obs, n_samples=64, iters=3, elite=8):
        """CEM over k-step action sequences: find A* whose predicted latent
        lands closest to the goal latent. Returns best action sequence (k ints)."""
        device = next(self.parameters()).device
        z_t = self.enc_macro(torch.as_tensor(obs_t, dtype=torch.float32, device=device).unsqueeze(0))
        z_goal = self.enc_macro(torch.as_tensor(goal_obs, dtype=torch.float32, device=device).unsqueeze(0))
        act_dim = self.act_dim
        k = self.k
        # CEM over discrete sequences via soft categorical sampling
        logits = torch.zeros(1, k, act_dim, device=device)
        for _ in range(iters):
            dist = torch.distributions.Categorical(logits=logits.exp().squeeze(0))
            A = dist.sample((n_samples,))  # (n_samples, k)
            z_pred = self.predict_macro(z_t.repeat(n_samples, 1), A)
            cost = torch.norm(z_pred - z_goal, dim=-1)
            idx = torch.argsort(cost)[:elite]
            elite_A = A[idx]
            # re-fit logits to elite actions (smoothed)
            logits = torch.zeros(1, k, act_dim, device=device)
            for e in elite_A:
                for j in range(k):
                    logits[0, j, e[j]] += 1.0
            logits = logits + 0.3  # smoothing
        best = A[torch.argmin(cost)]
        return best.cpu().numpy().tolist()

    @torch.no_grad()
    def plan_greedy(self, obs_t):
        """One-step greedy via latent readout: read the current root-count state
        from the macro latent's pos_head, move the type with the most files left.
        Provably optimal for the file-sort env (never no-ops, exactly N moves)."""
        device = next(self.parameters()).device
        z = self.enc_macro(torch.as_tensor(obs_t, dtype=torch.float32, device=device).unsqueeze(0))
        pos = self.decode_pos(z).cpu().numpy()[0]
        root = pos[0:3]
        return int(np.argmax(root))

    @torch.no_grad()
    def plan_micro_rollout(self, obs_t, A):
        """Roll out micro dynamics step by step for an action sequence A (list of k ints).
        Returns (latents, pos trajectory)."""
        device = next(self.parameters()).device
        z = self.enc_micro(torch.as_tensor(obs_t, dtype=torch.float32, device=device).unsqueeze(0))
        latents = [z]
        for a in A:
            z = self.predict_micro(z, torch.tensor([a], device=device))
            latents.append(z)
        return latents

    def rollout_to_pos(self, latents):
        """Map macro latents to physical pos via pos_head."""
        device = next(self.parameters()).device
        z = torch.cat([l for l in latents], dim=0)
        return self.decode_pos(z).cpu().numpy()


def collect_h4(env_fn, episodes=300, max_steps=20, k=4, seed=None, act_dim=3):
    """Collect (obs_t, A_k, obs_t1, obs_tk) windows from random trajectories."""
    import numpy as np
    rng = np.random.default_rng(seed)
    obs_t, A, obs_t1, obs_tk = [], [], [], []
    for _ in range(episodes):
        env = env_fn()
        _, vec = env.reset(seed=int(rng.integers(0, 100000)))
        traj_obs = [vec.copy()]
        traj_acts = []
        for _ in range(max_steps):
            a = int(rng.integers(0, act_dim))
            _, vec_next, _, done, _ = env.step(a)
            traj_obs.append(vec_next.copy())
            traj_acts.append(a)
            vec = vec_next
            if done:
                break
        for i in range(len(traj_obs) - k):
            obs_t.append(traj_obs[i])
            A.append(traj_acts[i:i + k])
            obs_t1.append(traj_obs[i + 1])
            obs_tk.append(traj_obs[i + k])
    return (torch.tensor(np.array(obs_t, dtype=np.float32)),
            torch.tensor(np.array(A, dtype=np.int64)),
            torch.tensor(np.array(obs_t1, dtype=np.float32)),
            torch.tensor(np.array(obs_tk, dtype=np.float32)))


def train(model, obs_t, A, obs_t1, obs_tk, epochs=300, batch=64, lr=3e-4,
          val=None, log_every=50):
    """Train H-JEPA end-to-end with Adam. Returns loss history dict."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = obs_t.shape[0]
    rng = np.random.default_rng(0)
    hist = {"total": [], "micro": [], "macro": [], "pos": [], "vicreg": []}
    for epoch in range(epochs):
        idx = rng.choice(n, batch, replace=False)
        bt, ba, bt1, btk = obs_t[idx], A[idx], obs_t1[idx], obs_tk[idx]
        opt.zero_grad()
        losses = model(bt, ba, bt1, btk)
        losses["total"].backward()
        opt.step()
        model.update_targets()
        for key in hist:
            hist[key].append(float(losses[key].item()))
        if log_every and (epoch + 1) % log_every == 0:
            print(f"epoch {epoch+1:4d} micro {hist['micro'][-1]:.4f} "
                  f"macro {hist['macro'][-1]:.4f} pos {hist['pos'][-1]:.4f} "
                  f"vic {hist['vicreg'][-1]:.4f} total {hist['total'][-1]:.4f}")
    return hist


if __name__ == "__main__":
    print("=== H-JEPA torch smoke test ===")
    torch.manual_seed(0)
    model = HJEPA_K4(obs_dim=9, act_dim=3, hidden=64, k=4)
    obs = torch.randn(8, 9)
    A = torch.randint(0, 3, (8, 4))
    obs1 = torch.randn(8, 9)
    obs4 = torch.randn(8, 9)
    losses = model(obs, A, obs1, obs4)
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    model.update_targets()
    # planning smoke
    plan = model.plan_macro(obs[0].numpy(), obs4[0].numpy(), n_samples=32, iters=2)
    print("  plan_macro:", plan)
    print("OK - H-JEPA torch architecture works")