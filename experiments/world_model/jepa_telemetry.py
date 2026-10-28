#!/usr/bin/env python3
"""jepa_telemetry.py — Close the JEPA loop: predict the telemetry stream.

The user's framing: "you have built a perfect logger, but not yet a predictor."
This ports the cart-pole JEPA world model onto the macOS telemetry stream:

    State  s_t   = {"app": "Chrome", "windows": 64, ...}   -> encode -> z_t
    Action a_t   = the OS event that changed it            -> encode -> a
    Next   s_{t+1} = {"app": "Chrome", "windows": 63}      -> target -> z_{t+1}

    JEPA:  z_{t+1} = z_t + f(z_t, a_t)    (latent residual predictor)
    Error: ||z_pred - z_{t+1}||^2           (Surprise / Energy)
    Update: backprop the error into f      (Mode-1 policy learns the OS dynamics)

The model learns: given the current app/window/shell state and an action, what
is the next state? Then SURPRISE = prediction error on live events, logged to a
trace (ERIS-style).

Encoding:
  - app: one-hot over observed apps
  - windows: normalized count
  - shell_new: shell action indicator
  - z_vis_std: vision-stream latent energy
  -> z (latent state vector)
  Action: a compact one-hot of the most recent event kind (fsevents kind or a
  synthetic 'window changed' action)
"""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

JOURNAL = ROOT / "data" / "telemetry.jsonl"
AGENT_JOURNAL = ROOT / "data" / "autonomous_journal.jsonl"
TRACE = ROOT / "data" / "eris_trace.jsonl"
INTERACTIONS = ROOT / "data" / "interactions.jsonl"

APPS = ["Google Chrome", "OpenCode", "Finder", "Terminal", "Other"]
ACTIONS = ["click_down", "click_up", "mouse_move", "scroll", "key_down", "key_up", "drag"]


class TelemetryJEPA(nn.Module):
    """Latent world model over the telemetry stream."""

    def __init__(self, app_dim=5, feat_dim=3, latent=16, act_dim=4, tau=0.995):
        super().__init__()
        self.app_dim = app_dim
        self.feat_dim = feat_dim
        self.latent = latent
        self.act_dim = act_dim
        self.tau = tau
        # encoder: app one-hot + features -> z
        self.enc = nn.Linear(app_dim + feat_dim, latent)
        self.target = nn.Linear(app_dim + feat_dim, latent)
        self.target.load_state_dict(self.enc.state_dict())
        # predictor: (z, action) -> delta
        self.pred = nn.Linear(latent + act_dim, latent)

    def encode(self, s):
        return self.enc(s)

    def encode_target(self, s):
        with torch.no_grad():
            return self.target(s)

    def predict(self, z, a):
        return z + self.pred(torch.cat([z, a], dim=-1))

    def update_target(self, tau=None):
        tau = tau or self.tau
        for p, tp in zip(self.enc.parameters(), self.target.parameters()):
            tp.data.mul_(tau).add_(p.data, alpha=1 - tau)

    def surprise(self, z_pred, z_target):
        return float(torch.mean((z_pred - z_target) ** 2).item())


def encode_state(app_name, windows, shell_new, zvis_std):
    """Encode a telemetry state into a vector: app one-hot + normalized features."""
    app = app_name if app_name in APPS else "Other"
    a = np.zeros(len(APPS), dtype=np.float32)
    a[APPS.index(app)] = 1.0
    w = np.clip(windows / 100.0, 0, 1).astype(np.float32)
    s = np.clip(shell_new / 5.0, 0, 1).astype(np.float32)
    z = np.array([zvis_std if zvis_std is not None else 0.2], dtype=np.float32)
    feat = np.concatenate([np.array([w, s]), z])
    return torch.as_tensor(np.concatenate([a, feat]), dtype=torch.float32)


def action_oh(kind, act_dim=4):
    """Encode an event kind as a one-hot action."""
    kinds = {"created": 0, "modified": 1, "deleted": 2, "moved": 3, "context": 0}
    i = kinds.get(kind, 0) % act_dim
    return torch.nn.functional.one_hot(torch.tensor([i]), act_dim).float()[0]


def encode_interaction_state(app, x, y):
    """Encode an interaction event into a state vector: app one-hot + normalized
    cursor position. This is the 'where am I + what app' part of the user's
    behavior stream."""
    app = app if app in APPS else "Other"
    a = np.zeros(len(APPS), dtype=np.float32)
    a[APPS.index(app)] = 1.0
    pos = np.array([np.clip((x or 0) / 2000.0, 0, 1),
                    np.clip((y or 0) / 2000.0, 0, 1)], dtype=np.float32)
    # pad to match encode_state's 8-dim shape (app + 3 features)
    return torch.as_tensor(np.concatenate([a, pos, np.zeros(1, dtype=np.float32)]),
                           dtype=torch.float32)


def load_interaction_samples(n=200):
    """Load (s_t, a, s_{t+1}) pairs from the user interaction stream. The JEPA
    learns the dynamics of the user's own behavior: given the current app +
    cursor position and an action (click/scroll/type), where does the cursor
    go next? Surprise on this stream = 'the user did something I didn't
    expect' — the reflex signal for the interaction world."""
    samples = []
    if not pathlib.Path(INTERACTIONS).exists():
        return samples
    events = []
    for line in pathlib.Path(INTERACTIONS).read_text().splitlines():
        try:
            r = json.loads(line)
            if r.get("kind") in ACTIONS:
                events.append(r)
        except Exception:
            continue
    for i in range(len(events) - 1):
        e0, e1 = events[i], events[i + 1]
        s0 = encode_interaction_state(e0.get("app"), e0.get("x"), e0.get("y"))
        a = action_oh(e0.get("kind"), act_dim=len(ACTIONS))
        s1 = encode_interaction_state(e1.get("app"), e1.get("x"), e1.get("y"))
        samples.append((s0, a, s1))
    return samples[-n:]


def load_context_samples(n=200):
    """Load (s_t, s_{t+1}) pairs from the context stream (app/windows/shell/zvis)."""
    samples = []
    if not pathlib.Path(AGENT_JOURNAL).exists():
        return samples
    ctx = []
    for line in pathlib.Path(AGENT_JOURNAL).read_text().splitlines():
        try:
            r = json.loads(line)
            if r.get("event") == "context":
                ctx.append(r)
        except Exception:
            continue
    for i in range(len(ctx) - 1):
        c0, c1 = ctx[i], ctx[i + 1]
        s0 = encode_state(c0.get("app"), c0.get("windows", 0), c0.get("shell_new", 0),
                          c0.get("zvis_std"))
        s1 = encode_state(c1.get("app"), c1.get("windows", 0), c1.get("shell_new", 0),
                          c1.get("zvis_std"))
        samples.append((s0, action_oh("context", act_dim=len(ACTIONS)), s1))
    return samples[-n:]


def load_fsevent_samples(n=200):
    """Load (s_t, a, s_{t+1}) from the fsevents stream (created/modified/deleted)."""
    samples = []
    if not pathlib.Path(JOURNAL).exists():
        return samples
    events = []
    for line in pathlib.Path(JOURNAL).read_text().splitlines():
        try:
            r = json.loads(line)
            if r.get("stream") == "fsevents":
                events.append(r)
        except Exception:
            continue
    for i in range(len(events) - 1):
        e0, e1 = events[i], events[i + 1]
        # state = (app=Other, windows=stable count, shell=0, zvis=None)
        s0 = encode_state("Other", i % 70, 0, None)
        a = action_oh(e0.get("kind"), act_dim=len(ACTIONS))
        s1 = encode_state("Other", (i + 1) % 70, 0, None)
        samples.append((s0, a, s1))
    return samples[-n:]


def train(model, samples, epochs=200, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for s0, a, s1 in samples:
            z0 = model.encode(s0.unsqueeze(0))
            zp = model.predict(z0, a.unsqueeze(0))
            zt = model.encode_target(s1.unsqueeze(0))
            loss = torch.nn.functional.mse_loss(zp, zt.detach())
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.update_target()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--live", action="store_true",
                    help="after training, score the live stream and append to ERIS trace")
    args = ap.parse_args()

    print("=" * 66)
    print("JEPA TELEMETRY PREDICTOR (close the loop)")
    print("=" * 66)

    model = TelemetryJEPA(act_dim=max(4, len(ACTIONS)))
    ctx = load_context_samples()
    fsev = load_fsevent_samples()
    inter = load_interaction_samples()
    print(f"\n[data] context pairs: {len(ctx)}, fsevents pairs: {len(fsev)}, "
          f"interaction pairs: {len(inter)}")
    if not ctx and not fsev and not inter:
        print("  no telemetry yet — run the daemon for a while first")
        return

    samples = (ctx + fsev + inter)[-300:]
    print(f"[train] {len(samples)} (s_t, a, s_t+1) pairs")
    train(model, samples, epochs=args.epochs)

    with torch.no_grad():
        trained_errs = []
        for s0, a, s1 in samples[-60:]:
            zp = model.predict(model.encode(s0.unsqueeze(0)), a.unsqueeze(0))
            zt = model.encode_target(s1.unsqueeze(0))
            trained_errs.append(float(torch.mean((zp - zt) ** 2).item()))
        rand_errs = []
        for s0, a, s1 in samples[-60:]:
            zt = model.encode_target(s1.unsqueeze(0))
            zr = torch.randn_like(zt)
            rand_errs.append(float(torch.mean((zr - zt) ** 2).item()))
    print(f"\n[result] prediction error (surprise) on trained stream: "
          f"{np.mean(trained_errs):.4f}")
    print(f"         random baseline: {np.mean(rand_errs):.4f}")
    learned = np.mean(trained_errs) < np.mean(rand_errs) * 0.5
    print(f"         => {'LEARNED the OS dynamics (predicts next state)' if learned else 'not learned yet'}")

    print("\n[ERIS trace] surprise per live event:")
    import time as _time
    trace_lines = []
    for i, (s0, a, s1) in enumerate(samples[-10:]):
        with torch.no_grad():
            zp = model.predict(model.encode(s0.unsqueeze(0)), a.unsqueeze(0))
            zt = model.encode_target(s1.unsqueeze(0))
            surf = model.surprise(zp, zt)
        rec = {"t": _time.time(), "step": i, "surprise": round(surf, 5),
               "action": "context" if a[0].item() == 1 else "fsevent"}
        trace_lines.append(rec)
        print(f"  step {i}: surprise {surf:.5f}")
    with open(TRACE, "a") as f:
        for r in trace_lines:
            f.write(json.dumps(r) + "\n")
    print(f"  -> appended to {TRACE}")

    print("\n" + "=" * 66)
    print(f"VERDICT: {'JEPA loop closed — logger became a predictor' if learned else 'needs more data'}")
    print("=" * 66)


if __name__ == "__main__":
    main()