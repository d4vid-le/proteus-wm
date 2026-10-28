# LE-GATES — Continual Learning for proteus (verified)

## What Proteus Gates is (from the codebase)
Two variations exist in the project:
1. the author's earlier hard-gate prototype — hard gates: `gateA_delta`/`gateB_delta`
   init 0 → gate==1.0 exactly, `freeze_gates()` sets delta=-1 (path off), freezes
   params. Behavior-preserving surgery (H-SURG-1 PASS, ppl delta 0.0000).
2. An earlier prototype — **trainable sigmoid gates**: `gateA_logit`/`gateB_logit`
   are `nn.Parameter`, `gate = sigmoid(logit)`, applied as residual scalars:
   `x = x + gateA * attn(norm(x))` / `x = x + gateB * mlp(norm(x))`.
   `freeze_gates()` → logit -10 (σ≈0), `unfreeze_gates()` → logit 0 (σ=0.5).

## The adaptation for proteus
Applied the **trainable residual gate** to the world-model's macro dynamics
predictor (the component that must adapt when the world changes):

```
z_{t+k} = z + (1-g)·old_delta(z,A) + g·new_delta(z,A)
g = sigmoid(gate_logit)          # trainable
old_delta frozen after domain A  # old knowledge protected, by construction
new_delta + gate_logit train on domain B
```

This mirrors the earlier prototype exactly: the gate scales the sub-network's contribution, so
at g=0 the old path flows unchanged (identity → no forgetting), and the model
*learns* how much new-path to mix in.

## Test: distribution shift — a new file type (.md) arrives

Base H-JEPA trained on txt/jpg/pdf (+keep). Shift: `.md` files appear, folding
into the txt channel. Measured macro latent MSE (lower = better prediction):

| Method | shift-MSE before→after | old-MSE before→after | shift learn | old drift |
|---|---|---|---|---|
| **Naive CL** (all weights) | 0.0113 → 0.0096 | 0.0011 → 0.0123 | **+15%** | **+1015%** (catastrophic forgetting) |
| **True Proteus Gates** | 0.0113 → 0.0083 | 0.0011 → 0.0030 | **+26%** | **+169%** (bounded, σ(gate)=0.179) |

## Reading the result
- **Proteus Gates learns the new domain BETTER** (+26% vs +15%) AND **protects the old
  domain ~6x better** (169% vs 1015% drift).
- The gate σ **learned down to 0.179** — the model actively chose to rely mostly
  on the old path with a small gated correction. It's not a hard switch; it's a
  learned, smooth trade-off. This is the trainable-gate advantage over a hard freeze.
- Earlier attempt (freeze enc_macro+pos_head, train only pred_macro) FAILED
  (shift got worse, 0.0113→0.0632) — the gate must sit on the dynamics delta as a
  residual, NOT freeze the encoder/readout. The residual structure is essential.

## Files
- the earlier prototype — original trainable sigmoid gates (source)
- the author's earlier hard-gate prototype (source)
- `/tmp/gates_compare2.py` — the head-to-head (naive vs true Proteus Gates)