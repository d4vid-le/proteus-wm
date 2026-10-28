# LE-GATES — Continual Learning for the Proteus world model (verified)

## What Le-Gates is

Le-Gates is the gated residual pattern used across the author's model
experiments. Two variants came before this world-model adaptation:

1. **Hard gates** (behavior-preserving surgery): `gateA_delta`/`gateB_delta`
   initialize to 0, so the gate equals 1.0 exactly; `freeze_gates()` sets the
   delta to -1 (path off) and freezes the parameters. Verified as behavior
   preserving: H-SURG-1 PASS, perplexity delta 0.0000.
2. **Trainable sigmoid gates** (smooth adaptation): `gateA_logit`/`gateB_logit`
   are `nn.Parameter`, `gate = sigmoid(logit)`, applied as residual scalars:
   `x = x + gateA * attn(norm(x))` / `x = x + gateB * mlp(norm(x))`.
   `freeze_gates()` → logit -10 (σ≈0), `unfreeze_gates()` → logit 0 (σ=0.5).

## The adaptation for Proteus
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
| **True Le-Gates** | 0.0113 → 0.0083 | 0.0011 → 0.0030 | **+26%** | **+169%** (bounded, σ(gate)=0.179) |

## Reading the result
- **Le-Gates learns the new domain BETTER** (+26% vs +15%) AND **protects the old
  domain ~6x better** (169% vs 1015% drift).
- The gate σ **learned down to 0.179** — the model actively chose to rely mostly
  on the old path with a small gated correction. It's not a hard switch; it's a
  learned, smooth trade-off. This is the trainable-gate advantage over a hard freeze.
- Earlier attempt (freeze enc_macro+pos_head, train only pred_macro) FAILED
  (shift got worse, 0.0113→0.0632) — the gate must sit on the dynamics delta as a
  residual, NOT freeze the encoder/readout. The residual structure is essential.

## Files
- `proteus/world/le_gates.py` — Le-Gates implementation for the world model
- `proteus/eval/run_evals.py` — eval 5 (growable Le-Gates) and eval 4 (continual)
- `docs/research/GROWABLE.md` — the growable-latent extension
- `docs/research/WO1_LIFELONG.md` — multi-shift slot routing