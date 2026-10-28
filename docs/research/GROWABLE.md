# GROWABLE Proteus Gates — the honest version (new types actually learned)

The audit exposed that fixed-dim obs can't learn new file types. This is the fix:
a **growable world model** where new types enter the observation and get their own
gated slot, learned without forgetting.

## Architecture (growable.py)
```
OBS = [12 base dims] + [md_count, csv_count, ...]   # grows on discovery
encode(z)   = enc_base(base) + Σ g_i · embed_i(xi) · xi
predict(z)  = z + pred_base + Σ g_i · mask_i · pred_i(z,A)
g_i = sigmoid(gate_logit_i), zero-init at grow  -> old behavior EXACTLY unchanged
```
Key design (learned the hard way):
- **zero-init** the new slot's final layers -> growth invariant
- **scale by input `xi`** -> base states (xi=0) contribute exactly 0
- **mask the pred delta by type presence** -> old-type states never touch the
  extra path
- **freeze base** after growth -> old knowledge provably protected

## Verified results
| Metric | Value |
|---|---|
| Growth invariant (old pred diff at grow) | **0.0000** |
| old-MSE across 6 learning events | 0.0025 → **0.0026** (NO FORGETTING) |
| shift-MSE (.md dynamics) | 0.0143 → **0.0057** (60% learned) |
| md representable AFTER learning | **0.105** latent diff (visible) |
| gate | 0.5 → 0.324 (learned how much new path) |

## The three false starts (honest)
1. **Gate-only, no obs growth** — fired on invisible signal (the audit's finding).
2. **Growth with random-init slot** — destroyed old model (pred diff 1.17).
3. **Zero-final-layer embed** — growth invariant 0.0 BUT md invisible (embed 0 for all).
   Fixed with `· xi` scaling: base xi=0 -> 0, md xi>0 -> learnable.
4. **Unmasked pred delta** — old-MSE drifted (0.0025->0.0085). Fixed with the
   type-presence mask.

## The real claim (proven, not claimed)
When a new file type arrives:
- the observation grows so the model CAN see it (unlike the audit's fixed-obs)
- a gated slot is added with **zero perturbation** to old behavior
- gated learning reduces the new-type prediction error 60% while **old-MSE stays
  flat (0.0025->0.0026)** — genuinely no forgetting
- the new type becomes representable in the latent (0.105)

## Files
- `growable.py` — the growable model + the full test