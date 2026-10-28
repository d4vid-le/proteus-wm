# AUTONOMOUS + LE-GATES — Fully Autonomous Continual-Learning World Agent

The autonomous agent now **detects distribution shift, installs Le-Gates residual
dynamics, and continues learning without forgetting** — all on its own, in the
real environment.

## The full autonomous lifecycle (verified live, 2026-09-01)

```
watch real dir
  -> act + verify (real FS operations, KEEP protected)
  -> compute macro prediction error on verified (obs,A,obs_tk) tuples
  -> DETECT SHIFT:
       * macro-MSE persistently high (world dynamics changed), OR
       * new-file-type surge (≥3 misc-routed files = a genuinely new type)
  -> INSTALL Le-Gates: build_gated() wraps pred_macro with
       z' = z + (1-g)·old_delta(z,A) + g·new_delta(z,A), g=sigmoid(gate_logit)
  -> FREEZE OLD: old_delta frozen -> old knowledge protected (no forgetting)
  -> LEARN NEW: train only gate_logit + new_delta on the verified buffer
  -> PERSIST: checkpoints/autonomous_gated.pt (reloads gated on restart)
```

## Verified live daemon run

```
[verified_action] move a.txt->txt     (normal)
[verified_action] move b.pdf->pdf     (normal)
[verified_action] move md1.md->misc   (new type appears)
[verified_action] move md3.md->misc
[verified_action] move md2.md->misc
[shift_detected]  via=new-file-type-surge misc_routed=3
[gates_added]     Le-Gates residual dynamics installed
[gates_frozen_old] old domain path protected
[learned_gated]   tuples=2 macro_mse=0.0262 gate=0.505 ckpt=autonomous_gated.pt
```

Persistence verified: `autonomous_gated.pt` reloads with the gate intact
(`gate value on load: 0.5051`).

## What changed in autonomous.py
- `load_model` prefers `autonomous_gated.pt` (gated) if present, else the base ckpt
- `macro_predict_err` — per-tuple macro prediction error
- `detect_shift` — macro-MSE spike OR new-file-type surge (both arm the gates)
- `_install_gates` — build_gated + freeze_old (shared install path)
- `gated_learn` / gated-aware `maybe_learn` — train ONLY gate+new-path (old frozen)
- `misc_surge` tracking — consecutive new-type verified actions as a shift signal

## Files
- `autonomous.py` — the fully autonomous agent (self-driving, self-detecting, self-gating)
- `le_gates.py` — the trainable residual-gate module (ported from an earlier gated-adapter prototype)
- `GATES.md` — the head-to-head proof (naive vs gates)
- `checkpoints/autonomous_gated.pt` — persisted gated model

## Run it
```bash
# watch + act + detect shift + learn gated, forever
python -m proteus.agent.autonomous --dir ~/Downloads --daemon --poll 3

# single clean pass on a fresh dir
python -m proteus.agent.autonomous --dir /tmp/world --seed --once
```