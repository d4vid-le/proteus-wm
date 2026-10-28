# HONEST AUDIT — Proteus Gates in the autonomous agent

## Verdict: the earlier claim was a shortcut. Now corrected.

When asked to "test it to see if your claim holds or you're taking shortcuts," the
audit found a **critical bug and a hollow claim**. Both are now fixed.

## Bug found: build_gated() destroyed the trained model

`build_gated()` replaced the trained `pred_macro` with a **random-init** GatedDynamics,
copying that random `old` into `new`. On an already-trained model this changed the
prediction by **1.17** — the agent would forget everything the moment a shift was
detected. Fixed: `build_gated` now copies the **trained** weights into `old` (verified
pred diff = **0.0** after build).

## Hollow claim: "learning new file types" via Proteus Gates

The daemon detected `.md` files (new-file-type surge) and logged `learned_gated`
with macro_mse dropping. But:
- the 12-dim obs **cannot encode** `.md`/`.csv` (they're invisible — moved to /misc)
- `md-state` vs `plain-state` latent diff = **0.0** — the model literally cannot see them
- so the "learning" trained on **noise**, and the gate stayed at ~0.5

That was a shortcut. New file types are **safely misc-routed by grounded logic**,
but the MODEL never learns them — and claiming it did was wrong.

## What Proteus Gates actually does (honest, verified)

On a **representable** shift (the output dynamics genuinely change in a way the
model can see):
- gate learns (0.5 → 0.53)
- err vs NEW target: **0.0002** (learns the new domain)
- err vs OLD target: **0.2465** (drifts on old — a representable shift redefines the
  same output space, so a single gate can't serve both perfectly; old WEIGHTS are
  untouched, but the mixed output drifts)

**The honest claim:** Proteus Gates lets a frozen-old-path model adapt to a representable
shift in the gated new path, with the old weights protected. It does NOT let a
fixed-dim-obs model learn genuinely new observation types, and "no forgetting" is
partial when the new domain redefines the same outputs.

## What the corrected daemon does
- Shift detection now requires **representable** change (macro-MSE spike over
  verified tuples WHERE the obs actually varied). New-file-type misc-routing no
  longer arms gates or logs false "learned" claims.
- Unknown types are safely misc-routed (grounded), never deleted — but the agent
  is honest that the model didn't learn them.
- Gates install only on real, representable dynamics shifts.

## Files changed
- `gates.py` — build_gated preserves trained weights (critical bug fix)
- `autonomous.py` — honest shift detection (representability check), no false
  learning claims from misc-surge