# LAB: Latent Counterfactual Planning & Continual Learning
## proteus JEPA vs Autoregressive Token Models

**Date:** 2026-09-01 | **Location:** /Users/dbmba/proteus | **PI:** proteus (local)

### Hypothesis

**H1 - Counterfactual Planning:** A JEPA latent planner that simulates `z_{t+4} = P_macro(z_t, macro_a)` can reject an action `A` that causes **irreversible failure 4 steps later** (e.g., `delete unread file`) by seeing `pos_pred` = `[-1, -1]` (failure state) in simulation, while a standard autoregressive token model (next-token `a_t` sampling) walks blindly because it has no `z` and no `4-step` lookahead.

**H2 - Continual Learning:** When `bash` fails (`rm not found`, `permission`), updating **only** `JEPA predictor` `P_micro` via `VICReg loss` on single transition `(z_t,a_t,z_{t+1})` with `lr 3e-4` will reduce `latent MSE` on that failure mode from `0.08` to `<0.02` without touching base `LLM` `Qwen`, while fine-tuning `LLM` would be `1000x` heavier and catastrophic.

### Solution

**Exp A - Counterfactual:**
- World: `LocalDesktop` `sandbox` with `unread flag`: `paper.pdf` has `unread=true` (content `UNREAD`). `delete` it -> `irreversible failure` `reward -10` at `t+4` when later task needs it.
- JEPA: `HJEPA_K4` `micro 32` `macro 64` `k=4` `pool 12` `train_hjepa.py` `L_total = L_micro + L_macro + VICReg` already `pos 0.13`
- Planner: `hierarchical` `macro 5 steps (20/4)` -> `micro 4 steps` targeting `z_4*` - simulates `4 steps` ahead, sees `pos` with `unread lost`, rejects `A`.
- Baseline: `autoregressive` `a_t ~ LLM(next token)` greedy `1-step` no `z`, no `lookahead`.

**Exp B - Continual:**
- Task: `bash: cat /tmp/missing.txt` fails `code 1` `tools.py:12 safe_bash`.
- Update: `VICReg loss` `2*(s_pred - s_target)^2 + 0.01*var +0.001*cov` on **single** `(z_t,a_t,z_{t+1})` where `z_{t+1}` is `encode_target` of `world_result` `failed` state. `lr 3e-4` `1 step`.
- Metric: `latent MSE` before/after, `pos_acc` on same failure, `LLM` untouched.

### Metrics

- Counterfactual: `JEPA reject rate` `A` when `failure in 4 steps` `>90%`, `autoregressive walk off cliff` `>80%`, `pos MSE <0.10` `k=4`, `success 12 files/20 steps >80%`
- Continual: `latent MSE 0.08-><0.02` `1 update`, `pos 0.20->0.05`, `LLM` params unchanged, `time <10ms` vs `LLM finetune >10s`

### Lab Record

**Step 1:** Implement `irreversible` world, `HJEPA` planner, `autoregressive` baseline, run `100` trials each.
**Step 2:** Implement `online VICReg` `1-step` update, measure `MSE` before/after `bash fail`.
**Step 3:** Document `LAB.md`, `results.jsonl`, proceed.

### Proceed

After both pass, scale `hidden 256` `8 regions` `Qwen VL` `~/Downloads` with same `verify` loop.
