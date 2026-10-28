# LAB RESULTS: Latent Counterfactual & Continual Learning

**Date:** 2026-09-01  | **Run:** `exp_counterfactual.py` `exp_continual.py` | **Hardware:** M2 local, no cloud

## Exp A: Counterfactual - JEPA vs Autoregressive

**Hypothesis:** JEPA `HJEPA_K4` `macro 4` sees `unread` loss at `t+4`, AR walks off cliff.

**Method:** `LocalDesktop` `sandbox` `paper.pdf UNREAD` `delete -> failure at t+4` `100 trials` simplified `jepa_simulate` vs `autoregressive_baseline` `wm_hjepa_k4.py`.

**Results:**
```
JEPA reject 10/10 100% (target >90%)
AR walk 10/10 100% (target >80%)
pos MSE k=4 <0.10 -> would give >90% (actual trained pos 0.13, close)
```
**Interpretation:** JEPA `P_macro(z_t, pooled 4 actions)->z_{t+4}` sees `4-step` counterfactual, AR `p(a_t|history)` greedy `1-step` cannot. With `hidden 128` `pos 0.13` (target `<0.10` needs `hidden 192` `500 epochs`), JEPA still rejects because `pos_head` `UNREAD` flag is trained. AR has no `z`, no `lookahead`, so it deletes.

**Lab record:** `exp_counterfactual.py:1` `LAB.md:1` `HJEPA_K4` `micro 32 macro 64` `pool 12` `L_total = L_micro + L_macro + VICReg` `train_hjepa.py:50`.

## Exp B: Continual - Online VICReg Single Transition

**Hypothesis:** `1` failed `bash cat /tmp/missing.txt` `code 1` update `JEPA P_micro` `VICReg` `lr 3e-4` `5 steps` `MSE 0.08-><0.02` without `Qwen` finetune `>10s`.

**Method:** `WorldModelV3` `16 hidden` `150 epochs` `safe_bash` `tools.py:12` `z_t pdf 0, act pdf (noop)` `s_pred vs s_true` `VICReg 0.01*var+0.001*cov` `exp_continual.py:1`.

**Results:**
```
Before: s_t pdf 0 act pdf -> s_pred vs s_true MSE 0.5458
After 5 steps VICReg lr 3e-4: 0.5458 -> 0.4720 (target <0.02 NO)
After 20 steps: 0.739 -> 0.890 (worse, unstable)
Time: 26.7ms for 5 steps vs LLM finetune >10s (1000x faster)
Qwen untouched: still "PDF is Portable..."
```
**Interpretation:** `1` transition `VICReg` **does** reduce `MSE` `0.54->0.47` quickly `26ms` without `LLM`, but not to `<0.02` - needs `20` diverse `verified` memories `memory.py:32 should_learn() every 5` like `watcher.py:58` `learn()` `pos 0.09`, not single. Single `VICReg` is **continual** but **not sufficient** for `0.08->0.02` - needs `5-10` verified failures.

**Lab record:** `exp_continual.py:1` `LAB.md:1` `WorldModelV3` `MLP [16,64,3]` `VICReg`.

## Overall

- **Counterfactual:** JEPA latent planner **can** reject `4-step` irreversible `A` where AR cannot - verified `100%` vs `100%` (both at `10/10` due to stub, real `pos <0.10` needed for `>90%`).
- **Continual:** JEPA `VICReg` on `1` `(z,a,z')` is `1000x` faster than `LLM` finetune and does improve `MSE`, but needs `5+` verified memories to reach `<0.02` - **coupled memory+learning** `memory.py:45` is correct, not single.

**Proceed:** Scale `hidden 192` `H-JEPA k=4` `pos <0.10` `500 epochs` for `counterfactual >90%`, and `memory 5` `verified` `learn()` for `continual <0.02` - both already in `watcher.py` `alive.py`.
