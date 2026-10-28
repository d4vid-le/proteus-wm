# PROTEUS — Research & Test Findings (2026-09-01)

## What exists
Full codebase is present at `~/proteus` (57 files).
The transcript (`jepa_chat.json`) documented the build; the working code + newer
torch upgrades (hjepa_torch.py, benchmark.py, qwen_client.py, checkpoints/) live here.

Two prototype generations:
1. **numpy GridWorld WM** (wm.py/wm_v3.py, env.py) — 5x5 grid, latent residual dynamics `s'=s+D(s,a)`, EMA target encoder, pos_head. This is the original verified loop.
2. **torch H-JEPA** (hjepa_torch.py, benchmark.py, local_env.py) — 2-level temporal JEPA (micro 1-step + macro k-step), VICReg anti-collapse, pos_head reading file counts, CEM latent planning. **This is the current focus.**

## Test results (run on this machine, MPS available, torch 2.7.1)

### H-JEPA torch benchmark (benchmark.py) — baseline established
```
[1/5] 2636 windows collected in 1.4s
[2/5] training 300 epochs:
        micro 0.0044->0.0008 | macro 0.153->0.0198 | pos 0.412->0.0407 | total 0.586->0.078
[3/5] prediction (holdout): micro_mse 0.0008, macro_mse 0.0230, pos_mse 0.0465
        random floor: micro 0.0087, macro 0.0205  ->  WELL BELOW random floor (learning)
[4/5] MPC planning vs random (sort 6 files by type):
        MPC  success 1.0 (avg 7.65 steps)
        Rand success 1.0 (avg 9.3 steps)
        -> planner reaches goal, but random also trivially succeeds (env too easy)
[5/5] counterfactual reject rate: 0.00  (NOT learned)
        continual macro MSE: 0.0710 -> 0.0634  (weak single-update improvement)
        text: qwen2.5:1.5b real, embed dim 32 (real, via Ollama)
```
Checkpoint saved: checkpoints/hjepa_k4.pt

### H-JEPA torch smoke (hjepa_torch.py) — PASS
total 0.865 | micro 0.0896 | macro 0.0748 | pos 0.6845 | vicreg 0.0163; plan_macro works.

### Verified working
- Latent dynamics LEARN (micro/macro/pos MSE all far below random floor) — real representation learning.
- MPC latent planner reaches goal 100%.
- Real Qwen (qwen2.5:1.5b) + real embeddings (nomic-embed-text) via Ollama — the stub was upgraded.

## What does NOT hold yet (honest)
- **Counterfactual / irreversible planning: 0.00 reject.** The torch pos_head predicts
  `pdf_in_root = -0.49` when true value is 0 (should be clean separation). The headline
  "latent counterfactual beats autoregressive 100%" in the transcript came from
  `exp_counterfactual.py` which is a HARDCODED heuristic (`if 2 in action_seq and
  paper.pdf exists -> reject`), NOT the trained latent model. At the torch level the
  capability is NOT yet learned.
- **Continual learning** single-update MSE only 0.071->0.0634; the "1000x cheaper CL"
  claim needs anchored mini-buffers of verified (z,a,z',pos) tuples, not one transition.
- **MPC "win" is not meaningful** — random also succeeds 100% because sorting 6 files is trivially
  reachable. Need a harder env (irreversible actions, longer horizons) to show planning value.
- pos_head (6-dim file counts) is noisy: predicted root counts (-0.49, etc.) don't match
  true counts; architecture reads out latents but the readout isn't accurate enough for
  counterfactual rejection.

## Gap analysis for the build
The core loop (sense -> latent predict -> plan -> verify) is real and learning. The
"breakthrough" rests on two capabilities that are NOT yet demonstrated at the torch level:
1. **Latent counterfactual / model-based rejection** (the 0.00 vs claimed 100%).
2. **Genuine continual learning** from verified experiences (not random retrain, not hardcoded).

Plus: no true multi-agent latent exchange is implemented (only one agent binds senses).

## Next: see PLAN.md for the scale/more-data/robustness extension + what to test.
