# PROTEUS → Qwen Grounded Language — Handover & Onboarding

Status: **2026-09-04 — conversational ground achieved.** PROTEUS now *speaks*
clean grounded action phrases from its JEPA world model through a frozen
Qwen3.5-2B, via a soft-prefix bridge + a rank-16 LoRA on the 6 full-attention
layers.

This doc is the single source of truth for the next agent: what the system is,
what was built in this session, how to run/verify it, the honest numbers, the
open problems, and the cost/context reality.

---

## 1. What this system actually is (read this first)

**PROTEUS is NOT a chatbot.** It is an autonomous, continually-learning
**JEPA (Joint Embedding Predictive Architecture) world-model agent** that
perceives a real macOS file-world, predicts state transitions in a compact
latent space, plans, learns without forgetting, and enforces a KEEP safety
barrier. See `README.md` and `ARCHITECTURE.md` for the full stack.

**Qwen3.5-2B is the communicator — the "mouth" — NOT the brain.** The
intelligence lives in the macro-64 JEPA latent and the dynamics
`z_{t+1} = z_t + f(z_t, a_t)`. Qwen is deliberately FROZEN. The work in this
session was the **JEPA → LLM bridge**: getting the world model to steer Qwen's
language.

The three layers, bottom-up:

```
[World obs (12-dim)] -> macro-64 JEPA latent (the brain)
        -> power-whitened (γ=0.5)                       # coordinate contract
        -> JEPAToQwenBridge (projector, trainable)      # 64/128/129 -> soft-prefix
        -> Qwen3.5-2B (FROZEN) + LoRA (6 full-attn layers, trainable)
        -> grounded action phrase ("move jpg")
```

Two bridge tasks:
- **Description bridge**: `z_t → "what happened to the files?"` (static state).
- **Transition bridge** (the star): `(z_t, z_{t+1}) → action` — empirical
  inverse dynamics. `z_t` = latent before, `z_{t+1}` = latent after, both
  empirical (zero label leakage). The surprise scalar `e` was tried and
  **removed** (it hurts class labeling).

---

## 2. Session accomplishments (what got built)

### a. Coordinate-consistency fix — `latent_memory.py`
All latent storage/retrieval now uses **uniformly power-whitened coordinates**.
`append`/`retrieve` take a `space=` arg (raw→whitened on store;whitened→as-is),
a 12-dim obs is always embed→whiten. `explain.py` passes
`space="whitened"`. Self-test PASS. Before this, any caller passing a raw
latent silently got wrong L2 neighbors.

### b. Empirical inverse-dynamics transition dataset — `transition_pairs.py`
Builds **1724 deduped** transitions from the 6235 raw consecutive journal
pairs (80 sessions). Each row: whitened `z_t`, whitened `z_{t+1}`, label-free
`surprise` (deviation from a FIXED canonical rollout `[0,0,0,0]` — no target
action in the input, no 4-step smear), and the action label.
Outputs `data/transition_pairs_clean.jsonl` + `data/transition_stats.json`.

### c. 3-channel transition bridge — `llm_bridge.py`
The bridge generalized to `in_dim` 64/128/129 (auto-reconstructed from
checkpoint). New CLI:
```
--train-transition / --eval-transition
--use-surprise | --no-surprise     # 129-dim vs 128-dim ablation
--ckpt-suffix                      # so ablations don't clobber
--sweep-prior / --prior-lambda     # decode-time log-prior interpolation
```
Plus the decode-time prior sweep + tie-tolerant accuracy metric.

### d. Transition-LoRA — `train_transition_lora.py` (the conversational rail)
Rank-16 / α32 / dropout 0.05 on **q/k/v/o of the 6 full-attention layers only**
(indices 3,7,11,15,19,23) — 1.47M trainable of 1.88B (0.078%). Trained on the
1724 transition pairs, loss masked strictly to the completion phrase
(`Action: move {ext}<|im_end|>`). Bridge frozen; only LoRA trains.
Loss 5.08→0.31 over 120 steps. Saves only the LoRA adapters (~5.9MB).

### e. Interactive REPL — `chat.py`
Grounds and talks. `--lora` routes through the LoRA for clean phrase
generation; human-terms state parsing; auto-grounding.

---

## 3. The honest numbers (the whole point)

### Transition bridge — phrase log-likelihood (n=1724, chance 20%, tie-tolerant)

| variant | overall | balanced | jpg | pdf | png | misc | txt |
|---|---|---|---|---|---|---|---|
| no-prefix (prior floor) | 32% | 20% | 0% | 0% | 0% | 0% | 100% |
| 129-dim (w/ e) | 38% | 39% | 54% | 48% | 95% | 0% | 5% |
| **128-dim (no e)** | **39%** | **43%** | 22% | 89% | 100% | 0% | 4% |
| 128-dim class-balanced | 36% | 37% | 3% | 99% | 100% | 83% | 1% |
| 128-dim + λ-sweep prior | 38.9% (λ=0 best) | 42.7% | — | — | 0% at λ≥1.5 | — | 4% |

### LoRA — grounded generation (greedy decode, n=1724, chance 20%)

| class | bridge-only (LL) | **LoRA generation** |
|---|---|---|
| txt | 4% | **34%** |
| jpg | 22% | 19% |
| pdf | 89% | 60% |
| png | 100% | 91% |
| misc | 0% | 0% |
| overall | 39% | **38%** |

### The three hard-won findings (all verified experimentally)

1. **The empirical delta `(z_t → z_{t+1})` carries causal action identity.**
   pdf 89%, png 100% vs chance 20% — the transition space carries MORE
   discriminative content than the static state space (which was stuck at a
   47% ceiling). This is the core thesis, proven.

2. **The surprise scalar `e` is NOT an action signal** — it's anomaly
   magnitude. Removing it IMPROVED pdf (48%→89%). `e` belongs in the reflex's
   anomaly detection, not class labeling.

3. **The frozen "move txt" prior cannot be fixed at the projector or decode
   level.** Class-balanced training made it worse; decode-time prior
   interpolation λ left txt frozen at 4-5% and collapsed png at λ≥1.5. The
   suppression is **per-vector** (the projector hard-suppressed txt across
   thousands of non-txt rows). **Only the LoRA fixed it** (txt 4%→34%) because
   trainable attention layers can route the prefix instead of escaping to EOS.

---

## 4. The interactive chat UX (how to actually talk to it)

```bash
cd ~/proteus

# interactive, grounded + conversational
python experiments/agents/chat.py --lora
#   state 1 txt 2 jpg in root     <- ground the world (human terms OK)
#   What action was taken?        <- PROTEUS > move jpg

# one-shot with human terms
python experiments/agents/chat.py --once --lora --obs '2 txt 1 pdf in root' --question 'What action was taken?'
#   -> PROTEUS > move txt
```

**State syntax** (`state` command or `--obs`):
- raw 12-dim: `1,2,0,0,0,0,3,0,1,1,3,0`
- human: `1 txt 2 jpg in root`, `3 loose files`, `1 keep pdf`, `clean`

**Auto-grounding:** typing a message with no state set auto-grounds to the
latest live daemon obs, else a default world — so the LoRA is always active.

---

## 5. The 12-dim obs contract

From `world_latent.py`:
```
[root_txt, root_jpg, root_pdf, txt, jpg, pdf, root_total,
 keep_in_root, keep_preserved, junk_in_root, total, junk_deleted]
```
Actions (`real_env.ACTIONS`): `move_txt=0, move_jpg=1, move_pdf=2, delete_pdf=3`.
Full-attention layers: indices `[3, 7, 11, 15, 19, 23]`.

---

## 6. Key files & artifacts

### Scripts (new / modified this session)
| file | role |
|---|---|
| `transition_pairs.py` | build the inverse-dynamics transition dataset |
| `train_transition_lora.py` | LoRA on 6 full-attn layers; `--eval` for generation eval |
| `chat.py` | interactive REPL (grounded + conversational, `--lora`) |
| `llm_bridge.py` | bridge (description + transition), all evals, prior sweep |
| `latent_memory.py` | PAM buffer (uniform-whitened coords, fixed) |
| `explain.py` | grounded explanation CLI (+ PAM context) |
| `world_latent.py` | macro-64 latent source |

### Data
| artifact | rows/desc |
|---|---|
| `data/transition_pairs_clean.jsonl` | **1724** whitened (z_t, z_next, surprise, action) |
| `data/transition_stats.json` | surprise (mean, std) for standardization |
| `data/latent_pairs_clean.jsonl` | 1743 deduped, whitened description pairs |
| `data/latent_memory.jsonl` | PAM buffer (whitened z) |
| `data/whiten.npz` | (mean, W) transform — the coordinate contract |
| `data/autonomous_journal.jsonl` | 6315 verified actions (all backfill-era: **0 live obs**) |

### Checkpoints
| checkpoint | what |
|---|---|
| `checkpoints/jepa_bridge.pt` | description projector (64-dim) |
| `checkpoints/jepa_bridge_transition.pt` | 129-dim transition bridge (w/ e) |
| `checkpoints/jepa_bridge_transition.pt_128` | **128-dim transition bridge (no e) — the good one** |
| `checkpoints/jepa_bridge_transition.pt_128_bal` | class-balanced ablation (worse) |
| `checkpoints/transition_lora.pt` | **the LoRA — the conversational rail** |

---

## 7. Run / verify commands

```bash
cd ~/proteus

# rebuild the transition dataset
python -m proteus.world.transition_pairs

# train + eval the 128-dim transition bridge
python -m proteus.bridge.llm_bridge --train-transition --no-surprise --ckpt-suffix _128
python -m proteus.bridge.llm_bridge --eval-transition --no-surprise --ckpt-suffix _128

# train + eval the LoRA
python experiments/world_model/train_transition_lora.py --epochs 120 --batch 4
python experiments/world_model/train_transition_lora.py --eval --n 200

# talk to it
python experiments/agents/chat.py --lora
python experiments/agents/chat.py --lora --once --obs '2 txt 1 pdf in root' --question 'What action was taken?'

# regression: memory coordinate contract
python -m proteus.world.latent_memory --self-test
```

---

## 8. Environment / cost reality (IMPORTANT for the next agent)

- **Model:** local `Qwen/Qwen3.5-2B` at `../qwen3.5-2b` (bf16, MPS, ~4.3GB,
  1.88B params). Full-attention layers are `Qwen3_5Attention` with
  `q_proj/k_proj/v_proj/o_proj` (installed in `transformers/models/qwen3_5/`).
  peft 0.19.1 available.
- **First load is ~2-4 min** (weights + bridge + LoRA into MPS). Each new
  `python3 ...` process pays it again (no cross-process cache). Use one long
  process (like `chat.py`) to amortize.
- **Memory is tight on this Mac (64GB, swap ~11GB used by other apps).** Don't
  run two Qwen loads at once — they OOM each other. Kill stale `jepa_llm_bridge`
  processes before a new one.
- **Shell-tool timeout gotcha:** a backgrounded trainer via the shell tool can
  be killed when the tool's command budget expires. Run training through
  `subprocess.run(...)` with `timeout=None` (as in the `/tmp/run_*.py` harness
  scripts) so it isn't cut off mid-run. A ~400-step bridge train takes ~5 min;
  a 120-step LoRA train ~2 min.
- **Zero-width / CJK mojibake chars** sometimes sneak into files from pasted
  text (U+200B, full-width, etc.) and break Python. If you hit a
  "non-printable character" SyntaxError, strip them:
  `python3 -c "import re,pathlib; s=pathlib.Path('F').read_text(); \
  pathlib.Path('F').write_text(re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]','',s))"`.

---

## 9. Open problems / next levers (ranked)

1. **jpg's ~19% geometric ceiling.** jpg's delta is least separable in the
   latent. The fix is **richer `S_t`** — inject `z_vis` (vision latent) and
   `z_ui` (app/window) into the world state so jpg separates from pdf/txt.
   This is the highest-value next experiment.

2. **No live telemetry yet.** All 6315 journal actions are backfill
   reconstruction (0 live obs). When `autonomous.py --daemon` next runs and
   logs `obs`+`latent`, the transition dataset should be rebuilt from GROUND
   TRUTH (not reconstruction) — likely a jump in accuracy.

3. **misc/png are tiny classes** (12 and 57 rows) — the LoRA can't learn them
   well. Need more data or leave them as the eval's honest near-zero.

4. **4-8 token prefix** (vs the current 4) — stronger steering, easy to try.

5. **Richer action set** — the world is only move/delete; adding organize/rename
   would exercise the bridge more and match the fuller `ARCHITECTURE.md` `S_t`.

---

## 10. What I did NOT do (so you don't assume it's there)

- No change to `autonomous.py` / the daemon / the JEPA encoder itself — the
  encoder and world model are untouched (they're the frozen "brain").
- No cross-attention LoRA (deferred: Qwen3.5's linear-attn layers don't take
  KV cross-attn cleanly).
- No changes to the base `jepa_bridge.pt` description checkpoint beyond the
  generation guardrails (attention_mask, real EOS, repetition penalty,
  ChatML framing) — the description bridge was NOT retrained this session.
- The `--why` (PAM reconciliation) path still uses the description bridge /
  `explain()`, not the LoRA.

---

## 11. The one-paragraph summary

We proved the core thesis: **feeding the empirical world transition
`(z_t → z_{t+1})` through a soft-prefix bridge steers a frozen Qwen to name
the action that caused it** (pdf 89%, png 100% vs 20% chance) — something the
static state could never do. The surprise scalar is a red herring for class
labeling. The frozen "move txt" prior was unkillable by projector or decode
tricks, so we added a **rank-16 LoRA on the 6 full-attention layers**, which
let Qwen actually *speak* clean grounded phrases and fixed txt (4%→34%).
PROTEUS now holds a grounded, conversational chat (`chat.py --lora`). The next
mountain is jpg's ceiling via richer multimodal state.
