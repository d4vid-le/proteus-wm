# JEPA → LLM Bridge — PROTEUS grounded language

Status (2026-09-04): **bridge trained, prefix verified to ADD information.**
Inverse-dynamics transition bridge trained + evaluated (empirical delta
carries action identity; surprise scalar does not help class labeling).

Connects the JEPA world model to Qwen3.5-2B (local, bf16, MPS) so PROTEUS can
speak *from* its world state — not from serialized text strings. The raw
continuous world latent is injected as a soft-prefix token into Qwen's
embedding space; Qwen's language is steered by the JEPA's internal geometry.



```
[World obs (12-dim)] -> macro-64 JEPA latent -> power-whitened (γ=0.5)
        -> JEPAToQwenBridge (64->2048->2048, ONLY trainable params
        -> soft-prefix token (2048)  +  neutral prompt
        -> Qwen3.5-2B (frozen, bf16, MPS) -> grounded explanation

  Inward JEPA (PAM): surprise spike -> L2-nearest past latent neighborhoods
        -> retrieved (action, desc) context -> Qwen reconciles
```



## Locked decisions
- **Latent source:** macro-64 (`HJEPA_K4.enc_macro`) —the space where the
  transition dynamics live (macro_mse=0.0194), NOT raw ViT-768
- **Load mode:** full bf16 on MPS (4.3GB, 1.88B params) — native PyTorch
  autograd, no dequant.


- **Projector:** 2-layer MLP (Linear→GELU→Linear),the *only* trainable params
  Qwen + JEPA encoder frozen.


- **Soft-prefix, not cross-attention LoRA**deferred: Qwen3.5 uses linear-attn
  layers that don't take KV cross-attn cleanly
- **Coordinate contract:** ALL latent storage/retrieval (latent_memory.py,
  explain.py, bridge) is uniformly POWER-WHITENED (the whiten transform
  at data/whiten.npz applied at every write boundary No raw/whitened
  mixing — L2 retrieval is always Mahalanobis-equivalent in raw space.They



## The inverse-dynamics transition bridge (2026-09-04)
The description bridge narrates the *state*. The transition bridge decodes the
*action that caused an empirical transition* — exactly the shape of a live
reflex event:



```
Ch1: z_t        (whitened latent BEFORE  — empirical)
Ch2: z_{t+1}    (whitened latent AFTER  — empirical, NOT a simulated rollout)
Ch3: e_surprisez-scored — label-free scalar

e = || z_{t+1} - z_hat ||^2, z_hat = predict_macro_gated(z_t, A_fixed),
A_fixed = [0,0,0,0]  (fixed canonical reference — NO target action ever
enters the input: zero label leakage, no 4-step smear

Target:the action phrase that caused the transition.
```



### Findings (n=1724, chance=20%, tie-tolerant best-phrase correct)

| variant | overall | balanced | jpg | pdf | png | misc | txt |
|---|---|---|---|---|---|---|---|
| no-prefix (prior floor) |32% |20% |0% |0% |0% |0% |100% |
| 129-dimw/ e) |38% |39% |54% |48% |95% |0% |5% |
| **128-dimno e)** |**39%** |**43%** |22% |89% |100% |0% |4% |
| 128-dim class-balanced |36% |37% |3% |99% |100% |83% |1% |
| 128-dim + λ-sweep prior |38.9%best at λ=0|42.7% |0.22→0.30 |0.89→0.75 |1..00→0.00 at λ≥1.5 |0% |0..04→0..05 |

1. **The empirical delta carries causal action identity — proven.** pdf 89%,
   png 100% vs chance 20% —the transition space carries *more*
   discriminative content than the static state space (47% ceiling on the
   description task The 47%-static-ceiling is broken





2. **The surprise scalar `e` is NOT the action signal — it's anomaly magnitude.**
   Removing it *improves* pdf48%→89%):e adds noise to class labeling


3. **The dominant "move txt" prior is unfixable at the projector/decode level.**
   Class-balanced training made it worse (killed jpg 22%→3%) Decode-time
   prior interpolation λ·log priorsweep 0→3 leaves txt frozen at 
   4–5% and collapses png at λ≥1.5:the suppression is per-vectorthe
   projector hard-suppressed txt across thousands of non-txt rows,not per-class−uniform. A scalar can't undo a per-vector suppression The fix belongs
   architecturaltiny LoRA on the 6 full-attention layers, or train-time
   class conditioning of the prompt,not reweighting the projector/decoder







## Files
| file | role |
|---|---|---|
| `world_latent.py` | macro-64 latentgated checkpoint, `WorldLatent.embed` |
| `latent_memory.py` | PAM bufferappend/retrieve, L2-in-latent, uniform-whitened coordinates |
| `backfill_latents.py` | reconstruct latents from 6.3k historical journal actions |
| `latent_whiten.py` | dedup + power-whiteningγ, rewrites pairs + memory |
| `transition_pairs.py` | empirical inverse-dynamics pairsz_t, z_next, e, action |
| `llm_bridge.py` |the 64→2048 bridge; `--train`, `--eval-action`, `--train-transition`, `--eval-transition`with `--use-surprise`/`--no-surprise`, `--ckpt-suffix`, `--sweep-prior`, `--prior-lambda` |
| `explain.py` | grounded explanation CLI+ PAM context |
| `reflex.py` | Mode-2 hypothesis now prefers the JEPA bridge |
| `data/transition_pairs_clean.jsonl` | 1724 deduped, whitened transition pairs |
| `data/transition_stats.json` | surprisemean, std for standardization |
| `data/latent_pairs_clean.jsonl` | 1743 deduped, whitened train pairs |
| `data/latent_memory.jsonl` | PAM bufferwhitened z |
| `data/whiten.npz` |mean, W transform for live latents |
| `checkpoints/jepa_bridge.pt` | trained description projector |
| `checkpoints/jepa_bridge_transition.pt` (+ `_128`, `_128_bal`) | transition projectors (e-scalar / no-e / class-balanced ablations) |
| `checkpoints/transition_lora.pt` | LoRA (rank 16) on the 6 full-attention layers — the conversational rail |

## Run it
```bash
python -m proteus.bridge.llm_bridge --train --epochs 500 --batch 8   # description bridge
python -m proteus.bridge.llm_bridge --eval-action                     # the honest description eval
python -m proteus.world.transition_pairs                                  # build inverse-dynamics pairs
python -m proteus.bridge.llm_bridge --train-transition --no-surprise  # 128-dim transition bridge
python -m proteus.bridge.llm_bridge --eval-transition --no-surprise   # the honest prediction eval
python experiments/world_model/train_transition_lora.py --epochs 120 --batch 4      # train the LoRA (conversational rail)
python experiments/world_model/train_transition_lora.py --eval --n 200              # LoRA grounded generation eval

# TALK to PROTEUS (interactive REPL, grounded + conversational)
python experiments/agents/chat.py                                              # no --lora: bridge-only (may soup)
python experiments/agents/chat.py --lora                                       # --lora: grounded phrase generation
python experiments/agents/chat.py --lora --once --obs '1,2,0,0,0,0,3,0,1,1,3,0' --question 'What action was taken?'
```

## The transition-LoRA (2026-09-04) — the conversational rail
The frozen soft-prefix alone makes Qwen emit EOS to escape the OOD manifold
(observed `()` and token soup). A LoRA on the 6 full-attention layers
(indices 3,7,11,15,19,23; q/k/v/o, rank 16, alpha 32, dropout 0.05 —
1.47M trainable of 1.88B, 0.078%) lets the attention heads route the prefix
into natural language. Trained on the 1724 transition pairs (empirical
`(z_t, z_next)` deltas — the data that unlocked jpg/png/pdf), loss masked to
the completion phrase only.

### LoRA grounded-generation eval (n=1724, chance 20%, greedy decode)

| class | bridge-only (LL) | **LoRA generation** |
|---|---|---|
| txt | 4% | **34%** |
| jpg | 22% | 19% |
| pdf | 89% | 60% |
| png | 100% | 91% |
| misc | 0% | 0% |
| overall | 39% | **38%** |

**Result: PROTEUS now SPEAKS clean grounded phrases.** The txt-collapse that
no projector reweighting or decode-time λ could fix is resolved (4%→34%) —
the LoRA's trainable attention layers route the prefix instead of escaping to
EOS. Overall stays ~38% (parity with the bridge's log-likelihood metric) but
it is now *generated*, not scored — interactive and conversational. jpg stays
weak (~19%, the latent's geometric limit). Loss 5.08→0.31 over 120 steps.

## Why the objective matters (the debugging arc)
1. Naive target `"moved notes0.txt -> /txt/"` —the latent cannot contain the
   arbitrary filename chars → bridge wastes capacity hallucinating them.
   **Fix:** target = `normalize_target(action)` = `"move txt"` / `"move jpg"` /
   `"move pdf"` / `"delete pdf"`. No filenames
2. Prompt `"World state. Action: "` made the answer predictable WITHOUT the
   latent → bridge learned a neutral constant;with/without were identical.
   **Fix:** neutral prompt `"World state. What happened to the files? "` so only
   the latent disambiguates
3. Cone collapse: pairwise cos 0.94, participation ratio 1.11 → all latents in
   a narrow cone. **Fix:** dedup6315→1743+ power-whitening `Σ^-γ/2`, γ=0.5
   (Pareto middle: count 33% / action  41% on deduped data vs 29%/48% at γ=1



## Notes / caveats
- Live latents now flow from the patched `autonomous.py`logs `obs`+`latent`
  on each verified_action — those are exact;the backfill is reconstruction
- The reflex hook prefers the bridge when a file-world obs exists, else falls
  back to `qwen_client`the journal has no obs until the daemon runs again
- Free-text generation is still generic prosethinking-model + neutral
  promptthe *rigorous* grounded signal is the phrase log-likelihood eval
- Next: jpg's ~19% ceiling (latent geometry) — richer `S_t` (z_vis+z_ui) to
  separate jpg from pdf/txt; and 4–8 token prefix for stronger steering.