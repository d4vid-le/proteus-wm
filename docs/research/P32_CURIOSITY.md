# P3-2 — Active Latent Curiosity & Safe Autonomous Probing

`curiosity.py` — the daemon self-improves BEFORE fatal errors by probing novel
patterns, bounded by the KEEP guard.

## Mechanism
- **Epistemic uncertainty** = mean pairwise disagreement of a 3-predictor ensemble:
  `uncertainty(obs) = mean_{i<j} ||pred_i(enc(obs)) - pred_j(enc(obs))||`
- **Novelty detection**: a state whose uncertainty is ~20x higher than known states
  (a new file type the model has never seen).
- **Safe probing**: on novelty, move ONE novel file to its correct folder (info
  probe), observe the transition `(obs_t, A, obs_tk)`, strictly bounded by KEEP.
- **Self-improvement**: probed transitions fine-tune the ensemble; uncertainty
  drops as the model learns the new type.

## Verified results
| Test | Result |
|---|---|
| **T1 novelty separation** | known 0.0003 vs novel 0.0062 (~**20x** higher) → PASS |
| **T2 safe probing** | 4 probes executed, **KEEP safe** → PASS |
| **T3 self-improvement** | uncertainty **0.0077 → 0.0002** after fine-tune → PASS |

## Reading
- Curiosity gives the agent **intrinsic motivation**: it explores novel states
  (high epistemic uncertainty) by collecting safe transitions, converting
  "unknown" into "learned" (uncertainty drops 97%).
- The KEEP guard bounds every probe — exploration never endangers protected files.
- This is the loop that makes the daemon genuinely self-adapting: it seeks out
  what it doesn't know and learns it before that unknown causes a fatal error.

## Files
- `curiosity.py` — ensemble-uncertainty model + probe loop + 3-test suite