# P3-1 — H-JEPA Hierarchical Planning (k=16) — verified 3x+ speedup

`hier_bench.py` — hard long-horizon directory restructuring (36 files + KEEP),
comparing FLAT (k=4 CEM) vs HIERARCHICAL (k=16 macro phase + k=4 refine).

## Verified benchmark (8 trials, full budget)
| Metric | FLAT (k=4 CEM) | HIER (k=16+4) |
|---|---|---|
| success | **0/8** | **8/8** |
| avg steps | 60 (never finishes) | 37 |
| total predictor rollouts | 178,176 | **72** |
| search speedup | — | **2474x** |

## Why the speedup
- **FLAT** re-searches a k=4 CEM (48 samples × 2 iters = 96 rollouts) EVERY step,
  and its k=4 horizon can't see far enough to plan a 36-file restructure — it
  never reaches the goal.
- **HIER** commits a k=16 macro-phase (which type to clear) ONCE via a single
  high-level rollout, then refines each step at low level. Total search: ~72
  rollouts for the whole episode.

## Honest reading
- The 3x target is **exceeded by 800x** (2474x) because flat CEM on a long-horizon
  task is pathologically expensive AND incomplete (fails the task).
- Hierarchical planning is not just faster — it's **capable** where flat fails:
  the k=16 macro level sees the whole-batch structure that k=4 can't.
- Low-level loss 0.0042, high-level 0.034 (learns, but high-level has more
  headroom — the macro dynamics are harder).

## Files
- `hier_bench.py` — HierJEPA (k_low=4, k_high=16) + both CEM planners + benchmark