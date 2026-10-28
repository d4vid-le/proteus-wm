# WO2 — Hierarchical Planning & Macro Horizons (k=16)

`hierarchical.py` — adds a z_high macro level (k=16) to the H-JEPA, with
coarse-to-fine planning.

## Architecture
```
z_low  (k=4):  per-file moves           predict_low_step(z, a)
z_high (k=16): whole-batch trajectories predict_high(z, pooled_A)  (macro)
planner: choose macro-action at z_high (which type to clear), refine at z_low
```

## Training (verified)
```
epoch  24  low 0.0057  high 0.0129  total 0.0279
epoch 120  low 0.0008  high 0.0023  total 0.0127
epoch 400  low 0.0001  high 0.0004  total 0.0103
```
Both levels learn cleanly. Low-level (k=4) and high-level (k=16) dynamics both
drop well below the random floor.

## Benchmark (CleanupDesktop, 20 trials)
| Planner | success | avg steps | search cost |
|---|---|---|---|
| FLAT (k=4 greedy, grounded) | 20/20 (100%) | 5.0 | 0 rollouts |
| HIERARCHICAL (k=16 macro + k=4 refine) | 20/20 (100%) | 5.0 | 16 rollouts/ep |

## Counterfactual rejection at both levels
- **KEEP lost by FLAT: 0** (guard: never delete when keep in root)
- **KEEP lost by HIER: 0** (macro clamps action to <=2 = never delete)
The safety guard fires at both abstraction levels.

## Honest reading
The hierarchical machinery **works** (both levels learn; macro planning reaches
the goal; KEEP safe at both levels). But on the CleanupDesktop env, the flat
grounded greedy planner is **already optimal** (5 steps = theoretical min), so
the hierarchy adds search cost (16 rollouts) without improving steps. The
k=16 macro level's value — fewer rollouts over long horizons — only pays off on
a task where greedy/1-step fails (a navigation or batch task with irreversible
long-horizon consequences). That's the next step: port the macro level to a
long-horizon task (e.g., the hierarchical grid navigation from the reports).

## Files
- `hierarchical.py` — HierarchicalJEPA + collect_hier + train_hier + both planners