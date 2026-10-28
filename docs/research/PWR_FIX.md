# PWR — Cross-World Planning: the honest fix (goal-directed data)

`cross_reason.py` — after the user pushed "fix it," the root cause of unstable
cross-world planning was found and fixed.

## The root cause (found by systematic testing, not guessing)
The per-world JEPA's MPC planning was seed-unstable (0-4/20). The cause was the
**training data distribution**: transitions collected from RANDOM wandering
don't correlate with "toward the goal," so the dynamics never learn goal-directed
latent structure. Distance-to-goal-latent isn't monotonic under random-move
dynamics.

## The fix: goal-biased (curriculum) training data
Biased actions toward the goal (70% greedy, 30% random) during data collection.
This teaches the predictor how latent states MOVE toward the goal.

## Verified improvement (reproducible across seeds)
| data | planning success (3 seeds) | mean |
|---|---|---|
| random wandering | [4, 0, 12] | ~5/20 |
| **goal-directed** | **[12, 14, 14]** | **~13/20** |

No more catastrophic seed failures (0-4/20). The agent reliably makes progress
and reaches the goal ~65% of the time — a real capability, versus near-random.

## Honest limits
- Planning is **~65% reliable, seed-dependent — NOT 20/20.** The 5×5 grid with
  4 discrete actions has local minima where greedy k-step MPC gets stuck.
- The value-head (V(z) = -dist-to-goal) did NOT improve over raw distance-MPC;
  raw distance on goal-trained dynamics is the better planner.
- This is a genuine, defensible result: goal-directed data turned unstable
  planning into ~65% reliable planning. Full reliability needs better search
  (CEM over full sequences, or a proper learned value on predicted states).

## What this means
The route to robust cross-world planning is **data curriculum** (goal-directed
trajectories) + **better search** — not just architecture. The dynamics learn
goal-direction when shown goal-directed transitions.

## Files
- `cross_reason.py` — goal-biased `collect_transitions`, working config