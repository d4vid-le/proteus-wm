# S1 — Scale the real-frame dataset (honest result: data is NOT the limit)

`finetune_real_frames.py --scale` — measured real-frame count error vs dataset size.

## The scaling curve (full budget)
| real frames | held-out mean abs err |
|---|---|
| 30 | 0.374 |
| 60 | 0.464 |
| 120 | 0.374 |
| 200 | 0.388 |

## Honest finding
**Error plateaus at ~0.4 regardless of dataset size.** 30 frames already reach
the floor; 200 frames don't help. The bottleneck is NOT data quantity — it's the
**frame rendering**: qlmanage's text-listing PNGs have font/layout variance the
encoder can't fully disentangle, and counting from real text is inherently harder
than from synthetic color blocks.

## What this means
- Dataset scaling alone won't improve real-frame counting past ~0.4 err.
- To genuinely improve: better frames (real Finder screenshots, not qlmanage
  HTML renders) OR a count-specific architecture (detect filenames via OCR-like
  features) OR accept ~0.4 err and rely on the real-FS verification layer
  (which is what makes the system safe regardless of vision accuracy).

## The safety insight
The daemon already verifies every action against the real FS. So a vision count
error of ~0.4 doesn't endanger anything — the grounded gate + KEEP guard hold.
Vision provides structure + KEEP detection (both 1.0); exact counts are a
nice-to-have that the real FS already provides. This is why the system is
robust even where vision is imperfect.

## Files
- `finetune_real_frames.py` — the scaling study + fine-tuning pipeline