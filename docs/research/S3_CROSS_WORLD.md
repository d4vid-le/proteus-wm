# S3 — Cross-World Transfer of the vision latent

`cross_world.py` — attacks the F1 AGI-falsification gap (world-kind transfer)
for VISION.

## Setup
- **Domain A**: file scenes (the projection was trained on these).
- **Domain B**: a completely different visual world — a grid with agent/goal/
  obstacles, labeled by a 2-class semantic (same line, no obstacle) that has
  NOTHING to do with file counts.

## Verified
| test | result |
|---|---|
| few-shot probe, 4 grid examples | test acc **0.68** |
| few-shot probe, 10 examples | **0.72** |
| few-shot probe, 30 examples | **0.75** |
| zero-shot frozen-feature separation | 0.23 (weak) |

## Reading (the honest transfer story)
- **The frozen pretrained backbone IS the cross-world-transfer part.** Its
  features carry general vision structure (layout, objects, color semantics)
  that a NEW domain can exploit with just a few examples: 4 grid images →
  68% accuracy on a task completely unrelated to files.
- **Zero-shot is weak (0.23)** — the frozen features don't separate the grid
  classes for free; transfer needs a tiny bit of learning (few-shot), which is
  how transfer learning actually works.
- **The file-specific projection does NOT transfer** (it maps to the file latent),
  but it was never meant to — the general part is the encoder.

## What this means for the F1 gap
The original falsification showed the FILE-MODEL (12-dim counts) fails world-
kind transfer. With the vision stack, the FROZEN ENCODER transfers structure
across worlds (few-shot 68%+). The remaining gap: the file-latent projection is
domain-specific, so full cross-world REASONING (planning on a new world's latent)
needs a new projection per world — but the perception generalizes. That's a real,
measurable step toward the F1 gap.

## Files
- `cross_world.py` — the few-shot/zero-shot transfer study