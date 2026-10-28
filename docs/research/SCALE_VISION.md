# SCALE VISION — honest scaling study

`scale_vision.py` (easy renderer) + `scale_vision_hard.py` (random-layout).

## The critical finding first
The FIRST renderer placed blocks at deterministic positions from counts → the
CNN trivially read "counts from pixel positions" → cosine 1.0 everywhere → no
scaling axis bit. Even worse, `seed_real_world` produces the SAME counts
(2/2/2) for every seed → **all states identical → z_macro degenerate → any
vision latent has cos ~0.99 to it**. The early "1.0 alignment" was measuring
alignment to a single point. That was a real measurement bug, now fixed.

## The honest result (random-layout, diverse states)
Fixed with a diverse generator (counts 0-4, keep on/off → 20 unique states) and
a HARD renderer (random blob positions, distractors):

| Test | cosine | keep-detection |
|---|---|---|
| data 100 | 0.999 | **1.00** |
| data 200 | 0.997 | **1.00** |
| data 400 | 0.988 | **1.00** |
| 0 distractors | 0.996 | 1.00 |
| 30 distractors | 0.995 | 1.00 |
| width 8 | 0.995 | 1.00 |
| width 32 | 0.991 | 1.00 |

## What scales and what doesn't
- **Keep-detection: saturated at 1.00** across every axis. The red KEEP block is
  trivially findable by even a width-8 CNN with 30 distractors. The task is below
  the model's capacity.
- **Alignment cosine: ~0.99** (real, not fake 1.0). It *slightly* drops with more
  data (0.999→0.988) — meaning the target latent is informative and the vision
  isn't trivially matching a constant.
- **Capacity: no effect** (width 8 vs 32 identical). The bottleneck is NOT the model.
- **Data: near-flat** (0.999→0.988). The task is easy enough that 100 pairs suffice.

## The honest bottleneck
Perception of these SYNTHETIC scenes saturates. The red-block-vs-not decision is
trivially learnable. The real perception bottleneck — real-world images with
lighting, occlusion, texture, real file icons — is not exercised by any renderer.
To genuinely scale perception, the next step is real frames (screencapture of a
real desktop) + a pretrained vision backbone (VLM), not more synthetic blobs.

## Files
- `scale_vision.py` — easy renderer (shows the degenerate-data trap)
- `scale_vision_hard.py` — random-layout + diverse states (the honest numbers)
- `vision.py` — the working vision encoder (4/4 tests)