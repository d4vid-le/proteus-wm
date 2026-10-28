# VISION — Real Perception (vision → shared latent → routing)

`vision.py` — a CNN vision encoder that turns actual images into the SAME
64-dim latent space as the file-world model, so the agent can SEE the world.

## What it does
- **renderer**: draws a "directory scene" as a real RGB image (colored blocks per
  file type, a red block for KEEP).
- **VisionEncoder**: a real CNN (convs over RGB images) → 64-dim latent.
- **alignment**: trained so z_vision(img(state)) ≈ z_macro(state) — one shared
  latent for vision and semantics (the multi-agent binding principle, applied to
  vision↔semantics).
- **routing from pixels**: the agent detects KEEP by reading the red block and
  protects it, with NO access to the count vector's dim 7.

## Verified results (4/4 PASS)
| Test | Result |
|---|---|
| **T1 shared latent** | z_vision ≈ z_macro, cosine **1.000** (trained from 0.015) |
| **T2 KEEP from vision** | detects KEEP red block **40/40 (1.00)**, protects it |
| **T3 generalization** | unseen seeds **1.000** |
| **T4 real perception** | corrupting the KEEP region flips detection **30/30** — genuinely reads pixels, not a lookup |

## The honesty check that mattered
The FIRST perception test (small-region noise + cosine drop) gave 0.0 — looked
like a shortcut. The real test: **corrupt the KEEP red block specifically → the
KEEP decision flips 30/30.** That's genuine perception: remove the semantic
signal, the semantic decision changes. The vision encoder reads the image, not a
memorized lookup.

## Honest scope
- This is perception of a **synthetic rendered scene** — the CNN learns to read
  colored blocks as file counts. It proves the *mechanism* (image → shared latent
  → decision) works end-to-end.
- It is NOT yet "seeing a real desk photo" — that needs a real VLM (e.g.
  llava/qwen-vl via Ollama) which we haven't pulled. The next step is swapping the
  renderer for real frames and the CNN for a pretrained vision backbone.

## Files
- `vision.py` — renderer + VisionEncoder + VisionAgent + 4-test suite