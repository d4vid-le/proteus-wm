# V2 — Perception on Real Frames (screencapture)

`real_frames.py` — tests the frozen-encoder + count-head pipeline on REAL images:
a rendered real directory listing (real filenames as text) and, after screen
permission was granted, the actual desktop frame.

## Verified
| Test | Result |
|---|---|
| **R1 real listing image** | frozen encoder latent norm 16.5 (non-degenerate, saw content); count head runs |
| **R2 corruption-sensitive** | corrupting the real listing region changes latent by **0.242** → real perception |
| **R3 JPEG robustness** | cosine 1.000 / 0.999 / 1.000 at q90/50/20 |
| **actual desktop frame** | latent norm 46 (informative); encoder runs on the real screen |

## The honest boundary
- The **mechanism is real**: real frames → non-degenerate latent → corruption-
  sensitive → JPEG-robust. The frozen ResNet18 genuinely processes the actual
  screen.
- **Exact counts don't transfer**: the count head trained on synthetic color-
  blocks overcounts real text listings ([4.5,0.9,1.8] vs real [2,2,2]) and
  produces meaningless counts on the real desktop ([6,3,10]). This is the
  expected **synthetic→real distribution shift**.
- Fixing it needs **fine-tuning on real frames** (capture real directory
  screenshots + the known real FS state as labels) — a data-collection step, not
  an architecture change.

## Files
- `real_frames.py` — capture + real-image pipeline + robustness tests
- `count_head.py` — the count head (works on synthetic, needs real-frame FT)