# VISION SUPER POWERS — on a real Finder window

`finder_vision.py` — per the architectural correction, vision does NOT count
files (the OS counts). Vision's true superpowers, all verified on a REAL Finder
window:

## The 5 superpowers (all PASS)
| # | Superpower | Result |
|---|---|---|
| F1 | **real Finder window capture** | screencapture the actual window via AppleScript bounds |
| F2 | **spatial latent** (UI/window affordances) | frozen ResNet feature map (14×14×256), layout-preserving, std 0.19 |
| F3 | **OOD / error detection** | injected dialog → anomaly score >> normal (2x+), flagged |
| F4 | **visual confirmation** | full-res pixel diff detects a file move (threshold ~1e-4) |
| F5 | **per-world latent planning** | spatial JEPA predicts next-layout transitions (err 1.7) |

## The architectural insight (the user's correction, now proven)
- **Don't use vision to count** — the frozen ImageNet features are too invariant
  for counting, and the OS already counts. Vision's value is SPATIAL:
  - **Affordances**: the 14×14 spatial map preserves WHERE things are (windows,
    icons, dialogs) — not a single pooled vector.
  - **Anomaly**: distance to the normal-layout manifold flags unexpected dialogs,
    permission popups, missing windows.
  - **Confirmation**: full-res pixel diff verifies a file visually moved — NOT
    frozen features (too invariant), a change-specific signal.

## Honest notes
- F4 needed the right tool: frozen features gave 0.0008 for an 80-diff change
  (too invariant); full-res pixel diff catches it. The spatial latent is for
  affordances/anomaly, pixel-diff is for change confirmation — different tools
  for different superpowers.
- F5's latent planning works but the prediction error (1.7) is non-trivial —
  the layout dynamics are complex (Finder animations, icon repositioning).
- Finder refresh timing matters (need ~1.5s sleep between captures).

## Files
- `finder_vision.py` — all 5 superpowers on a real Finder window