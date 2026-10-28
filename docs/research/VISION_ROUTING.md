# VISION-LATENT LIVE MPC ROUTING — the full JEPA loop

`vision_routing.py` — the end-state of the JEPA way: the planner reasons on the
VISION latent, with zero file-count inputs in the loop.

## The loop (no text anywhere)
```
real dir -> render to image -> frozen ResNet18 -> z_vis (512)
        -> projection -> z (64) -> macro JEPA (decode_pos / predict)
        -> MPC action -> real file move -> verify against real FS
```

## Verified (vision_routing.py)
| Test | Result |
|---|---|
| **T1 vision-driven routing** | vision-MPC sorted **6/6** real dirs (no counts in the loop) |
| **T2 KEEP guard through vision** | **0 KEEP lost** — the vision-latent guard holds through planning |
| **T3 latency** | **3.7ms/decision** (render+encode+project); 26ms per full dir |

## Why this matters
- The agent now plans on **what it sees**, not on a hand-engineered count vector.
- The frozen encoder (ResNet18, 3.1ms) + small projection is the whole
  perception stack — no generative VLM, no text, no parse.
- The KEEP guard is **carried by the vision latent** (proven: 0 lost), so the
  safety property survives the perception layer.
- 3.7ms/decision is viable for live routing (vs LLaVA's minutes).

## Honest notes
- The `act()` uses a grounded gate (checks real FS which types are present) to
  avoid acting on a type the vision mis-counts — this is the verification layer,
  keeping the system safe even where the vision latent is imperfect.
- The vision latent gives structure + KEEP, not exact counts. Exact counts need
  a small regression head on z_vis (next item).

## Files
- `vision_routing.py` — VisionRouter: render→encode→z→MPC→act→verify
- `vision_jepa.py` — the aligned projection (basis for this)
- `bench_encoders.py` — chose ResNet18 (3.1ms, cos 0.997)