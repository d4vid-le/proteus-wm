# S2 — Vision-Latent Counts wired into the live daemon

`autonomous.py --vision-count` — the daemon now selects actions from the
vision-latent count head (real-frame-finetuned), not just real counts.

## Verified live run
```
[vision_count] counts=vision-counts=[3, 0, 2] target=txt   -> moved notes0.txt
[verified_action] move notes0.txt->txt verified 1
[vision_count] counts=vision-counts=[0, 0, 1] target=pdf   -> moved junk.pdf
[verified_action] move junk.pdf->pdf verified 1
... 5/5 verified, world clean, KEEP preserved
```

## How it works
- `VisionCountRouter` renders the watched dir to a real frame, reads counts
  from the frozen-encoder + finetuned count head.
- The vision counts propose the next action; the daemon **gates by real-FS
  presence** (only acts on types actually there) and the **KEEP guard** holds.
- Every action is still verified against the real FS.

## The safety architecture
Vision is now in the loop, but the system is safe because:
1. **KEEP guard** — never delete when KEEP present (vision + real check)
2. **real-FS gating** — vision counts can't make the agent act on a type that
   isn't actually present
3. **verification** — every move is re-checked against the real directory

So vision imperfection (~0.4 count err) can't cause harm — it proposes, the
real world + guards confirm.

## Files
- `autonomous.py` — `--vision-count` flag + VisionCountRouter
- `checkpoints/count_head_real.pt` — the real-frame-finetuned count head