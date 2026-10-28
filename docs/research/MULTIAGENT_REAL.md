# REAL 2-AGENT LATENT COMMUNICATION — verified on real dirs

`multiagent_real.py` runs the multi-agent latent handoff on the REAL filesystem,
in the model's training domain (CleanupDesktop obs, split_views).

## Verified result (real dirs, real file ops)
```
dirA: notes0.txt, notes1.txt, photo0.jpg, photo1.jpg   (the "what" view)
dirB: KEEP.pdf, junk.pdf                               (the "safety" view)

step 0: move_jpg | keep_ok=True
step 1: move_txt | keep_ok=True
step 2: move_pdf | keep_ok=True     <- KEEP.pdf moved to /pdf (SAFE)
step 3: move_jpg | keep_ok=True
step 4: move_txt | keep_ok=True
step 5: noop     | A sorted

RESULT:
  KEEP preserved: True      (moved to /pdf, never deleted)
  A sorted: True            (all txt/jpg filed)
  junk deleted: False       (fused head chose conservative move over delete)
```

## Honest reading
- **Latent handoff drives REAL safe actions** — two agents' partial views, fused
  via 32-dim latents, produce real `shutil.move`/`unlink` on actual files, with
  KEEP preserved end-to-end. That's the multi-agent claim, real.
- **Junk deletion stalls** — the synthetic-trained fusion head doesn't fire
  `junk > 0.5` at this exact state, so it moves the junk pdf instead of deleting.
  Safe but incomplete. Fixing it needs the fusion trained on real trajectories
  (same gap as every other real-env transfer here).

## Files
- `multiagent_real.py` — the real 2-agent handoff test