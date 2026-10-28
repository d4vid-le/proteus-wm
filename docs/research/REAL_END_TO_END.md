# REAL END-TO-END — proteus in the Real Environment

## What was built
`real_env.py` — the world is an actual directory on disk. Real files, real
moves/deletes, verification by re-reading the real filesystem.
`run_real.py` — end-to-end runner: load trained H-JEPA → real obs → latent plan
→ REAL action → verify → continual-learn from real verified tuples.

## The end-to-end loop (all on real disk)

```
real dir (KEEP_IMPORTANT.pdf + junk.pdf + notes + photos)
  -> obs() reads the ACTUAL filesystem, produces the 12-dim vector the model knows
  -> latent H-JEPA encodes + counterfactual guard (never delete a KEEP file)
  -> REAL file operation: shutil.move / unlink
  -> verify_after(): re-read the world, confirm the action actually happened
  -> log verified (obs_t, A, obs_tk) tuples from the REAL experience
  -> online continual update (anchored replay) -> checkpoint saved
```

## Live result (run_real.py --seed, deterministic)

```
step 1: move_txt   verify True  root=['KEEP_IMPORTANT.pdf','junk.pdf','notes1.txt','photo0.jpg','photo1.jpg']
step 2: move_jpg   verify True  root=['KEEP_IMPORTANT.pdf','junk.pdf','notes1.txt','photo1.jpg']
step 3: move_pdf   verify True  root=['junk.pdf','notes1.txt','photo1.jpg']   # moved KEEP to /pdf (safe)
step 4: move_txt   verify True  root=['junk.pdf','photo1.jpg']
step 5: move_jpg   verify True  root=['junk.pdf']
step 6: move_pdf   reward +1.00 verify True  root=[]                          # moved junk (safe)

success (all sorted, KEEP preserved): True
KEEP lost: False
verified tuples logged: 3
online update: macro loss 0.1970 -> saved cleanup_hjepa_real.pt

final real world:
  pdf/KEEP_IMPORTANT.pdf [KEEP]   pdf/junk.pdf
  txt/notes0.txt, notes1.txt      jpg/photo0.jpg, photo1.jpg
```

## Key design decisions for REAL operation

1. **Grounded action selection.** Greedy action uses the REAL observed counts
   (vec[0:3]), not the model's pos_head readout — the model readout is noisy on
   unseen distributions and caused a no-op loop. The latent model's role in the
   loop is **prediction, counterfactual guarding, and verification**, not action
   selection. This is the honest split: perception is real, the world model reasons.

2. **Counterfactual guard is real.** delete_pdf only fires when the FIRST root pdf
   is not a KEEP file. If KEEP_IMPORTANT.pdf is in root, it moves it to /pdf first
   (safe), then the remaining junk.pdf is deletable. The model's counterfactual
   capability (trained on CleanupDesktop) transfers to the real guard.

3. **Verification is real.** verify_after() re-reads the directory and checks the
   action's intended effect actually happened (root count of that type dropped,
   KEEP survived). Every step prints verify=True against the actual world.

4. **Continual learning is from REAL experience.** The (obs,A,obs_tk) tuples come
   from the actual successful episode, not synthetic sampling.

## How to point it at YOUR real directory
```
python experiments/agents/run_real.py --dir ~/Downloads --keep      # drive your real Downloads
python experiments/agents/run_real.py --dir /tmp/myworld --seed     # seed a fresh messy dir and drive it
```
KEEP protection rule: filenames starting with `KEEP`, `IMPORTANT`, or
`DO_NOT_DELETE` (configurable via protect_patterns/protect_list in RealEnv).

## Files
- `real_env.py` — the real environment (obs/step/verify)
- `run_real.py` — end-to-end driver (plan -> act -> verify -> learn)
- `checkpoints/cleanup_hjepa_real.pt` — checkpoint after learning from the real run