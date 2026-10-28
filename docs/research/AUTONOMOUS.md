# AUTONOMOUS — proteus runs by itself

`autonomous.py` is a self-running world agent. No human in the loop:
it watches a real directory, notices mess, plans, acts, verifies, learns, and
keeps doing it forever — until the world is clean.

## Verified live run (daemon mode)

```
$ python -m proteus.agent.autonomous --dir /tmp/proteus-daemon --daemon --poll 1.0

[verified_action] move a.txt->txt    verified 1
[verified_action] move b.jpg->jpg    verified 1
[verified_action] move c.pdf->pdf    verified 1
[verified_action] move blob.xyz->misc  verified 1   (unknown ext -> /misc, never deleted)
[world_clean]     only protected KEEP files remain

final world:
  txt/a.txt  jpg/b.jpg  pdf/c.pdf  misc/blob.xyz  KEEP_SECRET.pdf  [untouched]
```

Files appeared over time; the daemon cleaned each as it appeared, routed the
unknown `.xyz` to `/misc` instead of deleting it, and **never touched
KEEP_SECRET.pdf**.

## Capabilities (all self-triggered)

| Capability | How |
|---|---|
| **Watch** | polls the real dir every `--poll` s; no command needed |
| **Decide** | grounded: acts on the most abundant loose category (real counts) |
| **Act** | real `shutil.move` to type folders (`txt/jpg/pdf`), unknown → `misc` |
| **Verify** | re-reads the world after every action; only logs `verified_action` if the loose count actually dropped |
| **Learn** | buffers verified `(obs, A, obs_tk)` tuples; at ≥4 runs an anchored-replay macro update and saves `checkpoints/autonomous_best.pt` |
| **Recover** | re-plans from whatever the world now is (files appearing mid-run are picked up) |
| **Protect** | never touches KEEP files (name patterns) — terminal state = "only protected remain" |
| **Cooldown** | won't hammer a failed action |
| **Journal** | every event → `data/autonomous_journal.jsonl` with session id + timestamp |

## Safety (hard rules)
- Never deletes a file. Moves only. Unknown formats → `/misc`, never removed.
- Never touches a file whose name starts with `KEEP` / `IMPORTANT` / `DO_NOT_DELETE`.
- Never operates outside the watched root.
- `max_acts` cap per pass prevents runaway.
- Cooldown before retrying a failed action.

## Usage
```bash
# watch + act forever (fully autonomous)
python -m proteus.agent.autonomous --dir ~/Downloads --daemon --poll 3

# single clean-up pass on a real dir
python -m proteus.agent.autonomous --dir ~/Desktop/messy --once

# seed a messy demo world and clean it
python -m proteus.agent.autonomous --dir /tmp/world --seed --once
```

## What's already proven upstream (the reasoning brain)
- latent H-JEPA world model (prediction, counterfactual rejection, MPC)
- multi-agent latent binding (spatial info survives 32-dim latent)
- OOD transfer + compositional generalization
- continual learning with zero forgetting

## Architecture (full stack)
```
real world dir
  └─ RealEnv (real_env.py): obs/step/verify against actual FS
  └─ AutonomousAgent (autonomous.py): decide/act/verify/learn loop
  └─ H-JEPA latent model (hjepa_torch.py): prediction + counterfactual guard
  └─ continual learning (anchored replay) -> checkpoints/autonomous_best.pt
  └─ journal (data/autonomous_journal.jsonl): full audit trail
```

## Next (to push further)
1. **Real vision**: screencapture → VLM → latent, so it sees instead of counts.
2. **Real language**: Qwen narrates and grounds every verified action.
3. **Multi-agent**: two autonomous processes with complementary views exchanging
   latents over the same real dir.
4. **Policy**: distill the verified trajectories into a reflex policy `π(z)→a`
   for sub-ms decisions.