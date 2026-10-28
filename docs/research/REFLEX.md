# REFLEX — surprise-triggered corrective actions (intrinsic motivation)

`reflex.py` closes the final loop: surprise from the JEPA telemetry predictor
isn't just logged — it **triggers real corrective action**.

## The loop
```
JEPA predictor (Mode-1, fast) → surprise = ||z_pred − z_target||²
  → ERIS trace (timestamped)
  → reflex watches:  surprise > threshold?
      LOW  → Mode-1 handles silently (save compute)
      HIGH → wake Mode-2 (Qwen) → hypothesize cause
           → Mode-1 correct(): probe real OS state + investigate + act
           → verified schema → ERIS memory
```

## Corrective actions (verified)
On a surprise spike, the reflex:
1. **probe-system-state** — real app + window count (Chrome, pid 702, 63 windows)
2. **check-system-log-errors** (big spikes) — reads real macOS logs, found:
   - `duetexpertd` BiomeFoundation file-handler errors
   - `launchd` failed lookup: `com.apple.tiswitcher`
3. **check-top-processes** — CPU 5.69% user, 83.78% idle
4. **dry-run / execute** — `--dry-run` logs only (safe default); `--execute` allows mutation

## The safety model
- **Default = dry-run**: the reflex investigates and logs but never mutates.
  `--execute` is the explicit opt-in for real corrective mutations.
- All probe actions are read-only (system state, logs, top). No destructive
  operations are wired in by default.

## The three honesty fixes (per review)

1. **Daemon invocation** — `--daemon` isn't a flag; the default (no `--once`) is
   the loop. To live on the Mac:
   ```bash
   nohup python -m proteus.agent.reflex --std-mult 5 &
   # or a launchd plist
   ```
2. **Adaptive threshold is the default** — `--std-mult k` (default 3) triggers on
   rolling mean + k·σ. The ambient floor runs 0.0009–0.0013; a fixed threshold
   below that (0.0005) would wake Mode-2 on every event. Now guarded: a fixed
   threshold below 0.001 is raised to 0.001. Verified: 0.00084/0.00090/0.00095 →
   silent; 0.00102 → triggers; 0.13981 (≈100σ) fires under any sane k.
3. **Hypotheses are VERIFIED, not assumed** — Qwen's answer is abduction. Before
   it lands as "verified", the reflex probes cheap ground-truth:
   - `log show --predicate 'crash OR fault'` (crash/fault lines)
   - `ps -axo pid,comm` (process count)
   - status = `verified` only if the probe corroborates; else `unverified-abduction`.

## The honesty metric
**Verification rate** = fraction of Mode-2 hypotheses the probe confirms.
Printed LIVE every cycle in daemon mode: `[verification rate] N/M (P%) | pending K`.

## Deferred verification (the unverified lifecycle)
"Unverified" is a PENDING QUEUE, not a dead end. Crash logs land late, so an
abduction that failed its first probe may be corroborated minutes later:
- re-probe after N minutes (default 120s)
- **promote** → verified if the late crash/fault log corroborates
- **decay** → dropped after 10x the delay if never confirmed
Verified: unverified → reprobe → promoted (unverified 1→0, verified 0→1).

This is dual grounding operationalized: fast latent intuition (Mode-2 Qwen)
proposes, hard-wired kernel truth (probe) disposes.

## Usage
```bash
# watch + investigate (safe dry-run), adaptive threshold, UNBUFFERED
nohup python -u -m proteus.agent.reflex --std-mult 5 --correct &

# allow real corrective mutations
python -u -m proteus.agent.reflex --correct --execute

# one pass (no daemon)
python -m proteus.agent.reflex --once
```
Note: use `-u` for unbuffered output so the live verification rate flushes to the log.

## Files
- `reflex.py` — SurpriseReflex: reason (hypothesize+verify) + correct + reprobe (deferred)
- `data/eris_trace.jsonl` — timestamped surprise stream
- `data/eris_memory.jsonl` — verified/unverified schemas with probe evidence