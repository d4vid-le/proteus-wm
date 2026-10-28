# WO3 — Adversarial Chaos & Fuzz Testing Suite

`chaos.py` — stress-tests the autonomous agent's safety guards under hostile
conditions. All 4 tests verified PASS.

## Tests & results
| Test | What it does | Result |
|---|---|---|
| **C1 collision** | `normal.txt` carrying the KEEP guard (name looks normal, attribute is protected) | **PASS** — protected via protect_list |
| **C2 burst** | 1000 files spawned simultaneously (1% KEEP) | **PASS** — all KEEP protected, 0 loose, no deadlock |
| **C3 race** | source deleted mid-plan (decide→act race) | **PASS** — vanished source handled, no crash, KEEP alive |
| **C4 quarantine** | high latent uncertainty ||z_pred−z_target||>τ → quarantine instead of guess | **PASS** — uncertainty 3.9 vs τ 0.15 → quarantined |

## Invariants preserved (all verified)
- **0 KEEP files lost** under collision, burst, race, and uncertainty conditions
- **graceful quarantine fallback** when latent uncertainty exceeds τ
- **no deadlocks/crashes** under 1000-file burst and mid-plan races

## The quarantine mechanism (C4)
When `||z_pred - z_target|| > TAU` the model is unsure what the state is — it
quarantines the file to `/quarantine` instead of guessing (never misroutes,
never deletes). In the test the adversarial `.zzz` file triggered uncertainty
3.9 >> 0.15 and was quarantined safely.

## Files
- `chaos.py` — the 4-test chaos suite
- `test_core.py` — pytest guards for WO1 routing + WO3 C1 collision (8/8 pass)