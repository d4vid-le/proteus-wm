# Developer Artifact Janitor — reclaim dead build cache

Status (2026-09-04): **implemented + verified in sandbox; real scan done.**

A new proteus "world" that reclaims disk space by clearing dead build cache
and virtual environments from dormant dev projects. Slots into the existing
JEPA stack: 12-dim obs, 4 actions, same `HJEPA_K4(obs_dim=12, act_dim=4)`
latent, safety barrier, and journal format.

## State (obs, 12-dim)
```
[root_pycache, root_nodemod, root_cargo, root_venv,
 pycache_n, nodemod_n, cargo_n, venv_n,
 stale_repos, has_protected, reclaimable_gb, clean_complete]
```

## Actions (4)
| idx | action | what it removes |
|---|---|---|
| 0 | `clean_pycache` | `__pycache__/`, `*.pyc`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache` |
| 1 | `clean_node_modules` | `node_modules/` in stale repos |
| 2 | `cargo_clean` | `target/` (Rust build output) |
| 3 | `clean_venv` | `venv/`, `.venv/`, `env/` |

`build/` and `dist/` are deliberately EXCLUDED — some projects commit them as
source, so they're ambiguous.

## Staleness
A project is "dormant" if it has a VCS/manifest marker (`.git`, `package.json`,
`Cargo.toml`, `pyproject.toml`, ...) AND:
- git project: last commit **author date** > 30 days (mtime is unreliable —
  git checkout updates it), OR
- non-git project: newest non-cache file mtime > 30 days.

## Safety barrier (mirrors the KEEP barrier)
- Only known-safe, regenerable cache dirs are ever removed.
- A repo is skipped entirely if `git status --porcelain` shows ANY change to
  tracked files (untracked `.DS_Store`/caches don't count).
- Never touches KEEP/IMPORTANT/DO_NOT_DELETE files or `.git`.
- `dry_run=True` by default; `--execute` required to actually delete.

## Verification loop
`du -sk` before/after each action. If reclaimed > 500MB AND git stays
pristine → `verified_action` logged to `autonomous_journal.jsonl` with
`obs` (12) + `latent` (64) + `reclaimed_bytes`, so the JEPA can learn and
Qwen can explain. Small/no-op actions are NOT logged.

## Real scan (~/Development)
33 dormant projects, **~1,014 MB reclaimable** (107 `node_modules` dirs).
Qwen's grounded explanation:
> "To reclaim 1014.7 MB of disk space, clean up the 107 node modules
> directories."

## Files
| file | role |
|---|---|
| `dev_janitor.py` | the world: scan / stale-detect / actions / verify / daemon |
| `monitor.py` | shows a JANITOR section (reclaimed MB, verified actions) |

## Run it
```bash
python experiments/agents/dev_janitor.py --root ~/Development --scan          # report only
python experiments/agents/dev_janitor.py --root ~/Development --clean --dry-run  # what WOULD be deleted
python experiments/agents/dev_janitor.py --root ~/Development --clean --execute   # actually reclaim
python experiments/agents/dev_janitor.py --root ~/Development --explain          # Qwen explanation
python experiments/agents/dev_janitor.py --root ~/Development --daemon --interval 3600  # scheduled (dry-run)
```

## Notes / caveats
- The JEPA→Qwen bridge is trained on the file-cleanup world, so the janitor's
  explanation is grounded in the actual scan facts (stale count, reclaimable
  MB) rather than the latent — the bridge would need janitor-world training
  data to steer it.
- Verified in a sandbox: 600MB node_modules → `verified: 1` → journal entry
  with 64-dim latent; small actions correctly skipped; tracked files intact.
- Real `--execute` on ~/Development is destructive (deletes node_modules in
  107 dormant repos) — run `--dry-run` first and confirm the list.