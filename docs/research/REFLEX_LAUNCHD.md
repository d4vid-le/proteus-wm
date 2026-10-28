# Running the reflex unattended (launchd)

The reflex is now a **launchd agent** — it survives reboots, restarts on crash,
and logs to a file (no nohup needed).

## Status: RUNNING
```
launchctl list | grep proteus
21785  0  com.proteus.reflex     (PID 21785, exit 0 = healthy)
```
Log: `data/reflex_daemon.log` — live verification rate every cycle.

## Manage it
```bash
# stop it
launchctl unload ~/Library/LaunchAgents/com.proteus.reflex.plist

# start it again
launchctl load ~/Library/LaunchAgents/com.proteus.reflex.plist

# check status
launchctl list | grep proteus

# watch the live honesty metric
tail -f ~/proteus/data/reflex_daemon.log

# edit the config (e.g. --std-mult, --execute), then reload:
#   edit the plist in ~/Library/LaunchAgents/ then:
launchctl unload ~/Library/LaunchAgents/com.proteus.reflex.plist
launchctl load ~/Library/LaunchAgents/com.proteus.reflex.plist
```

## What it runs
```
python -u -m proteus.agent.reflex --std-mult 5 --correct
```
- **adaptive surprise threshold** (mean + 5σ) — ambient ~0.001 is silent, real
  spikes fire
- **corrective actions** enabled (dry-run by default: probe + log, no mutation)
- **deferred verification** — unverified abductions re-probed, promoted/decayed
- **live verification rate** — the honesty metric, printed every cycle

## Files
- `~/Library/LaunchAgents/com.proteus.reflex.plist` — the launchd agent
- `data/reflex_daemon.log` / `.err` — stdout/stderr
- `data/eris_trace.jsonl` — surprise stream
- `data/eris_memory.jsonl` — verified/unverified schemas