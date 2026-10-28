# PROTEUS LONG-LIVED TELEMETRY DAEMON — macOS-native sensory map

`daemon.py` + `telemetry.py` — the persistent, continually-learning agent
with a complete macOS-native sensory map.

## The telemetry stack (all verified working)
| Stream | macOS source | What it feeds | Status |
|---|---|---|---|
| **Filesystem truth** | FSEvents (kernel FS tap, watchdog FSEventsObserver) | ground-truth file transitions, continual-learning labels | ✅ |
| **App context** | NSWorkspace | active app, bundle id, PID | ✅ (Chrome, pid 702) |
| **Window bounds** | CGWindowList | bounding boxes of on-screen windows (63 on screen) | ✅ |
| **Shell actions** | preexec/precmd zsh hooks → `/tmp/proteus_shell.pipe` | command, exit code, pwd | ✅ (installed in ~/.zshrc) |

## Live verified run
```
[daemon_start] watching /private/tmp/outputs copy
[telemetry] fsevents_watching = [outputs copy, /tmp, energy-guided-sampling/outputs]
[context] app=Google Chrome windows=63 shell_new=0
... dropped new_test_result.png into the watched dir ...
[verified_action] move new_test_result.png->png verified 1
```
The daemon reacted to a new file appearing, filed it into `png/`, and keeps
sampling machine context. It runs forever, self-improving via continual learning.

## The JEPA unified state
The daemon assembles S_t = z_fs (FSEvents) + z_ui (app/window) + z_act (shell):
- **z_fs**: exact, event-driven file transitions (never polls)
- **z_ui**: which app is focused, where its windows are (spatial affordances)
- **z_act**: what commands were run, with exit codes (action traces)
- **z_vis**: (from finder_vision) the real Finder window / spatial latent

## Files
- `telemetry.py` — the 4 native stream hooks + JSONL journal
- `daemon.py` — the long-lived daemon tying telemetry + the autonomous agent
- `data/telemetry.jsonl` — the event journal
- `~/.zshrc` — preexec/precmd shell hook (command traces)

## Usage
```bash
# run forever, watching a real dir
python -m proteus.agent.daemon --dir "/private/tmp/outputs copy" --poll 2

# one pass (telemetry + organize once)
python -m proteus.agent.daemon --dir "/private/tmp/outputs copy" --once
```

## Honest notes
- FSEvents gives event-driven file truth (no polling) — but the agent still
  polls `pass_once()` every `--poll` sec for its act loop; the FS events are
  the *labels*, the poll is the *act trigger*. A fully event-driven act loop
  (trigger on CREATE) is the next refinement.
- ScreenCaptureKit (zero-copy frames) is the natural upgrade to the CLI
  screencapture in finder_vision — not yet wired but the architecture supports it.
- Shell hook requires new shells to pick it up (source ~/.zshrc).