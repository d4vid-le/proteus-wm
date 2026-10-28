# K1+K2 — ScreenCaptureKit upgrade + event-driven act loop

## K1 — ScreenCaptureKit (zero-copy frames)
`telemetry.sck_capture()` wraps a compiled Swift tool (`sck_capture.swift`) that
uses **SCScreenshotManager** (ScreenCaptureKit) to capture a display or window
directly to PNG from the compositor — no CLI screencapture disk bottleneck.

Verified:
- Swift tool compiles and captures the screen → valid PNG (765KB)
- Wired into Python: `telemetry.sck_capture()` → spatial encoder → 14×14×256 layout
  latent (std 0.192) — the SCK frame feeds the vision stack.

Honest note: for ONE-SHOT capture, CLI screencapture (0.107s) is actually faster
than SCK (0.212s) because SCK re-fetches shareable content each run. SCK's real
advantage is STREAMING — a persistent SCStream that delivers repeated frames
zero-copy without re-fetching. The one-shot wrapper is the first step; a
streaming capture is the future upgrade.

## K2 — Event-driven act loop (FSEvents trigger, no polling)
`daemon.py` no longer polls `pass_once()` blindly. It:
- **drains FSEvents** from the telemetry journal, acts only on `created`/`moved`
  file events in the watched root (`fs_trigger`)
- **coalesces** burst events into one act pass
- samples app/window/shell context on a 5s cadence
- **goes quiet when the world is clean** — no more "stuck" spam on protected files

Verified live:
```
[fs_trigger] files=['.../event_driven_test.png'] -> move event_driven_test.png->png verified 1
... dropped event_driven_test.png ...
MOVED ✓
stuck count in last lines: 0   <- quiet, event-driven
```

## Files
- `sck_capture.swift` — ScreenCaptureKit capture tool
- `telemetry.py` — `sck_capture()` + `compile_sck()` wrappers
- `daemon.py` — event-driven act loop