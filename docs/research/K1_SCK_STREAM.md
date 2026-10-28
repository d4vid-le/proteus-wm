# K1 — ScreenCaptureKit STREAMING capture (zero-copy frames)

Upgrade from one-shot SCK to a **persistent `SCStream`** — the real advantage:
shareable content is fetched ONCE, then frames arrive from the compositor
continuously without re-fetching.

## Verified
- `sck_stream.swift` compiled → persistent SCStream → rolling frame file
- **frames update live**: sizes change over time (191847→193854→194010 bytes) =
  real streaming, not a static capture
- Python `telemetry.SCKStream` wrapper: start → first frame lands → grab latest →
  feed spatial encoder → valid 14×14×256 layout latent (std 0.213) → stop cleanly

## Wired into the daemon (`daemon.py --vision-stream`)
```
[daemon_start]
[vision_stream] started: True
[fs_trigger] 4 files (event-driven)
[context] app=OpenCode windows=69 zvis_std=0.2127   <- spatial latent from STREAMED frame
```
The daemon now continuously samples z_vis (the spatial latent) from the zero-copy
stream, alongside FSEvents (z_fs), NSWorkspace (z_ui), and shell (z_act) — the
full JEPA unified state S_t, event-driven and streamed.

## Honest notes
- **Race-condition fix (critical)**: reading the frame while Swift was mid-write
  gave `zvis_std: null` + "unrecognized data stream contents" (Pillow). Fixed two
  ways: (1) **atomic rename in Swift** — write to `.tmp` then `replaceItemAt`
  (APFS rename is atomic, reader never sees a half-written frame); (2) **Python
  retry guard** (`telemetry.safe_read_image`) — verify + retry on mid-write.
  Result: **0/8 null reads**, z_vis std healthy (~0.212) on every tick, 0 errors.
- The vision-stream FSEvents noise is filtered out of the journal (the frame is
  rewritten ~10x/sec; only meaningful file events are logged).
- Swift stdout is buffered; the frame file mtime/size is the reliable signal.

## Files
- `sck_stream.swift` — the persistent SCStream tool (atomic rename write)
- `telemetry.py` — `SCKStream` class + `safe_read_image` race-safe reader + FSEvents noise filter
- `daemon.py` — `--vision-stream` flag wiring z_vis into the loop
- `finder_vision.py` — race-safe `encode_spatial` via the safe reader