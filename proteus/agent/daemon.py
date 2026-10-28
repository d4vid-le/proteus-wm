#!/usr/bin/env python3
"""daemon.py — The long-lived, continually-learning proteus daemon.

Combines the autonomous agent with the macOS telemetry streams:
  - FSEvents: event-driven file truth (no polling) on the watched dir
  - NSWorkspace + CGWindowList: app/window context (the z_ui part)
  - Shell pipe: command/action traces (the z_act part)
  - The autonomous agent: organizes the dir, learns new types, protects artifacts
  - Continual learning: every verified action feeds the model

The daemon runs FOREVER, reacting to real events and self-improving.

Usage:
  python -m proteus.agent.daemon --dir sandbox --daemon
  (Ctrl-C to stop)
"""
import argparse
import pathlib
import sys
import time

import proteus.perception.telemetry as telemetry
from proteus.agent.autonomous import AutonomousAgent


class ProteusDaemon:
    def __init__(self, root, poll=2.0, protect_ext=(".glb", ".json", ".pt"),
                 journal=telemetry.JOURNAL, vision_stream=False):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.agent = AutonomousAgent(str(root), poll=poll,
                                     protect_ext=protect_ext)
        self.telemetry = telemetry
        self.shell_seen = 0
        self.fs_events = 0
        self.vision_stream = vision_stream
        self.vstream = None

    def start_telemetry(self):
        """Hook the FSEvents kernel stream for the watched dir only. Watching
        /tmp broadly causes endless 'modified' spam (SCK frames, temp files) —
        only the target dir's events matter for acting."""
        dirs = [str(self.root)]
        self.obs = telemetry.watch_fsevents(dirs, callback=self.on_fs_event)
        self.agent.log("telemetry", fsevents_watching=dirs)
        if self.vision_stream:
            # persistent ScreenCaptureKit stream -> rolling frame for the spatial encoder
            telemetry.compile_sck()
            self.vstream = telemetry.SCKStream("/tmp/proteus_vision_stream.png", width=640)
            ok = self.vstream.start()
            self.agent.log("vision_stream", started=ok)
        return self.obs

    def vision_latent(self):
        """Feed the current streamed frame to the spatial encoder (z_vis)."""
        if not self.vstream:
            return None
        f = self.vstream.latest()
        if not f:
            return None
        try:
            from proteus.perception.finder_vision import SpatialEncoder
            enc = SpatialEncoder()
            return enc.encode_spatial(f)
        except Exception as e:
            self.agent.log("vision_stream_err", exc=str(e)[:80])
            return None

    def on_fs_event(self, ev):
        self.fs_events += 1
        # if a file appeared in the watched root, trigger a cleanup pass
        if ev.get("is_dir") is False and ev.get("kind") in ("created", "moved"):
            if str(self.root) in str(ev.get("path", "")) and "/" not in ev.get("path", "").replace(str(self.root), "").strip("/").split("/")[0][0:0]:
                pass
        return ev

    def sample_context(self):
        """One unified system-state snapshot: z_ui (app/windows) + z_act (shell)."""
        s = telemetry.sample_system_state()
        shell = telemetry.read_shell_pipe()
        new_shell = shell[self.shell_seen:]
        self.shell_seen = len(shell)
        for c in new_shell:
            self.agent.log("shell_action", **{k: c.get(k) for k in ("cmd", "exit", "pwd")})
        s["shell_new"] = len(new_shell)
        return s

    def run(self, daemon=True):
        self.agent.log("daemon_start", root=str(self.root))
        self.start_telemetry()
        # event-driven: FSEvents CREATE/MOVE in the watched root triggers an act
        pending = []
        last_context = 0
        try:
            while True:
                # 1. drain FS events from the journal, act only on relevant ones
                acted = False
                for ev in self._drain_fs_events():
                    if ev.get("is_dir") is False and ev.get("kind") in ("created", "moved"):
                        p = ev.get("path", "")
                        if str(self.root) in p and not any(
                                seg in p.split("/")[-2:-1] or p.split("/")[-1] in ("",)
                                for seg in []):
                            pending.append(p)
                if pending:
                    # coalesce: one act pass for all pending files
                    self.agent.log("fs_trigger", files=pending[:5], count=len(pending))
                    self.agent.pass_once()
                    pending = []
                    acted = True

                # 2. sample app/window/shell context — but ONLY when there's
                #    activity. When the world is clean and idle, go SILENT
                #    (no context spam). Log on a slow cadence only if active.
                now = time.time()
                busy = acted or not self._world_clean()
                if busy and now - last_context >= 5.0:
                    ctx = self.sample_context()
                    zvis = self.vision_latent()
                    self.agent.log("context", app=(ctx.get("app") or {}).get("name"),
                                   windows=ctx.get("windows"), shell_new=ctx.get("shell_new", 0),
                                   zvis_std=round(float(zvis.std()), 4) if zvis is not None else None)
                    last_context = now

                # 3. if not event-driven-triggered, still do a slow sweep when idle
                #    so the initial state gets organized, then go quiet (no "stuck" spam)
                if not acted and not self._world_clean():
                    self.agent.pass_once()

                if not daemon:
                    break
                time.sleep(1.0 if busy else 3.0)  # faster when busy, slow when idle
        except KeyboardInterrupt:
            self.agent.log("daemon_stop", fs_events=self.fs_events)
        finally:
            try:
                self.obs.stop()
                self.obs.join(timeout=5)
            except Exception:
                pass
            if self.vstream:
                try:
                    self.vstream.stop()
                except Exception:
                    pass

    def _drain_fs_events(self):
        """Read new FSEvents entries from the telemetry journal since last read."""
        import json as _json
        lines = telemetry.JOURNAL.read_text().splitlines() if telemetry.JOURNAL.exists() else []
        new = lines[self._journal_pos:] if hasattr(self, "_journal_pos") else lines
        self._journal_pos = len(lines)
        out = []
        for ln in new:
            try:
                rec = _json.loads(ln)
                if rec.get("stream") == "fsevents":
                    out.append(rec)
            except Exception:
                continue
        return out

    def _world_clean(self):
        """True if the watched root has no loose (non-protected) files."""
        loose = [p for p in self.root.glob("*")
                 if p.is_file() and not self.agent.env.is_keep(p.name)
                 and p.suffix.lower() not in self.agent.protect_ext]
        return len(loose) == 0

    def save(self):
        self.agent.save_state()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="sandbox")
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--protect-ext", default=".glb,.json,.pt")
    ap.add_argument("--vision-stream", action="store_true",
                    help="enable persistent ScreenCaptureKit vision stream")
    args = ap.parse_args()

    print("=" * 64)
    print("PROTEUS LONG-LIVED TELEMETRY DAEMON")
    print("=" * 64)
    print(f"  watching: {args.dir}")
    print(f"  telemetry: FSEvents + NSWorkspace + shell pipe"
          + (" + SCK vision stream" if args.vision_stream else ""))
    d = ProteusDaemon(args.dir, poll=args.poll,
                   protect_ext=tuple(e.strip() for e in args.protect_ext.split(",")),
                   vision_stream=args.vision_stream)
    d.run(daemon=not args.once)
    d.save()
    print("\n  final stats:", d.agent.stats)


if __name__ == "__main__":
    main()