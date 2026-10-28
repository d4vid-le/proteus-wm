#!/usr/bin/env python3
"""telemetry.py — macOS-native telemetry streams for proteus.

Hooks the 4 highest-leverage macOS APIs into a unified, event-driven sensory
map that feeds the JEPA unified state S_t = z_fs + z_ui + z_act + z_vis.

  1. FILESYSTEM TRUTH — FSEvents (kernel FS tap, <1ms, event-driven)
     watchdog's FSEventsObserver: CREATE/MODIFY/DELETE/RENAME/MOVE events
     -> ground-truth file transitions for continual learning labels.
  2. APP CONTEXT — NSWorkspace (active app, bundle id, PID)
  3. WINDOW BOUNDS — CGWindowList (bounding boxes of on-screen windows)
  4. SHELL ACTIONS — reads /tmp/proteus_shell.pipe (preexec/precmd hooks)

All events are appended to a JSONL journal the daemon ingests.
"""
from proteus.paths import ROOT
import json
import pathlib
import subprocess
import sys
import time

JOURNAL = ROOT / "data" / "telemetry.jsonl"


def log(stream, **kw):
    rec = {"t": time.time(), "stream": stream, **kw}
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


# ---------------- 1. FSEvents (filesystem truth) ----------------
# Paths to ignore: high-frequency self-generated noise (the vision stream frame
# is rewritten ~10x/sec by ScreenCaptureKit, which would flood the journal).
IGNORE_SUBSTRINGS = ("proteus_vision_stream", ".DS_Store", "proteus_shell.pipe")


def watch_fsevents(paths, callback=None, recursive=True):
    """Watch real directories via the kernel FSEvents tap (no polling)."""
    from watchdog.observers.fsevents import FSEventsObserver
    from watchdog.events import FileSystemEventHandler

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            # skip self-generated noise (vision stream, DS_Store, shell pipe)
            p = str(event.src_path)
            if any(s in p for s in IGNORE_SUBSTRINGS):
                return
            ev = {
                "kind": event.event_type,  # created/modified/deleted/moved
                "path": p,
                "is_dir": event.is_directory,
            }
            if event.event_type == "moved":
                ev["dest"] = str(event.dest_path)
            log("fsevents", **ev)
            if callback:
                callback(ev)
            return ev

    observer = FSEventsObserver()
    handler = Handler()
    for p in paths:
        observer.schedule(handler, str(p), recursive=recursive)
    observer.start()
    return observer


# ---------------- 2. App context (NSWorkspace) ----------------
def active_app():
    from AppKit import NSWorkspace
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return {
        "name": app.localizedName(),
        "bundle": app.bundleIdentifier(),
        "pid": app.processIdentifier(),
    }


# ---------------- 3. Window bounds (CGWindowList) ----------------
def window_bounds():
    from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
    wins = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    out = []
    for w in wins or []:
        bounds = w.get("kCGWindowBounds", {})
        out.append({
            "owner": w.get("kCGWindowOwnerName", ""),
            "layer": w.get("kCGWindowLayer", 0),
            "bounds": [bounds.get("X", 0), bounds.get("Y", 0),
                       bounds.get("Width", 0), bounds.get("Height", 0)],
        })
    return out


# ---------------- 4. Shell actions (preexec/precmd pipe) ----------------
def read_shell_pipe(pipe_path="/tmp/proteus_shell.pipe", nonblock=True):
    """Read shell action traces appended by the preexec/precmd zsh hooks.
    The pipe is a JSONL file; we read new lines since last read."""
    p = pathlib.Path(pipe_path)
    if not p.exists():
        return []
    lines = p.read_text().splitlines()
    if not lines:
        return []
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return events


def sample_system_state():
    """One snapshot of the current machine state (the z_ui + z_act part)."""
    state = {"app": None, "windows": 0, "shell": []}
    try:
        state["app"] = active_app()
    except Exception as e:
        state["app"] = {"error": str(e)[:50]}
    try:
        state["windows"] = len(window_bounds())
    except Exception as e:
        state["windows"] = -1
    return state


# ---------------- ScreenCaptureKit (zero-copy frames) ----------------
SCK_BIN = "/tmp/sck_capture"  # compiled Swift ScreenCaptureKit one-shot tool
SCK_STREAM_BIN = "/tmp/sck_stream"  # compiled Swift streaming tool


def compile_sck():
    """Compile the Swift ScreenCaptureKit tools if not present."""
    import subprocess as sp
    ok = True
    for src, dst in (("sck_capture.swift", SCK_BIN), ("sck_stream.swift", SCK_STREAM_BIN)):
        src_p = ROOT / src
        if pathlib.Path(dst).exists() or not src_p.exists():
            continue
        r = sp.run(["swiftc", "-O", str(src_p), "-o", dst], capture_output=True, text=True)
        ok = ok and r.returncode == 0
    return ok


def safe_read_image(path_or_bytes, retries=3, delay=0.02):
    """Read an image safely, retrying if it's mid-write (race between the Swift
    stream writing a frame and Python reading it). Returns a PIL Image or None."""
    from PIL import Image
    for attempt in range(retries):
        try:
            img = Image.open(path_or_bytes)
            img.verify()          # quick integrity check
            return Image.open(path_or_bytes)  # re-open after verify
        except (OSError, Exception):
            if attempt < retries - 1:
                time.sleep(delay)
    return None


def sck_capture(out_path, window_id=None, display_id=None, width=None):
    """Capture a frame via ScreenCaptureKit (zero-copy from the compositor).
    Returns True on success. Uses the compiled Swift one-shot tool."""
    import subprocess
    cmd = [SCK_BIN, str(out_path)]
    if window_id is not None:
        cmd += ["--window", str(window_id)]
    if display_id is not None:
        cmd += ["--display", str(display_id)]
    if width is not None:
        cmd += ["--width", str(width)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    return r.returncode == 0 and pathlib.Path(out_path).exists()


class SCKStream:
    """Persistent ScreenCaptureKit stream. Writes the latest frame to a rolling
    file continuously (zero-copy frames from the compositor, no re-fetch)."""

    def __init__(self, out_path, width=None, display_id=None, window_id=None):
        self.out = pathlib.Path(out_path)
        self.bin = SCK_STREAM_BIN
        self.cmd = [self.bin, str(out_path)]
        if width:
            self.cmd += ["--width", str(width)]
        if display_id is not None:
            self.cmd += ["--display", str(display_id)]
        if window_id is not None:
            self.cmd += ["--window", str(window_id)]
        self.proc = None

    def start(self):
        import subprocess as sp
        if not pathlib.Path(self.bin).exists():
            compile_sck()
        self.proc = sp.Popen(self.cmd, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        # wait for the first frame to land
        import time
        for _ in range(30):
            if self.out.exists() and self.out.stat().st_size > 1000:
                return True
            time.sleep(0.2)
        return False

    def latest(self):
        """Return the path to the current frame (always fresh)."""
        return self.out if self.out.exists() else None

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None


if __name__ == "__main__":
    print("=" * 60)
    print("MACOS TELEMETRY STREAM TEST")
    print("=" * 60)
    s = sample_system_state()
    print(f"\n[app] {s['app']}")
    print(f"[windows] {s['windows']} on screen")
    print(f"[shell] {read_shell_pipe()} (from preexec/precmd hooks if any)")
    # brief FSEvents watch demo
    print("\n[starting FSEvents watch on /tmp/proteus-finder for 3s...]")
    obs = watch_fsevents(["/tmp/proteus-finder"])
    time.sleep(3)
    obs.stop()
    obs.join(timeout=5)
    print("\n[telemetry] events written to", JOURNAL)
    print("=" * 60)