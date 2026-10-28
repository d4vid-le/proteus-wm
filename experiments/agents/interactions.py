#!/usr/bin/env python3
"""interactions.py — Record ALL user interactions (clicks, typing, mouse, context)
for continual learning.

Captures the full interaction stream the JEPA can learn from:
  - Mouse clicks (button, position, app, window)
  - Mouse moves / drags (trajectory)
  - Keyboard presses (key, modifiers, app)
  - App/window context at event time (NSWorkspace + CGWindowList)
  - Timestamps for temporal structure

Uses CGEventTap (macOS event tap) — needs Accessibility permission.
Events are written to data/interactions.jsonl as a structured learning stream.

Usage:
  python3 interactions.py                 # record forever (Ctrl-C to stop)
  python3 interactions.py --seconds 60    # record 60s then stop
  python3 interactions.py --summary       # summarize what's been captured
"""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import argparse
import json
import pathlib
import sys
import time

OUT = ROOT / "data" / "interactions.jsonl"

# CGEvent constants (from Quartz)
try:
    from Quartz import (
        CGEventTapCreate, CGEventTapLocation, kCGEventTapOptionListenOnly,
        kCGHIDEventTap, CGEventGetIntegerValueField, kCGEventSourceStateHIDSystemState,
        kCGEventMouseMoved, kCGEventLeftMouseDown, kCGEventLeftMouseUp,
        kCGEventRightMouseDown, kCGEventKeyDown, kCGEventKeyUp,
        kCGMouseEventNumber, kCGMouseEventClickState, kCGKeyboardEventKeycode,
        CGEventTapEnable, kCGEventLeftMouseDragged, kCGEventRightMouseDragged,
        kCGEventOtherMouseDown, kCGEventScrollWheel, kCGEventFlagsChanged,
        CGEventGetLocation, kCGMouseEventDeltaX, kCGMouseEventDeltaY,
        CFMachPortCreateRunLoopSource,
    )
    # needed to keep the event tap running
    import ctypes
    from CoreFoundation import (
        CFRunLoopGetCurrent, CFRunLoopAddSource, kCFRunLoopDefaultMode, CFRunLoopRunInMode,
    )
except Exception as e:
    print(f"CGEvent import failed: {e}")
    sys.exit(1)


def active_context():
    """Capture current app + front window at event time."""
    ctx = {}
    try:
        import proteus.perception.telemetry as telemetry
        ctx["app"] = telemetry.active_app()
        ctx["windows"] = len(telemetry.window_bounds())
    except Exception as e:
        ctx["error"] = str(e)[:60]
    return ctx


_secure_cache = {"t": 0.0, "val": False}


def _secure_input():
    """True if the frontmost app's focused element is a secure text field
    (password field). Keycodes are skipped for those so plaintext passwords
    never reach the log. Throttled to ~2Hz (AX calls are not free)."""
    now = time.time()
    if now - _secure_cache["t"] < 0.5:
        return _secure_cache["val"]
    val = False
    try:
        from AppKit import NSWorkspace
        from ApplicationServices import (
            AXUIElementCreateApplication, AXUIElementCopyAttributeValue,
            kAXFocusedUIElementAttribute, kAXRoleAttribute, kAXSecureTextFieldRole,
            kAXValueAttribute,
        )
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        el = AXUIElementCreateApplication(app.processIdentifier())
        err, focused = AXUIElementCopyAttributeValue(el, kAXFocusedUIElementAttribute, None)
        if not err and focused is not None:
            err, role = AXUIElementCopyAttributeValue(focused, kAXRoleAttribute, None)
            if not err and role == kAXSecureTextFieldRole:
                val = True
            if not val:
                err, v = AXUIElementCopyAttributeValue(focused, kAXValueAttribute, None)
                if not err and v and ("•" in v or "●" in v):
                    val = True
    except Exception:
        pass
    _secure_cache["t"] = now
    _secure_cache["val"] = val
    return val


def write_event(kind, **fields):
    rec = {"t": time.time(), "kind": kind, **fields}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


class InteractionRecorder:
    def __init__(self, events_path=OUT, write_apps=True):
        self.events_path = pathlib.Path(events_path)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.last_ctx = {}
        self.tap = None
        self.write_apps = write_apps

    def _ctx(self):
        """Throttled app/window context (only fetch every N events to save CPU)."""
        if self.count % 10 == 0:
            self.last_ctx = active_context()
        return self.last_ctx

    def handle_event(self, event, *args):
        etype = CGEventGetIntegerValueField(event, kCGMouseEventNumber)
        # Get keycode or unicode for keyboard
        kind = None
        fields = {}
        if etype == kCGEventMouseMoved:
            kind = "mouse_move"
            loc = CGEventGetLocation(event)
            fields["x"], fields["y"] = float(loc.x), float(loc.y)
        elif etype == kCGEventLeftMouseDown:
            kind = "click_down"
            fields["button"] = "left"
            loc = CGEventGetLocation(event)
            fields["x"], fields["y"] = float(loc.x), float(loc.y)
            fields["click_state"] = CGEventGetIntegerValueField(event, kCGMouseEventClickState)
        elif etype == kCGEventLeftMouseUp:
            kind = "click_up"
            fields["button"] = "left"
            loc = CGEventGetLocation(event)
            fields["x"], fields["y"] = float(loc.x), float(loc.y)
        elif etype == kCGEventRightMouseDown:
            kind = "click_down"
            fields["button"] = "right"
            loc = CGEventGetLocation(event)
            fields["x"], fields["y"] = float(loc.x), float(loc.y)
        elif etype == kCGEventKeyDown:
            kind = "key_down"
            if not _secure_input():
                fields["keycode"] = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
        elif etype == kCGEventKeyUp:
            kind = "key_up"
            if not _secure_input():
                fields["keycode"] = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
        elif etype == kCGEventScrollWheel:
            kind = "scroll"
        elif etype == kCGEventLeftMouseDragged:
            kind = "drag"
            fields["button"] = "left"
            loc = CGEventGetLocation(event)
            fields["x"], fields["y"] = float(loc.x), float(loc.y)
        if kind is None:
            return event
        ctx = self._ctx()
        rec = write_event(kind, **fields, **ctx)
        self.count += 1
        return event

    def start(self):
        """Start the event tap (needs Accessibility permission). Listen for all
        mouse + keyboard events. Adds a run-loop source so callbacks fire."""
        mask = (1 << kCGEventMouseMoved) | (1 << kCGEventLeftMouseDown) | \
               (1 << kCGEventLeftMouseUp) | (1 << kCGEventRightMouseDown) | \
               (1 << kCGEventKeyDown) | (1 << kCGEventKeyUp) | \
               (1 << kCGEventScrollWheel) | (1 << kCGEventLeftMouseDragged) | \
               (1 << kCGEventRightMouseDragged)
        self.tap = CGEventTapCreate(
            kCGHIDEventTap,
            kCGEventTapOptionListenOnly,
            kCGEventTapOptionListenOnly,
            mask,
            self.handle_event,
            None,
        )
        if self.tap is None:
            print("ERROR: Cannot create event tap — check Accessibility permission.")
            print("System Settings -> Privacy & Security -> Accessibility")
            return False
        CGEventTapEnable(self.tap, True)
        # create the run-loop source from the mach port
        self.runloop_source = CFMachPortCreateRunLoopSource(None, self.tap, 0)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), self.runloop_source, kCFRunLoopDefaultMode)
        print("Interaction recorder started. Recording to", self.events_path)
        return True

    def stop(self):
        if self.tap is not None:
            CGEventTapEnable(self.tap, False)

    def summary(self):
        from collections import Counter
        kinds = Counter()
        n = 0
        if self.events_path.exists():
            for line in self.events_path.read_text().splitlines():
                try:
                    e = json.loads(line)
                    kinds[e["kind"]] += 1
                    n += 1
                except Exception:
                    pass
        return {"total": n, "by_kind": dict(kinds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0, help="record N seconds then stop (0=forever)")
    ap.add_argument("--summary", action="store_true", help="print summary and exit")
    args = ap.parse_args()

    if args.summary:
        r = InteractionRecorder()
        s = r.summary()
        print(f"Interactions captured: {s['total']}")
        for k, v in sorted(s["by_kind"].items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")
        return

    rec = InteractionRecorder()
    if not rec.start():
        return
    try:
        if args.seconds:
            # run the run loop for the duration, then stop
            deadline = time.time() + args.seconds
            while time.time() < deadline:
                CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.5, False)
            rec.stop()
            s = rec.summary()
            print(f"\nCaptured {s['total']} interactions in {args.seconds}s")
            for k, v in sorted(s["by_kind"].items(), key=lambda x: -x[1]):
                print(f"  {k}: {v}")
        else:
            print("Recording... press Ctrl-C to stop")
            while True:
                CFRunLoopRunInMode(kCFRunLoopDefaultMode, 1.0, False)
    except KeyboardInterrupt:
        rec.stop()
        s = rec.summary()
        print(f"\nCaptured {s['total']} interactions total")
        for k, v in sorted(s["by_kind"].items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()