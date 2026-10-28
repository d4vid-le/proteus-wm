// input_tap.swift — Capture user interactions (clicks, typing, mouse) to JSONL.
//
// A robust CGEventTap (the Quartz event tap) that listens for mouse and
// keyboard events system-wide and appends them to a JSONL file. Handles the
// run loop + permissions properly (unlike the Python CGEventTap attempt).
//
// PRIVACY: by default keycodes are NOT written — only that a key was pressed
// (kind + app + timestamp). Pass --keycodes to also record the keycode, but
// even then keystrokes in secure text fields (password fields, detected via
// the Accessibility API) are redacted to {secure: true} with no keycode.
//
// Requires: Accessibility permission for the terminal/Python process.
//
// Usage:
//   swiftc -O input_tap.swift -o /tmp/input_tap
//   /tmp/input_tap <output.jsonl> [--seconds N] [--keycodes]
import Foundation
import CoreGraphics
import AppKit
import ApplicationServices

// JSONL writer (thread-safe via a simple lock)
final class JSONLWriter {
    let fh: FileHandle
    let lock = NSLock()
    init(path: String) {
        FileManager.default.createFile(atPath: path, contents: nil)
        self.fh = FileHandle(forWritingAtPath: path)!
    }
    func write(_ dict: [String: Any]) {
        lock.lock()
        if let data = try? JSONSerialization.data(withJSONObject: dict),
           let s = String(data: data, encoding: .utf8) {
            fh.write((s + "\n").data(using: .utf8)!)
        }
        lock.unlock()
    }
}

// App/window context at event time
func activeAppName() -> String {
    return NSWorkspace.shared.frontmostApplication?.localizedName ?? "?"
}

// True if the frontmost app's focused element is a secure text field
// (password field, etc.). macOS exposes this via the Accessibility API.
func isSecureInput() -> Bool {
    guard let app = NSWorkspace.shared.frontmostApplication else { return false }
    let appRef = AXUIElementCreateApplication(app.processIdentifier)
    var focused: CFTypeRef?
    guard AXUIElementCopyAttributeValue(appRef, kAXFocusedUIElementAttribute as CFString, &focused) == .success,
          let focusedEl = focused else { return false }
    let el = focusedEl as! AXUIElement
    var role: CFTypeRef?
    if AXUIElementCopyAttributeValue(el, kAXRoleAttribute as CFString, &role) == .success,
       let roleStr = role as? String, roleStr == "AXSecureTextField" {
        return true
    }
    // Some apps don't set the secure role but mask the value with bullets.
    var value: CFTypeRef?
    if AXUIElementCopyAttributeValue(el, kAXValueAttribute as CFString, &value) == .success,
       let val = value as? String, val.contains("•") || val.contains("●") {
        return true
    }
    return false
}

// Throttled secure-field check (AX calls are not free; cache ~0.5s).
var lastSecureCheck = Date.distantPast
var lastSecureResult = false
func isSecureInputCached() -> Bool {
    if Date().timeIntervalSince(lastSecureCheck) > 0.5 {
        lastSecureResult = isSecureInput()
        lastSecureCheck = Date()
    }
    return lastSecureResult
}

// Whether to record keycodes (global so the C callback can read it without
// capturing context). Set from the --keycodes flag.
var gLogKeycodes = false

// The event tap callback
func makeCallback(_ writer: JSONLWriter) -> CGEventTapCallBack {
    return { proxy, type, event, refcon in
        let w = Unmanaged<JSONLWriter>.fromOpaque(refcon!).takeUnretainedValue()
        let now = Date().timeIntervalSince1970
        var rec: [String: Any] = ["t": now, "app": activeAppName()]

        switch type {
        case .mouseMoved:
            let p = event.location
            rec["kind"] = "mouse_move"; rec["x"] = p.x; rec["y"] = p.y
        case .leftMouseDown:
            let p = event.location
            rec["kind"] = "click_down"; rec["button"] = "left"; rec["x"] = p.x; rec["y"] = p.y
            rec["click_count"] = event.getIntegerValueField(.mouseEventClickState)
        case .leftMouseUp:
            let p = event.location
            rec["kind"] = "click_up"; rec["button"] = "left"; rec["x"] = p.x; rec["y"] = p.y
        case .rightMouseDown:
            let p = event.location
            rec["kind"] = "click_down"; rec["button"] = "right"; rec["x"] = p.x; rec["y"] = p.y
        case .keyDown:
            rec["kind"] = "key_down"
            if isSecureInputCached() {
                rec["secure"] = true          // redact: no keycode
            } else if gLogKeycodes {
                rec["keycode"] = event.getIntegerValueField(.keyboardEventKeycode)
            }
        case .keyUp:
            rec["kind"] = "key_up"
            if isSecureInputCached() {
                rec["secure"] = true
            } else if gLogKeycodes {
                rec["keycode"] = event.getIntegerValueField(.keyboardEventKeycode)
            }
        case .leftMouseDragged:
            let p = event.location
            rec["kind"] = "drag"; rec["button"] = "left"; rec["x"] = p.x; rec["y"] = p.y
        case .scrollWheel:
            rec["kind"] = "scroll"
        default:
            return Unmanaged.passRetained(event)
        }
        w.write(rec)
        return Unmanaged.passRetained(event)
    }
}

let args = CommandLine.arguments
guard args.count >= 2 else {
    print("usage: input_tap <output.jsonl> [--seconds N] [--keycodes]")
    exit(2)
}
let outPath = args[1]
var seconds: Double = 0
if let i = args.firstIndex(of: "--seconds"), i + 1 < args.count {
    seconds = Double(args[i + 1]) ?? 0
}
let logKeycodes = args.contains("--keycodes")
gLogKeycodes = logKeycodes

let writer = JSONLWriter(path: outPath)
let ref = UnsafeMutableRawPointer(Unmanaged.passRetained(writer).toOpaque())

// Listen to all events we care about
let mask = (1 << CGEventType.mouseMoved.rawValue) |
           (1 << CGEventType.leftMouseDown.rawValue) |
           (1 << CGEventType.leftMouseUp.rawValue) |
           (1 << CGEventType.rightMouseDown.rawValue) |
           (1 << CGEventType.keyDown.rawValue) |
           (1 << CGEventType.keyUp.rawValue) |
           (1 << CGEventType.leftMouseDragged.rawValue) |
           (1 << CGEventType.scrollWheel.rawValue)

let callback = makeCallback(writer)
guard let tap = CGEvent.tapCreate(
    tap: .cghidEventTap,
    place: .headInsertEventTap,
    options: .listenOnly,
    eventsOfInterest: CGEventMask(mask),
    callback: callback,
    userInfo: ref
) else {
    print("ERROR: cannot create event tap — check Accessibility permission.")
    exit(1)
}

let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
CFRunLoopAddSource(CFRunLoopGetCurrent(), source, .defaultMode)
CGEvent.tapEnable(tap: tap, enable: true)
print("input_tap started -> \(outPath)  keycodes=\(logKeycodes ? "ON" : "OFF")")

if seconds > 0 {
    let deadline = Date().addingTimeInterval(seconds)
    while Date() < deadline {
        RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.2))
    }
} else {
    print("recording... Ctrl-C to stop")
    RunLoop.current.run()  // run forever
}
print("stopped")
