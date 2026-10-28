// sck_capture.swift — ScreenCaptureKit capture-to-PNG (zero-copy from compositor)
//
// Uses ScreenCaptureKit (SCScreenshotManager) to capture a display or window
// directly to a PNG, avoiding the CLI `screencapture` disk bottleneck.
// This is the modern, hardware-accelerated path.
//
// Usage:
//   sck_capture <output.png> [--display <id>] [--window <id>] [--width <px>]
//
// Requires macOS 14+ (SCScreenshotManager). Run with screen-recording permission.
import Foundation
import ScreenCaptureKit
import AppKit

func die(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(2)
}

func main() {
    let args = CommandLine.arguments
    guard args.count >= 2 else {
        die("usage: sck_capture <output.png> [--display <id>] [--window <id>] [--width <px>]")
    }
    let outPath = args[1]

    var displayID: CGDirectDisplayID? = nil
    var windowID: CGWindowID? = nil
    var width: Int? = nil
    var i = 2
    while i < args.count {
        switch args[i] {
        case "--display": if i+1 < args.count { displayID = UInt32(args[i+1]) ?? nil; i += 1 }
        case "--window": if i+1 < args.count { windowID = UInt32(args[i+1]) ?? nil; i += 1 }
        case "--width": if i+1 < args.count { width = Int(args[i+1]) ?? nil; i += 1 }
        default: break
        }
        i += 1
    }

    // Determine capture content
    let sem = DispatchSemaphore(value: 0)
    var resultPNG: Data? = nil

    let config = SCStreamConfiguration()
    if let w = width { config.width = w }

    func grab(_ filter: SCContentFilter) {
        SCScreenshotManager.captureImage(contentFilter: filter, configuration: config) { image, error in
            guard let image = image else {
                FileHandle.standardError.write("capture error: \(String(describing: error))\n".data(using: .utf8)!)
                sem.signal()
                return
            }
            let rep = NSBitmapImageRep(cgImage: image)
            resultPNG = rep.representation(using: .png, properties: [:])
            sem.signal()
        }
    }

    Task {
        do {
            let content = try await SCShareableContent.current
            if let wid = windowID {
                if let win = content.windows.first(where: { $0.windowID == wid }) {
                    grab(SCContentFilter(desktopIndependentWindow: win))
                } else { die("window not found") }
            } else if let did = displayID {
                if let d = content.displays.first(where: { $0.displayID == did }) {
                    grab(SCContentFilter(display: d, excludingWindows: []))
                } else { die("display not found") }
            } else if let d = content.displays.first {
                grab(SCContentFilter(display: d, excludingWindows: []))
            } else { die("no display") }
        } catch {
            die("no content: \(error)")
        }
    }

    _ = sem.wait(timeout: .now() + 10)
    guard let png = resultPNG else { die("capture produced no image") }
    do {
        try png.write(to: URL(fileURLWithPath: outPath))
        print("wrote \(outPath) (\(png.count) bytes)")
    } catch {
        die("write failed: \(error)")
    }
}

main()