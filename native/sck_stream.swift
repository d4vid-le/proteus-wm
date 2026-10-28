// sck_stream.swift — persistent ScreenCaptureKit stream -> rolling PNG frames
//
// Runs a persistent SCStream (ScreenCaptureKit) that captures frames
// continuously and writes the latest frame to a fixed output file on each
// tick. This is the zero-copy STREAMING path: shareable content is fetched ONCE,
// then frames arrive from the compositor without re-fetching — unlike the
// one-shot tool which re-fetches content every call.
//
// Usage:
//   sck_stream <output.png> [--fps <n>] [--display <id>] [--window <id>] [--width <px>]
//
// Writes output.png continuously. Ctrl-C to stop.
import Foundation
import ScreenCaptureKit
import AppKit

var running = true
signal(SIGINT) { _ in running = false }

func die(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(2)
}

final class Streamer: NSObject, SCStreamOutput, SCStreamDelegate {
    let outURL: URL
    let width: Int?
    private var stream: SCStream?
    private var frameCount = 0
    private var lastWrite = Date.distantPast

    init(outPath: String, width: Int?) {
        self.outURL = URL(fileURLWithPath: outPath)
        self.width = width
        super.init()
    }

    func start(filter: SCContentFilter) {
        let cfg = SCStreamConfiguration()
        cfg.capturesAudio = false
        cfg.showsCursor = false
        if let w = width { cfg.width = w }
        let s = SCStream(filter: filter, configuration: cfg, delegate: self)
        do {
            try s.addStreamOutput(self, type: .screen, sampleHandlerQueue: .main)
        } catch {
            die("addStreamOutput failed: \(error)")
        }
        s.startCapture { error in
            if let error = error { die("startCapture failed: \(error)") }
            print("stream started")
        }
        self.stream = s
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, running else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        // throttle to ~fps (avoid writing every compositor frame)
        let now = Date()
        guard now.timeIntervalSince(lastWrite) >= 0.1 else { return }  // ~10 fps max
        lastWrite = now
        frameCount += 1
        let ci = CIImage(cvPixelBuffer: pixelBuffer)
        let ctx = CIContext()
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return }
        let rep = NSBitmapImageRep(cgImage: cg)
        if let png = rep.representation(using: .png, properties: [:]) {
            // ATOMIC WRITE: write to temp, then rename. On APFS rename() is
            // atomic, so a reader never sees a half-written frame.
            let tempURL = outURL.deletingPathExtension().appendingPathExtension("tmp")
            do {
                try png.write(to: tempURL)
                _ = try? FileManager.default.replaceItemAt(outURL, withItemAt: tempURL)
            } catch {
                // skip this frame; don't crash the stream
            }
        }
        if frameCount % 30 == 0 {
            print("captured \(frameCount) frames -> \(outURL.path)")
        }
    }

    func stop() {
        stream?.stopCapture { _ in }
    }
}

let args = CommandLine.arguments
guard args.count >= 2 else { die("usage: sck_stream <output.png> [--fps n] [--display id] [--window id] [--width px]") }
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

Task {
    do {
        let content = try await SCShareableContent.current
        let streamer = Streamer(outPath: outPath, width: width)
        if let wid = windowID {
            guard let win = content.windows.first(where: { $0.windowID == wid }) else { die("window not found") }
            streamer.start(filter: SCContentFilter(desktopIndependentWindow: win))
        } else if let did = displayID {
            guard let d = content.displays.first(where: { $0.displayID == did }) else { die("display not found") }
            streamer.start(filter: SCContentFilter(display: d, excludingWindows: []))
        } else if let d = content.displays.first {
            streamer.start(filter: SCContentFilter(display: d, excludingWindows: []))
        } else { die("no display") }
        // keep alive
        while running {
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
        exit(0)
    } catch {
        die("failed: \(error)")
    }
}

RunLoop.main.run()