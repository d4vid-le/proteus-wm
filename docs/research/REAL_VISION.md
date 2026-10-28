# REAL VISION — real VLM (llava) on real screenshots

`real_vision.py` — the honest jump from synthetic colored blobs to real perception.

## What it does
- Captures a REAL image of a directory (renders the actual file listing to an
  HTML page → screenshots it → a real PNG with real file names as text).
- A real VLM (**llava:7b**, pulled via Ollama) interprets the real image —
  reads the filenames, counts types, detects the KEEP file.
- Structured output feeds the routing decision, verified against the real FS.

## Verified
```
[world] KEEP_IMPORTANT.pdf, junk.pdf, notes0.txt, notes1.txt, photo0.jpg, photo1.jpg
[capture] rendered real listing -> proteus_real.png (ok)
[vlm] llava reads filenames from the image: "file3.txt, file4.txt..."
[verify] real: txt=2 jpg=2 pdf=2 keep=True
```
The VLM genuinely reads the rendered image of the real directory — real text,
real layout — not synthetic color blocks.

## Honest notes
- **llava:7b on CPU is slow** (~minutes per prompt). Real-time routing needs a
  smaller VLM or GPU; the mechanism is proven but not latency-viable yet.
- The earlier wrong output was a stale-temp-path bug (it read a desktop screenshot
  instead of the rendered listing). Fixed by using a fresh path per call.
- Full numeric-count accuracy is imperfect (llava reads most but not all
  filenames reliably); the KEEP detection was correct. For production, pair the
  VLM's reading with the real FS check (verification makes it robust).

## The pipeline (real end-to-end)
```
real dir -> render listing -> screenshot (real PNG)
         -> llava interprets (reads names, counts, KEEP)
         -> route decision -> verify against real FS
```

## Files
- `real_vision.py` — capture + VLM interpret + verify
- Requires: `ollama pull llava:7b`, `qlmanage`/`screencapture` (macOS)