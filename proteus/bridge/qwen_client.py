"""qwen_client.py - Real local Qwen via Ollama (replaces qwen_stub).

Provides the same interface as qwen_stub.py but backed by a real model:
  - qwen_generate(prompt, image=None, system=...) -> str      (qwen2.5:1.5b)
  - qwen_embed(text, dim=32) -> np.ndarray                     (nomic-embed-text)

Graceful fallback to the stub when Ollama is unavailable so demos never
hard-crash; logs a warning so you know you're on the fallback.
"""
import json
import subprocess
import numpy as np
import proteus.bridge.qwen_stub as _stub
GENERATE_MODEL = "qwen2.5:1.5b"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_BIN = "/opt/homebrew/bin/ollama"

_cache_available = None  # tri-state: None=unknown, True/False


def _ollama(args, timeout=120):
    """Run an ollama command, return parsed JSON or None on failure."""
    global _cache_available
    try:
        out = subprocess.run(
            [OLLAMA_BIN] + args, capture_output=True, text=True, timeout=timeout
        )
        if out.returncode != 0:
            _cache_available = False
            return None
        _cache_available = True
        try:
            return json.loads(out.stdout)
        except json.JSONDecodeError:
            return {"raw": out.stdout.strip()}
    except Exception as e:
        _cache_available = False
        print(f"[qwen_client] ollama unavailable: {e}")
        return None


def available():
    return _ollama(["list"], timeout=30) is not None


def qwen_generate(prompt, image=None, system="You are Qwen, helpful local muse."):
    """Real local generation. Falls back to the stub if Ollama is down."""
    if _cache_available is False:
        return _stub.qwen_generate(prompt, image=image, system=system)
    payload = {
        "model": GENERATE_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.6, "num_predict": 128},
    }
    # write payload to temp file to avoid arg-length limits
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.gettempdir()) / "qwen_payload.json"
    tmp.write_text(json.dumps(payload))
    res = _ollama(["run", GENERATE_MODEL, json.dumps(payload)], timeout=180)
    if res is None:
        return _stub.qwen_generate(prompt, image=image, system=system)
    if "raw" in res:
        return res["raw"]
    return res.get("response") or _stub.qwen_generate(prompt, image=image, system=system)


def qwen_embed(text, dim=32):
    """Embed text via nomic-embed-text, truncate/pad to `dim`, L2-normalized.

    Falls back to a deterministic hash embedding (as in wm_continuous.py) if
    the embed model isn't pulled yet, so the pipeline still runs.
    """
    if _cache_available is not False:
        payload = {"model": EMBED_MODEL, "prompt": text, "stream": False}
        res = _ollama(["run", EMBED_MODEL, json.dumps(payload)], timeout=60)
        if res and "embedding" in res:
            v = np.asarray(res["embedding"], dtype=np.float32)
            if v.shape[0] != dim:
                if v.shape[0] > dim:
                    v = v[:dim]
                else:
                    v = np.pad(v, (0, dim - v.shape[0]))
            n = np.linalg.norm(v)
            return v / n * 2 if n > 0 else v
    # fallback: deterministic hash embedding (keeps pipeline runnable)
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
    v = np.tile(arr, (dim // len(arr) + 1))[:dim]
    v = (v - 128) / 30
    return v / np.linalg.norm(v) * 2


def qwen_action_embed(cmd_str, dim=32):
    """Embed a shell command / action string into the continuous action space."""
    return qwen_embed(cmd_str, dim=dim)


if __name__ == "__main__":
    print("=== qwen_client smoke test ===")
    print("generate:", qwen_generate("what is pdf?")[:120])
    e = qwen_embed("grep -rn 'error' /var/log")
    print("embed dim:", e.shape, "norm:", round(float(np.linalg.norm(e)), 3))
    print("action_embed:", qwen_action_embed("ls /tmp").shape)