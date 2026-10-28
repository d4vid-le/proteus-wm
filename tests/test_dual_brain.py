#!/usr/bin/env python3
"""test_dual_brain.py — Verify the Pi + proteus dual-brain stack end-to-end.

Walks through every layer:
  1. MCP server alive (tools/list)
  2. Each System-1 tool returns real data
  3. The KEEP safety barrier BLOCKS (the critical one)
  4. Pi (System 2) actually calls the tools and reasons

Usage:
  python3 test_dual_brain.py            # test all layers
  python3 test_dual_brain.py --pi       # also run Pi integration tests (needs ollama qwen3.6:35b)
"""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import argparse
import json
import pathlib
import subprocess
import sys

MCP = ROOT / "mcp_server.py"


def mcp_call(method, params=None):
    """Call the MCP server over stdio. Returns the result dict."""
    req = {"jsonrpc": "2.0", "id": 1, "method": method,
           "params": params or {}}
    r = subprocess.run(["python3", str(MCP)], input=json.dumps(req) + "\n",
                       capture_output=True, text=True, timeout=60)
    try:
        resp = json.loads(r.stdout.strip().splitlines()[-1])
        return resp.get("result", {})
    except Exception as e:
        return {"error": str(e), "raw": r.stdout[-200:]}


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""))
    return cond


def test_mcp():
    print("\n[1] MCP server alive")
    r = mcp_call("initialize")
    ok = check("initialize", r.get("serverInfo", {}).get("name") == "proteus-jepa")
    r = mcp_call("tools/list")
    tools = [t["name"] for t in r.get("tools", [])]
    check("tools/list has 9 tools", len(tools) == 9, f"({len(tools)}: {', '.join(tools[:4])}...)")
    return ok


def test_tools():
    print("\n[2] System-1 tools return real data")
    results = {}
    results["world_state"] = mcp_call("tools/call", {"name": "jepa_world_state", "arguments": {}})
    w = results["world_state"].get("content", [{}])[0].get("text", "")
    check("jepa_world_state", "app" in w, w[:50])

    results["surprise"] = mcp_call("tools/call", {"name": "jepa_surprise", "arguments": {}})
    s = results["surprise"].get("content", [{}])[0].get("text", "")
    check("jepa_surprise", "surprise" in s, s[:50])

    results["curiosity"] = mcp_call("tools/call", {"name": "jepa_curiosity", "arguments": {}})
    c = results["curiosity"].get("content", [{}])[0].get("text", "")
    check("jepa_curiosity", "novelty" in c, c[:60])

    results["daemon"] = mcp_call("tools/call", {"name": "jepa_daemon_status", "arguments": {}})
    d = results["daemon"].get("content", [{}])[0].get("text", "")
    check("jepa_daemon_status", "reflex" in d, d[:60])
    return results


def test_safety():
    print("\n[3] KEEP safety barrier (THE critical test)")
    # blocked case
    r = mcp_call("tools/call", {"name": "jepa_safety_check",
                                "arguments": {"action": "delete", "path": "KEEP.pdf"}})
    blocked = r.get("isError", False)
    text = r.get("content", [{}])[0].get("text", "")
    ok_blocked = check("delete KEEP.pdf → BLOCKED", blocked and "SAFETY BLOCKED" in text,
                       text if text else "no text")
    # safe case
    r2 = mcp_call("tools/call", {"name": "jepa_safety_check",
                                 "arguments": {"action": "delete", "path": "notes.txt"}})
    safe = not r2.get("isError", True)
    check("delete notes.txt → SAFE", safe)
    return ok_blocked


def test_pi():
    print("\n[4] Pi (System 2) calls the tools and reasons")
    try:
        r = subprocess.run(
            ["pi", "--provider", "ollama", "--model", "qwen3.6:35b-a3b-nvfp4", "--print",
             "Call mcp_proteus_jepa_jepa_safety_check with action delete and path KEEP.pdf. "
             "Report exactly what it returns."],
            capture_output=True, text=True, timeout=180)
        out = r.stdout + r.stderr
        print(f"  Pi output: {out[:200]}")
        ok = "SAFETY BLOCKED" in out or "BLOCKED" in out
        check("Pi correctly reports the block", ok)
        return ok
    except FileNotFoundError:
        print("  [SKIP] pi not found")
        return None
    except subprocess.TimeoutExpired:
        print("  [FAIL] Pi timed out (model loading may take a while)")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi", action="store_true", help="run Pi integration tests")
    args = ap.parse_args()

    print("=" * 60)
    print("DUAL-BRAIN STACK TEST (Pi + proteus)")
    print("=" * 60)

    results = []
    results.append(test_mcp())
    results.append(test_safety())
    test_tools()
    if args.pi:
        results.append(test_pi())

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    passes = sum(1 for r in results if r)
    print(f"  critical tests passed: {passes}/{len(results)}")
    print("  (world_state/surprise/curiosity/daemon are informational)")
    print("=" * 60)


if __name__ == "__main__":
    main()