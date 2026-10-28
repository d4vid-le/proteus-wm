"""mcp_client.py — MCP client over stdio. Talks to mcp_server.py.

Spawns the JEPA MCP server as a subprocess and communicates via
JSON-RPC 2.0 over stdin/stdout. Handles initialize, tools/list, tools/call.
"""
from proteus.paths import ROOT
import json
import subprocess
import sys
import threading
import time
import pathlib



class MCPClient:
    """Stdio MCP client for proteus JEPA tools."""

    def __init__(self, server_script=None, timeout=15):
        self.server_script = server_script or str(ROOT / "mcp_server.py")
        self.timeout = timeout
        self.proc = None
        self._req_id = 0
        self._lock = threading.Lock()
        self.tools = {}
        self._started = False

    def start(self):
        """Spawn the MCP server process and do the initialize handshake."""
        self.proc = subprocess.Popen(
            [sys.executable, self.server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # initialize
        resp = self._send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "pi-harness", "version": "1.0"},
        })
        if "error" in resp:
            raise RuntimeError(f"MCP init failed: {resp['error']}")

        # tools/list
        resp = self._send("tools/list", {})
        if "result" in resp and "tools" in resp["result"]:
            for tool in resp["result"]["tools"]:
                self.tools[tool["name"]] = tool

        self._started = True
        return resp

    def call(self, tool_name, arguments=None):
        """Call a tool by name. Returns the result dict."""
        if not self._started:
            self.start()
        resp = self._send("tools/call", {
            "name": tool_name,
            "arguments": arguments or {},
        })
        return resp.get("result", resp)

    def _send(self, method, params):
        """Send a JSON-RPC request, read response."""
        with self._lock:
            self._req_id += 1
            req_id = self._req_id

        req = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        line = json.dumps(req) + "\n"

        try:
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            return {"error": f"write failed: {e}"}

        # read response (blocking with timeout)
        start = time.time()
        while time.time() - start < self.timeout:
            try:
                raw = self.proc.stdout.readline()
                if not raw:
                    time.sleep(0.05)
                    continue
                resp = json.loads(raw.strip())
                if resp.get("id") == req_id:
                    return resp
                # not our response, skip
            except json.JSONDecodeError:
                continue
            except Exception as e:
                return {"error": f"read failed: {e}"}

        return {"error": f"timeout waiting for response ({self.timeout}s)"}

    def stop(self):
        """Terminate the server process."""
        if self.proc:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None
            self._started = False

    def __del__(self):
        self.stop()


def quick_test():
    """Smoke test: start server, list tools, call one."""
    print("=== MCP Client Smoke Test ===")
    client = MCPClient()
    try:
        client.start()
        print(f"Tools available: {len(client.tools)}")
        for name in sorted(client.tools.keys()):
            print(f"  - {name}")

        print("\nCalling jepa_world_state...")
        result = client.call("jepa_world_state")
        print(f"Result: {json.dumps(result, indent=2)[:300]}")

        print("\nCalling jepa_surprise...")
        result = client.call("jepa_surprise")
        print(f"Result: {json.dumps(result, indent=2)[:300]}")

        print("\nCalling jepa_daemon_status...")
        result = client.call("jepa_daemon_status")
        print(f"Result: {json.dumps(result, indent=2)[:300]}")

        print("\n=== PASS ===")
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.stop()


if __name__ == "__main__":
    quick_test()
