#!/usr/bin/env python3
"""pi_harness.py — Pi agent harness: Qwen + proteus JEPA via MCP.

The main loop:
  1. User types a message
  2. Message goes to Qwen (via Ollama) with tool schemas
  3. If Qwen outputs a tool call, route it to JEPA via MCP
  4. Tool result feeds back to Qwen for the next turn
  5. Repeat until Qwen gives a final text response

Usage:
  python -m proteus.bridge.pi_harness                    # interactive mode
  python -m proteus.bridge.pi_harness --once "what do you see?"  # single query
  python -m proteus.bridge.pi_harness --model qwen2.5:1.5b       # specify model
"""
import argparse
import json
import subprocess
import sys
import time
import os
import pathlib
import readline  # enables arrow keys in input()

from proteus.bridge.mcp_client import MCPClient
from proteus.bridge.tool_parser import parse_tool_calls, strip_tool_calls
from proteus.bridge.system_prompt import get_system_prompt

OLLAMA_BIN = "/opt/homebrew/bin/ollama"
DEFAULT_MODEL = os.environ.get("PROTEUS_MODEL", "Qwen2.5-Coder-14B-Instruct")
MAX_TOOL_ROUNDS = 6  # prevent infinite loops
MAX_TOKENS = 512
TEMPERATURE = 0.6


def ollama_chat(model, messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    """Generate response using mlx_lm for local models, Ollama for remote."""
    # Check if model is a local path (MLX model)
    if pathlib.Path(model).exists():
        return mlx_generate(model, messages, temperature, max_tokens)
    # Fallback to Ollama
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    try:
        result = subprocess.run(
            [OLLAMA_BIN, "run", model, json.dumps(payload)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return f"[error: ollama returned {result.returncode}] {result.stderr[:200]}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[error: ollama timed out]"
    except FileNotFoundError:
        return "[error: ollama not found at /opt/homebrew/bin/ollama]"


_mlx_model = None
_mlx_tokenizer = None

def mlx_generate(model_path, messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    """Generate using mlx_lm with model caching."""
    global _mlx_model, _mlx_tokenizer
    if _mlx_model is None:
        from mlx_lm import load
        print(f"  [loading {pathlib.Path(model_path).name}...]")
        _mlx_model, _mlx_tokenizer = load(model_path)
    from mlx_lm import generate as mlx_gen
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=temperature)
    text = _mlx_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = mlx_gen(
        _mlx_model, _mlx_tokenizer,
        prompt=text,
        max_tokens=max_tokens,
        sampler=sampler,
    )
    return response


def format_tool_result(tool_name, result):
    """Format a JEPA MCP result for Qwen to read."""
    if isinstance(result, dict):
        # extract useful parts
        if "content" in result:
            # MCP response format
            contents = result["content"]
            if isinstance(contents, list) and contents:
                text = contents[0].get("text", "")
                try:
                    parsed = json.loads(text)
                    # add context for empty results
                    if tool_name == "jepa_organize" and parsed.get("count", -1) == 0:
                        return "Directory is already organized or empty. No files to move."
                    if tool_name == "jepa_world_state":
                        app = parsed.get("app", {}).get("name", "unknown")
                        return f"Active app: {app}. Recent events: {len(parsed.get('recent_fs_events', []))} file events."
                    if tool_name == "jepa_surprise":
                        surprises = parsed.get("recent_surprises", [])
                        return f"Recent surprises: {len(surprises)} signals."
                    if tool_name == "jepa_daemon_status":
                        reflex = parsed.get("reflex_running", False)
                        daemon = parsed.get("daemon_running", False)
                        return f"Reflex: {'running' if reflex else 'stopped'}. Daemon: {'running' if daemon else 'stopped'}."
                    return json.dumps(parsed, indent=1)
                except json.JSONDecodeError:
                    return text
            return str(contents)
        if "isError" in result and result["isError"]:
            return f"SAFETY BLOCKED: {result.get('content', 'action not safe')}"
        return json.dumps(result, indent=1)
    return str(result)


def run_conversation(model, mcp_client, user_input=None):
    """Run one conversation turn (potentially multi-tool).

    Returns the final text response from Qwen.
    """
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": user_input},
    ]

    for round_num in range(MAX_TOOL_ROUNDS):
        # call Qwen
        response = ollama_chat(model, messages)
        print(f"\n[Pi]: {response}")

        # parse tool calls
        tool_calls = parse_tool_calls(response)
        if not tool_calls:
            # no tools, this is the final response
            return response

        # repeat detection: break if same tool+args called twice
        call_sig = json.dumps([(n, a) for n, a, _ in tool_calls])
        if call_sig == getattr(run_conversation, "_last_call", None):
            print("  [break: repeated call]")
            return strip_tool_calls(response)
        run_conversation._last_call = call_sig

        # execute tools
        for tool_name, args, span in tool_calls:
            print(f"\n[Tool]: {tool_name}({json.dumps(args)})")

            # safety check for dangerous actions
            if tool_name == "jepa_safety_check":
                pass  # let it through, it's the check itself
            elif tool_name in ("jepa_organize",) and not args.get("dir"):
                print("  [warn] missing dir argument")
                continue

            # call via MCP
            try:
                result = mcp_client.call(tool_name, args)
                result_text = format_tool_result(tool_name, result)
            except Exception as e:
                result_text = f"error: {e}"

            print(f"  [Result]: {result_text[:200]}")

            # add tool result to conversation
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Tool result for {tool_name}: {result_text}"})

    return response


def main():
    parser = argparse.ArgumentParser(description="Pi agent harness")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--once", type=str, default=None, help="Single query mode")
    parser.add_argument("--no-mcp", action="store_true", help="Skip MCP, text-only Qwen")
    args = parser.parse_args()

    print(f"=== Pi Agent Harness ===")
    print(f"Model: {args.model}")
    print(f"MCP: {'disabled' if args.no_mcp else 'enabled'}")
    print()

    # check ollama
    try:
        subprocess.run([OLLAMA_BIN, "list"], capture_output=True, timeout=10)
    except FileNotFoundError:
        print("ERROR: Ollama not found. Install: brew install ollama")
        print("Then pull a model: ollama pull qwen2.5:1.5b")
        sys.exit(1)

    # start MCP client
    mcp = None
    if not args.no_mcp:
        print("Starting JEPA MCP server...")
        try:
            mcp = MCPClient(timeout=15)
            mcp.start()
            print(f"Connected: {len(mcp.tools)} tools available")
            for name in sorted(mcp.tools.keys()):
                print(f"  - {name}")
        except Exception as e:
            print(f"WARNING: MCP start failed: {e}")
            print("Continuing without JEPA tools...")
            mcp = None
        print()

    try:
        if args.once:
            # single query
            print(f"> {args.once}")
            response = run_conversation(args.model, mcp, args.once)
        else:
            # interactive
            while True:
                try:
                    user_input = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nBye.")
                    break
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    print("Bye.")
                    break
                if user_input.lower() == "tools":
                    if mcp:
                        for name in sorted(mcp.tools.keys()):
                            print(f"  {name}")
                    else:
                        print("  MCP not connected")
                    continue
                if user_input.lower() == "status":
                    if mcp:
                        result = mcp.call("jepa_daemon_status")
                        print(json.dumps(result, indent=2)[:500])
                    else:
                        print("MCP not connected")
                    continue

                response = run_conversation(args.model, mcp, user_input)

    finally:
        if mcp:
            mcp.stop()
            print("MCP server stopped.")


if __name__ == "__main__":
    main()
