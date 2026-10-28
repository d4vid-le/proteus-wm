"""tool_parser.py — Parse tool calls from Qwen's text output.

Handles multiple formats since Qwen 1.5b is inconsistent:
  [TOOL: tool_name()]
  [TOOL: tool_name(arg=val, arg2=val2)]
  [TOOL: tool_name("arg val")]
  <tool>tool_name</tool><args>{"key":"val"}</args>
"""
import re
import ast
import json


# Pattern 1: [TOOL: name(args)]
BRACKET_RE = re.compile(
    r'\[TOOL:\s*(\w+)\(([^)]*)\)\]',
    re.IGNORECASE
)

# Pattern 2: tool name in angle brackets
TAG_RE = re.compile(
    r'<tool>(\w+)</tool>\s*<args>(.*?)</args>',
    re.IGNORECASE | re.DOTALL
)

# Pattern 3: markdown code block with TOOL
CODEBLOCK_RE = re.compile(
    r'```(?:bash)?\s*\n?\s*TOOL:\s*(\w+)\(([^)]*)\)\s*\n?\s*```',
    re.IGNORECASE | re.DOTALL
)

# Pattern 4: bare function call
BARE_RE = re.compile(
    r'(?<!\w)(\w+)\(([^)]*)\)\s*$',
    re.MULTILINE
)

TOOL_NAMES = {
    "jepa_vision_energy", "jepa_world_state", "jepa_surprise",
    "jepa_safety_check", "jepa_organize", "jepa_plan",
    "jepa_learn", "jepa_curiosity", "jepa_daemon_status",
    "jepa_latent_handoff", "jepa_intent",
}


def parse_args(raw):
    """Parse argument string into dict. Handles multiple formats."""
    raw = raw.strip()
    if not raw:
        return {}

    # try JSON first
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    # try as keyword args: key=val, key2="val with spaces"
    result = {}
    try:
        # split on commas, but respect quotes
        parts = re.findall(r'(\w+)\s*=\s*(".*?"|\'.*?\'|[^\s,]+)', raw)
        for key, val in parts:
            # strip quotes
            val = val.strip().strip("\"'")
            # try to parse as number/bool
            if val.lower() in ("true", "false"):
                result[key] = val.lower() == "true"
            else:
                try:
                    result[key] = int(val)
                except ValueError:
                    try:
                        result[key] = float(val)
                    except ValueError:
                        result[key] = val
        if result:
            return result
    except Exception:
        pass

    # fallback: treat whole string as single positional arg
    # for tools like jepa_safety_check(action="delete", path="/foo")
    try:
        # literal_eval can handle some cases
        result = ast.literal_eval(f"({raw},)")
        if isinstance(result, tuple) and len(result) == 1:
            return {"arg": result[0]}
    except Exception:
        pass

    # last resort: return raw string
    return {"raw": raw}


def parse_tool_calls(text):
    """Extract tool calls from Qwen's response text.

    Returns list of (tool_name, args_dict, match_span) tuples.
    match_span is (start, end) in the original text for replacement.
    """
    calls = []

    # pattern 1: [TOOL: name(args)]
    for m in BRACKET_RE.finditer(text):
        name = m.group(1).lower()
        if name in TOOL_NAMES:
            args = parse_args(m.group(2))
            calls.append((name, args, (m.start(), m.end())))

    # pattern 2: <tool>name</tool><args>...</args>
    for m in TAG_RE.finditer(text):
        name = m.group(1).lower()
        if name in TOOL_NAMES:
            try:
                args = json.loads(m.group(2))
            except json.JSONDecodeError:
                args = parse_args(m.group(2))
            calls.append((name, args, (m.start(), m.end())))

    # pattern 3: markdown code blocks
    for m in CODEBLOCK_RE.finditer(text):
        name = m.group(1).lower()
        if name in TOOL_NAMES:
            args = parse_args(m.group(2))
            calls.append((name, args, (m.start(), m.end())))

    # pattern 4: bare calls (only if no other calls found)
    if not calls:
        for m in BARE_RE.finditer(text):
            name = m.group(1).lower()
            if name in TOOL_NAMES:
                args = parse_args(m.group(2))
                calls.append((name, args, (m.start(), m.end())))

    # deduplicate by position
    seen = set()
    unique = []
    for name, args, span in calls:
        if span not in seen:
            seen.add(span)
            unique.append((name, args, span))

    return unique


def extract_tool_block(text):
    """Return True if text contains any tool call."""
    return bool(parse_tool_calls(text))


def strip_tool_calls(text):
    """Remove tool call syntax from text, return clean text."""
    text = BRACKET_RE.sub("", text)
    text = TAG_RE.sub("", text)
    # clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def quick_test():
    """Test parser with various Qwen output formats."""
    tests = [
        # format 1: bracket
        'Let me check. [TOOL: jepa_vision_energy()]\n',
        # format 2: bracket with args
        '[TOOL: jepa_safety_check(action="delete", path="/tmp/foo.pdf")]',
        # format 3: multiple
        '[TOOL: jepa_world_state()] then [TOOL: jepa_surprise()]',
        # format 4: tag
        '<tool>jepa_plan</tool><args>{"goal": "organize Downloads"}</args>',
        # format 5: mixed text + tool
        'I see a messy directory. [TOOL: jepa_organize(dir="/tmp/test")] Done!',
        # format 6: no tool (should return empty)
        'Just a normal response with no tools.',
    ]

    print("=== Tool Parser Test ===")
    for i, t in enumerate(tests):
        calls = parse_tool_calls(t)
        print(f"\nTest {i+1}: {t[:60]}...")
        if calls:
            for name, args, span in calls:
                print(f"  -> {name}({args})")
        else:
            print(f"  -> (no tools)")
    print("\n=== PASS ===")


if __name__ == "__main__":
    quick_test()
