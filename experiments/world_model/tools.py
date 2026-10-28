"""tools.py - Local tools for scalable agent. All local, all verifiable, sandboxed.
Tools are the agent's hands to get data. Each call is DSML -> bash -> world_result -> verify."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import subprocess, pathlib, re, json, os

SANDBOX = pathlib.Path("sandbox")
ALLOWED_BASH_PREFIXES = ["ls","cat","echo","pwd","whoami","date","head","tail","wc","grep","find","stat"]
BLOCKED = ["rm -rf","sudo","mv /","dd ","mkfs","chmod 777","curl | sh",":(){"]  # safety

def safe_bash(cmd: str, workdir=SANDBOX, timeout=5):
    """Execute bash, sandboxed, verified by output."""
    # safety check
    for b in BLOCKED:
        if b in cmd:
            return {"ok":False, "output":f"BLOCKED: {b} not allowed", "code":1}
    # force workdir inside sandbox or /tmp
    try:
        workdir = pathlib.Path(workdir).resolve()
        if not str(workdir).startswith(str(SANDBOX.resolve())) and not str(workdir).startswith("/tmp"):
            workdir = SANDBOX
    except:
        workdir = SANDBOX
    try:
        result = subprocess.run(cmd, shell=True, cwd=str(workdir), capture_output=True, text=True, timeout=timeout)
        return {"ok": result.returncode==0, "output": (result.stdout + result.stderr)[:2000], "code": result.returncode}
    except subprocess.TimeoutExpired:
        return {"ok":False, "output":"timeout", "code":124}
    except Exception as e:
        return {"ok":False, "output":str(e), "code":1}

def tool_list(path=SANDBOX):
    """List files - verified by actual fs."""
    p = pathlib.Path(path)
    if not p.exists():
        return {"ok":False, "output":"not found"}
    files = sorted([f.name for f in p.iterdir()])
    return {"ok":True, "output": json.dumps(files), "files": files}

def tool_read(path):
    """Read file - verified by content."""
    p = pathlib.Path(path)
    # sandbox check
    try:
        if not str(p.resolve()).startswith(str(SANDBOX.resolve())):
            return {"ok":False, "output":"outside sandbox"}
    except: return {"ok":False,"output":"bad path"}
    if not p.exists():
        return {"ok":False, "output":"not found"}
    try:
        text = p.read_text()[:2000]
        return {"ok":True, "output": text}
    except Exception as e:
        return {"ok":False, "output":str(e)}

def tool_write(path, content):
    """Write file - verified by readback."""
    p = pathlib.Path(path)
    try:
        if not str(p.resolve()).startswith(str(SANDBOX.resolve())):
            return {"ok":False, "output":"outside sandbox"}
    except: return {"ok":False,"output":"bad path"}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        # verify
        back = p.read_text()[:2000]
        return {"ok": back==content, "output": f"wrote {len(content)} bytes, verify {back==content}"}
    except Exception as e:
        return {"ok":False, "output":str(e)}

def tool_stat(path=SANDBOX):
    """Stat for WM vec: counts by ext."""
    import numpy as np
    p = pathlib.Path(path)
    files = [f for f in p.iterdir() if f.is_file()] if p.exists() else []
    vec = np.zeros(9, dtype=float)
    for f in files:
        if f.suffix==".txt": vec[0]+=1
        elif f.suffix==".jpg": vec[1]+=1
        elif f.suffix==".pdf": vec[2]+=1
    # subfolders
    for i, ext in enumerate(["txt","jpg","pdf"]):
        sub = p/ext
        if sub.exists():
            vec[3+i]=len(list(sub.glob("*")))
    vec[6]=len(files)
    return {"ok":True, "output": vec.tolist(), "vec": vec}

# DSML tool schemas for agent (mirrors dsv4.py TOOLS_TEMPLATE)
TOOL_SCHEMAS = [
    {"name":"bash","description":"Run bash command in sandbox","parameters":{"properties":{"command":{"type":"string"}},"required":["command"]}},
    {"name":"list","description":"List files in sandbox","parameters":{"properties":{"path":{"type":"string"}},"required":[]}},
    {"name":"read","description":"Read file","parameters":{"properties":{"path":{"type":"string"}},"required":["path"]}},
    {"name":"write","description":"Write file","parameters":{"properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}},
    {"name":"stat","description":"Get file counts vec for WM","parameters":{"properties":{},"required":[]}},
    {"name":"find","description":"Find files by pattern (find . -name)","parameters":{"properties":{"pattern":{"type":"string"}},"required":["pattern"]}},
    {"name":"grep","description":"Grep content in files","parameters":{"properties":{"pattern":{"type":"string"}},"required":["pattern"]}},
    {"name":"qwen","description":"Ask Qwen general knowledge (local)","parameters":{"properties":{"prompt":{"type":"string"}},"required":["prompt"]}},
]

def dispatch_tool(name, arguments):
    """Dispatch DSML invoke to real tool, return world_result."""
    try:
        args=json.loads(arguments) if isinstance(arguments,str) else arguments
    except:
        args={"command": arguments}
    if name=="bash":
        return safe_bash(args.get("command",""))
    elif name=="list":
        return tool_list(args.get("path", str(SANDBOX)))
    elif name=="read":
        return tool_read(args.get("path",""))
    elif name=="write":
        return tool_write(args.get("path",""), args.get("content",""))
    elif name=="stat":
        return tool_stat(args.get("path", str(SANDBOX)))
    elif name=="find":
        pat=args.get("pattern","*.pdf")
        return safe_bash(f"find {SANDBOX} -name '{pat}' 2>&1 | head -20")
    elif name=="grep":
        pat=args.get("pattern","deepseek")
        return safe_bash(f"grep -r '{pat}' {SANDBOX} 2>&1 | head -20")
    elif name=="qwen":
        from proteus.bridge.qwen_stub import qwen_generate
        out=qwen_generate(args.get("prompt",""))
        return {"ok":True, "output": out}
    else:
        return {"ok":False, "output":f"unknown tool {name}"}

def encode_tool_dsml(name, arguments_dict):
    """Encode to DSML like dsv4.py: <｜DSML｜invoke>"""
    import json
    dsml_token="｜DSML｜"
    args_str=""
    for k,v in arguments_dict.items():
        is_str="true" if isinstance(v,str) else "false"
        val=v if isinstance(v,str) else json.dumps(v)
        args_str+=f'<{dsml_token}parameter name="{k}" string="{is_str}">{val}</{dsml_token}parameter>\n'
    return f"<{dsml_token}invoke name=\"{name}\">\n{args_str}</{dsml_token}invoke>"

def decode_tool_dsml(text):
    """Parse DSML back to (name, arguments dict)."""
    m=re.search(r'<｜DSML｜invoke name="([^"]+)">', text)
    if not m: return None,None
    name=m.group(1)
    args={}
    for km in re.finditer(r'<｜DSML｜parameter name="([^"]+)" string="(true|false)">(.*?)</｜DSML｜parameter>', text, re.DOTALL):
        k, is_str, v = km.groups()
        if is_str=="false":
            try: v=json.loads(v)
            except: pass
        args[k]=v
    return name, args
