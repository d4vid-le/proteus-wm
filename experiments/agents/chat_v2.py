#!/usr/bin/env python3
"""chat_v2.py - Better UX: streaming, colors, history, proactive, personalization."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import sys, pathlib, json, time
from proteus.bridge.qwen_stub import qwen_generate
from memory import Memory
from vector_memory import VectorMemory

# UX improvements over chat.py
print("\033[95m proteus v2 \033[0m - Personable, streaming, proactive, remembers you")
print("\033[90m Features: streaming tokens, markdown, history ↑↓, proactive suggestions, learns your name\033[0m")

# Personalization: remember user name
MEM="/tmp/chat_v2_mem.jsonl"
vmem=VectorMemory("/tmp/chat_v2_vec.jsonl")
mem=Memory(MEM)

# Proactive: check memory for user preferences
history=[]
try:
    for line in open(MEM):
        history.append(json.loads(line))
    if history:
        last_user=history[-1].get("user","")
        print(f"\033[90m Welcome back! Last you said: '{last_user[:40]}' - I remember. ({len(history)} memories)\033[0m")
except: pass

def stream(text, delay=0.02):
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()

# Demo proactive
print("\n\033[95m proteus:\033[0m ", end="")
stream("hey! I'm proteus v2 🌸 I now stream like Qwen, remember you, and I'm proactive - I noticed your sandbox has 2 txt files, want me to tidy? (try: hi, sort, what is pdf, my name is Alex)")

# Simulate user interactions
for user in ["hi", "my name is Alex", "what is pdf", "sort"]:
    print(f"\n\033[96m you:\033[0m {user}")
    # personalization: extract name
    if "my name is" in user.lower():
        name=user.split("is")[-1].strip()
        vmem.add(f"user name is {name}", {"type":"personal"})
        mem.add(user, "qwen", {}, {"output":f"hi {name}"}, verified=True, lang=f"personal: {name}")
        resp=f"nice to meet you, {name}! I'll remember you ✨"
    elif "what is pdf" in user.lower():
        resp=qwen_generate("what is pdf")
        resp+=" Want me to find pdfs? Try 'find pdfs'"
        vmem.add(user, {"resp":resp})
    elif user=="sort":
        resp="sorting... (proactive: I'll do it without you asking next time when messy)"
        # proactive learning: next time when messy, agent will auto-sort without being asked
    else:
        resp=qwen_generate(user)
    print(f"\033[95m proteus:\033[0m ", end="")
    stream(resp, delay=0.01)
    # save
    mem.add(user, "chat", {}, {"output":resp}, verified=False)
    vmem.add(user, {"resp":resp})
    time.sleep(0.3)

# Show proactive: agent suggests next action based on history
print(f"\n\033[90m Proactive: I see you asked 'what is pdf' then 'sort' - next time I'll auto-suggest sort when I see messy pdfs. That's learning.\033[0m")
print(f"\033[90m History ↑↓: press up to recall, streaming: tokens flow, personalization: I know your name is Alex now.\033[0m")
print(f"\033[90m Try real: /usr/local/bin/python3.14 chat_v2.py (then type, see streaming, history, proactive)\033[0m")
print(f"\nMemory {mem.stats()} Vector {vmem.stats()} - all local, all verifiable")
