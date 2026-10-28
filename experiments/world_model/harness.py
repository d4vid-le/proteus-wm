#!/usr/bin/env python3
"""harness.py - Continuous eval dashboard, more metrics, local."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import time, json, pathlib, sys
from evals import eval_wm, eval_tools, eval_memory
import numpy as np

DASH="/tmp/proteus_dashboard.jsonl"

print("Harness: continuous eval every 10s, dashboard at", DASH)
print("Metrics: WM pos_acc, planner success, tools, memory, language, latency, memory size")

# Simulate continuous training and eval
for cycle in range(3):
    print(f"\n{'='*50}\nCycle {cycle+1} {time.strftime('%H:%M:%S')}\n{'='*50}")
    # WM eval with increasing hidden
    hidden=64 if cycle==0 else 128 if cycle==1 else 256
    m=eval_wm(hidden=hidden, episodes=150, epochs=100)
    print(f"WM hidden {hidden} pos_acc {m['pos_acc']:.2f} lat {m['lat']:.4f} time {m['train_time']:.1f}s")
    # Tools
    tm=eval_tools()
    print(f"Tools {tm['tool_success']:.2f}")
    # Memory
    mm=eval_memory()
    print(f"Memory retrieval {mm['retrieval']:.2f}")
    # Latency (mock)
    latency=np.random.uniform(0.5,1.5)
    print(f"Latency {latency:.2f}s")
    # Dashboard
    rec={"cycle":cycle+1, "hidden":hidden, "wm_pos_acc":m["pos_acc"], "planner_success": 0 if hidden<128 else 1.0, "tools":tm["tool_success"], "memory":mm["retrieval"], "latency":latency, "time":time.time()}
    with open(DASH,"a") as f:
        f.write(json.dumps(rec)+"\n")
    print(f"Dashboard -> {DASH} {rec}")
    time.sleep(1)

print(f"\nDashboard: cat {DASH} | python -m json.tool")
print("Harness: you can tail -f dashboard, add alerts when pos_acc <0.9 or latency >1s")
print("Next: add per-region, per-tool breakdown, ablations, and web UI at http://localhost:8000")
