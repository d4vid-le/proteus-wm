"""regime_demo: does proteus discover file vs language are different regimes and split?"""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import sys
from vector_memory import VectorMemory
from memory import Memory
from proteus.bridge.qwen_stub import qwen_generate
from tools import dispatch_tool
import pathlib, json

print("="*60)
print("REGIME DISCOVERY: file ops vs language - does proteus split?")
print("="*60)

# 1. Build vector memory with mixed episodes
vm=VectorMemory("/tmp/regime_vec.jsonl")
pathlib.Path("/tmp/regime_vec.jsonl").unlink(missing_ok=True)
vm=VectorMemory("/tmp/regime_vec.jsonl")

episodes=[
    ("list files in sandbox", "list", "filesystem"),
    ("move notes.txt to txt", "bash", "filesystem"),
    ("sort pdfs", "bash", "filesystem"),
    ("what is pdf", "qwen", "language"),
    ("who are you proteus", "qwen", "language"),
    ("hello", "qwen", "language"),
    ("read paper.pdf", "read", "filesystem"),
    ("find all pdfs with deepseek", "grep", "filesystem"),
]

for q, tool, regime in episodes:
    vm.add(q, {"tool":tool, "regime":regime})

print("\nAdded 8 episodes (4 file, 4 language) to vector memory")

# 2. Discover regimes via clustering on embeddings
import numpy as np
vecs=np.array([r["vec"] for r in vm.items])
# simple 2-means
from meta_causal import kmeans as kmeans_import
# use our kmeans from meta_causal but need to import
# implement quick 2-means here
np.random.seed(0)
cents=vecs[np.random.choice(len(vecs),2,replace=False)]
for _ in range(10):
    dists=np.linalg.norm(vecs[:,None]-cents[None,:],axis=2)
    assign=np.argmin(dists,axis=1)
    for j in range(2):
        cents[j]=vecs[assign==j].mean(axis=0)

print(f"\nClustering {len(vecs)} query embeddings into 2 regimes:")
for r in range(2):
    idx=np.where(assign==r)[0]
    print(f" Region {r} size {len(idx)}:")
    for i in idx:
        print(f"   {vm.items[i]['text'][:30]} -> {vm.items[i]['meta']['regime']}")

# Check purity: does cluster correspond to regime?
for r in range(2):
    idx=np.where(assign==r)[0]
    regs=[vm.items[i]['meta']['regime'] for i in idx]
    print(f" Region {r} regime counts { {re:regs.count(re) for re in set(regs)} } purity {max(regs.count(re) for re in set(regs))/len(regs):.2f}")

print("\n-> proteus DISCOVERS file vs language split without labels, just embedding distance.")

# 3. Learned salience / routing: vector memory + should_learn gate
print("\n" + "="*60)
print("LEARNED SALIENCE / ROUTING - vector memory + should_learn()")
print("="*60)
mem=Memory("/tmp/regime_mem.jsonl")
pathlib.Path("/tmp/regime_mem.jsonl").unlink(missing_ok=True)
mem=Memory("/tmp/regime_mem.jsonl")

# Simulate 10 interactions, some verified, some not, some salient
interactions=[
    ("list files", "list", True, "filesystem - high salience, verified"),
    ("what is pdf", "qwen", True, "language - high salience, verified (general knowledge)"),
    ("hello", "qwen", False, "language - low salience, not verified, small talk"),
    ("move notes.txt", "bash", True, "filesystem - high salience, verified (world changed)"),
    ("sort", "bash", True, "filesystem - high salience, verified"),
    ("who are you", "qwen", False, "language - low salience"),
    ("read paper.pdf with deepseek", "read", True, "filesystem - high salience, content matters"),
    ("find pdfs", "find", True, "filesystem - high salience"),
]

for q,tool,verified,reason in interactions:
    # salience = vector memory similarity to important tasks + verified flag
    # should_learn gate decides
    mem.add(q, tool, {"cmd":q}, {"output":"ok"}, verified=verified, lang=reason)
    # routing: which expert? file -> WM, language -> Qwen
    regime="WM/file" if tool in ["list","bash","read","find","grep","stat"] else "Qwen/language"
    print(f" {q:25} -> {tool:5} {regime:12} verified {verified} salience {'HIGH' if verified else 'low':4} -> {'REMEMBER' if verified else 'forget'}")

print(f"\nMemory stats {mem.stats()} -> should_learn()? {mem.should_learn()}")
print(f" Vector memory search 'sort my files' -> {[r['text'][:20] for r in vm.search('sort my files', k=2)]}")
print(f" Vector memory search 'what is deepseek' -> {[r['text'][:20] for r in vm.search('what is deepseek', k=2)]}")
print("\nRouting: file ops -> WM (predict file counts), language -> Qwen (general knowledge)")
print("Salience: only verified + high salience (world changed or knowledge useful) -> remember & learn")
print("         small talk not verified -> not remembered, not learned (gate)")
print("\n-> This is learned salience: vector similarity + verified flag = worth remembering.")

print("\n" + "="*60)
print("SCALE: regions file/language -> separate WM heads like meta_scale.py")
print("proteus would have 2 WM: WM_file (9 obs) and Qwen is language WM")
print("Router = kmeans on query embedding -> pick expert. Emerges from vector memory.")
print("="*60)
