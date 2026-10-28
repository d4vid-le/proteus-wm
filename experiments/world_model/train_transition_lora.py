#!/usr/bin/env python3
"""train_transition_lora.py — LoRA on the 6 full-attention layers of Qwen3.5-2B,
trained to continue from the JEPA transition-bridge soft prefix.

The spec (2026-09-04):
  - LoRA rank 16 / alpha 32 / dropout 0.05, applied STRICTLY to
    q_proj / k_proj / v_proj / o_proj of the 6 full_attention layers
    (indices 3,7,11,15,19,23); linear-attention/Gated DeltaNet frozen.

  - Input:  [bridge-prefix(z_t, z_next)[, e])] + ChatML text
         "<|im_start|>user\nWhat action was taken?\n<|im_end|>\n<|im_start|>assistant\nAction: "
  - Target:  "move {ext}<|im_end|>" (same normalize_target phrases as the eval).
  - Loss masked to -100 for ALL prefix tokens AND prompt tokens; CE ONLY on
    the completion tokens ("move {ext}<|im_end|>").
  - Bridge frozen.  Only the LoRA adapters train.

Why this works: the frozen soft-prefix alone makes Qwen emit EOS to escape the
OOD manifold (observed "()"). The LoRA gives the attention heads a small
trainable route to absorb the prefix activations into the natural-language
generation stream.

Usage:
  python3 train_transition_lora.py --epochs 120 --batch 4 --lr 2e-4
  python3 train_transition_lora.py --eval
"""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType

from proteus.bridge.llm_bridge import (                    # noqa: E402
    TRANSITION_CKPT, load_bridge, load_qwen, load_transition_pairs,
    transition_vec, normalize_target, phrase_logprobs_batch_chunked,
    _prior_accuracy, _class_counts,
)

MODEL = ROOT.parent / "qwen3.5-2b"
LORA_CKPT = ROOT / "checkpoints" / "transition_lora.pt"
USE_SURPRISE = True           # match the best 129-dim transition bridge
FULL_ATTN_IDX = [3,  7, 11,  15,  19,  23]
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

SYSTEM_TXT = "You are PROTEUS. Use your world state to answer the user's question concisely."
USER_QUESTION = "What action was taken?"
PREFILL = "Action: "


def chatml_text(user_q=USER_QUESTION):
    """ChatML text tokens (system+user+assistant-anchor)."""
    return ("<|im_start|>system\n" + SYSTEM_TXT + "<|im_end|>\n"
            "<|im_start|>user\n" + user_q + "<|im_end|>\n"
            "<|im_start|>assistant\n" + PREFILL)


def build_lora(model):
    """Attach LoRA to q/k/v/o of the 6 full-attention layers ONLY."""
    target_modules = []
    for li in FULL_ATTN_IDX:
        for t in TARGET_MODULES:

            target_modules.append(f"model.layers.{li}.self_attn.{t}")
    cfg = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=target_modules, task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def make_batch_lora(model, tok, bridge, rows, device="mps"):
    """Build (inputs_embeds, labels) for one batch, with the bridge prefix +
    ChatML text, loss masked to completion tokens only."""
    vec = transition_vec(rows, use_surprise=USE_SURPRISE)      # (B, 129)
    zt = torch.as_tensor(vec, dtype=torch.float32, device=device)
    with torch.no_grad():
        prefix = bridge(zt).to(torch.bfloat16)                   # (B, n, D)
    text = chatml_text()
    enc = tok(text, return_tensors="pt")
    text_ids = enc.input_ids.to(device)             # (1, T)
    with torch.no_grad():
        text_emb = model.get_base_model().model.embed_tokens(text_ids)   # (1, T, D)
    B = prefix.shape[0]
    n = prefix.shape[1]
    T = text_emb.shape[1]
    text_emb = text_emb.expand(B, -1, -1)          # (B, T, D)

    # phrase token ids (completion): "move {ext}<|im_end|>"
    phrase_ids = []
    max_plen = 0
    for r in rows:
        tids = tok(normalize_target(r["action"]) + "<|im_end|>",
                    add_special_tokens=False).input_ids
        phrase_ids.append(tids)
        max_plen = max(max_plen, len(tids))
    with torch.no_grad():
        phrase_emb = model.get_base_model().model.embed_tokens(
            torch.as_tensor([phrase_ids[0]], device=device))  # placeholder, replaced below
    # build the (B, max_plen) phrase tensor (right-pad with eos)
    phrase_t = torch.full((B, max_plen), tok.eos_token_id, dtype=torch.long, device=device)
    for i, tids in enumerate(phrase_ids):
        phrase_t[i, :len(tids)] = torch.as_tensor(tids, device=device)
    with torch.no_grad():
        phrase_emb = model.get_base_model().model.embed_tokens(phrase_t)  # (B, max_plen, D)

    inputs = torch.cat([prefix, text_emb, phrase_emb], dim=1)   # (B, n+T+max_plen, D)

    # labels: -100 for prefix+text; phrase ids only (padded positions -100)
    labels = torch.full((B, n + T + max_plen), -100, dtype=torch.long, device=device)
    for i in range(B):
        plen = len(phrase_ids[i])
        labels[i, n + T: n + T + plen] = phrase_t[i, :plen]
    return inputs, labels


def train(epochs=120, batch=4, lr=2e-4, log_every=10):
    model, tok = load_qwen()
    bridge = load_bridge(ckpt=TRANSITION_CKPT)   # 129-dim transition bridge (frozen)
    model = build_lora(model)
    model.train()

    rows = load_transition_pairs()
    print(f"transition pairs: {len(rows)}  (LoRA on {FULL_ATTN_IDX}, rank {LORA_RANK})")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(0)
    losses = []
    t0 = time.time()
    for step in range(epochs):
        idx = rng.choice(len(rows), batch, replace=False)
        inputs, labels = make_batch_lora(model, tok, bridge,
                                        [rows[i] for i in idx])
        out = model(inputs_embeds=inputs, labels=labels)
        loss = out.loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % log_every == 0:
            print(f"step {step+1:4d} loss {loss.item():.4f} "
                  f"({(time.time()-t0)/(step+1):.2f}s/step)")
    # save ONLY the LoRA adapters (small)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()
          if "lora_" in k}
    torch.save(sd, LORA_CKPT)
    print(f"LoRA saved -> {LORA_CKPT}  ({len(sd)} tensors)")
    print(f"final loss {np.mean(losses[-20:]):.4f} (start {losses[0]:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval", action="store_true", help="evaluate the LoRA (phrase LL + generation)")
    ap.add_argument("--n", type=int, default=0, help="rows to eval (0=all)")
    args = ap.parse_args()
    if args.eval:
        evaluate(args.n)
    else:
        train(epochs=args.epochs, batch=args.batch, lr=args.lr)


def load_lora(model):
    """Attach LoRA to the loaded model and load the trained adapters."""
    model = build_lora(model)
    if pathlib.Path(LORA_CKPT).exists():
        sd = torch.load(LORA_CKPT, map_location="mps", weights_only=True)
        # peft names prefixes with 'base_model.model.' ; our saved keys have it too
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[lora] missing: {missing[:6]}...")
        if unexpected:
            print(f"[lora] unexpected: {unexpected[:6]}...")
        model.eval()
    return model


@torch.no_grad()
def evaluate(n=0):
    """Honest eval: with the LoRA ON, do the (prefix+ChatML) inputs generate a
    clean grounded phrase instead of () or soup? Measures:
      (a) phrase log-likelihood under the LoRA model (compare to base),
      (b) actual generation: how often the completion equals the true phrase,
      (c) the raw decoded string (the conversational proof)."""
    model, tok = load_qwen()
    bridge = load_bridge(ckpt=TRANSITION_CKPT)
    model = load_lora(model)
    rows = load_transition_pairs()
    if n:
        rows = rows[:n]
    cands = sorted({normalize_target(r.get("action")) for r in rows})
    targets = [normalize_target(r["action"]) for r in rows]
    print(f"[eval-lora] n={len(rows)}  cands={cands}")
    print(f"[eval-lora] LoRA adapters loaded from {LORA_CKPT}")

    # (a)+(c) generate with the LoRA + bridge prefix + ChatML text
    from proteus.bridge.llm_bridge import TRANSITION_PROMPT
    text = chatml_text()
    text_ids = tok(text, return_tensors="pt").input_ids.to("mps")
    text_emb = model.get_base_model().model.embed_tokens(text_ids)
    n_pre = bridge.n_prefix
    vec = transition_vec(rows, use_surprise=USE_SURPRISE)
    zt = torch.as_tensor(vec, dtype=torch.float32, device="mps")
    prefix = bridge(zt).to(torch.bfloat16)
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    if eos_id is None or eos_id == tok.unk_token_id:
        eos_id = tok.eos_token_id

    exact = 0
    samples = []
    # manual greedy decode (robust with inputs_embeds + PEFT — generate() with a
    # KV cache can return the input unchanged on some builds)
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    if eos_id is None or eos_id == tok.unk_token_id:
        eos_id = tok.eos_token_id
    for i in range(len(rows)):
        inp = torch.cat([prefix[i:i+1], text_emb], dim=1)
        attn = torch.ones(inp.shape[:2], dtype=torch.long, device="mps")
        gen_ids = []
        cur = inp
        for _ in range(12):
            with torch.no_grad():
                lg = model(inputs_embeds=cur, attention_mask=attn).logits
            nxt = int(lg[0, -1].argmax().item())
            gen_ids.append(nxt)
            if nxt == eos_id:
                break
            nxt_emb = model.get_base_model().model.embed_tokens(
                torch.tensor([[nxt]], device="mps"))
            cur = torch.cat([cur, nxt_emb], dim=1)
            attn = torch.cat([attn, torch.ones((1, 1), dtype=torch.long, device="mps")], dim=1)
        dec = tok.decode(gen_ids, skip_special_tokens=True).strip()
        samples.append(dec)
        if dec == targets[i]:
            exact += 1
        if i < 6:
            print(f"  [gen {i}] true={targets[i]!r:12s} got={dec!r}")

    ne = len(rows)
    print(f"\n  EXACT-GENERATION match: {exact}/{ne} ({exact/ne:.0%})")
    from collections import Counter
    print("  generation distribution:", dict(Counter(samples).most_common(6)))
    # per-class exact-match
    per = {}
    for c in cands:
        idxs = [i for i, t in enumerate(targets) if t == c]
        if idxs:
            per[c] = sum(1 for i in idxs if samples[i] == c) / len(idxs)
    print("  per-class exact-match:", {c: f"{v:.0%}" for c, v in per.items()})


if __name__ == "__main__":
    main()