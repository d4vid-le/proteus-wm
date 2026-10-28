#!/usr/bin/env python3
"""llm_bridge.py — JEPA latent -> Qwen soft-prefix bridge (the projector).

The locked spec: macro-64 JEPA world latent -> 2-layer MLP -> projected prefix
token (d_qwen = 2048) -> prepended to Qwen's input_embeds. Freeze Qwen (bf16,
MPS) + freeze the JEPA encoder. Train ONLY the bridge so it learns to make
Qwen describe the world given the latent.

Loss: cross-entropy on the description tokens, with the prefix + prompt
positions masked (-100) so the gradient flows only through what the latent
should explain.

The projector output is re-scaled to the mean token-embedding norm so Qwen
does not see an OOD-magnitude vector (verified necessary in Stage 0: raw
random noise generates degenerate output).

Usage:
  python -m proteus.bridge.llm_bridge --train --epochs 600 --batch 8
  python -m proteus.bridge.llm_bridge --eval          # answer-vs-truth with/without prefix
"""
from proteus.paths import ROOT
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

MODEL = ROOT.parent / "qwen3.5-2b"
PAIRS = ROOT / "data" / "latent_pairs_clean.jsonl"
TRANSITION_PAIRS = ROOT / "data" / "transition_pairs_clean.jsonl"
WHITEN = ROOT / "data" / "whiten.npz"
BRIDGE_CKPT = ROOT / "checkpoints" / "jepa_bridge.pt"
TRANSITION_CKPT = ROOT / "checkpoints" / "jepa_bridge_transition.pt"

JEPA_DIM = 64   # whitened latent dim (raw 64 -> PCA-whitened k=64)
QWEN_DIM = 2048
PROMPT = "World state. What happened to the files? "
# Inverse-dynamics prompt: neutral, so ONLY the latent channels disambiguate.
TRANSITION_PROMPT = "World state. What action was taken? "
N_PREFIX = 4    # number of soft-prefix tokens (scaling lever: 1 -> 4)


class JEPAToQwenBridge(nn.Module):
    """in_dim -> 2048 -> n_prefix*2048 projector (the only trainable params).
    Emits n_prefix soft tokens per world state so the LLM has more capacity to
    be steered by the latent.

    in_dim is the latent input width:
      64   = single state latent (description bridge)
      128  = empirical inverse dynamics (z_t, z_{t+1}) — no surprise
      129  = inverse dynamics + surprise scalar (z_t, z_{t+1}, e)
    """

    def __init__(self, in_dim=JEPA_DIM, qwen_dim=QWEN_DIM, n_prefix=N_PREFIX):
        super().__init__()
        self.in_dim = in_dim
        self.n_prefix = n_prefix
        self.proj = nn.Sequential(
            nn.Linear(in_dim, qwen_dim),
            nn.GELU(),
            nn.Linear(qwen_dim, n_prefix * qwen_dim),
        )
        # scale learned during training to match token-embedding norm
        self.register_buffer("scale", torch.ones(1))

    def forward(self, z):
        p = self.proj(z)                     # (B, n_prefix*D)
        return p.view(-1, self.n_prefix, QWEN_DIM) * self.scale  # (B, n, D)


# ---------------- whitening transform (for live 64-dim latents) ----------------
_whiten = None


def load_whiten(path=WHITEN):
    """Load the PCA-whitening transform fit on training latents.
    Returns callable: raw (N,64) L2-normalized -> (N,16) whitened."""
    global _whiten
    if _whiten is not None:
        return _whiten
    d = np.load(path)
    mean, W = d["mean"], d["W"]

    def _apply(Z):
        Z = np.asarray(Z, dtype=np.float32)
        if Z.ndim == 1:
            Z = Z[None, :]
        Zw = (Z - mean) @ W
        Zw = Zw / (np.linalg.norm(Zw, axis=1, keepdims=True) + 1e-8)
        return Zw[0] if Z.shape[0] == 1 else Zw
    _whiten = _apply
    return _apply


# ---------------- model / tokenizer (cached singletons) ----------------
_model = None
_tok = None


def load_qwen():
    global _model, _tok
    if _model is not None:
        return _model, _tok
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.bfloat16, device_map="mps",
        trust_remote_code=True)
    _model.eval()
    for p in _model.parameters():
        p.requires_grad_(False)
    return _model, _tok


@torch.no_grad()
def token_embed_norm(model, tok, n=64):
    """Mean L2 norm of real token embeddings (used to scale the prefix)."""
    ids = torch.randint(0, tok.vocab_size, (1, n), device="mps")
    emb = model.model.embed_tokens(ids)
    return float(emb.norm(dim=-1).mean())


def load_bridge(ckpt=BRIDGE_CKPT):
    """Load the bridge, reconstructing n_prefix AND in_dim from the saved
    projector widths so a 1-token/4-token and a 64/128/129-dim checkpoint all
    load correctly."""
    if pathlib.Path(ckpt).exists():
        sd = torch.load(ckpt, map_location="mps", weights_only=True)
        # proj.0.weight shape = (2048, in_dim); proj.2.weight = (n_pre*2048, 2048)
        in_dim = sd["proj.0.weight"].shape[1]
        n_pre = sd["proj.2.weight"].shape[0] // QWEN_DIM
        b = JEPAToQwenBridge(in_dim=in_dim, n_prefix=n_pre)
        b.load_state_dict(sd)
    else:
        b = JEPAToQwenBridge()
    b.to("mps")  # keep the tiny bridge in fp32; cast output to bf16 at the seam
    return b


# ---------------- dataset ----------------
def normalize_target(action):
    """Strip the unpredictable entropy (filenames) from the training label.
    The latent knows ACTION TYPE + extension, not the arbitrary filename chars.
    Returns the grounded phrase the bridge should learn to emit. The phrasing
    is kept SHORT and category-distinct so the latent (not the LM prior) must
    disambiguate which action happened."""
    a = action or ""
    if a.startswith("move "):
        body = a[5:]
        if "->" in body:
            ext = body.split("->", 1)[1].strip()
            return f"move {ext}"
    if "delete" in a:
        return "delete pdf"
    return "move misc"


def load_pairs(limit=None):
    rows = []
    if pathlib.Path(PAIRS).exists():
        for line in pathlib.Path(PAIRS).read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("desc"):
                    r["target"] = normalize_target(r.get("action", ""))
                    rows.append(r)
            except Exception:
                continue
    if limit:
        rows = rows[:limit]
    return rows


def load_transition_pairs(limit=None):
    """Load empirical inverse-dynamics pairs. Returns rows with 'target' set
    and a helper to assemble the bridge input vector.

    Input vector (in_dim):
      use_surprise=True  -> concat(z_t(64), z_next(64), surprise_std(1)) = 129
      use_surprise=False -> concat(z_t(64), z_next(64))                 = 128
    The surprise channel is the standardized label-free scalar (no action
    label ever enters the input)."""
    rows = []
    if pathlib.Path(TRANSITION_PAIRS).exists():
        for line in pathlib.Path(TRANSITION_PAIRS).read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("action"):
                    r["target"] = normalize_target(r.get("action", ""))
                    rows.append(r)
            except Exception:
                continue
    if limit:
        rows = rows[:limit]
    return rows


def transition_vec(rows, use_surprise=True):
    """(N, in_dim) input matrix for the transition bridge."""
    zt = np.asarray([r["latent"] for r in rows], dtype=np.float32)
    zn = np.asarray([r["latent_next"] for r in rows], dtype=np.float32)
    if use_surprise:
        e = np.asarray([r["surprise_std"] for r in rows], dtype=np.float32)
        return np.concatenate([zt, zn, e[:, None]], axis=-1)
    return np.concatenate([zt, zn], axis=-1)


def make_batch(model, tok, bridge, latents, descs, device="mps",
               prompt=PROMPT):
    """Build (inputs_embeds, labels) for one batch.
    Shape: [prefix(n) | prompt | desc], loss on desc positions only.
    Prompt and desc are tokenized SEPARATELY then concatenated, so the desc
    boundary is exact (no BPE merge, no off-by-one with n-token prefixes)."""
    pids = tok(prompt, add_special_tokens=True).input_ids   # fixed
    nctx = len(pids)
    dids = [tok(d, add_special_tokens=False).input_ids for d in descs]
    L = nctx + max((len(x) for x in dids), default=0)
    B = len(descs)
    inp_ids = torch.full((B, L), tok.eos_token_id, dtype=torch.long, device=device)
    desc_lens = []
    for i, d in enumerate(dids):
        inp_ids[i, :nctx + len(d)] = torch.as_tensor(pids + d, device=device)
        desc_lens.append(len(d))

    with torch.no_grad():
        emb = model.model.embed_tokens(inp_ids)          # (B, L, 2048)
    z = torch.as_tensor(np.asarray(latents, dtype=np.float32), device=device)
    prefix = bridge(z).to(torch.bfloat16)                # (B, n, 2048)
    n = prefix.shape[1]

    inputs = torch.cat([prefix, emb], dim=1)             # (B, n+L, D)
    # labels: -100 everywhere; predict each desc token from the prior position
    labels = torch.full((B, n + L), -100, dtype=torch.long, device=device)
    for i in range(B):
        nd = desc_lens[i]
        # desc token t sits at inputs position n+nctx-1+t (after prefix+prompt);
        # logits at position n+nctx-1+t predict it.
        labels[i, n + nctx - 1: n + nctx - 1 + nd] = inp_ids[i, nctx: nctx + nd]
    return inputs, labels


def train(epochs=600, batch=8, lr=3e-4, log_every=50):
    model, tok = load_qwen()
    bridge = load_bridge()
    emb_norm = token_embed_norm(model, tok)
    bridge.scale.fill_(emb_norm)
    print(f"token embed norm = {emb_norm:.2f}; scaled prefix to match")

    rows = load_pairs()
    print(f"pairs: {len(rows)}")
    latents = np.asarray([r["latent"] for r in rows], dtype=np.float32)
    descs = [r["target"] for r in rows]

    opt = torch.optim.Adam(bridge.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    losses = []
    t0 = time.time()
    for step in range(epochs):
        idx = rng.choice(len(rows), batch, replace=False)
        inputs, labels = make_batch(model, tok, bridge,
                                    latents[idx], [descs[i] for i in idx])
        out = model(inputs_embeds=inputs, labels=labels)
        loss = out.loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % log_every == 0:
            print(f"step {step+1:4d} loss {loss.item():.4f} "
                  f"({(time.time()-t0)/(step+1):.2f}s/step)")
    torch.save(bridge.state_dict(), BRIDGE_CKPT)
    print(f"bridge saved -> {BRIDGE_CKPT}")
    print(f"final loss {np.mean(losses[-20:]):.4f} (start {losses[0]:.4f})")


# ---------------- eval ----------------
NUM_RE = r"\d+"


def answer(model, tok, bridge, latent, question, with_prefix=True, max_new=8,
           raw=False):
    """Generate an answer. latent is a 64-dim vector.
    raw=True  -> latent came from the live world (WorldLatent.embed), so apply
                 the PCA-whitening transform before the bridge.
    raw=False -> latent is already whitened( from latent_pairs_clean.jsonl.

    Conversational rail (fixed 2026-09-04): the soft prefix comes FIRST, then
    the ChatML text tokens (system+user+assistant-anchor) are appended after
    it — [prefix] + [text]. Generation continues from the anchor, and stops at
    the REAL "<|im_end|>" token (the template's EOS, not tok.eos_token_id)."""
    # 1. real template EOS (tokenizer knows "<|im_end|>" as a control token)
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    if eos_id is None or eos_id == tok.unk_token_id:
        eos_id = tok.eos_token_id

    # 2. ChatML text tokens (system + user + assistant anchor)
    sys_txt = "You are PROTEUS. Use your world state to answer the user's question concisely."
    user_txt = question
    prefill = "Action: "
    chatml = ("<|im_start|>system\n" + sys_txt + "<|im_end|>\n"
               "<|im_start|>user\n" + user_txt + "<|im_end|>\n"
               "<|im_start|>assistant\n" + prefill)
    enc = tok(chatml, return_tensors="pt")
    text_ids = enc.input_ids.to("mps")
    with torch.no_grad():
        text_emb = model.model.embed_tokens(text_ids)   # (1, T, 2048)
        if with_prefix:
            z = np.asarray(latent, dtype=np.float32)
            if raw:
                z = load_whiten()(z)
            zt = torch.as_tensor(np.asarray([z], dtype=np.float32), device="mps")
            prefix = bridge(zt).to(torch.bfloat16)            # (1, n, D)
            inp = torch.cat([prefix, text_emb], dim=1)     # [prefix]+[text]
        else:
            inp = text_emb
        attn = torch.ones(inp.shape[:2], dtype=torch.long, device="mps")
        out = model.generate(inputs_embeds=inp, attention_mask=attn,
                            max_new_tokens=max_new,
                            do_sample=True, temperature=0.4, top_p=0.9,
                            eos_token_id=eos_id, pad_token_id=tok.eos_token_id,
                            early_stopping=True,
                            repetition_penalty=1.3, no_repeat_ngram_size=3)
    return tok.decode(out[0], skip_special_tokens=True)


def action_id_of(action):
    a = action or ""
    if "delete" in a:
        return 3
    for i, t in enumerate(("txt", "jpg", "pdf")):
        if f"->{t}" in a:
            return i
    return 4


def _token_ids(tok_, phrase):
    """Token ids for a phrase, correctly handling the leading space (BPE):
    the model emits ' txt' after 'Action:', not 'txt'. Tokenize the space."""
    return tok_(" " + phrase, add_special_tokens=False).input_ids


@torch.no_grad()
def phrase_logprobs_batch(model, bridge, latents, cands):
    """Log-likelihood of EVERY candidate phrase for a batch of latents, in ONE
    model forward (the scaled-eval speedup: ~14k forwards -> ~batch-count
    forwards).

    Returns (N, C) matrix: logprob of candidate c for latent-row i. latents is
    (N, d) or None for the no-prefix baseline (then returns the same row
    broadcast for all N — the baseline doesn't depend on the latent).
    """
    global _tok
    if _tok is None:
        load_qwen()
    tok = _tok
    pids = tok(PROMPT, add_special_tokens=True).input_ids
    nctx = len(pids)
    # tokenize every candidate (leading space intact), no BPE boundary merge
    cand_ids = [tok(c, add_special_tokens=False).input_ids for c in cands]
    max_n = max(len(x) for x in cand_ids)
    C = len(cands)
    D = QWEN_DIM

    N = latents.shape[0] if latents is not None else 1
    npre = bridge.n_prefix if latents is not None else 0

    # build the full batch: (N*C, npre + nctx + max_n) inputs_embeds
    with torch.no_grad():
        # prompt+phrase ids (padded right)
        seq = torch.full((N, C, nctx + max_n), tok.eos_token_id, dtype=torch.long,
                         device="mps")
        for c, fids in enumerate(cand_ids):
            seq[:, c, :nctx + len(fids)] = torch.as_tensor(
                pids + fids, device="mps")
        ids = seq.view(N * C, nctx + max_n)
        emb = model.model.embed_tokens(ids)          # (N*C, nctx+max_n, D)

        if npre:
            z = torch.as_tensor(latents, dtype=torch.float32, device="mps")
            pre = bridge(z).to(torch.bfloat16)       # (N, npre, D)
            pre = pre.unsqueeze(1).expand(-1, C, -1, -1).reshape(N * C, npre, D)
            emb = torch.cat([pre, emb], dim=1)       # (N*C, npre+nctx+max_n, D)

        out = model(inputs_embeds=emb)
        logits = out.logits.float()                  # (N*C, T, V)

    lp = torch.zeros((N, C), dtype=torch.float32, device="mps")
    for c, fids in enumerate(cand_ids):
        n = len(fids)
        # rows for this candidate: r*C + c
        rows = torch.arange(N, device="mps") * C + c
        # phrase token i predicted at logits[ npre + nctx - 1 + i ]
        sel = logits[rows, npre + nctx - 1: npre + nctx - 1 + n]  # (N, n, V)
        ls = torch.log_softmax(sel, dim=-1)
        tgt = torch.as_tensor(fids, device="mps")
        lp[:, c] = ls[:, torch.arange(n, device="mps"), tgt].mean(dim=-1)
    return lp.cpu().numpy()


@torch.no_grad()
def phrase_logprob(model, bridge, latent, phrase):
    """Single-candidate log-likelihood (kept for introspection); the scaled
    eval uses phrase_logprobs_batch."""
    lps = phrase_logprobs_batch(
        model, bridge,
        np.asarray([latent], dtype=np.float32) if latent is not None else None,
        [phrase])
    return float(lps[0, 0])


def phrase_logprobs_batch_chunked(model, bridge, latents, cands, chunk=128):
    """Chunked version of phrase_logprobs_batch to bound memory on MPS when
    scoring many rows at once (the scaled full-set eval)."""
    if latents is None:
        return phrase_logprobs_batch(model, bridge, None, cands)
    N = latents.shape[0]
    out = np.zeros((N, len(cands)), dtype=np.float32)
    for s in range(0, N, chunk):
        out[s:s + chunk] = phrase_logprobs_batch(
            model, bridge, latents[s:s + chunk], cands)
    return out


def eval_action(n=0):
    """The bridge's REAL task: grounded action explanation. Score all candidate
    action phrases under the model WITH and WITHOUT the latent prefix; check if
    the prefix raises the rank / probability of the TRUE phrase. The prompt has
    zero file info, so any gain must come from the latent (proxied through the
    bridge).

    n=0 -> run on the FULL dataset (the scaled test). Candidates are derived
    dynamically from the target classes actually present (not hardcoded), so
    the accuracy is measured against the real label set the bridge must
    discriminate."""
    model, tok = load_qwen()
    bridge = load_bridge()
    rows = load_pairs()

    # dynamic candidate set = all distinct normalized targets present
    cands = sorted({normalize_target(r.get("action")) for r in rows})
    print(f"[eval] candidates ({len(cands)}): {cands}")

    rng = np.random.default_rng(2)
    idx = rng.choice(len(rows), min(n, len(rows)) if n else len(rows),
                     replace=False)
    # batch all rows into one parallel forward (scaled eval)
    lat = np.asarray([rows[i]["latent"] for i in idx], dtype=np.float32)
    print(f"[eval] batched forward over {len(idx)} rows x {len(cands)} cands ...")
    Lw = phrase_logprobs_batch_chunked(model, bridge, lat, cands)     # with
    Lwo = phrase_logprobs_batch(model, bridge, None, cands)            # without

    n_eval = len(idx)
    # no-prefix baseline is row-independent (prompt alone) — Lwo is (1, C)
    Lwo_b = Lwo[0]
    w_corr = sum(Lw[i][cands.index(normalize_target(rows[idx[i]]["action"]))]
                 == Lw[i].max() for i in range(n_eval))
    wo_corr = sum(Lwo_b[cands.index(normalize_target(rows[idx[i]]["action"]))]
                  == Lwo_b.max() for i in range(n_eval))
    ne = n_eval or 1
    print(f"\n=== BRIDGE PHRASE EVAL (n={n_eval}, chance={1/len(cands):.0%}) ===")
    print(f"  WITH prefix:    best-phrase correct {w_corr}/{n_eval} "
          f"({w_corr/ne:.0%})")
    print(f"  WITHOUT prefix: best-phrase correct {wo_corr}/{n_eval} "
          f"({wo_corr/ne:.0%})")
    # per-class breakdown
    print("  per-class (with | without):")
    for t in sorted({normalize_target(rows[i]["action"]) for i in idx}):
        sub = [i for i in range(n_eval)
               if normalize_target(rows[idx[i]]["action"]) == t]
        wc = sum(Lw[i][cands.index(t)] == Lw[i].max() for i in sub)
        oc = sum(Lwo_b[cands.index(t)] == Lwo_b.max() for i in sub)
        print(f"    {t:12s} {wc}/{len(sub)} ({wc/max(len(sub),1):.0%})  |  "
              f"{oc}/{len(sub)} ({oc/max(len(sub),1):.0%})")
    if w_corr > wo_corr:
        print("  => PREFIX ADDS GROUNDED PHRASE (latent steers the LLM's language)")
    else:
        print("  => no gain over no-prefix yet")


def train_transition(epochs=400, batch=8, lr=3e-4, log_every=50,
                     use_surprise=True, ckpt_suffix=""):
    """Train the EMPIRICAL INVERSE-DYNAMICS bridge. Inputs are empirical
    (z_t, z_{t+1}[, e]) — NO simulated rollout, NO action label in the input.
    The target is the action phrase that caused the transition."""
    model, tok = load_qwen()
    ckpt = pathlib.Path(str(TRANSITION_CKPT) + ckpt_suffix)
    if pathlib.Path(ckpt).exists():
        bridge = load_bridge(ckpt=ckpt)
    else:
        bridge = JEPAToQwenBridge(in_dim=129 if use_surprise else 128)
    bridge.to("mps")
    emb_norm = token_embed_norm(model, tok)
    bridge.scale.fill_(emb_norm)
    print(f"token embed norm = {emb_norm:.2f}; scaled prefix to match")

    rows = load_transition_pairs()
    print(f"transition pairs: {len(rows)}  (in_dim={bridge.in_dim})")
    latents = transition_vec(rows, use_surprise=use_surprise)
    descs = [r["target"] for r in rows]
    prompt = TRANSITION_PROMPT

    opt = torch.optim.Adam(bridge.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    losses = []
    t0 = time.time()
    # class-balanced sampling: sample each target class equally per batch so
    # the dominant "move txt" prior doesn't get over-suppressed (the
    # prior-suppression overcorrection that crushed txt in the unweighted runs).
    classes = sorted({d for d in descs})
    class_idx = {c: [i for i, d in enumerate(descs) if d == c] for c in classes}
    print(f"class-balanced sampling over {len(classes)} classes "
          f"(sizes {[len(class_idx[c]) for c in classes]})")
    for step in range(epochs):
        picks = []
        per = batch // len(classes)
        rem = batch % len(classes)
        for c in classes:
            ci = class_idx[c]
            if not ci:
                continue
            n = per + (1 if rem else 0)
            rem = max(rem - (1 if rem else 0), 0)
            picks.extend(rng.choice(ci, min(n, len(ci)), replace=False).tolist())
        if len(picks) < batch:
            picks.extend(rng.choice(len(rows), batch - len(picks), replace=False).tolist())
        idx = np.asarray(picks, dtype=np.int64)
        inputs, labels = make_batch(model, tok, bridge,
                                    latents[idx], [descs[i] for i in idx],
                                    prompt=prompt)
        out = model(inputs_embeds=inputs, labels=labels)
        loss = out.loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % log_every == 0:
            print(f"step {step+1:4d} loss {loss.item():.4f} "
                  f"({(time.time()-t0)/(step+1):.2f}s/step)")
    torch.save(bridge.state_dict(), ckpt)
    print(f"transition bridge saved -> {ckpt} (in_dim={bridge.in_dim}, "
          f"use_surprise={use_surprise})")
    print(f"final loss {np.mean(losses[-20:]):.4f} (start {losses[0]:.4f})")


def _prior_accuracy(Lw, targets, cands, lam=0.0, prior=None):
    """Best-phrase accuracy under a log-prior decode-time interpolation:
        score_c = Lw[:,c] + lam * log(prior_c).
    Lw: (N, C) phrase logprobs. targets: (N,) true candidate labels (strings).
    Returns (overall, per_class, balanced) where balanced = unweighted
    mean over per-class accuracies (counts the rare classes fairly)."""
    L = np.asarray(Lw, dtype=np.float64).copy()
    if lam != 0.0 and prior is not None:
        lp = np.log(np.asarray(prior, dtype=np.float64) + 1e-12)
        L = L + lam * lp[None, :]
    L = np.asarray(L, dtype=np.float64)
    tgt = np.asarray([cands.index(t) for t in targets], dtype=np.int64)
    # tie-tolerant: "correct" = true label attains the max (any of several maxes).
    # (argmax-first-wins would under-count rows where pdf ties another class.)
    is_max = L == L.max(axis=1, keepdims=True)
    correct = is_max[np.arange(len(tgt)), tgt]
    overall = float(correct.mean())
    per = {}
    for i, c in enumerate(cands):
        mask = tgt == i
        per[c] = float(correct[mask].mean()) if mask.any() else 0.0
    balanced = float(np.mean(list(per.values())))
    return overall, per, balanced


def _class_counts(targets, cands):
    """Empirical class frequencies (for the decode-time prior)."""
    n = len(targets)
    return [float(targets.count(c) / n if n else 0.0) for c in cands]


def eval_transition(n=0, use_surprise=True, ckpt_suffix="",
                    prior_lambda=None, sweep_prior=False):
    """THE PREDICTION HARNESS: can Qwen name the action that caused an
    EMPIRICAL transition, given only (z_t, z_{t+1}[, e]) through the bridge?

    The prompt has zero file/action info, so any gain must come from the
    latent channels. Candidates are the action phrases actually present in
    the data. Reports WITH vs WITHOUT prefix + per-class, and is runnable in
    BOTH ablations (use_surprise True/False) so we can see how much the
    empirical delta alone carries vs the surprise scalar.

    Decode-time prior correction: with --prior-lambda λ (or --sweep-prior,
    a table over λ) the empirical class prior is interpolated into the
    phrase logprobs at eval time only (score = loglik + λ·log prior),
    testing whether a single scalar rescues the txt-collapse without
    retraining — the fix for the prior-suppression overcorrection found in
    the per-class experiments."""
    model, tok = load_qwen()
    bridge = load_bridge(ckpt=pathlib.Path(str(TRANSITION_CKPT) + ckpt_suffix))
    rows = load_transition_pairs()
    cands = sorted({normalize_target(r.get("action")) for r in rows})
    print(f"[eval-transition] candidates ({len(cands)}): {cands}")
    print(f"[eval-transition] in_dim={bridge.in_dim} "
          f"use_surprise={use_surprise}")

    rng = np.random.default_rng(2)
    idx = rng.choice(len(rows), min(n, len(rows)) if n else len(rows),
                     replace=False)
    lat = transition_vec([rows[i] for i in idx], use_surprise=use_surprise)
    print(f"[eval-transition] batched forward over {len(idx)} rows x "
          f"{len(cands)} cands ...")
    Lw = phrase_logprobs_batch_chunked(model, bridge, lat, cands)     # with
    Lwo = phrase_logprobs_batch(model, bridge, None, cands)            # without

    n_eval = len(idx)
    Lwo_b = Lwo[0]
    targets = [normalize_target(rows[idx[i]]["action"]) for i in range(n_eval)]
    prior = _class_counts([normalize_target(r.get("action")) for r in rows], cands)

    # ---- base report (λ=0: current behavior) ----
    overall_w, per_w, bal_w = _prior_accuracy(Lw, targets, cands, 0.0)
    overall_wo, per_wo, bal_wo = _prior_accuracy(Lwo_b[None, :] * np.ones((n_eval, 1)), targets, cands, 0.0)
    ne = n_eval or 1
    print(f"\n=== INVERSE-DYNAMICS BRIDGE EVAL (n={n_eval}, "
          f"chance={1/len(cands):.0%}, use_surprise={use_surprise}) ===")
    print(f"  WITH prefix:    best-phrase correct {int(overall_w*n_eval)}/{n_eval} "
          f"({overall_w:.0%})")
    print(f"  WITHOUT prefix: best-phrase correct {int(overall_wo*n_eval)}/{n_eval} "
          f"({overall_wo:.0%})")
    print("  per-class (with | without):")
    for t in sorted({normalize_target(rows[i]["action"]) for i in idx}):
        wc = int(per_w[t] * sum(1 for x in targets if x == t))
        oc = int(per_wo[t] * sum(1 for x in targets if x == t))
        n_sub = sum(1 for x in targets if x == t)
        print(f"    {t:12s} {wc}/{n_sub} ({per_w[t]:.0%})  |  "
              f"{oc}/{n_sub} ({per_wo[t]:.0%})")
    if overall_w > overall_wo:
        print("  => PREFIX ADDS INVERSE-DYNAMICS GROUNDING "
              "(empirical delta steers the LLM's action naming)")
    else:
        print("  => no gain over no-prefix yet")

    # ---- decode-time prior interpolation ----
    if sweep_prior or prior_lambda is not None:
        lams = ([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
                 if sweep_prior else [prior_lambda])
        hdr = " ".join(f"{c[:5]:>6s}" for c in cands)
        print(f"\n  DECODE-TIME PRIOR INTERPOLATION (prior={[round(p,3) for p in prior]})")
        print(f"    {'λ':>5s} {'overall':>8s} {'balanced':>8s}  {hdr}")
        best_o, best_b = (0.0, 0.0,); best_lo = best_lb = None
        for lam in lams:
            ov, per, bal = _prior_accuracy(Lw, targets, cands, lam, prior)
            tag_o = ""
            if ov > best_o: best_o = ov; best_lo = lam; tag_o = " *"
            if bal > best_b: best_b = bal; best_lb = lam; tag_b = " *"
            row = (f"{lam:5.2f} {ov:8.1%} {bal:8.1%}  "
                  f"{' '.join(f'{per[c]:.2f}' for c in cands)}")
            print("    " + row + (f"  [best-ov{tag_o}]" if lam == best_lo else ""))
        print(f"    best overall: λ={best_lo} ({best_o:.1%})  |  best balanced: "
              f"λ={best_lb} ({best_b:.1%})")


def eval_answers(n=80):
    """The honest test: does the prefix ADD information? Ask a question that
    is answerable ONLY from the latent (a file count), with and without the
    projected prefix.

    Measures TWO things, because a prefix that just parrots the training prior
    is not 'adding world state':
      (a) ANSWER RATE — how often the model produces any number at all.
          WITHOUT the latent, Qwen literally cannot count (it has no count in
          the prompt) and typically starts a thinking loop. WITH the prefix it
          commits to a number. A higher answer-rate WITH = the latent steers
          generation.
      (b) ACCURACY on the counts it does output. If the prefix were pure
          noise/prior, accuracy wouldn't beat the mode. We need BOTH: the
          prefix must elicit an answer AND that answer must track the truth."""
    model, tok = load_qwen()
    bridge = load_bridge()
    rows = load_pairs()
    rng = np.random.default_rng(1)
    idx = rng.choice(len(rows), min(n, len(rows)), replace=False)

    import re
    stats = {"w_ans": 0, "w_corr": 0, "wo_ans": 0, "wo_corr": 0}
    n_eval = 0
    for i in idx:
        r = rows[i]
        obs = r["obs"]
        truth = str(int(obs[0]))
        question = "How many .txt files are in the root directory? Answer with a number only."
        with_p = answer(model, tok, bridge, r["latent"], question, True, max_new=16)
        without_p = answer(model, tok, bridge, r["latent"], question, False, max_new=16)
        m_w = re.search(r"\d+", with_p)
        m_wo = re.search(r"\d+", without_p)
        n_eval += 1
        if m_w:
            stats["w_ans"] += 1
            stats["w_corr"] += (m_w.group(0) == truth)
        if m_wo:
            stats["wo_ans"] += 1
            stats["wo_corr"] += (m_wo.group(0) == truth)
    ne = n_eval or 1
    print(f"\n=== BRIDGE EVAL (n={n_eval}) ===")
    print(f"  WITH prefix:    answered {stats['w_ans']}/{n_eval} "
          f"({stats['w_ans']/ne:.0%}), correct {stats['w_corr']}/{n_eval} "
          f"({stats['w_corr']/ne:.0%})")
    print(f"  WITHOUT prefix: answered {stats['wo_ans']}/{n_eval} "
          f"({stats['wo_ans']/ne:.0%}), correct {stats['wo_corr']}/{n_eval} "
          f"({stats['wo_corr']/ne:.0%})")
    # conditional accuracy: of the answers given, how often correct
    ca_w = stats['w_corr'] / stats['w_ans'] if stats['w_ans'] else 0
    ca_wo = stats['wo_corr'] / stats['wo_ans'] if stats['wo_ans'] else 0
    print(f"  conditional accuracy (correct / answered): "
          f"WITH {ca_w:.0%}  WITHOUT {ca_wo:.0%}")
    print(f"  answer-rate gain (WITH - WITHOUT): {stats['w_ans']-stats['wo_ans']}/+"
          f"{n_eval} = {(stats['w_ans']-stats['wo_ans'])/ne:+.0%}")
    if stats['w_ans'] > stats['wo_ans']:
        print("  => PREFIX STEERS GENERATION (elicits answers the prompt alone can't)")
    else:
        print("  => prefix not yet eliciting answers — recheck bridge")
    if ca_w > ca_wo:
        print("  => PREFIX ADDS ACCURATE WORLD STATE (conditional acc beats no-prefix)")
    else:
        print("  => conditional acc not beating no-prefix yet — prefix may be prior-biased")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-action", action="store_true")
    ap.add_argument("--train-transition", action="store_true")
    ap.add_argument("--eval-transition", action="store_true")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ckpt-suffix", default="",
                    help="checkpoint suffix (e.g. _128) so ablations don't clobber")
    ap.add_argument("--prior-lambda", type=float, default=None,
                    help="decode-time log-prior interpolation λ (score = loglik + λ·log prior)")
    ap.add_argument("--sweep-prior", action="store_true",
                    help="sweep λ over a grid and report the best decode-time prior")
    sp = ap.add_mutually_exclusive_group()
    sp.add_argument("--use-surprise", dest="use_surprise", action="store_true",
                    default=True, help="inverse-dynamics input includes e_surprise (129 dim)")
    sp.add_argument("--no-surprise", dest="use_surprise", action="store_false",
                    help="inverse-dynamics input is empirical delta only (128 dim)")
    args = ap.parse_args()
    if args.train:
        train(epochs=args.epochs, batch=args.batch)
    if args.eval:
        eval_answers()
    if args.eval_action:
        eval_action()
    if args.train_transition:
        train_transition(epochs=args.epochs, batch=args.batch,
                         use_surprise=args.use_surprise, ckpt_suffix=args.ckpt_suffix)
    if args.eval_transition:
        eval_transition(use_surprise=args.use_surprise, ckpt_suffix=args.ckpt_suffix,
                        prior_lambda=args.prior_lambda, sweep_prior=args.sweep_prior)