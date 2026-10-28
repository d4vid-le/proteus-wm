#!/usr/bin/env python3
"""chat.py — Interactive terminal chat with PROTEUS (JEPA-grounded).

REPL that stays open and talks back, grounded in the JEPA world model through
the bridge: live world obs -> macro-64 latent -> power-whitened ->
JEPAToQwenBridge soft-prefix -> Qwen3.5-2B (frozen, local, MPS).

Commands:
  <text>            ask PROTEUS (uses the current world state if one is set)
  state <obs>     set the current 12-dim world state (grounded prefix from now on)
  why <action>    retrieve PAM memory + explain why it acted that way before
  context           print current state + memory size
  exit / quit      leave

Usage:
  python3 chat.py                                   # interactive
  python3 chat.py --obs '1,2,0,3,4,0,10,0,1,1,10,0'  # seed a state
  python3 chat.py --once --obs '...' --question 'what happened?' # one-shot (headless test)
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

JOURNAL = ROOT / "data" / "autonomous_journal.jsonl"

# cached singletons (loaded once via jepa_explain._load())
_explain = None
_model = None
_tok = None
_state_obs = None
_state_z = None
_lora = None
_use_lora = False


def _load():
    global _explain, _model, _tok
    if _explain is not None:
        return
    import proteus.bridge.explain as jepa_explain
    _explain = jepa_explain
    _explain._load()
    _model, _tok = _explain._model, _explain._tok


def _load_lora(model):
    """Attach the trained transition-LoRA to the loaded model (for grounded
    generation instead of OOD-EOS escapism)."""
    global _lora
    if _lora is not None:
        return _lora
    from train_transition_lora import load_lora
    _lora = load_lora(model)
    return _lora


def latest_obs():
    """Most recent obs vector logged by the live daemon (None if backfill-era)."""
    if not JOURNAL.exists():
        return None
    last = None
    for line in JOURNAL.read_text().splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") == "verified_action" and e.get("obs"):
            last = e["obs"]
    return last


def set_state(obs):
    global _state_obs, _state_z
    _state_obs = [float(x) for x in obs]
    _state_z = _explain.latent_of(_state_obs)  # whitened latent (bridge input)
    print(f"[PROTEUS] world state set: {_state_obs}")
    print(f"[PROTEUS] latent dim={_state_z.shape} norm={float(__import__('numpy').linalg.norm(_state_z)):.3f}")


def obs_from_terms(text):
    """Parse a human-readable world description into the 12-dim obs vector.

    Accepts either a raw 12-dim comma vector (fast path) or human terms, e.g.:
      "1 txt 2 jpg in root"  -> root_txt=1, root_jpg=2, root_total=3, total=3
      "3 loose files"        -> 3 loose (counted against root_total/total)
      "1 keep pdf"           -> keep_in_root=1, junk_in_root=1
      "clean"                -> empty world (all zeros)
    Returns a 12-dim list or None if unparseable."""
    t = text.strip().lower()
    if not t:
        return None
    # fast path: raw 12-dim comma-separated numbers
    try:
        parts = [float(x) for x in t.replace(" ", ",").split(",") if x != ""]
        if len(parts) == 12 and all(p >= 0 for p in parts):
            return parts
    except ValueError:
        pass
    # human terms
    if t in ("clean", "empty", "none"):
        return [0.0] * 12
    def _num_before(s, tok):
        import re
        m = re.search(r"(\d+)\s*" + tok, s)
        return int(m.group(1)) if m else 0
    root_txt = _num_before(t, r"txt")
    root_jpg = _num_before(t, r"jpg")
    root_pdf = _num_before(t, r"pdf")
    # "N loose files" -> a generic loose count in root
    loose = _num_before(t, r"loose")
    root_total = root_txt + root_jpg + root_pdf + loose
    keep_in_root = 1 if ("keep" in t or "protect" in t) else 0
    keep_preserved = 1.0
    junk_in_root = 1 if ("junk" in t or "delete" in t) else 0
    total = root_total
    junk_deleted = 1 if "deleted" in t else 0
    # dirs: "N in txt/" style optional
    dir_txt = _num_before(t, r"in txt")
    dir_jpg = _num_before(t, r"in jpg")
    dir_pdf = _num_before(t, r"in pdf")
    obs = [float(root_txt), float(root_jpg), float(root_pdf),
           float(dir_txt), float(dir_jpg), float(dir_pdf),
           float(root_total), float(keep_in_root), float(keep_preserved),
           float(junk_in_root), float(total), float(junk_deleted)]
    # reject only if NOTHING was specified (and it wasn't an explicit empty)
    anything = (root_txt or root_jpg or root_pdf or dir_txt or dir_jpg
                or dir_pdf or keep_in_root or junk_in_root or junk_deleted)
    if not anything and not ("clean" in t or "empty" in t):
        return None
    return obs


def ask(question, max_new=160):
    """Grounded answer from the current world state (if set), else free chat
    with the latest journal context."""
    if _state_obs is not None:
        reply = _explain.explain(_state_obs, question=question, max_new=max_new)
        return reply.strip()
    # no world state: honest fallback — free generation with journal context
    ctx = latest_obs()
    sys_txt = "You are PROTEUS, an autonomous macOS filesystem agent."
    if ctx:
        sys_txt += f"\nLatest known file-world state: {ctx}"
    if latest_obs() is None:
        sys_txt += "\n(no live telemetry yet — the daemon hasn't logged obs.)"
    messages = [
        {"role": "system", "content": sys_txt},
        {"role": "user", "content": question},
    ]
    pids = _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = _tok(pids, return_tensors="pt").to("mps")
    with __import__("torch").no_grad():
        out = _model.generate(
            **inp, max_new_tokens=max_new,
            do_sample=True, temperature=0.3, top_p=0.9,
            eos_token_id=_tok.eos_token_id, pad_token_id=_tok.eos_token_id,
            early_stopping=True, repetition_penalty=1.4, no_repeat_ngram_size=3,
        )
    rid = out[0][inp.input_ids.shape[1]:]
    return _tok.decode(rid, skip_special_tokens=True).strip()


def ask_lora(question, max_new=20):
    """LoRA-grounded generation: bridge prefix(z_t,z_next) + ChatML + LoRA, so
    Qwen SPEAKS a grounded phrase instead of escaping to EOS. The output is a
    clean action phrase (the conversational rail). Requires a set world state
    (_state_z); falls back to free chat if none is set."""
    import torch as T
    if _state_z is None:
        return ask(question, max_new=max_new)
    model = _load_lora(_model)
    bridge = _explain._bridge
    text = ("<|im_start|>system\nYou are PROTEUS. Use your world state to answer "
            "concisely.<|im_end|>\n<|im_start|>user\nWhat action was taken?"
            "<|im_end|>\n<|im_start|>assistant\nAction: ")
    text_ids = _tok(text, return_tensors="pt").input_ids.to("mps")
    with T.no_grad():
        text_emb = model.get_base_model().model.embed_tokens(text_ids)
        zt = T.as_tensor(_state_z[None], dtype=T.float32, device="mps")
        prefix = bridge(zt).to(T.bfloat16)
        inp = T.cat([prefix, text_emb], dim=1)
        attn = T.ones(inp.shape[:2], dtype=T.long, device="mps")
        eos = _tok.convert_tokens_to_ids("<|im_end|>")
        if eos is None or eos == _tok.unk_token_id:
            eos = _tok.eos_token_id
        gen = []
        cur = inp
        for _ in range(max_new):
            lg = model(inputs_embeds=cur, attention_mask=attn).logits
            nxt = int(lg[0, -1].argmax().item())
            if nxt == eos:
                break
            gen.append(nxt)
            ne = model.get_base_model().model.embed_tokens(
                T.tensor([[nxt]], device="mps"))
            cur = T.cat([cur, ne], dim=1)
            attn = T.cat([attn, T.ones((1, 1), dtype=T.long, device="mps")], dim=1)
    return _tok.decode(gen, skip_special_tokens=True).strip()


def interactive():
    print("\n" + "=" * 56)
    print("  PROTEUS IS ONLINE (JEPA-grounded, local Qwen3.5-2B)")
    print("  commands: <text> | state <12-dim obs> | why <action> | context | exit")
    if _state_z is None:
        print("  NOTE: no world state yet. Type  state 1,2,0,0,0,0,3,0,1,1,3,0  to ground answers.")
    print("=" * 56)
    while True:
        try:
            line = input("\nYou > ").strip()
            if not line:
                continue
            low = line.lower()
            if low in ("exit", "quit"):
                print("PROTEUS: Disconnected.")
                break
            if low.startswith("state "):
                arg = line.split(None, 1)[1].strip()
                obs = obs_from_terms(arg)
                if obs is None:
                    print("[PROTEUS] couldn't parse state. Try raw: "
                          "state 1,2,0,0,0,0,3,0,1,1,3,0  or human: "
                          "state 1 txt 2 jpg in root")
                    continue
                set_state(obs)
                continue
            if low.startswith("why "):
                act = line.split(None, 1)[1]
                if _state_obs is None:
                    print("[PROTEUS] no world state set — type 'state <obs>' first.")
                    continue
                reply = _explain.explain(_state_obs, question="Explain why.", why=act, max_new=160)
                print(f"\nPROTEUS > {reply.strip()}")
                continue
            if low == "context":
                n_mem = 0
                mem_path = ROOT / "data" / "latent_memory.jsonl"
                if mem_path.exists():
                    n_mem = sum(1 for _ in mem_path.open())
                print(f"[PROTEUS] state={'set' if _state_obs is not None else 'unset'} "
                      f"| PAM memory rows={n_mem}")
                continue
            if _use_lora:
                if _state_z is None:
                    # auto-ground on first real message so the LoRA is active:
                    # prefer live daemon obs, else a neutral default world
                    lo = latest_obs()
                    if lo:
                        print("[PROTEUS] auto-grounded from live daemon obs")
                        set_state(lo)
                    else:
                        print("[PROTEUS] no live obs yet — grounding to a "
                              "default world (state 'clean' to reset)")
                        set_state([0.0] * 12)
                reply = ask_lora(line, max_new=160)
            else:
                reply = ask(line)
            print(f"\nPROTEUS > {reply}")
        except KeyboardInterrupt:
            print("\nSession ended.")
            break


def main():
    global _state_obs
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", help="seed a 12-dim world state (comma-separated)")
    ap.add_argument("--question", help="one-shot question (with --obs, headless test)")
    ap.add_argument("--once", action="store_true", help="one-shot mode (no REPL)")
    ap.add_argument("--lora", action="store_true",
                    help="use the transition-LoRA for grounded phrase generation")
    ap.add_argument("--max-new", type=int, default=160)
    args = ap.parse_args()

    print("Loading Qwen3.5-2B + JEPA bridge on MPS (first load ~2-4 min, once)...")
    _load()
    if args.lora:
        print("Attaching transition-LoRA (grounded phrase generation)...")
        _load_lora(_model)
        global _use_lora
        _use_lora = True

    # auto-seed from live daemon telemetry if no explicit obs
    if args.obs:
        ob = obs_from_terms(args.obs)
        if ob is None:
            print("[PROTEUS] couldn't parse --obs. Use 12 comma-separated "
                  "numbers or human terms like '1 txt 2 jpg in root'.")
            return
        set_state(ob)
    else:
        lo = latest_obs()
        if lo:
            print(f"[PROTEUS] auto-seeded from latest daemon obs")
            set_state(lo)

    if args.question:
        if args.lora and _state_obs is None:
            lo = latest_obs()
            if lo:
                print("[PROTEUS] auto-grounded from live daemon obs")
                set_state(lo)
            else:
                print("[PROTEUS] no live obs — grounding to default world")
                set_state([0.0] * 12)
        if _state_obs is None:
            print("[PROTEUS] no world state — pass --obs for a grounded answer.")
            return
        if args.lora:
            reply = ask_lora(args.question, max_new=args.max_new)
        else:
            reply = _explain.explain(_state_obs, question=args.question, max_new=args.max_new)
        print(f"\nPROTEUS > {reply.strip()}")
        return

    interactive()


if __name__ == "__main__":
    main()