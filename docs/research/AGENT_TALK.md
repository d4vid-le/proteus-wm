# AGENT_TALK — PROTEUS speaks on its own (two-way communication)

Status: **2026-09-06 — first working agent-initiated voice.** PROTEUS can now
speak without being prompted, reusing its existing grounded JEPA->Qwen "mouth"
(`jepa_explain` / `jepa_llm_bridge` / `reflex`). No new model, no research.

The user goal (verbatim): *"creating an agent that might one day learn to
communicate back to me in natural conversational language."* This doc records
where that stands, the log-health check, what was built, the design/plan, and
the honest blocker.

---

## 1. What was built this session — `agent_talk.py`

A small, self-contained CLI that launches the agent's "mouth" on two **self-
triggered** voices (agent-initiated, not human-prompted):

| voice | when it speaks | how it's grounded | needs live obs? |
|---|---|---|---|
| `surprise` | it noticed something (prediction-error spike) | `reflex.SurpriseReflex._jepa_hypothesis` (JEPA->Qwen bridge + PAM memory) | **YES** — without live obs it stays silent ("no private language") |
| `action` | it did something (verified action logged) | the **verified** action label (real ground truth) | **NO** — works now |

Run it (loads Qwen once, ~2-4 min, ~4.3GB on MPS — run **instead of** `chat.py`):

```bash
python -m proteus.agent.agent_talk --watch                    # both voices
python -m proteus.agent.agent_talk --watch --voices action    # after-action only
python -m proteus.agent.agent_talk --once --action 'moved photo.jpg -> /jpg/'   # headless test
python -m proteus.agent.agent_talk --once --surprise 0.05     # headless test (needs obs to speak)
```

Reuses: `jepa_explain._load()` (Qwen+bridge), `reflex.SurpriseReflex`
(adaptive threshold = mean+3*std, so only real spikes speak), a shared pace
guard (`--min-interval`, default 30s), and a hard grounding rule (silent if it
can't ground). Agent-initiated messages append to `data/comm.jsonl`; emit is
pluggable (`--emit print|none`). **Verified:** `py_compile` OK; after-action
logic exercised (moved/deleted emit, noop silent, pace respected).

---

## 2. Log-health check (proteus, "on for a few days")

Checked `data/` while writing this. Honest findings:

- **Healthy while running:** reflex verification rate **100%**; world reached
  "clean" (only protected KEEP files remain). 12,462 journal actions, 1,743
  PAM-memory rows, 1,715 interaction events, 30,281 reflex-log lines.
- **BUT the journal has ZERO live `obs`.** `records_with_obs = 0 / 12,462`.
  Every row carries `action`/`detail` but **no 12-dim obs vector**. The patch
  that logs `obs`+`latent` on each `verified_action` (see `HANDOVER.md` §9) is
  **not active** in what's on disk.
- **Daemons have gone quiet ~1.5-2 days.** Last writes: reflex log & eris
  ≈ 2 days ago, journal ≈ 1.5 days, interactions ≈ 1.2 days. So proteus is
  **not currently running** (no fresh writes today).
- Small data-quality note: `eris_memory.jsonl` hypotheses contain stray ANSI
  cursor-control sequences (`\u001b[8D\u001b[K`) — likely captured from terminal
  output in `qwen_client`. Cosmetic, but worth stripping.

**Implication:** the `surprise` voice can't speak yet (nothing to ground on),
and the whole "agent talks to you" goal is blocked **not** by the generator but
by **live obs not flowing**. Fixing that is the #1 priority (see plan).

---

## 3. The two-way communication design (the plan)

The architecture for "learns to communicate back in natural language," layered
on what already exists:

```
observe (webcam + machine telemetry) -> JEPA encoder -> z -> world model
   -> [ INITIATOR ] decides "do I have something to say?"      (self-triggered)
   -> [ grounding CHECK ] assertion that can't be anchored -> downgrade to question
   -> [ mouth ] bridge (soft-prefix z -> frozen Qwen + LoRA) -> natural language
   -> [ deliver ] surface where the user sees it (opencode / chat / notification)
   -> [ outcome loop ] label did the exchange help -> tune initiator + wording
```

- **Initiator** = a small policy over the agent's own epistemic signals:
  surprise (prediction error / reflex spike), novelty (min L2 distance to PAM
  memory), uncertainty (low prediction confidence → ask, not assert), presence
  change (webcam), plus a pace guard. This is the "write back on its own" part.
- **Grounding check (hard rule)** — an assertion must have a groundable anchor
  (obs/z/event + confidence ≥ τ); otherwise it becomes a question or stays
  silent. This is the enforceable version of "no private language."
- **Outcome loop** — when the user replies or ground truth is verified, label
  the message; use `(provenance, kind, confidence, outcome)` to tune the
  initiator thresholds and the message phrasing so the agent's communication
  *improves with data*. This is the "emergent" part — a grounded reward loop.
- **Two-way** — a reply path (`direction: user`) in the same long-lived process,
  so exchanges become a language game, not one-shot.

### Phased plan

1. **P0 (now): get live `obs` flowing.** Run `autonomous.py --daemon`
   (or `daemon.py`) so each `verified_action` logs `obs`+`latent`. This
   unblocks everything (the `surprise` voice, grading, and richer grounding).
2. **P1 (done, minimal): agent speaks after actions** via verified labels
   (`agent_talk.py --voices action`). Works today without obs.
3. **P2: message schema + full initiator + outcome loop** — structured
   `data/comm.jsonl` messages (direction/kind/grounding/outcome), a novelty +
   surprise initiator, and the outcome-driven tuning.
4. **P3: co-learn the protocol** — open prompts, outcome-labeled conversations,
   so the agent learns *which* language works (the language-game learning).
5. **P4: webcam presence + vision-first grounding + optional SCG** once the
   stream is rich and continual learning is wired to live video.

### Honest hard limits
- The model is a **base** Qwen3.5-2B (unified vision-language, but used
  **text-only** here) — the LoRA makes it speak *grounded phrases*, not rich
  open conversation. Deeper back-and-forth is a later fine-tune/chat variant.
- Qwen loads once (~4.3GB MPS); run `agent_talk` **instead of** `chat.py`.
- Grounded `surprise` needs live obs; until then the `action` voice is the live
  one, and that's genuinely grounded in the verified label.

---

## 4. How to run / verify

```bash
cd ~/proteus
python3 -m py_compile proteus/agent/agent_talk.py                 # syntax
python -m proteus.agent.agent_talk --once --action 'moved photo.jpg -> /jpg/'  # log a spoken line
python -m proteus.agent.agent_talk --watch --voices action       # speak after each verified action
```

Notes: keep the shell command alive during the 2-4 min first Qwen load so it
isn't cut off; don't run `chat.py` simultaneously. Messages land in
`data/comm.jsonl`.

---

## 5. Experimental build — live-obs run (2026-09-06) — CONFIRMED

The blocker is **not a bug** — the code already logs obs+latent:

- `autonomous.py` (line ~475) passes `obs=vec.tolist(), latent=latent` on every
  `verified_action`. The 12,462 journal rows on disk are **pre-patch/backfill**
  (0 obs); the current code is correct.
- Ran one real single-pass cleanup (gated/Proteus Gates checkpoint): **40 acts / 40
  verified, 1 shift detected, 13 continual-learning updates**. The journal rows
  carried a **12-dim `obs` + 64-dim `latent`** → **PASS**.
- To make the grounded `surprise` voice fire, just run the live agent so new rows
  carry obs+latent:
  ```bash
  python -m proteus.agent.autonomous --seed --dir /real/dir        # journal stays in data/, NOT the watched dir
  python -m proteus.agent.daemon --dir /real/dir --vision-stream
  ```
- Caution: don't place the journal **inside** the watched directory — the test
  run above watched a dir that contained `journal.jsonl`, so the agent moved its
  own journal. Harmless here, avoid in real use.

## 6. Plan to build a general agent (honest roadmap)

Preface: I won't claim a timeline to "AGI" — that's not something anyone can
guarantee. This is the honest engineering path toward a **general, embodied,
continually-learning agent that talks to you grounded**, which is a coherent
skeleton for generality (and matches where the field's best guesses point:
world-model-centric, embodied). The current proteus is a strong **M0**.

The thesis: **vision-first JEPA world model = the core (System 1); LLM Qwen =
the interface (System 2); continual learning (Proteus Gates) + capacity growth (SCG)
let it keep learning without forgetting; grounded two-way communication is how
it connects to you.**

- **M0 (done):** vision-first JEPA world model + Proteus Gates continual learning +
  safety barrier (0 violations on KEEP-deletion).
- **M1 (now):** grounded autonomous two-way communication — `agent_talk.py`
  (surprise + after-action voices), with the P0 requirement: run the live agent
  so `obs`+`latent` flow and the `surprise` voice fires. Then the outcome loop
  (learn *when* to speak and *what* to say).
- **M2:** richer perception — live webcam **video** into `z_vis` (not just
  images), fused with machine telemetry, so it *sees you in time*.
- **M3:** longer-horizon + cross-world generalization — multi-step rollout
  consistency (fix the documented ~65% cross-world), compositional worlds.
- **M4:** amortized acquisition — learn an **adapter-generator** so a "new world"
  becomes an inference event (235 → ≤50 transitions) instead of a training event.
- **M5:** the pieces proteus's own `not_agi.py` names as missing — self-model,
  user-model, goal-setting, rule abstraction. A self/user model is what lets it
  communicate *about you* and *about itself*, not just about files.
- **M6:** honesty — keep live verification rate as the ground-truth metric, and
  re-run `not_agi.py` at every milestone.

**The three things to do next (this week), in order:**
1. Get **live `obs`+`latent`** flowing against a real dir (M1's unblocker).
2. Rebuild the transition dataset from that ground truth (docs say it'll be a
   jump in accuracy over the backfill reconstruction).
3. Run `agent_talk.py --watch` against the live stream — the agent starts
   talking, and you start two-way.
