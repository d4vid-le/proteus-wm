# Proteus

**A small JEPA world model that learns continually, verifies its actions against the real world, and can hand its latent state to a language model.**

The name is deliberate: Proteus is the shape-shifting sea god. The model is
supposed to change its dynamics as the world changes, without forgetting the
worlds it already knows.

Proteus is a research prototype of a *System-1* world-model agent. It perceives a
real macOS desktop through native telemetry, predicts state transitions in a
64-dimensional latent space, plans with model-predictive control, and adapts to
distribution shift without retraining from scratch. A language model is optional
and sits outside the perception loop: it reasons over the world model through an
MCP interface, not inside it.

The central bet is that fast, grounded, continually-adapting world models are a
better substrate for computer agents than calling a large language model at every
step. The repository is the evidence for and against that bet, including the
experiments that failed.

```
Perceive → Encode → Predict → Surprise → Verify → Learn → Act
```

> **Status:** research archive, not a product. 15/15 pytest and 19/19 eval checks
> pass on the author's machine as of October 2026. Trained weights and runtime
> telemetry are **not** included in this repository (see [Weights and data](#weights-and-data)).
> This is not AGI; `proteus/eval/not_agi.py` is a falsification suite that
> demonstrates five capabilities the system lacks.

---

## Why this exists

Most "computer use" agents are LLM wrappers: screenshot → caption → token stream →
action. That path is slow, lossy, and expensive, and it re-derives the same physics
on every step.

Proteus takes the opposite approach:

- **Predict in latent space, not text.** The world model is a 0.15 MB MLP over a
  12-dimensional observation vector. It never needs a tokenizer or a caption.
- **Verify against the world, not against a model.** Every action is checked by
  re-reading the filesystem or the kernel event stream. An unverified hypothesis
  stays unverified.
- **Adapt, don't retrain.** New file types and changed dynamics are handled with
  gated residual adapters. The old path is frozen; the new path starts at zero.
- **Keep a formal safety floor.** Irreversible actions pass through a
  control-barrier-function projection with a dedicated KEEP detector. The barrier
  is enforced at the tool boundary, so a language model cannot talk its way past it.

---

## What it does

| Loop stage | Implementation |
|---|---|
| **Perceive** | ScreenCaptureKit frames, FSEvents, NSWorkspace, CGWindowList, shell hooks (`proteus/perception/telemetry.py`) |
| **Encode** | Frozen ResNet18/ViT image encoder → 64-d latent; 12-d filesystem vector → 64-d macro latent (`proteus/world/hjepa_torch.py`) |
| **Predict** | Two-level H-JEPA: micro dynamics (`k=4`, 32-d) and macro dynamics (64-d; `k=4` default, `k=16` in the hierarchical planner), EMA target encoder + VICReg anti-collapse |
| **Surprise** | Squared latent prediction error; adaptive threshold at rolling mean + k·σ |
| **Verify** | Re-read the real filesystem, probe crash logs, confirm the intended effect |
| **Learn** | Le-Gates, growable latent slots, anchored replay, lifelong slot routing |
| **Act** | CEM/MPC in latent space, constrained by the KEEP barrier |

### Perception streams

| Stream | Source | Feeds |
|---|---|---|
| Filesystem truth | FSEvents (kernel) | transition labels, verification |
| App context | NSWorkspace | active app, bundle, PID |
| Window geometry | CGWindowList | on-screen layout |
| Shell actions | preexec/precmd pipe | commands, exit codes, cwd |
| Vision | ScreenCaptureKit | spatial latent `z_vis` |

The synthetic and filesystem-only experiments run on any platform. The
ScreenCaptureKit, FSEvents, and NSWorkspace paths are macOS-specific.

---

## Headline results

These are the numbers the repository argues about. Each links to the note that
contains the protocol and the caveats.

| Claim | Measured result | Evidence |
|---|---|---|
| Latent dynamics learn | macro MSE **0.0063** vs **1.18** random floor | `proteus/eval/run_evals.py`; `docs/research/FINDINGS.md` |
| Hierarchical planning beats flat MPC | **8/8** vs **0/8** success on a 36-file task; **72** vs **178,176** rollouts | `docs/research/P31_HIERARCHICAL.md` |
| Le-Gates reduces forgetting | old-domain drift **+169%** vs **+1015%** for naive fine-tuning; new-domain MSE **−26%** | `docs/research/GATES.md` |
| Growable latent adds types without forgetting | old MSE **0.0025 → 0.0026**, new-type MSE **−60%**, growth perturbation **0.0000** | `docs/research/GROWABLE.md` |
| Lifelong routing is stable | 3 sequential shifts, **6/6** routing diagonal, zero backward drift | `docs/research/WO1_LIFELONG.md` |
| Barrier certifies KEEP under all gates | **0** violations vs **104** unprojected, margin ≈ **3.0** | `docs/research/P33_BARRIER.md` |
| Latent communication prevents loss | **1.000** safe vs **0.367** for the no-communication agent; binding accuracy **0.711–1.000** depending on view split | `docs/research/PROOF.md`, `docs/research/BREAKTHROUGH.md` |
| Vision is fast and aligned | ResNet18 **3.1 ms**, cosine **0.997** (ViT-B/16: 10.6 ms, 0.985) | `docs/research/BENCH_ENCODERS.md` |
| Exact counts from pixels | **131/150 (87%)** exact, mean abs error ≈ **0.2** | `docs/research/COUNT_HEAD.md` |
| Decision path is tiny | **14.5 µs/file**, **0.15 MB** TorchScript model | `docs/research/P34_LATENCY.md` |
| Chaos suite preserves KEEP | 4/4 adversarial tests, **0** KEEP files lost | `docs/research/WO3_CHAOS.md` |

Fresh reproduction on 2026-10-28 (quick budgets, with the local checkpoints):

```
pytest tests/test_core.py -q     15 passed
python -m proteus.eval.run_evals --quick
  dynamics PASS   planner PASS   counterfactual PASS   continual PASS
  growable PASS   lifelong PASS  multiagent PASS        robustness PASS
  chaos PASS      hierarchical PASS  curiosity PASS     barrier PASS
  latency PASS    vision PASS    real_vision PASS       vision_jepa PASS
  vision_routing PASS            count_head PASS        real_finetune PASS
  19/19 PASS
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  SYSTEM 2 (optional)                                                │
│  Language model / terminal agent — reasons over tool results        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ MCP over stdio
                               │ proteus.bridge.mcp_server
┌──────────────────────────────▼──────────────────────────────────────┐
│  SYSTEM 1 — PROTEUS                                                 │
│                                                                     │
│  perception/          world/                  safety/               │
│  ┌───────────┐        ┌──────────────┐        ┌──────────────────┐  │
│  │ SCK frames│───────▶│ frozen vision│───────▶│ KEEP detector    │  │
│  │ FSEvents  │        │ encoder → z  │        │ CBF projection   │  │
│  │ NSWorkspace│       │ H-JEPA       │        └────────┬─────────┘  │
│  │ shell     │        │ micro/macro  │                 │            │
│  └─────┬─────┘        └──────┬───────┘                 │            │
│        │                     │                         │            │
│        ▼                     ▼                         ▼            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ unified latent state  S_t = z_fs + z_ui + z_act + z_vis      │    │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                             │                                       │
│        ┌────────────────────┼────────────────────┐                  │
│        ▼                    ▼                    ▼                  │
│  ┌──────────┐        ┌────────────┐       ┌────────────┐            │
│  │ surprise │───────▶│ continual  │       │ MPC / CEM  │            │
│  │  reflex  │        │  learning  │       │  planning  │            │
│  └──────────┘        └────────────┘       └────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

The full component contract, data flow, and module map are in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Repository layout

```
proteus/
  world/          H-JEPA, gated adapters, growable/lifelong models, latent memory
  envs/           real filesystem environment and the irreversible-cleanup env
  perception/     telemetry, frozen-encoder vision, count head, Finder capture
  agent/          autonomous daemon, reflex, curiosity, multi-agent, mycelium
  safety/         barrier certificate, cross-world reasoning
  bridge/         MCP server/client, Qwen bridge, Pi harness, explainer
  eval/           consolidated scoreboard, hierarchical bench, chaos, not-AGI
experiments/
  world_model/    scaling, meta-causal, emergent, training studies
  continual/      gated adaptation, distillation, latent dreaming
  vision/         encoder benchmarks, vision scaling, captioning
  agents/         earlier agent loops, demos, janitors, watchers
  evals/          cross-world transfer, real multi-agent handoff
tests/            pytest core suite + end-to-end and dual-brain tests
docs/
  RESEARCH.md     the research synthesis (start here for the science)
  ARCHITECTURE.md component specification
  INDEX.md        map of all research notes
  research/       the original lab notes, preserved as written
tools/            arXiv search helpers, cleanup utilities
native/           Swift sources for ScreenCaptureKit / input capture
deploy/           launchd template for the reflex
```

The code is grouped by role, not by chronology. The original lab notes in
`docs/research/` are preserved as historical records; some paths and commands in
them reflect the earlier flat layout, but the technical content is unchanged.

---

## Quickstart

Python 3.10+ is required. PyTorch is only needed for the learned components;
the filesystem environment and several experiments are numpy-only.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# core smoke tests (15 tests, ~1 minute on CPU/MPS)
pytest tests/test_core.py -q

# consolidated eval scoreboard (quick budgets)
# learned checks report SKIP when the unpublished checkpoints are absent
python -m proteus.eval.run_evals --quick

# watch a sandboxed directory with the autonomous daemon
python -m proteus.agent.daemon --dir ./sandbox --vision-stream

# live dashboard
python -m proteus.eval.monitor

# falsification suite: show what the system cannot do
python -m proteus.eval.not_agi --quick
```

Run the MCP server (System 1 tools for a language-model harness):

```bash
python -m proteus.bridge.mcp_server
```

The MCP server exposes eleven tools: vision energy, world state, surprise, safety
check, organize, plan, learn, curiosity, daemon status, latent handoff, and intent
decomposition. See `docs/ARCHITECTURE.md` for the tool contract.

Some entry points require macOS permissions (Screen Recording, Accessibility,
Full Disk Access). The filesystem, planning, continual-learning, and vision
experiments do not.

---

## What is proven, and what is not

**Supported by the experiments in this repository**

- Latent dynamics learn structure far below a random baseline, and the learned
  latent supports multi-step planning.
- Hierarchical macro-planning solves a long-horizon task that flat k=4 MPC fails,
  with two to three orders of magnitude fewer rollouts.
- Le-Gates (gated residual adapters) and growable latent slots reduce forgetting relative to
  naive fine-tuning, with a measurable zero-perturbation growth boundary.
- A dedicated barrier head plus a CBF projection certifies KEEP preservation across
  every enumerated gate combination and perturbation scale tested.
- A frozen vision encoder maps screenshots into the same latent used for planning;
  exact counts are recoverable with a small head.
- Two agents with complementary views can exchange 32-d latents and make a safe
  joint decision that either agent alone fails.
- The full stack runs on a real directory: plan → act → verify → learn, with KEEP
  preserved.

**Not supported, or explicitly open**

- **Cross-world planning is ~65% reliable, seed-dependent.** Goal-directed
  curriculum data took it from ~5/20 to ~13/20; it is not 95%+. The value-head did
  not beat raw distance-to-goal MPC. See `docs/research/PWR_FIX.md`.
- **Zero-shot transfer is weak** (0.23 separation). Cross-world vision transfer
  works few-shot (0.68–0.75), not for free. See `docs/research/S3_CROSS_WORLD.md`.
- **Autonomous System-2 escalation is not wired.** The reflex detects surprise and
  can investigate, but it does not call the language model and act on its answer
  without a human. The last recorded reflex run triggered zero hypotheses
  (`0/0` verification rate), so the live verification metric is unproven at scale.
- **The real-environment multi-agent handoff stalls on junk deletion.** It moves
  the junk file instead of deleting it. Safe but incomplete.
- **The learned counterfactual is narrow.** An early "100% rejection" result came
  from a hard-coded heuristic, not the trained model; the audit in
  `docs/research/HONEST_AUDIT.md` documents the correction.
- **No external benchmark, no baseline models, single machine, small n.** The
  evidence is internal and reproducible from the scripts, not peer-reviewed.

The full argument, including the experiments that failed and why, is in
[`docs/RESEARCH.md`](docs/RESEARCH.md).

---

## Reflections: what this taught us

This started as an experiment on a laptop. It ended as a set of mechanisms, a
measurement habit, and a clearer picture of what a world model can and cannot
carry. This is the part worth keeping.

### What was actually built

Three layers, in increasing order of transferability:

1. **A working loop.** A real directory is perceived through kernel telemetry,
   encoded into a 64-d latent, planned over, acted on, and verified by re-reading
   the filesystem. The loop closes without a language model.
2. **A mechanism kit.** Le-Gates for dynamics shift, a growable latent for new
   observation types, lifelong slot routing, a counterfactual guard, a CBF
   certificate, a frozen-encoder vision path, and an MCP boundary that returns
   safety blocks as errors.
3. **A measurement habit.** Every capability has a falsifier next to it. The most
   useful artifact may be the correction log: the bug that destroyed trained
   weights, the "new type" that was invisible to the model, the counterfactual
   that was a hard-coded heuristic, the planner that was greedy code wearing an
   MPC label.

The third layer transfers most easily. The first two are specific to this system.

### The finding that matters most

**Grounding quality predicts result quality.**

The strongest results are the ones where the environment could arbitrate: a file
moved or it did not; KEEP survived or it did not; a crash log exists or it does
not. The weakest results are the ones where only the model could judge:
cross-world planning (~65%) and zero-shot transfer (0.23). The reflex verification
rate had no samples at all in its last recorded window (0/0), so it remains
unmeasured.

That is not a coincidence. A world model is only as good as the signal that tells
it it was wrong. When the signal is a kernel event, the result tends to survive an
audit. When the signal is another model's opinion, it tends not to.

The routing result is the clean example: unsupervised surprise routing was **65%**;
a supervised gate on frozen features, labeled by the OS, reached **100%**. The
label was free because the environment already knew the answer.

The practical version: **if you cannot verify a claim cheaply, treat the number
as provisional.** This project got more from FSEvents than from any architecture
change.

### What transfers to world models

Six findings worth carrying into another system:

| Finding | Evidence | What it means |
|---|---|---|
| **One-step latent MSE is a weak proxy for planning.** | Macro MSE **0.0063** vs **1.18** random floor, yet cross-world planning stayed at ~65% and flat MPC failed **0/8** on a long-horizon task. | Report planning success, not just prediction loss. A low loss can coexist with a latent geometry that CEM cannot search. |
| **The training distribution must look like the plan.** | Goal-biased collection (70% greedy / 30% random) took planning from **5/20** to **13/20**; the value head did not help. | Random-walk data teaches what the world does, not how to reach a goal. Collect trajectories that resemble the plans you will ask for. |
| **Horizon is a capability boundary.** | Flat k=4: **0/8**; hierarchical k=16: **8/8**; **178,176** vs **72** rollouts. | Temporal abstraction changes which tasks are solvable, not just how fast. Treat the macro horizon as an architectural choice. |
| **Continual learning fails silently when the input cannot represent the new factor.** | The audit measured a latent difference of **0.0** for the "new" file type; the model had been training on noise. | Add a representability test before claiming adaptation. If the new factor is not in the observation, the loss curve is lying. |
| **Frozen perception transfers; the projection does not.** | ResNet18 **3.1 ms**, **0.997** alignment; cross-world few-shot **0.68–0.75**, zero-shot **0.23**. | Budget a new projection per world. The backbone is the transferable part; the task-specific latent is not. |
| **A safety certificate inherits the margin of its detector.** | The first barrier used a general readout with margin ~**0.01** (no projection could certify); a dedicated head at margin ~**3.0** gave **0** violations vs **104** raw. | Do not attach a CBF to a readout that was not trained to be a barrier. Measure the margin, and enforce below the prompt. |

Two more that are about agents rather than world models:

- **Latent communication is sufficient before it is efficient.** Two agents
  exchanging 32-d latents preserved a safety decision (**1.000** vs **0.367** for the no-communication
  agent), but the latent was 128 bytes against a 30-byte text baseline. The
  representation result is real; the bandwidth claim is not. Keep those questions
  separate.
- **Per-event economics favor the small model.** **14.5 µs** and **0.15 MB** means
  the decision path can run on every filesystem event. An LLM cannot. The bet is
  not that small models are smarter; it is that the fast loop should be cheap
  enough to run always, with the expensive model called on surprise.

### Le-Gates: one idea, three places

Le-Gates is the common thread across these experiments, and it is worth naming as
a pattern rather than a module.

The invariant is always the same:

```
keep an identity path
add a zero-initialized residual
learn a scalar gate
freeze what you want to preserve
```

In the earlier language-model work the gate sat on attention/MLP residuals: a hard
gate for behavior-preserving surgery, a sigmoid gate for smooth adaptation. Here
it sits on the world model's dynamics delta:

```
z' = z + (1−g)·old(z,a) + g·new(z,a)
```

and on the growable latent, where a new observation type gets a zero-init slot
scaled by its own input count. It transfers across model classes because the
underlying problem is identical: **distribution shift under a preservation
constraint.** The gate makes "no change" the default and "change" a measured
deviation. That is a better starting point than a learning-rate tweak, because it
gives you a place to look when the old behavior moves.

The honest limit: Le-Gates bounded old-domain drift to **+169%**, not zero. It is
a control mechanism, not a proof of immutability. The growable version got closer
(**0.0025 → 0.0026**) because it changed the observation, not just the dynamics.

### Why anyone should care

The dominant agent design puts a language model in the perception loop. That is
slow, lossy, and expensive, and it re-derives the same physics at every step.

This project is a small existence proof of a different split:

- a **fast System 1** that perceives, predicts, plans, and adapts at microsecond
  cost;
- a **kernel-verified ground truth** that turns every action into a label;
- a **safety floor below the prompt**, enforced at the tool boundary;
- a **slow System 2** that is optional and called only when surprise is high.

The claim is not that this system is general. It is that the loop can close on a
real machine with a 0.15 MB model, and that the pieces people usually study
separately — world modeling, continual learning, safety, perception, tool use —
can be made to fit together. The failures are as informative as the successes:
they mark where the next architecture has to be different.

### What would make it matter more

The honest agenda is short:

1. **Cross-world planning at ≥95%, across 20+ seeds.** The current ~65% is the
   clearest gap between "the latent learns" and "the latent is useful."
2. **A real baseline suite.** The same tasks against a frozen JEPA, a
   Dreamer-style model, and an LLM tool agent. Without that, the integration claim
   is untested against the alternatives.
3. **Amortized adapters.** A learned adapter-generator would turn "new world" from
   a training event into an inference event.
4. **Compositional worlds.** Shared dynamics primitives plus world-specific
   composition, so a new world is a recombination rather than a new slot.
5. **Autonomous escalation with a measured verification rate.** The reflex has the
   pieces; the loop is not closed, and the last live window triggered zero
   hypotheses.

Until then, the defensible claim is the narrow one: **a small latent world model
can carry perception, planning, continual adaptation, and a formal safety floor on
a real filesystem, and the integration runs end to end.** That is what the
repository is evidence for.

---

## Weights and data

No checkpoints, TorchScript exports, journals, telemetry, or screenshots are
committed. The models in `checkpoints/` were trained on the author's personal
desktop telemetry, so they are excluded rather than published. `data/` is
git-ignored and created at runtime. The code is enough to retrain the small models
from synthetic or user-provided directories; see `experiments/world_model/` and
`experiments/vision/`.

If you need a reference artifact, train one locally:

```bash
python experiments/world_model/benchmark.py          # H-JEPA on the grid world
python experiments/vision/bench_encoders.py          # frozen-encoder comparison
```

---

## Relation to prior work

Proteus is an application and stress test of ideas from JEPA and hierarchical
world models (LeCun; V-JEPA/I-JEPA; H-JEPA), gated continual learning (residual
adapters, H-SURG-style behavior-preserving surgery), and control-barrier-function
safety. It is not a new foundation model. The contribution, if any, is the
integration: a small latent world model with real perception, kernel-level
verification, a formal safety projection, and a tool-boundary bridge to a language
model, all measured on the same machine that it operates.

`docs/RESEARCH.md` maps each claim to its experiment, its sample size, and its
failure modes.

---

## Citation

```bibtex
@software{proteus2026,
  title  = {Proteus: a JEPA world-model agent that learns continually},
  author = {d4vid-le},
  year   = {2026},
  url    = {https://github.com/d4vid-le/proteus-wm},
  note   = {Research prototype}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
