# Proteus: research synthesis

This document is the argument. It states what was tested, what was measured, what
held, what did not, and which parts of the system are load-bearing.

The original lab notes are preserved in `docs/research/`. This synthesis is the
shortest path for a researcher who wants to evaluate the work without reading
fifty files.

---

## 1. Abstract

Proteus is a small, continually-learning world-model agent. It represents a
desktop filesystem as a 12-dimensional observation, compresses it to a 64-d
latent, learns two timescales of latent dynamics, plans by sampling action
sequences in that latent space, and verifies every action by re-reading the real
filesystem. It adds three mechanisms that are usually studied separately:

1. **Gated residual adaptation** for distribution shift, with the old dynamics
   path frozen.
2. **A growable latent** that adds observation slots for new file types without
   perturbing old behaviour.
3. **A control-barrier-function safety projection** with a dedicated KEEP
   detector, enforced at the tool boundary rather than in a prompt.

The system was built and measured on one Apple-silicon laptop, with real
ScreenCaptureKit/FSEvents telemetry and real file operations. The models are
tiny: the decision path is 0.15 MB and runs at 14.5 µs per file. The point is not
scale. The point is whether a fast, grounded, self-adapting latent model can carry
the parts of agency that language models are bad at.

---

## 2. Research questions

- **Q1. Do latent dynamics learn structure that supports planning?**
  If the predictor only memorizes transitions, multi-step CEM should not reach
  goals; if it learns dynamics, it should.
- **Q2. Does hierarchy buy capability, not just speed?**
  Flat k=4 MPC versus a k=16 macro phase with k=4 refinement, on a long-horizon
  file-restructuring task.
- **Q3. Can the model adapt to a changed world without catastrophic forgetting?**
  Three mechanisms: gated residual adapters, growable observation slots, and
  multi-slot lifelong routing.
- **Q4. Can irreversible-action safety be certified rather than prompted?**
  A trained barrier head plus a projection that removes the hazard-normal
  component of any adapter perturbation.
- **Q5. Can perception be latent-first?**
  A frozen image encoder projected into the same 64-d space, avoiding the
  image→text→parse pipeline.
- **Q6. Can two partial-view agents communicate through latents alone?**
  Exchange 32-d latents and make a joint delete/keep decision that neither can
  make alone.
- **Q7. Do these pieces compose on a real machine?**
  Plan → act → verify → learn, with KEEP preserved, under real telemetry.

---

## 3. System under test

| Component | Size / setting |
|---|---|
| Macro encoder / predictor | 12 → 128 → 64 MLP, residual dynamics `z' = z + f(z, a)` |
| Micro level | 1-step, 32-d, k=4 |
| Macro level | k-step, 64-d, k=16 in hierarchical experiments |
| Target encoder | EMA, τ = 0.995 |
| Anti-collapse | VICReg variance + covariance terms |
| Vision encoder | frozen ResNet18 (512-d) or ViT-B/16 (768-d), linear/MLP projection to 64-d |
| Training | Adam, synthetic or real transition tuples, single MPS device |
| Environment | `CleanupDesktop` (12-d counts, 4 actions) and a real directory |
| Telemetry | FSEvents, NSWorkspace, CGWindowList, shell hooks, ScreenCaptureKit |
| Decision path | TorchScript, 0.15 MB, 14.5 µs/file |

The observation vector is:

```
[txt_root, jpg_root, pdf_root, txt_total, jpg_total, pdf_total,
 misc_root, keep_in_root, keep_preserved, junk_present, ..., ...]
```

Actions are `move_txt`, `move_jpg`, `move_pdf`, `delete_pdf`. KEEP files are
protected by filename prefix (`KEEP`, `IMPORTANT`, `DO_NOT_DELETE`) and by the
learned barrier.

Two environments are used:

- **`CleanupDesktop`** — a controlled, synthetic-but-real-filesystem environment
  with irreversible deletes. Seeds are fixed per trial.
- **A real directory** — seeded for deterministic runs, or pointed at a user
  directory. Verification re-reads the actual filesystem after every action.

---

## 4. Protocol notes

- Most experiments are **single-machine, single-seed-family** runs. Where a result
  is seed-sensitive, the note says so.
- The synthetic environments are small. Success rates near 1.0 on a 6-file
  directory are not evidence of general capability; they are evidence that the
  mechanism works in a controlled setting.
- The eval scoreboard (`proteus/eval/run_evals.py`) is a **regression harness**,
  not a benchmark. Some checks are model-based (dynamics, counterfactual,
  barrier); others exercise grounded environment code (the greedy planner) or
  synthetic binding models. Claims below cite the dedicated study, not the
  scoreboard, unless the scoreboard is the study.
- "Random floor" means the MSE of an isotropic random latent against the target
  latent. It is a sanity floor, not a trained baseline.
- No external baseline model was run for most tasks. The comparisons are
  internal: naive fine-tuning vs gated adaptation, flat vs hierarchical planning,
  communication vs no communication.

---

## 5. Results

### 5.1 Latent dynamics learn (Q1)

The macro predictor is trained on real `(obs_t, A, obs_{t+k})` tuples from
`CleanupDesktop`. On a sample of 512 transitions:

| Metric | Value |
|---|---|
| Macro latent MSE | **0.0063** |
| Isotropic random-latent floor | **1.187** |
| Ratio | ~189× below floor |

Training curves from the original benchmark run: micro 0.0044 → 0.0008, macro
0.153 → 0.0198, position head 0.412 → 0.0407, total 0.586 → 0.078.

**What this supports:** the encoder/predictor learn non-trivial structure, and the
latent is predictable several steps ahead. **What it does not support:** that the
latent is a sufficient statistic for arbitrary planning. The cross-world planning
result below is the counterweight.

### 5.2 Hierarchical planning (Q2)

`proteus/eval/hier_bench.py`, 8 trials, 36 files plus a KEEP file:

| Planner | Success | Mean steps | Predictor rollouts |
|---|---|---|---|
| Flat k=4 CEM | **0/8** | 60 (timeout) | 178,176 |
| Hierarchical k=16 + k=4 | **8/8** | 37 | **72** |

The flat planner is not just slower; its horizon cannot see the whole-batch
structure, so it never finishes. The hierarchical planner commits to a macro
phase (which type to clear) once, then refines.

**Caveat:** 8 trials, one task family, one seed family. The 2474× rollout ratio is
a consequence of the flat planner failing and exhausting its budget, not a
like-for-like speed comparison.

### 5.3 Continual learning (Q3)

#### Gated residual adapters

A new file type (`.md`) changes the dynamics. Two adaptation strategies:

| Method | New-domain MSE | Old-domain drift |
|---|---|---|
| Naive fine-tuning (all weights) | 0.0113 → 0.0096 (−15%) | 0.0011 → 0.0123 (**+1015%**) |
| Gated residual adapter | 0.0113 → 0.0083 (−26%) | 0.0011 → 0.0030 (**+169%**) |

The gate `g = sigmoid(logit)` multiplies a residual branch:
`z' = z + (1−g)·old(z,a) + g·new(z,a)`. The old branch is frozen; the new branch
and the gate train. The learned gate settled at σ ≈ 0.179, so the model chose a
small correction rather than replacing the old dynamics.

**Honest reading:** the gated model learns the new domain better and drifts ~6×
less, but 169% old-domain drift is not zero forgetting. The stronger no-forgetting
result is in the growable model below.

An earlier attempt that froze the encoder and readout but trained the predictor
made the shift worse (0.0113 → 0.0632). The residual placement matters: the gate
must sit on the dynamics delta.

#### Growable latent

The audit found that a fixed 12-d observation cannot represent a new file type at
all, so "learning" a new type was training on noise. The fix grows the observation:

```
obs = [12 base dims] + [md_count, csv_count, ...]
encode(z)  = enc_base(base) + Σ g_i · embed_i(x_i) · x_i
predict(z) = z + pred_base + Σ g_i · mask_i · pred_i(z, A)
```

| Metric | Value |
|---|---|
| Growth perturbation on old predictions | **0.0000** |
| Old MSE across 6 learning events | **0.0025 → 0.0026** |
| New-type shift MSE | **0.0143 → 0.0057** (−60%) |
| New type representable after learning | 0.105 latent separation |

Three false starts are documented in `docs/research/GROWABLE.md`: gate-only
without observation growth (invisible signal), random-initialized slot (destroyed
the old model), and zero-initialized embedding without input scaling (the new type
stayed invisible). The final design uses zero-initialized final layers, scales the
new path by the input count, and masks the prediction delta by type presence.

#### Lifelong routing

Three sequential shifts (`.md`, `.py`, `.tar.gz`), one slot each:

| Shift | Before | After | Relative drop |
|---|---|---|---|
| md | 0.0115 | 0.0057 | 50% |
| py | 0.0092 | 0.0070 | 23% |
| tar.gz | 0.0100 | 0.0085 | 15% |

Backward drift: md unchanged after py and tar.gz; py unchanged after tar.gz.
Routing confusion matrix: 6/6 diagonal. Later shifts show smaller relative drops
because the base predictor already covers txt-like dynamics; absolute error stays
below 0.01.

**Caveat:** the shifts are synthetic type additions in a 12+2-d observation. This
is not open-world continual learning.

### 5.4 Safety (Q4)

#### Counterfactual guard

On `CleanupDesktop`, the model rejects action sequences that delete a KEEP file.
The scoreboard reports 1.0 rejection and 1.0 acceptance on 20–30 seeded trials.
This number depends on the checkpoint and the environment; the earlier audit found
that an apparent 100% rejection in an early experiment came from a hard-coded
`if 2 in action_seq and paper.pdf exists: reject` heuristic, not from the model.
The corrected model-based test is in `run_evals.py`.

#### Barrier certificate

A dedicated barrier head `B(z)` is trained with a hinge margin to output +1 for
KEEP-present and −1 for KEEP-absent. For any adapter perturbation `Δz`, the
projection removes the hazard-normal component:

```
Δz_safe = Δz − (Δz · n̂) n̂,   n = ∇B(z)
```

Enumerating all gate combinations (σ ∈ {0, 0.5, 1}² = 9) over 37 real KEEP-present
states and perturbation scales 0.05–0.4:

| Perturbation scale | Projected violations | Raw violations | Barrier margin |
|---|---|---|---|
| 0.05 | **0** | 0 | 2.70 |
| 0.10 | **0** | 64 | 3.22 |
| 0.20 | **0** | 76 | 2.88 |
| 0.40 | **0** | 50 | 3.48 |

**What this supports:** for the tested states and perturbation bounds, no gate
combination crosses the KEEP boundary after projection. **What it does not
support:** a proof for arbitrary states. The guarantee is only as strong as the
barrier margin on the state distribution; a distribution shift outside the
training support can invalidate it. The first attempt used the general position
head as the barrier; its margin was ~0.01, and no projection could certify it.
The dedicated head with margin ~3.0 is the reason the certificate works.

#### Chaos suite

Four adversarial tests (`proteus/eval/chaos.py`): a KEEP file with a normal-looking
name, a 1000-file burst with 1% KEEP, a source deleted mid-plan, and an
out-of-distribution file that triggers high latent uncertainty. All four pass with
zero KEEP files lost. The uncertainty case quarantines the file instead of
guessing (uncertainty 3.9 vs threshold 0.15).

### 5.5 Perception (Q5)

Frozen-encoder benchmark on MPS, 8 encoders:

| Encoder | Latency | Alignment cosine |
|---|---|---|
| **ResNet18** | **3.1 ms** | **0.997** |
| ResNet34 | 4.1 ms | 0.996 |
| MobileNetV3-Small | 4.4 ms | 0.984 |
| ViT-B/32 | 5.3 ms | 0.968 |
| ViT-B/16 | 10.6 ms | 0.985 |

The image is projected into the same 64-d latent used by the macro world model.
KEEP detection from the vision latent is 20/20 in the recorded run. A small count
head recovers exact counts: 131/150 (87%) exact, mean absolute error ~0.2,
generalizing to unseen seeds.

**Honest notes:** the projection aligns directions, not magnitudes. Before the
count head, the vision latent could not regress counts accurately (predicted
10.65, 15.91, 0.39 for true 2, 2, 2). The 87% figure is on synthetic rendered
scenes, not arbitrary screenshots. Cross-world transfer is weak zero-shot and
useful only few-shot:

| Cross-world probe | Accuracy |
|---|---|
| 4 examples | 0.68 |
| 10 examples | 0.72 |
| 30 examples | 0.75 |
| zero-shot separation | 0.23 |

The frozen backbone carries general structure; the file-specific projection does
not transfer. That is expected, but it bounds the "perception generalizes" claim.

### 5.6 Latent communication (Q6)

Two agents see complementary views of the same world. Agent A sees positions and
counts; Agent B sees safety semantics. They exchange 32-d latents; a coordinator
fuses them and chooses delete vs move.

| Configuration | Safe decision | KEEP lost |
|---|---|---|
| Latent exchange | **1.000** | **0.000** |
| Agent A alone (greedy) | 0.367 | **0.633** |
| Ground-truth text handoff (oracle) | 1.000 | 0.000 |

60 cooperative trials. A separate spatial-binding test reports 1.000 accuracy for
reconstructing B's `keep_exists` from A's latent (MSE 0.0009), and 0.318 KEEP hits
for A alone.

**Caveats:** the compression ratio is not a win. The 128-byte latent is larger
than a 30-byte text representation of the same state. The honest claim is
format-free, bandwidth-competitive communication that preserves the safety
decision, not "fewer tokens". Binding accuracy in the earlier 60-trial version was
0.711, not 1.0; the 1.0 result comes from a spatial variant with a better-posed
view split. The real-filesystem handoff preserves KEEP but stalls on junk
deletion (it moves the junk file instead of deleting it), which is safe but
incomplete.

### 5.7 Integration, latency, and the reflex (Q7)

- The real-environment loop plans, executes with `shutil.move`/`unlink`, and
  verifies by re-reading the directory. In the recorded seeded run, six actions
  all verified true, KEEP was preserved, and three verified tuples were replayed
  for an online update (macro loss 0.1970).
- TorchScript export of the decision path: 14.5 µs/file, 0.15 MB, bit-identical
  to eager (diff 0.0).
- The reflex watches surprise, probes system state and crash logs, and marks a
  hypothesis `verified` only when the probe corroborates it. The design is
  implemented; the live evidence is thin. The last recorded daemon log ends with
  `[verification rate] 0/0 (0%)`: it triggered no hypotheses in that window, so
  the verification-rate metric is not yet demonstrated at scale.
- The MCP bridge exposes nine System-1 tools over stdio. The safety barrier is
  returned as an MCP error (`isError: true`) so a language model cannot
  misread a block as a suggestion. This is a design argument, not an A/B result.

---

## 6. Negative results and corrections

These are the parts that make the rest credible. Each was found by an explicit
audit, not by a reviewer.

| Correction | What was wrong | What changed |
|---|---|---|
| **`build_gated` destroyed trained weights** | The gated adapter copied a random-initialized module into the "old" path; prediction changed by 1.17 after installation | Copy the trained weights into the frozen old path; verified diff 0.0 |
| **"Learned new file types" was hollow** | The 12-d observation could not encode `.md`/`.csv`; latent difference was 0.0, so training used noise | Grow the observation; add a zero-perturbation gated slot; verify representability (0.105) |
| **Counterfactual "100%" was hard-coded** | An early experiment used an `if` heuristic, not the model | Rebuilt the test around the learned position head; documented in `HONEST_AUDIT.md` |
| **Cross-world planning was 0–4/20** | Training data came from random wandering, so dynamics never learned goal direction | Goal-biased curriculum (70% greedy, 30% random) raised it to 12–14/20 (~65%), still seed-dependent |
| **Vision latent could not count** | The projection aligned directions, not magnitudes | Added a count head: 87% exact, ~0.2 MAE |
| **Cross-world zero-shot failed** | Frozen features alone did not separate a new world's classes (0.23) | Few-shot probe reaches 0.68–0.75; the file-specific projection remains non-transferable |
| **Real multi-agent junk deletion stalled** | The fusion head did not fire on the real state | Documented as safe-but-incomplete; needs real-trajectory training |
| **Reflex verification rate unproven** | Last live window had no hypotheses (0/0) | Report the metric; do not claim a rate |
| **Scoreboard planner is greedy** | The label says "MPC" but the check uses grounded counts | Treat the scoreboard as a regression harness; cite `hier_bench.py` for learned planning |

A related honesty artifact is `proteus/eval/not_agi.py`. It demonstrates five
capabilities the system lacks: world-kind transfer, rule re-pointing, self-reasoning,
goal-setting, and raw-image perception in the base model. It reports 5/5 failures.
The vision stack adds image perception, but that is a separate encoder, not the
12-d base model; the falsification is about the core agent, not the add-on.

---

## 7. Threats to validity

1. **Single machine, single author, single seed family.** No external replication.
2. **Small controlled worlds.** `CleanupDesktop` is a 6-file world. High success
   rates there do not imply open-world competence.
3. **Internal baselines only.** Naive fine-tuning, flat planning, and
   no-communication are the comparisons. No V-JEPA, Dreamer, or LLM-agent baseline
   was run.
4. **Synthetic perception for the headline vision numbers.** Rendered scenes and
   a Finder window, not a broad screenshot distribution.
5. **The "random floor" is weak.** An isotropic random latent is not a trained
   predictor; a real baseline would be stronger.
6. **Seed sensitivity in planning.** The ~65% cross-world result is an average
   over three seeds with a wide range (12, 14, 14 after the fix; 4, 0, 12 before).
7. **The barrier certificate is empirical.** It enumerates gates and states, not
   all possible latents.
8. **The reflex and dual-brain loop are integration prototypes**, not measured
   user-facing systems.
9. **Weights and telemetry are not published**, so exact reproduction of the
   recorded numbers requires retraining on your own data. The scripts are
   deterministic given seeds and a checkpoint.

---

## 8. Open problems, in priority order

1. **Multi-step rollout consistency.** The dynamics predictor drifts over long
   rollouts, so CEM terminal scoring uses drifted latents. Goal-directed data
   helps; the proposed fix is to train predicted latents to stay on the encoded
   manifold. This is the main research gap.
2. **Autonomous System-2 escalation.** Wire surprise → language-model
   investigation → verified correction → schema update, without a human in the
   loop. The pieces exist; the loop is not closed.
3. **Amortized adapter acquisition.** Per-world adapters do not scale. A learned
   adapter generator would turn "new world" from a training event into an
   inference event (235 → ≤50 transitions).
4. **Compositional worlds.** Factor dynamics into shared primitives plus
   world-specific composition, so a new world is a recombination rather than a
   new slot.
5. **A real baseline suite.** Run the same tasks against a frozen JEPA, a
   Dreamer-style model, and an LLM tool agent. Without this, the integration
   claim is untested against the alternatives.

---

## 9. Reproduction map

Run from the repository root. Commands assume `pip install -e ".[dev]"`.

| Result | Script | Command |
|---|---|---|
| Core mechanisms | `tests/test_core.py` | `pytest tests/test_core.py -q` |
| Full scoreboard | `proteus/eval/run_evals.py` | `python -m proteus.eval.run_evals --quick` |
| Latent dynamics | `run_evals.py` eval 1 | `python -m proteus.eval.run_evals --quick` |
| Hierarchical planning | `proteus/eval/hier_bench.py` | `python -m proteus.eval.hier_bench` |
| Gated adaptation | `experiments/continual/exp_continual.py` | `python experiments/continual/exp_continual.py` |
| Growable latent | `proteus/world/growable.py` | `python -m proteus.world.growable` |
| Lifelong routing | `proteus/world/lifelong.py` | `python -m proteus.world.lifelong` |
| Barrier certificate | `proteus/safety/barrier.py` | `python -m proteus.safety.barrier` |
| Chaos suite | `proteus/eval/chaos.py` | `python -m proteus.eval.chaos` |
| Vision encoders | `experiments/vision/bench_encoders.py` | `python experiments/vision/bench_encoders.py` |
| Count head | `proteus/perception/count_head.py` | `python -m proteus.perception.count_head` |
| Cross-world transfer | `experiments/evals/cross_world.py` | `python experiments/evals/cross_world.py` |
| Multi-agent | `proteus/agent/multiagent.py` | `python -m proteus.agent.multiagent` |
| Real environment | `experiments/agents/run_real.py` | `python experiments/agents/run_real.py --seed` |
| Not-AGI falsification | `proteus/eval/not_agi.py` | `python -m proteus.eval.not_agi --quick` |

Some scripts expect a trained checkpoint that is not published. The scoreboard
will skip or warn for missing checkpoints; retrain with the training scripts in
`experiments/world_model/`.

---

## 10. What would change the conclusion

The work would be substantially stronger with any of:

- Cross-world planning at ≥95% across 20+ seeds, with the same dynamics model.
- A trained baseline (V-JEPA or Dreamer) beaten on the same file task.
- Barrier certification on states outside the training support, or a proof for
  the reachable set.
- Reflex escalation measured over days with a nonzero verification rate and a
  correction that prevents a real failure.
- A real desktop task where the latent model outperforms an LLM tool agent on
  latency and error rate at equal success.

Absent those, the defensible claim is narrower: **a small latent world model can
carry perception, planning, continual adaptation, and a formal safety floor on a
real filesystem, and the integration runs end-to-end.** That is the claim this
repository supports.

---

## 11. Relation to prior work

- **JEPA / H-JEPA** (LeCun; V-JEPA, I-JEPA): predict in latent space, use an EMA
  target and anti-collapse regularizers, and plan over multiple timescales. Proteus
  is an application of these ideas to a filesystem world, not a new architecture.
- **Gated continual learning** (residual adapters, H-SURG-style behavior-preserving
  surgery): freeze the old path, add a zero-initialized residual, learn a gate.
  The growable-latent extension is the part that goes beyond a single gate.
- **Control barrier functions**: a learned barrier plus a projection that removes
  the hazard-normal component. The novelty here is enforcing the projection at an
  MCP tool boundary, so the language model cannot bypass it.
- **World-model agents** (Dreamer, MPC/CEM): the planner is standard CEM over
  sampled action sequences. The difference is the scale and the verification loop.

The repository does not claim a new learning rule, a new architecture, or a new
benchmark. It claims an integration and a set of measurements, including the
measurements that failed.

---

## 12. Availability

Code: MIT. Weights, telemetry, journals, and screenshots: not published, because
they were collected from the author's personal desktop and used for training.
`data/` and `checkpoints/` are git-ignored and recreated at runtime. The
experiments that do not need a pretrained checkpoint run from scratch; the ones
that do will say so.
