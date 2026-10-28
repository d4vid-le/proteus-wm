# Proteus architecture

## System overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  SYSTEM 2 (optional)                                                 │
│  Language / reasoning / terminal / tool calling                      │
│  Qwen or any MCP-capable harness                                     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ MCP over stdio
                               │ 11 tools: jepa_*
┌──────────────────────────────▼───────────────────────────────────────┐
│  SYSTEM 1 — PROTEUS                                                  │
│                                                                      │
│  perception/               world/                  safety/           │
│  ┌────────────┐            ┌──────────────┐        ┌───────────────┐ │
│  │ SCK frames │───────────▶│ frozen vision│───────▶│ KEEP detector │ │
│  │ FSEvents   │            │ encoder → z  │        │ CBF projection│ │
│  │ NSWorkspace│            │ H-JEPA       │        └───────┬───────┘ │
│  │ shell      │            │ micro/macro  │                │         │
│  └─────┬──────┘            └──────┬───────┘                │         │
│        │                          │                        │         │
│        ▼                          ▼                        ▼         │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  unified latent state  S_t = z_fs + z_ui + z_act + z_vis      │    │
│  └───────────────────────────┬──────────────────────────────────┘    │
│                              │                                       │
│        ┌─────────────────────┼─────────────────────┐                 │
│        ▼                     ▼                     ▼                 │
│  ┌────────────┐       ┌────────────┐        ┌────────────┐           │
│  │  surprise  │──────▶│ continual  │        │ MPC / CEM  │           │
│  │   reflex   │       │  learning  │        │  planning  │           │
│  └────────────┘       └────────────┘        └────────────┘           │
└──────────────────────────────────────────────────────────────────────┘
```

Proteus is System 1: fast, latent, grounded. The language model is System 2:
slow, symbolic, optional. The boundary is an MCP server. The safety barrier is
enforced on the System-1 side and returned to System 2 as an error, so a tool call
cannot reinterpret a block as advice.

## Package layout

| Module | Role | Key interface |
|---|---|---|
| `proteus/world/hjepa_torch.py` | Two-level H-JEPA world model | `HJEPA_K4.enc_macro`, `predict_macro`, `decode_pos` |
| `proteus/world/le_gates.py` | Le-Gates (gated residual adapters) | `build_gated`, `GatedDynamics`, `predict_macro_gated` |
| `proteus/world/growable.py` | New observation slots + gated branches | `GrowableJEPA.add_type` |
| `proteus/world/lifelong.py` | Multi-shift slot registry | `LifelongJEPA.add_slot`, `route` |
| `proteus/world/world_latent.py` | Canonical latent space | `WorldLatent` |
| `proteus/world/latent_memory.py` | Predictive associative memory | `LatentMemory` |
| `proteus/world/world_registry.py` | Supervised world routing | `WorldGate.route/plan` |
| `proteus/world/transition_pairs.py` | Empirical inverse-dynamics dataset | `build_pairs` |
| `proteus/envs/real_env.py` | Real filesystem as the world | `RealEnv.obs/step/verify` |
| `proteus/envs/cleanup_env.py` | Irreversible-action test world | `CleanupDesktop` |
| `proteus/perception/telemetry.py` | macOS sensory streams | `watch_fsevents`, `SCKStream`, `safe_read_image` |
| `proteus/perception/vision_jepa.py` | Frozen encoder → 64-d latent | `FrozenVisionJEPA`, `build_frozen_vit` |
| `proteus/perception/count_head.py` | Exact counts from pixels | `CountVision` |
| `proteus/perception/finder_vision.py` | Finder-window capture and OOD detection | `capture_finder` |
| `proteus/agent/autonomous.py` | Self-driving file agent | `AutonomousAgent.pass_once` |
| `proteus/agent/daemon.py` | Long-lived telemetry agent | `ProteusDaemon.run` |
| `proteus/agent/reflex.py` | Surprise → verify → correct | `SurpriseReflex.reason/reprobe/correct` |
| `proteus/agent/curiosity.py` | Safe novelty probing | `probe_loop` |
| `proteus/agent/multiagent.py` | Latent handoff experiment | `run_trials` |
| `proteus/agent/mycelium.py` | Provenance graph | `Mycelium` |
| `proteus/safety/barrier.py` | CBF certificate | `BarrierHead`, `project_safe` |
| `proteus/safety/cross_reason.py` | Per-world planning | `PerWorldJEPA`, `plan_cem` |
| `proteus/bridge/mcp_server.py` | MCP server (dual-brain bridge) | 11 `jepa_*` tools |
| `proteus/bridge/llm_bridge.py` | Latent → soft-prefix projector | `LatentBridge` |
| `proteus/bridge/explain.py` | Grounded explanation | `explain_state` |
| `proteus/bridge/pi_harness.py` | System-2 harness | `PiHarness` |
| `proteus/eval/run_evals.py` | Consolidated scoreboard | `main` |
| `proteus/eval/hier_bench.py` | Hierarchical vs flat planning | `HierJEPA` |
| `proteus/eval/chaos.py` | Adversarial suite | `run_all` |
| `proteus/eval/monitor.py` | Live dashboard | `snapshot()`, `render()` |
| `proteus/eval/not_agi.py` | Falsification suite | `main` |

## Model specification

```
encoder E:   O → z            obs_dim → hidden → z_macro=64
target E*:   O → z*           EMA of E, τ = 0.995
predictor f: (z, a) → Δz      residual: z_{t+1} = z + f(z, a)
micro level:  1-step, z_micro=32, k=4
macro level:  k-step, z_macro=64, k=4 or k=16
pos_head:    z → physical counts (goal-relevant readout)
VICReg:      variance + covariance (anti-collapse)

loss = ||z_pred − z_target*||² + pos_loss + VICReg
```

## Continual-learning specification (Le-Gates)

```
gate = sigmoid(gate_logit)                 # trainable
z' = z + (1−g)·old(z, a) + g·new(z, a)     # residual adapter
old frozen after domain A                  # behaviour protected
```

For a genuinely new observation type, the observation grows:

```
obs = [base dims] + [new_type_count, ...]
encode(z)  = enc_base(base) + Σ g_i · embed_i(x_i) · x_i
predict(z) = z + pred_base + Σ g_i · mask_i · pred_i(z, A)
```

The new slot is zero-initialized and scaled by the input count, so growth is
exactly invariant on old states. The base is frozen after growth. Lifelong
routing adds a slot per shift and freezes all previous slots.

## Safety specification

```
B(z) = keep_detector(z)          # dedicated head, hinge margin ~3.0
n = ∇B(z)                        # hazard normal
Δz_safe = Δz − (Δz · n̂) n̂        # remove the keep-reducing component
certificate: B(z + Δz_safe) ≥ 0  # enumerated over gates and perturbation scales
```

The certificate is empirical: it covers the tested state distribution and
perturbation bound, not all possible latents. A general position head with margin
~0.01 cannot certify; the dedicated high-margin head is required.

## MCP tool contract

| Tool | Returns | System-1 role |
|---|---|---|
| `jepa_vision_energy` | spatial latent statistics | vision signal |
| `jepa_world_state` | app, windows, FSEvents | machine context |
| `jepa_surprise` | recent prediction errors | anomaly signal |
| `jepa_safety_check` | `BLOCKED` (`isError: true`) or `SAFE` | safety guard |
| `jepa_organize` | verified file moves | file structuring |
| `jepa_plan` | latent MPC plan | planning |
| `jepa_learn` | online update acknowledgement | continual learning |
| `jepa_curiosity` | novelty / exploration drive | intrinsic motivation |
| `jepa_daemon_status` | reflex / daemon state | ops telemetry |
| `jepa_latent_handoff` | latent exchanged between agents | multi-agent |
| `jepa_intent` | decomposed intent | planning bridge |

## One full cycle

1. FSEvents fires: a file appeared in the watched directory.
2. `proteus.agent.daemon` drains the event and calls `pass_once()`.
3. `RealEnv.obs()` reads the real filesystem into the 12-d vector.
4. `HJEPA_K4` encodes `z` and predicts the next latent.
5. The planner chooses an action (grounded greedy, or CEM in latent space).
6. The KEEP barrier checks the action; a violation returns BLOCKED.
7. `RealEnv.step()` performs the real file operation.
8. `verify()` re-reads the filesystem; the action is verified or failed.
9. Verified `(z, a, z', pos)` tuples enter the continual-learning buffer.
10. Surprise is computed; the reflex stays silent or escalates.
11. System 2 can call any `jepa_*` tool to reason over the state.

## Verification state

| Suite | Result on 2026-10-28 |
|---|---|
| `pytest tests/test_core.py -q` | 15/15 PASS |
| `python -m proteus.eval.run_evals --quick` | 19/19 PASS |
| `python -m proteus.eval.not_agi --quick` | 5/5 AGI checks failed (as intended) |
| Reflex daemon | not running in this archive |
| Safety barrier | BLOCKED on KEEP (returned as `isError: true`) |

## Gaps

1. **Cross-world planning ~65%, seed-dependent.** Predictor drift over rollouts
   limits CEM. Proposed fix: multi-step rollout consistency.
2. **Autonomous escalation is not wired.** The reflex can investigate but does not
   call System 2 and act on the answer without a human.
3. **Amortized acquisition.** Per-world adapters do not scale; a learned
   adapter-generator would turn a new world into an inference event.
4. **Compositional worlds.** Factor dynamics into shared primitives plus
   world-specific composition.

See [`RESEARCH.md`](RESEARCH.md) for the evidence and `docs/research/` for the
original lab notes.
