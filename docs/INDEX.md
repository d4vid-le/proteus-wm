# Documentation index

Start with [`../README.md`](../README.md) for the overview and
[`RESEARCH.md`](RESEARCH.md) for the research synthesis. This page is the map to
the original lab notes, which are preserved as written and are the primary record
of each experiment.

## Synthesis

| Document | Contents |
|---|---|
| [`RESEARCH.md`](RESEARCH.md) | Questions, protocol, results, negative results, threats to validity, reproduction map |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Component contracts, data flow, model/safety specifications |
| [`PLAN.md`](PLAN.md) | The original roadmap and phase status |

## Foundational results

| Document | Contents |
|---|---|
| [`research/BREAKTHROUGH.md`](research/BREAKTHROUGH.md) | Spatial information survives 32-d latent compression; OOD and compositional binding |
| [`research/PROOF.md`](research/PROOF.md) | Multi-agent latent communication, 60 trials |
| [`research/SCALE.md`](research/SCALE.md) | Early scaling study |
| [`research/SCALE_UP.md`](research/SCALE_UP.md) | Scale-up study |
| [`research/FINDINGS.md`](research/FINDINGS.md) | First consolidated findings (contains superseded numbers; kept for the record) |
| [`research/HONEST_AUDIT.md`](research/HONEST_AUDIT.md) | The audit that found the gated-adapter bug and the hollow new-type claim |

## Continual learning

| Document | Contents |
|---|---|
| [`research/GATES.md`](research/GATES.md) | Le-Gates: naive vs gated forgetting |
| [`research/GROWABLE.md`](research/GROWABLE.md) | Growable latent: new types with zero growth perturbation |
| [`research/WO1_LIFELONG.md`](research/WO1_LIFELONG.md) | Three sequential shifts, routing matrix |
| [`research/AUTONOMOUS_GATES.md`](research/AUTONOMOUS_GATES.md) | Gate installation in the live agent |

## Planning and safety

| Document | Contents |
|---|---|
| [`research/P31_HIERARCHICAL.md`](research/P31_HIERARCHICAL.md) | k=16 hierarchical planning vs flat k=4 |
| [`research/WO2_HIERARCHICAL.md`](research/WO2_HIERARCHICAL.md) | Hierarchical planning benchmark |
| [`research/P33_BARRIER.md`](research/P33_BARRIER.md) | Control-barrier certificate |
| [`research/PWR_CROSS_REASON.md`](research/PWR_CROSS_REASON.md) | Cross-world reasoning |
| [`research/PWR_FIX.md`](research/PWR_FIX.md) | The seed-instability fix and its limits |
| [`research/P32_CURIOSITY.md`](research/P32_CURIOSITY.md) | Uncertainty-driven safe probing |
| [`research/WO3_CHAOS.md`](research/WO3_CHAOS.md) | Adversarial fuzz suite |

## Perception and vision

| Document | Contents |
|---|---|
| [`research/VISION.md`](research/VISION.md) | Synthetic-vision baseline |
| [`research/VISION_JEPA.md`](research/VISION_JEPA.md) | Frozen encoder → latent, no text |
| [`research/VISION_ROUTING.md`](research/VISION_ROUTING.md) | MPC on the vision latent |
| [`research/VISION_SUPERPOWERS.md`](research/VISION_SUPERPOWERS.md) | Finder capture, anomaly detection, visual confirmation |
| [`research/BENCH_ENCODERS.md`](research/BENCH_ENCODERS.md) | Eight frozen encoders compared |
| [`research/COUNT_HEAD.md`](research/COUNT_HEAD.md) | Exact counts from pixels |
| [`research/REAL_VISION.md`](research/REAL_VISION.md) | VLM on real screenshots |
| [`research/REAL_FRAMES.md`](research/REAL_FRAMES.md) | Real desktop frames |
| [`research/SCALE_VISION.md`](research/SCALE_VISION.md) | Vision scaling study |
| [`research/S1_SCALE_REAL_FRAMES.md`](research/S1_SCALE_REAL_FRAMES.md) | Real-frame dataset scaling |
| [`research/S3_CROSS_WORLD.md`](research/S3_CROSS_WORLD.md) | Few-shot cross-world transfer |

## Telemetry, reflex, daemon

| Document | Contents |
|---|---|
| [`research/DAEMON.md`](research/DAEMON.md) | Long-lived telemetry agent |
| [`research/REFLEX.md`](research/REFLEX.md) | Surprise → verify → correct |
| [`research/REFLEX_LAUNCHD.md`](research/REFLEX_LAUNCHD.md) | Running the reflex unattended |
| [`research/K1_SCK_STREAM.md`](research/K1_SCK_STREAM.md) | ScreenCaptureKit streaming |
| [`research/K1_K2_SCK_EVENTLOOP.md`](research/K1_K2_SCK_EVENTLOOP.md) | SCK plus event-driven loop |
| [`research/S2_DAEMON_VISION_COUNTS.md`](research/S2_DAEMON_VISION_COUNTS.md) | Vision counts in the daemon |
| [`research/AGENT_TALK.md`](research/AGENT_TALK.md) | Agent-initiated speech |
| [`research/MYCELIUM.md`](research/MYCELIUM.md) | Provenance graph |

## Real environment and multi-agent

| Document | Contents |
|---|---|
| [`research/REAL_END_TO_END.md`](research/REAL_END_TO_END.md) | Plan → act → verify → learn on a real directory |
| [`research/MULTIAGENT_REAL.md`](research/MULTIAGENT_REAL.md) | Two real agents, latent handoff |
| [`research/PHASE3.md`](research/PHASE3.md) | Phase 3 plan and results |
| [`research/JEPA_LLM_BRIDGE.md`](research/JEPA_LLM_BRIDGE.md) | Latent-to-LM soft-prefix bridge |

## Operations and housekeeping

| Document | Contents |
|---|---|
| [`research/HANDOVER.md`](research/HANDOVER.md) | Long-form handover from the build |
| [`research/LAB.md`](research/LAB.md) | Lab notebook |
| [`research/LAB_RESULTS.md`](research/LAB_RESULTS.md) | Early lab results |
| [`research/JANITOR.md`](research/JANITOR.md) | Tmp-directory janitor |
| [`research/P34_LATENCY.md`](research/P34_LATENCY.md) | TorchScript latency and footprint |
| [`research/AUTONOMOUS.md`](research/AUTONOMOUS.md) | Autonomous-agent design notes |
| [`research/MULTIAGENT_REAL.md`](research/MULTIAGENT_REAL.md) | Real multi-agent run |

## Code entry points

| Path | Purpose |
|---|---|
| `pytest tests/test_core.py -q` | 15 core smoke tests |
| `python -m proteus.eval.run_evals --quick` | 19-check scoreboard |
| `python -m proteus.agent.daemon --dir DIR` | Autonomous daemon |
| `python -m proteus.eval.monitor` | Live dashboard |
| `python -m proteus.bridge.mcp_server` | MCP server |
| `python -m proteus.eval.not_agi --quick` | Falsification suite |

## A note on the notes

The files in `research/` were written during the build, in chronological order.
Some contain numbers that were later corrected, and some refer to the project by
its earlier working title. Where a note conflicts with `RESEARCH.md`, the synthesis
is the current statement; the note is the historical record. The corrections are
listed in section 6 of `RESEARCH.md`.
