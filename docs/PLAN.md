# PROTEUS — Build Plan: Scale, More Data, Make Robust

## Status (2026-09-02): DUAL-BRAIN COMPLETE — 15/15 pytest, 19/19 eval, 3/3 dual-brain

## The full system (all documented in README.md + ARCHITECTURE.md)
- H-JEPA world model, continual learning (Le-Gates/growable/lifelong),
  MPC/CEM planning, counterfactual rejection, formal KEEP safety barrier,
  real perception (JEPA-way vision + SCK streaming), macOS telemetry
  (FSEvents/NSWorkspace/shell), the surprise reflex, and the Pi dual-brain
  stack via MCP (11 System-1 tools).

## Current gaps (the open work)
1. **Cross-world planning ~65% → 95%+** — predictor drift over multi-step
   rollouts. Fix: multi-step rollout consistency loss.
2. **Autonomous System-2 escalation** — reflex → auto-call Pi → act.
3. **Amortized acquisition** — adapter-generator (235 → ≤50 transitions).
4. **Compositional worlds** — shared primitives + recombination.

## History (phases completed)
- Phases 0-5: hardening, data, continual learning, counterfactual, multi-agent,
  robustness, chaos — 19/19 eval scoreboard
- WO1/2/3: lifelong shifts, hierarchical k=16, chaos fuzzing
- Phase 3: hierarchical, curiosity, barrier certificate, latency
- Vision: synthetic → real frames → JEPA-way frozen encoder → SCK streaming
- Dual-brain: proteus MCP server inside Pi (System 1 for System 2)

## Phase 0 — Lock the core (hardening)
- [x] Real checkpoint persistence + load path (hjepa_proteus.pt, cleanup_hjepa.pt, gated)
- [x] Seed-stable everything (torch.manual_seed + numpy rng threaded through)
- [x] Unit smoke test CI-able (test_core.py — 6 tests: HJEPA, gates, growable, realenv)

## Phase 1 — More / real data
- [x] Irreversible env (cleanup_env.py)
- [x] Scale data + verified tuples (train_cleanup.py, continual_learn.py)
- [ ] Real VLM vision path (vision is still hash stub) — DEFER (needs mlx VLM)

## Phase 2 — Continual learning REAL
- [x] Verified tuples + anchored replay + zero forgetting (continual_learn.py)
- [x] Le-Gates trainable residual gates (proteus/world/le_gates.py, GATES.md)
- [x] Growable model: new types learned without forgetting (growable.py, GROWABLE.md)

## Phase 3 — Counterfactual / model-based rejection
- [x] Reject 1.0, accept safe 1.0, selector 1.0 (train_cleanup.py)
- [x] JEPA keep_lost 0.0 vs naive 0.70

## Phase 4 — Multi-agent latent communication
- [x] Synthetic: spatial binding 1.0, 3-agent fusion 1.0, OOD transfer 1.0, dropout graceful
      (multiagent.py, PROOF.md, SCALE.md, SCALE_UP.md)
- [x] Real env: autonomous daemon + Le-Gates + honest shift detection (AUTONOMOUS_GATES.md)
- [x] Real multi-agent: 2 agents on REAL dirs, latent handoff, KEEP preserved, A sorted
      (multiagent_real.py, MULTIAGENT_REAL.md)

## Phase 5 — Robustness + eval harness
- [x] Consolidated eval entry (run_evals.py — 7/7 PASS scoreboard)
- [x] Long-run stability: seed stability + no KEEP loss (robustness PASS)

## What to test for (the scoreboard) — ALL VERIFIED
| Capability | Result | Target | Status |
|---|---|---|---|
| Latent dynamics learn | macro 0.0063 vs floor 1.18 | below floor | ✅ PASS |
| MPC planner success | 1.0 (hard env) | >90% | ✅ PASS |
| Counterfactual reject | 1.0 | >90% | ✅ PASS |
| Continual MSE <0.02 | 0.0060, forget -0.0006 | <0.02, no forget | ✅ PASS |
| Le-Gates growable | 59% learn, old flat | learn, no forget | ✅ PASS |
| Multi-agent latent | 1.0 synthetic + real dirs | 2-agent pass | ✅ PASS |
| Eval harness | run_evals.py 7/7 | single scoreboard | ✅ PASS |
| Robustness | 3/3 seeds clean, 0 keep lost | no drift | ✅ PASS |

## Remaining (deferred / next frontier)
- Real VLM vision (screencapture → VLM → latent)
- Multi-agent real: junk-deletion completion (train fusion on real trajectories)
- Scale to larger obs / more file types (growable already supports this)

## Option-2 Work Orders (2026-09-01) — ALL COMPLETE
- **WO1 Multi-shift lifelong** (`lifelong.py`) — 3 sequential shifts (md→py→tar.gz),
  multi-gate registry + softmax routing, freeze-on-shift. Result: routing diagonal
  100%, zero forgetting (md/py flat), all final MSE <0.01. PASS.
- **WO2 Hierarchical planning k=16** (`hierarchical.py`) — z_high macro level over
  k=16 trajectories + coarse-to-fine MPC. Result: both levels learn (low 0.0001,
  high 0.0004), KEEP safe at both abstraction levels. PASS (note: flat greedy is
  already optimal on the simple env; hierarchy's value needs a harder task).
- **WO3 Chaos & fuzz** (`chaos.py`) — adversarial collision, 1000-file burst, race
  conditions, quarantine fallback. Result: 4/4 PASS, 0 KEEP lost, all invariants
  preserved. PASS.

## Consolidated scoreboard (run_evals.py)
| Test | Result |
|---|---|
| 1 dynamics | PASS (macro 0.0063 vs floor 1.18) |
| 2 MPC hard env | PASS (1.00) |
| 3 counterfactual reject | PASS (1.00) |
| 4 continual learning | PASS (0.0066, forget 0.000) |
| 5 growable Le-Gates | PASS (49% learn, old flat) |
| 5b lifelong multi-shift | PASS (WO1) |
| 6 multi-agent | PASS (1.00) |
| 7 robustness | PASS (3/3 seeds, 0 lost) |
| 8 chaos & fuzz | PASS (WO3) |
| 9 hierarchical k=16 | PASS (P3-1, 152x+ speedup) |
| 10 active curiosity | PASS (P3-2) |
| 11 safety barrier cert | PASS (P3-3, 0 violations) |
| 12 latency & footprint | PASS (P3-4, 14.5us, 0.15MB) |
| **Status** | **13/13 GREEN** |

## Phase 3 (2026-09-01) — Hierarchical, Self-Exploring, High-Performance — COMPLETE
- **P3-1 H-JEPA k=16** (`hier_bench.py`) — 2474x search speedup on hard task;
  hier capable where flat CEM fails. `P31_HIERARCHICAL.md`
- **P3-2 curiosity + probing** (`curiosity.py`) — ensemble uncertainty detects
  novelty (20x), safe KEEP-bounded probes, self-improvement (unc 0.0077→0.0002).
  `P32_CURIOSITY.md`
- **P3-3 barrier certificate** (`barrier.py`) — CBF projection + dedicated
  trained barrier head (margin 3.0); 0 violations across all gate combos.
  `P33_BARRIER.md`
- **P3-4 latency/footprint** (`latency.py`) — TorchScript export, 14.5us ingest,
  0.15MB. `P34_LATENCY.md`

## Vision (2026-09-01) — Real Perception — COMPLETE
- **Synthetic vision** (`vision.py`) — CNN reads synthetic scenes, 4/4 tests, shared latent. `VISION.md`
- **Scaling study** (`scale_vision.py`, `scale_vision_hard.py`) — caught the degenerate-data bug; honest result: synthetic scenes saturate, real bottleneck is real-world images. `SCALE_VISION.md`
- **Real vision** (`real_vision.py`) — real VLM (llava:7b) + real screenshot of a real directory; reads filenames, detects KEEP. `REAL_VISION.md`
- **THE JEPA WAY** (`vision_jepa.py`) — frozen ViT-B/16 → z_vis(768) → project → z(64) → macro JEPA. **10.4ms** (vs LLaVA minutes, ~96x), alignment 0.98, KEEP-from-vision 1.00, no text in the loop. `VISION_JEPA.md`
- **Daemon integration** (`autonomous.py --vision`) — vision cross-check wired in.
- Tests: pytest 13/13, eval 16/16.

## Consolidated scoreboard (run_evals.py)
| # | Test | Result |
|---|---|---|
| 1 | dynamics | PASS |
| 2 | MPC hard env | PASS |
| 3 | counterfactual reject | PASS |
| 4 | continual learning | PASS |
| 5 | growable Le-Gates | PASS |
| 5b | lifelong multi-shift | PASS (WO1) |
| 6 | multi-agent | PASS |
| 7 | robustness | PASS |
| 8 | chaos & fuzz | PASS (WO3) |
| 9 | hierarchical k=16 | PASS (P3-1) |
| 10 | active curiosity | PASS (P3-2) |
| 11 | safety barrier cert | PASS (P3-3) |
| 12 | latency & footprint | PASS (P3-4) |
| 13 | vision (synthetic) | PASS |
| 14 | real vision (VLM capture) | PASS |
| 15 | vision-jepa (frozen ViT→latent) | PASS (10.4ms, cos 0.98) |
| 16 | vision-latent MPC routing | PASS (sorted real dirs, 0 KEEP lost, 3.7ms) |
| 17 | vision count head | PASS (mean abs err 0.2, 87% exact) |
| 18 | real-frame fine-tuning | PASS (1.34→0.42, 68% better on real frames) |
| **Status** | **19/19 GREEN** | |

## Scale-out round (2026-09-01) — COMPLETE
- **S1 Scale real-frame dataset** — honest: error plateaus ~0.4; bottleneck is frame rendering, not data. `S1_SCALE_REAL_FRAMES.md`
- **S2 Vision counts in daemon** (`--vision-count`) — routes on vision-latent counts, gated by real-FS + KEEP. `S2_DAEMON_VISION_COUNTS.md`
- **S3 Cross-world transfer** (`cross_world.py`) — frozen encoder transfers (few-shot 4→68%, 30→75%). `S3_CROSS_WORLD.md`
- **Real project test** — organized 58 PNGs in a copy of the Flux outputs dir, .glb protected, 16 experiment subfolders intact. Added png type + `--protect-ext`.
- **Vision superpowers** (`finder_vision.py`) — real Finder window: spatial latent, OOD/error detection, visual confirmation, per-world spatial-JEPA planning. 5/5 PASS. `VISION_SUPERPOWERS.md`

## Next frontier
- **World routing DONE** (`world_registry.py`) — supervised gate on frozen features:
  **100% routing accuracy** (the OS world label makes it trivially separable).
  Planning is real-but-seed-sensitive (5/5 with good seed, ~20% default) — the
  MPC goal-gradient depends on latent structure, which varies with init.
- **Cross-world planning robustness** — the open problem: make MPC reliably
  reach goals regardless of seed (value-head / more data / better dynamics loss)
- Multi-world agent: gate + per-world JEPAs (routing solved, planning hardening)

## Long-lived daemon (2026-09-01)
- `daemon.py --dir ".../outputs copy" --vision-stream --daemon` — event-driven
  (FSEvents trigger), persistent ScreenCaptureKit vision stream (z_vis), app/window/
  shell telemetry, continual learning. Full JEPA state S_t = z_fs + z_ui + z_act + z_vis.