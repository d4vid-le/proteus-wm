#!/usr/bin/env python3
"""reflex.py — The surprise reflex (intrinsic motivation).

The final loop: surprise from the JEPA telemetry predictor is the Intrinsic
Cost / Curiosity trigger.

    LOW surprise (< threshold):  Mode-1 (fast reflex) handles it. Do nothing.
                                 Save compute.
    HIGH surprise (> threshold): An anomaly. Wake Mode-2 (the slow planner,
                                 Qwen) to reason about WHY the transition was
                                 surprising, form a hypothesis, and log it back
                                 to ERIS memory as a verified schema.

Modes:
  Mode-1: the JEPA world-model predictor (jepa_telemetry) — fast, continuous
  Mode-2: a reasoning pass over the surprising trace window (via qwen_client)
          that hypothesizes the cause + proposes a corrective action.

The watcher tails eris_trace.jsonl, maintains a rolling surprise baseline
(adaptive threshold = mean + 3*std), and fires Mode-2 on spikes.
"""
from proteus.paths import ROOT
import argparse
import json
import pathlib
import sys
import time

import numpy as np

TRACE = ROOT / "data" / "eris_trace.jsonl"
MEMORY = ROOT / "data" / "eris_memory.jsonl"


class SurpriseReflex:
    def __init__(self, threshold=None, std_mult=3.0, window=50, reprobe_delay=120):
        self.std_mult = std_mult
        self.window = window
        self.baseline = []  # rolling surprise history
        self.pos = 0
        self.n_verified = 0
        self.n_unverified = 0
        self.pending = {}  # key -> {rec, next_probe_t} for deferred verification
        self.reprobe_delay = reprobe_delay  # seconds before re-probing an abduction
        # guard: a fixed threshold below the ambient floor would wake Mode-2 on
        # every event (noise). Enforce a sane minimum if a fixed threshold is set.
        if threshold is not None and threshold < 0.0005:
            print(f"[warn] fixed threshold {threshold} is below ambient floor (~0.001); "
                  f"raising to 0.001 to avoid noise-triggering")
            threshold = 0.001
        self.threshold = threshold  # fixed override if set

    def _adaptive_threshold(self):
        if self.threshold is not None:
            return self.threshold
        if len(self.baseline) < 5:
            return 0.001  # default before enough data
        mu = float(np.mean(self.baseline))
        sd = float(np.std(self.baseline))
        return mu + self.std_mult * sd

    def ingest(self, rec):
        """Process one surprise record. Returns (triggered, threshold, surprise)."""
        surf = rec.get("surprise", 0)
        try:
            surf = float(surf)
        except Exception:
            surf = 0.0
        # compute threshold from PRIOR baseline only — otherwise the current
        # spike inflates mean/std and suppresses itself (0.13 < 0.28 bug).
        th = self._adaptive_threshold()
        triggered = surf > th
        self.baseline.append(surf)
        if len(self.baseline) > self.window:
            self.baseline = self.baseline[-self.window:]
        return triggered, th, surf

    def _last_obs_from_journal(self):
        """Best-effort: most recent file-world obs from the journal (the
        autonomous agent now logs `obs` on every verified_action). Returns a
        12-dim obs vector or None."""
        import json
        if not pathlib.Path(ROOT / "data" / "autonomous_journal.jsonl").exists():
            return None
        last = None
        for line in pathlib.Path(ROOT / "data" / "autonomous_journal.jsonl") \
                .read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") == "verified_action" and e.get("obs"):
                last = e["obs"]
        return last

    def _jepa_hypothesis(self, surf):
        """JEPA-grounded Mode-2 hypothesis: project the current file-world
        state through the JEPA->Qwen bridge + PAM memory, and let Qwen explain
        the surprise from the actual world state (not free-text speculation).
        Falls back to qwen_client on any error."""
        try:
            obs = self._last_obs_from_journal()
            if obs is None:
                return None
            import proteus.bridge.explain as jepa_explain
            q = (f"The JEPA world model recorded a SURPRISE spike of {surf:.4f} "
                 "in this file-world state. Explain what state transition or "
                 "anomaly may have caused it.")
            return jepa_explain.explain(obs, question=q, why=None, max_new=80)[:400]
        except Exception as e:
            return None

    def reason(self, rec):
        """Mode-2: explain the surprise spike via Qwen, VERIFY it against cheap
        ground-truth probes, and only then log as 'verified'. Returns dict."""
        import subprocess
        surf = rec.get("surprise", 0)
        # 1. hypothesize (abduction) — prefer the JEPA-grounded bridge
        try:
            hyp = self._jepa_hypothesis(surf)
            if hyp is None:
                import proteus.bridge.qwen_client as qwen_client
                prompt = (
                    "You are the Mode-2 planner of an autonomous macOS agent. "
                    f"The JEPA world model just recorded a SURPRISE spike of {surf:.4f} "
                    "(prediction error on the app/window/file telemetry stream). "
                    "Hypothesize: what kind of OS event or anomaly could cause this "
                    "surprising state transition? Be brief, one hypothesis."
                )
                hyp = qwen_client.qwen_generate(prompt)[:300]
        except Exception as e:
            hyp = f"(qwen unavailable: {str(e)[:50]})"
        # strip terminal control codes (qwen sometimes echoes \x1b[5D\x1b[K
        # which was polluting eris_memory.jsonl)
        try:
            import re
            hyp = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", hyp or "")
            hyp = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", hyp)[:300]
        except Exception:
            pass

        # 2. VERIFY with cheap ground-truth probes (not just abduction)
        evidence = {}
        try:
            r = subprocess.run(
                ["log", "show", "--last", "3m", "--predicate",
                 "eventMessage CONTAINS 'crash' OR eventMessage CONTAINS 'fault'",
                 "--style", "compact"], capture_output=True, text=True, timeout=10)
            crash_lines = [l for l in (r.stdout or "").splitlines() if l.strip()]
            # drop log-show chrome (headers like 'Filtering the log data...')
            crash_lines = [l for l in crash_lines
                           if "Filtering" not in l and "Log:" not in l]
            evidence["crash_or_fault"] = len(crash_lines)
            evidence["crash_sample"] = crash_lines[:2]
        except Exception as e:
            evidence["crash_probe_err"] = str(e)[:50]
        try:
            r = subprocess.run(["ps", "-axo", "pid,comm"], capture_output=True,
                               text=True, timeout=10)
            lines = [l for l in (r.stdout or "").splitlines() if l.strip()]
            # drop ps header line (PID COMMAND)
            if lines and lines[0].strip().lower().startswith("pid"):
                lines = lines[1:]
            proc_count = len(lines)
            evidence["process_count"] = proc_count
        except Exception as e:
            evidence["ps_err"] = str(e)[:50]

        # 3. verdict: verified only if probe corroborates in a sane range.
        #    log-show floods (26k lines in 3m) mean the predicate matched
        #    everything — treat as unreliable, not verified.
        n_crash = evidence.get("crash_or_fault", 0)
        verified = 0 < n_crash < 100
        if n_crash >= 100:
            evidence["probe_unreliable"] = f"crash count {n_crash} absurd — predicate flooded"
        status = "verified" if verified else "unverified-abduction"

        rec_out = {
            "t": time.time(),
            "surprise": surf,
            "triggered_by": rec.get("action"),
            "hypothesis": hyp,
            "evidence": {k: v for k, v in evidence.items() if k != "crash_sample"},
            "status": status,
        }
        with open(MEMORY, "a") as f:
            f.write(json.dumps(rec_out) + "\n")
        return rec_out

    def correct(self, rec, dry_run=True):
        """Mode-1 corrective action on a surprise spike. Investigates the real
        OS state and takes safe corrective steps. dry_run=True logs only (safe).
        Returns a list of (action, result) steps."""
        import subprocess
        steps = []
        surf = rec.get("surprise", 0)

        # 1. snapshot the actual machine state (the ground truth behind the surprise)
        probe = {}
        try:
            import proteus.perception.telemetry as telemetry
            # sample_system_state already degrades gracefully when
            # pyobjc (AppKit/Quartz) is missing — use it instead of
            # calling active_app()/window_bounds() directly.
            probe = telemetry.sample_system_state()
            if isinstance(probe.get("app"), dict) and "error" in probe["app"]:
                probe["app_unavailable"] = "pyobjc missing: pip install pyobjc-framework-Cocoa pyobjc-framework-Quartz"
        except Exception as e:
            probe = {"error": str(e)[:60], "app_unavailable": True}
        steps.append(("probe-system-state", probe))

        # 2. check for anomalies the OS reports (crash reports, high load)
        if surf > 0.01:  # big spike — investigate deeper
            try:
                r = subprocess.run(
                    ["log", "show", "--last", "2m", "--predicate", "eventMessage CONTAINS 'error'",
                     "--style", "compact"],
                    capture_output=True, text=True, timeout=10)
                err_lines = [l for l in (r.stdout or "").splitlines() if l.strip()][:5]
                steps.append(("check-system-log-errors", err_lines))
            except Exception as e:
                steps.append(("check-system-log-errors", str(e)[:60]))
            try:
                r = subprocess.run(["top", "-l", "1", "-n", "5", "-stats", "command,cpu"],
                                   capture_output=True, text=True, timeout=10)
                top = [l.strip() for l in (r.stdout or "").splitlines() if "%" in l][:5]
                steps.append(("check-top-processes", top))
            except Exception as e:
                steps.append(("check-top-processes", str(e)[:60]))

        # 3. take the corrective action (safe, non-destructive)
        if not dry_run:
            # e.g., if a known temp junk dir grew, clean it; if a process is
            # pegging CPU, note it. All reads/notes — nothing destructive.
            try:
                import proteus.perception.telemetry as telemetry
                telemetry.log("reflex_corrective", surprise=surf,
                              action="system-investigated", probe=probe)
                steps.append(("log-correction", "logged to telemetry"))
            except Exception as e:
                steps.append(("log-correction", str(e)[:60]))
        else:
            steps.append(("dry-run", "no mutation (dry_run=True)"))
        return steps

    def reprobe(self):
        """Deferred verification: re-probe unverified abductions after the delay.
        Crash logs often land late, so an abduction that failed its first probe
        may be corroborated minutes later. Promote to verified, or decay after
        a max age. Returns list of (key, action) outcomes."""
        import subprocess
        now = time.time()
        outcomes = []
        # decay: drop abductions older than 10x the reprobe delay (never confirmed)
        decay_age = self.reprobe_delay * 10
        for key in list(self.pending.keys()):
            entry = self.pending[key]
            if now - entry["t"] > decay_age:
                outcomes.append((key, "decayed"))
                del self.pending[key]
                continue
            if now < entry["next_probe"]:
                continue
            # re-probe: check if a crash/fault has since landed
            try:
                r = subprocess.run(
                    ["log", "show", "--last", "5m", "--predicate",
                     "eventMessage CONTAINS 'crash' OR eventMessage CONTAINS 'fault'",
                     "--style", "compact"], capture_output=True, text=True, timeout=10)
                lines = [l for l in (r.stdout or "").splitlines() if l.strip()]
                lines = [l for l in lines
                         if "Filtering" not in l and "Log:" not in l]
                crash_n = len(lines)
                # flooded predicate (100+ lines) = unreliable, not corroboration
                if crash_n >= 100:
                    crash_n = 0
            except Exception:
                crash_n = 0
            if crash_n > 0:
                # promote: the late landing corroborated the abduction
                outcomes.append((key, "promoted"))
                self.n_unverified -= 1
                self.n_verified += 1
                del self.pending[key]
                with open(MEMORY, "a") as f:
                    f.write(json.dumps({
                        "t": now, "event": "deferred-verify", "key": key,
                        "status": "verified", "crash_or_fault": crash_n,
                    }) + "\n")
            else:
                # not yet corroborated — schedule another probe
                entry["next_probe"] = now + self.reprobe_delay
        return outcomes

    def run(self, daemon=True, correct=False, dry_run=True):
        print(f"reflex watching {TRACE} (adaptive threshold = mean + {self.std_mult}*std)")
        print(f"  corrective actions: {'ENABLED' if correct else 'disabled'}"
              f"{' (dry-run)' if correct and dry_run else ''}")
        if not pathlib.Path(TRACE).exists():
            print("no trace yet — run jepa_telemetry.py first")
            return
        # the per-cycle honesty print used to fire every second (86k lines/day
        # -> 7MB log -> diff viewer stack overflow). Throttle: print on change
        # or every 60 cycles.
        cycle = 0
        last_rate_sig = None
        while True:
            lines = pathlib.Path(TRACE).read_text().splitlines()
            for ln in lines[self.pos:]:
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                self.pos += 1
                triggered, th, surf = self.ingest(rec)
                if triggered:
                    out = self.reason(rec)
                    print(f"\n[REFLEX] SURPRISE SPIKE: {surf:.5f} > threshold {th:.5f}")
                    print(f"  Mode-2 hyp: {out['hypothesis'][:120]}")
                    print(f"  probe evidence: {out['evidence']}")
                    print(f"  -> status: {out['status']}")
                    if out['status'] == "verified":
                        self.n_verified += 1
                    else:
                        self.n_unverified += 1
                        # enqueue for deferred verification
                        key = f"{out['t']}:{surf:.4f}"
                        self.pending[key] = {"rec": out, "t": out["t"],
                                             "next_probe": out["t"] + self.reprobe_delay}
                    if correct:
                        steps = self.correct(rec, dry_run=dry_run)
                        for act, res in steps:
                            print(f"  [correct] {act}: {str(res)[:120]}")
                else:
                    print(f"[Mode-1] surprise {surf:.5f} < {th:.5f} — handle silently")
            # deferred verification + live honesty metric each cycle
            if self.pending:
                for key, act in self.reprobe():
                    print(f"[deferred-verify] {key}: {act}")
            total = self.n_verified + self.n_unverified
            rate = self.n_verified / total if total else 0.0
            cycle += 1
            rate_sig = (self.n_verified, self.n_unverified, len(self.pending))
            if rate_sig != last_rate_sig or cycle % 60 == 0:
                print(f"  [verification rate] {self.n_verified}/{total} ({rate:.0%}) "
                      f"| pending {len(self.pending)}")
                last_rate_sig = rate_sig
            if not daemon:
                break
            time.sleep(1.0)
        # final honesty metric
        total = self.n_verified + self.n_unverified
        rate = self.n_verified / total if total else 0.0
        print(f"\n[verification rate] Mode-2 hypotheses confirmed by probe: "
              f"{self.n_verified}/{total} ({rate:.0%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None,
                    help="fixed surprise threshold (default: adaptive mean+3std)")
    ap.add_argument("--std-mult", type=float, default=3.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--correct", action="store_true",
                    help="enable corrective actions on surprise spikes")
    ap.add_argument("--execute", action="store_true",
                    help="allow real mutations (default: dry-run, log-only)")
    args = ap.parse_args()

    print("=" * 64)
    print("PROTEUS SURPRISE REFLEX (intrinsic motivation)")
    print("=" * 64)
    r = SurpriseReflex(threshold=args.threshold, std_mult=args.std_mult)
    try:
        r.run(daemon=not args.once, correct=args.correct, dry_run=not args.execute)
    except KeyboardInterrupt:
        print("\nreflex stopped")


if __name__ == "__main__":
    main()