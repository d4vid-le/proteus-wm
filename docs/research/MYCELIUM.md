# MYCELIUM — Mycelial provenance graph (symbiosis layer)

Status: **2026-09-06 — built and verified.** Turns the existing PAM latent
memory (`data/latent_memory.jsonl`) + the autonomous journal
(`data/autonomous_journal.jsonl`) into an **append-only provenance graph**
where every fact / relationship carries 5 deterministic, verifiable evidence
fields. This is the "symbiosis" substrate: human-substrate (authorized ground
truth) and AI structure (latent/concepts) are **grown together** and nothing is
ever deleted.

## The 5 evidence fields (per fact / edge)

| field | meaning | how it's computed |
|---|---|---|
| `exact source id` | canonical, addressable hash/UUID | `proteus:` + sha256(action)[:16] |
| `exact same full revisions` | append-only revision chain ("dormant history") | each observation = a revision; dedup by `rev_id`; never removed |
| `exact relationship labels` | controlled vocabulary | `REL_LABELS` (`preceded_by`, `similar_z`, `revised`, `derived_from`, `authorized_by`, `co_occurs`) |
| `exact deterministic edge weight` | reproducible, not stochastic | `similar_z`: cosine(rep_z_a, rep_z_b) @4dp; `preceded_by`: 1/(1+gap) @6dp |
| `exact owner authorization` | signature only the owner can forge | sha256(parts + owner_key) — the agent can't self-authorize |

`owner_key` lives in `data/owner.key` (created once, keep secret). `--verify`
recomputes every authorization and flags any mismatch (currently **0
violations** of 10,174).

## The z→source pointer

The JEPA latent (the "dormant activation") is an **index**, never trusted alone.
`mycelium.py --z-to-source <obs-or-latent>` embeds (if obs) / whitens (if
latent) and returns the k nearest sources with their evidence trail
(source_id, relationship, weight, history_len, latest_revision, owner_authorized).
So a latent activation **traverses the mycelial web back to the exact source**.

Append-only: rebuilds MERGE (dedupe by rev_id), never delete. The **current
derived relationship** is the latest revision per source; the full history stays
dormant and recoverable via `--history <id>`.

> **Lean default:** the DERIVED graph keeps only the last `--max-revisions`
> (default **8**) per source so the file stays small. The **full** source history
> is always preserved in `autonomous_journal.jsonl` + `latent_memory.jsonl`;
> only the graph's redundant copies are trimmed. Pass `--max-revisions 0` to keep
> the entire dormant history in the graph too.

## Build from real data (verified)

```
sources         1094   (distinct actions)
revisions       8058   (dormant history across PAM + journal)
edges           9080   (1093 preceded_by lineage + 7987 similar_z topology)
owner_authorized 1094  (--verify: 0 violations)
```

## Run it

```bash
cd ~/proteus
python -m proteus.agent.mycelium --build                                   # ingest PAM+journal -> graph
python -m proteus.agent.mycelium --stats                                   # graph metrics
python -m proteus.agent.mycelium --z-to-source '1,1,1,1,1,1,4,1,1,0,7,0'   # obs -> nearest sources + evidence
python -m proteus.agent.mycelium --z-to-source --latent '<64-dim>'         # latent -> sources
python -m proteus.agent.mycelium --history proteus:dd3292350af9142b        # full revision chain (dormant history)
python -m proteus.agent.mycelium --verify                                  # recompute all owner authorizations
```

Graph written to `data/mycelium.jsonl`.

## Next (natural integration)

When the agent speaks (`agent_talk.py` / `chat.py`), attach the matching mycelium
evidence (source_id, revision, relationship label, weight, authorization) so
every statement is **provenance-grounded** — the agent can't assert anything that
doesn't trace to an owner-authorized source. That closes the loop: the "mouth"
(the bridge) + this "nervous system" (provenance graph) together make symbiosis
auditable.
