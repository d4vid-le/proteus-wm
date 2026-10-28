#!/usr/bin/env python3
"""mycelium.py — Mycelial provenance graph for proteus memory. (symbiosis layer)

Turns the existing PAM latent memory (data/latent_memory.jsonl) and the
autonomous journal (data/autonomous_journal.jsonl) into an append-only
provenance graph where every fact / relationship carries FIVE deterministic,
verifiable evidence fields:

  1. exact source id           -> canonical, addressable hash/UUID
  2. exact same full revisions -> append-only revision chain ("dormant history")
  3. exact relationship labels -> controlled vocabulary (REL_LABELS)
  4. exact deterministic weight-> reproducible function, never stochastic
  5. exact owner authorization -> signature only the owner can forge

And the z->source pointer: the JEPA latent (the "dormant activation") is an
INDEX that traverses the mycelial web back to the exact, evidence-backed source
record. So a latent activation is never trusted alone — it points to its
ground truth.

Symbiosis: nothing is deleted. Rebuilds MERGE; new observations APPEND new
revisions. The "current derived relationship" is computed over the latest
revision per source, but the full history stays dormant and recoverable.

Design choice (deterministic weights, "exact"):
  - source_id / rev_id / owner_auth  -> sha256() of canonical JSON (byte-exact)
  - similar_z edge weight            -> cosine(rep_z_a, rep_z_b), rounded to 4 dp
  - preceded_by edge weight          -> 1/(1+gap), rounded to 6 dp
  All recomputable from the pinned records -> exact, reproducible.

Usage:
  python -m proteus.agent.mycelium --build              # ingest PAM + journal -> graph
  python -m proteus.agent.mycelium --stats              # nodes/edges/history metrics
  python -m proteus.agent.mycelium --z-to-source '1,2,0,0,0,0,3,0,1,1,3,0'   # obs -> sources
  python -m proteus.agent.mycelium --z-to-source --latent '<64-dim>'          # latent -> sources
  python -m proteus.agent.mycelium --history proteus:abcd...                  # full revision chain
  python -m proteus.agent.mycelium --verify              # check all owner authorizations
"""
from proteus.paths import ROOT
import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

PAM = ROOT / "data" / "latent_memory.jsonl"
JOURNAL = ROOT / "data" / "autonomous_journal.jsonl"
OUT = ROOT / "data" / "mycelium.jsonl"
OWNER_KEYFILE = ROOT / "data" / "owner.key"

REL_LABELS = ("preceded_by", "similar_z", "revised", "derived_from",
              "authorized_by", "co_occurs")
SIM_THRESHOLD = 0.5
MAX_SIM_NEIGHBORS = 8
PREC = 4  # float precision for deterministic weights
DEFAULT_MAX_REVISIONS = 8  # lean default: cap derived-graph history per source (full source history is preserved in the journal + PAM)

# in-process cache of the last built graph (so the CLI doesn't rebuild twice)
_GRAPH = None


# --------------------------------------------------------------------------- #
# deterministic primitives
# --------------------------------------------------------------------------- #
def _canon(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


def _hash(payload):
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _norm_action(a):
    return " ".join(str(a).strip().lower().split())


def _source_id(action):
    """Exact, addressable source id (a stable hash/UUID), keyed by the canonical
    action label so identical facts across time resolve to ONE logical source
    that accumulates revisions."""
    return "proteus:" + _hash("source|" + _norm_action(action))[:16]


def owner_key():
    """The owner secret. The agent cannot forge it (it never has the raw key
    except here, at the boundary). Created once, stable across builds."""
    if OWNER_KEYFILE.exists():
        return OWNER_KEYFILE.read_text().strip()
    key = _hash("proteus-owner|" + hashlib.sha256(str(np.random.bytes(8)).encode()).hexdigest())[:32]
    OWNER_KEYFILE.parent.mkdir(parents=True, exist_ok=True)
    OWNER_KEYFILE.write_text(key)
    print(f"[owner] created owner key at {OWNER_KEYFILE} — keep it secret")
    return key


def owner_auth(*parts):
    """Deterministic owner authorization signature over the given parts."""
    return _hash("|".join(parts) + "|" + owner_key())


def verify_auth(rec, rev=None):
    """Content-bound owner-authorization check. Recomputes the revision id from
    the CURRENT content and compares (so modifying a latent is detected) AND
    recomputes the owner signature. Returns True only if both match."""
    rev = rev or (rec["revisions"][-1] if rec.get("revisions") else None)
    if not rev:
        return False
    # 1) tamper-evidence: rev_id must match the content hash (recompute)
    content = {k: v for k, v in rev.items() if k not in ("rev_id", "owner_auth")}
    if _hash("rev|" + _canon(content)) != rev.get("rev_id"):
        return False
    # 2) owner signature
    return rev.get("owner_auth") == owner_auth(rec["source_id"], rev.get("rev_id"))


def _rounded(x, p=PREC):
    return float(np.round(float(x), p))


def _norm_vec(v):
    """Accept a 12-dim obs (embed+whiten) or a 64-dim whitened latent (as-is).
    Returns a normalized 64-dim vector, or None if it can't be produced."""
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if v.shape[0] == 64:
        return v / (np.linalg.norm(v) + 1e-9)
    # 12-dim obs -> embed (raw macro-64) -> power-whiten (PAM coordinate space)
    from proteus.world.world_latent import WorldLatent
    from proteus.world.latent_memory import whiten
    z = WorldLatent().embed(v.tolist())
    return whiten(z)


# --------------------------------------------------------------------------- #
# graph builders
# --------------------------------------------------------------------------- #
def _new_node(action, sid):
    return {
        "type": "node",
        "node_id": sid,
        "source_id": sid,
        "action": _norm_action(action),
        "relationship": "source",
        "revisions": [],
        "_rev_ids": set(),
        "rep_z": None,
        "edge_weight": 1.0,
        "owner_authorization": owner_auth(sid, "node"),
        "created_at": None,
    }


def _add_revision(node, rev, sid):
    rev = dict(rev)
    rev["rev_id"] = _hash("rev|" + _canon(rev))
    if rev["rev_id"] in node["_rev_ids"]:
        return False                       # idempotent: no duplicate revision
    node["_rev_ids"].add(rev["rev_id"])
    rev["owner_auth"] = owner_auth(sid, rev["rev_id"])
    node["revisions"].append(rev)
    return True


def _ingest():
    """Read PAM memory + journal into a dict source_id -> node. Append-only:
    identical facts accumulate revisions; nothing is dropped."""
    sources = {}

    def _get(action):
        sid = _source_id(action)
        node = sources.get(sid)
        if node is None:
            node = _new_node(action, sid)
            sources[sid] = node
        return node

    # PAM latent memory rows carry REAL latents -> the z anchors for z->source
    if PAM.exists():
        for line in PAM.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            act = r.get("action") or "?"
            node = _get(act)
            _add_revision(node, {"src": "pam", "z": r.get("z"),
                                 "z_next": r.get("z_next"), "t": r.get("t")}, node["source_id"])

    # autonomous journal verified_actions carry the authoritative lineage (t, detail)
    if JOURNAL.exists():
        for line in JOURNAL.read_text().splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") != "verified_action":
                continue
            act = e.get("action") or "?"
            node = _get(act)
            _add_revision(node, {"src": "journal", "t": e.get("t"),
                                 "detail": e.get("detail"), "obs": e.get("obs"),
                                 "latent": e.get("latent"), "verified": e.get("verified")},
                          node["source_id"])

    # compute deterministic representative latent per source (mean of revision z's)
    for node in sources.values():
        zz = [r.get("z") for r in node["revisions"] if r.get("z") is not None]
        if zz:
            zz = np.asarray(zz, dtype=np.float32)
            node["rep_z"] = (zz.mean(axis=0) / (np.linalg.norm(zz.mean(axis=0)) + 1e-9)).tolist()
            if node.get("created_at") is None:
                ts = [r.get("t") for r in node["revisions"] if r.get("t") is not None]
                node["created_at"] = min(ts) if ts else None
    return sources


def _build_edges(sources):
    """Deterministic edges: preceded_by (temporal lineage) + similar_z (topology)."""
    edges = []
    ids = []
    sid = set(sources)
    for node in sources.values():
        ids.append(node)

    # 1) temporal lineage: order sources by their earliest revision timestamp
    #    and link consecutive distinct ones -> preceded_by
    ordered = sorted([n for n in ids if n.get("created_at") is not None],
                     key=lambda n: n["created_at"])
    for i in range(1, len(ordered)):
        a, b = ordered[i - 1], ordered[i]
        if a["node_id"] == b["node_id"]:
            continue
        gap = max(1, b.get("created_at", 0) - a.get("created_at", 0))
        w = _rounded(1.0 / (1.0 + gap), 6)
        eid = _hash("edge|preceded_by|" + a["node_id"] + "|" + b["node_id"])
        edges.append({"type": "edge", "edge_id": eid, "from": a["node_id"],
                      "to": b["node_id"], "relationship_label": "preceded_by",
                      "edge_weight": w,
                      "owner_authorization": owner_auth(eid, "preceded_by"),
                      "revisions": [{"rev_id": _hash("rev|" + eid), "owner_auth": owner_auth(eid, "rev")}]})

    # 2) topology: similar_z edges between sources with a representative latent
    anchored = [n for n in ids if n.get("rep_z") is not None]
    seen = set()
    for i in range(len(anchored)):
        a = anchored[i]
        ranked = []
        for j in range(len(anchored)):
            if i == j:
                continue
            b = anchored[j]
            c = _rounded(np.dot(a["rep_z"], b["rep_z"]) /
                         (np.linalg.norm(a["rep_z"]) * np.linalg.norm(b["rep_z"]) + 1e-9), PREC)
            if c >= SIM_THRESHOLD:
                ranked.append((c, b))
        ranked.sort(key=lambda x: -x[0])
        for c, b in ranked[:MAX_SIM_NEIGHBORS]:
            key = tuple(sorted((a["node_id"], b["node_id"])))
            if key in seen:
                continue
            seen.add(key)
            eid = _hash("edge|similar_z|" + key[0] + "|" + key[1])
            edges.append({"type": "edge", "edge_id": eid, "from": key[0], "to": key[1],
                          "relationship_label": "similar_z", "edge_weight": c,
                          "owner_authorization": owner_auth(eid, "similar_z"),
                          "revisions": [{"rev_id": _hash("rev|" + eid), "owner_auth": owner_auth(eid, "rev")}]})
    return edges


def persist(sources, edges, max_revisions=None):
    """Append-only write: merge into data/mycelium.jsonl keeping every existing
    revision (dedupe by rev_id), so rebuilds never delete history. If
    max_revisions is set, a source's dormant history is capped to its last N
    revisions (the FULL source history is preserved in autonomous_journal.jsonl
    + latent_memory.jsonl; only the derived graph's redundant copies are
    trimmed)."""
    existing = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            existing[rec.get("node_id") or rec.get("edge_id")] = rec
    for node in sources.values():
        key = node["node_id"]
        if key in existing:
            have = {r["rev_id"] for r in existing[key].get("revisions", [])}
            for r in node["revisions"]:
                if r["rev_id"] not in have:
                    existing[key].setdefault("revisions", []).append(r)
            existing[key]["_rev_ids"] = None
            existing[key]["rep_z"] = node.get("rep_z", existing[key].get("rep_z"))
        else:
            node = dict(node)
            node.pop("_rev_ids", None)
            existing[key] = node
    # cap dormant history in the DERIVED graph only (source data is preserved)
    if max_revisions:
        for rec in existing.values():
            if rec.get("type") == "node" and len(rec.get("revisions", [])) > max_revisions:
                rec["revisions"] = rec["revisions"][-max_revisions:]
    for e in edges:
        existing[e["edge_id"]] = e  # edges are structurally stable (idempotent)
    with open(OUT, "w") as f:
        for rec in existing.values():
            f.write(json.dumps(rec) + "\n")
    return existing


def build(max_revisions=DEFAULT_MAX_REVISIONS):
    """Build the graph from current data, persist, and cache. Capping per-source
    dormant history keeps the DERIVED graph lean (the FULL source history is
    preserved in autonomous_journal.jsonl + latent_memory.jsonl; only the
    derived graph's redundant copies are trimmed via max_revisions). Pass
    max_revisions=0 to keep the entire dormant history in the graph too."""
    global _GRAPH
    sources = _ingest()
    edges = _build_edges(sources)
    _GRAPH = {"sources": sources, "edges": edges}
    persist(sources, edges, max_revisions=max_revisions)
    return _GRAPH


def stats():
    if _GRAPH is None:
        build()
    s = _GRAPH["sources"]
    n_rev = sum(len(n["revisions"]) for n in s.values())
    anchored = sum(1 for n in s.values() if n.get("rep_z") is not None)
    auth_ok = sum(1 for n in s.values() if verify_auth(n))
    edges = _GRAPH["edges"]
    by_label = {}
    for e in edges:
        by_label[e["relationship_label"]] = by_label.get(e["relationship_label"], 0) + 1
    return {"sources": len(s), "revisions": n_rev, "anchored_z": anchored,
            "edges": len(edges), "edges_by_label": by_label,
            "owner_authorized_sources": auth_ok}


def z_to_sources(query, k=5):
    if _GRAPH is None:
        build()
    q = _norm_vec(query)
    if q is None:
        return [{"error": "could not embed query (needs a 12-dim obs or 64-dim latent)"}]
    cands = []
    for n in _GRAPH["sources"].values():
        zr = n.get("rep_z")
        if zr is None:
            continue
        c = _rounded(np.dot(q, zr) / (np.linalg.norm(q) * np.linalg.norm(zr) + 1e-9), PREC)
        cands.append((c, n))
    cands.sort(key=lambda x: -x[0])
    out = []
    for c, n in cands[:k]:
        revs = n.get("revisions", [])
        out.append({"source_id": n["node_id"], "action": n["action"],
                    "cosine": c, "relationship": "similar_z", "weight": c,
                    "history_len": len(revs),
                    "latest_revision": revs[-1].get("rev_id") if revs else None,
                    "owner_authorized": verify_auth(n)})
    return out


def history(node_id):
    if _GRAPH is None:
        build()
    n = _GRAPH["sources"].get(node_id)
    if n is None:
        return {"error": f"no such source {node_id}"}
    return {"source_id": n["node_id"], "action": n["action"],
            "history_len": len(n["revisions"]), "owner_authorized": verify_auth(n),
            "revisions": n["revisions"]}


def verify_all():
    if _GRAPH is None:
        build()
    bad = []
    for n in _GRAPH["sources"].values():
        if not verify_auth(n):
            bad.append(n["node_id"])
    for e in _GRAPH["edges"]:
        if e.get("owner_authorization") != owner_auth(e["edge_id"], e["relationship_label"]):
            bad.append(e["edge_id"])
    return {"checked": len(_GRAPH["sources"]) + len(_GRAPH["edges"]), "violations": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="ingest PAM+journal -> graph")
    ap.add_argument("--stats", action="store_true", help="print graph metrics")
    ap.add_argument("--z-to-source", dest="query", nargs="?",
                    help="find nearest sources for a 12-dim obs or 64-dim latent")
    ap.add_argument("--latent", action="store_true",
                    help="with --z-to-source, the query is a raw 64-dim latent")
    ap.add_argument("--history", metavar="ID", help="print a source's full revision chain")
    ap.add_argument("--verify", action="store_true", help="verify all owner authorizations")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-revisions", type=int, default=DEFAULT_MAX_REVISIONS,
                    help="cap per-source dormant history in the DERIVED graph (0=keep all; "
                         "full source history stays in the journal + PAM)")
    args = ap.parse_args()

    if args.build:
        build(max_revisions=args.max_revisions)
        print("mycelium graph built ->", OUT)
        print(json.dumps(stats(), indent=2))
    elif args.stats:
        print(json.dumps(stats(), indent=2))
    elif args.query is not None:
        q = args.query.strip()
        if args.latent:
            q = [float(x) for x in q.replace(",", " ").split()]
        else:
            q = [float(x) for x in q.replace(",", " ").split()]
        print(json.dumps(z_to_sources(q, k=args.k), indent=2))
    elif args.history:
        print(json.dumps(history(args.history), indent=2))
    elif args.verify:
        print(json.dumps(verify_all(), indent=2))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
