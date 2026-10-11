# Vision (draft)

> **Status: draft for the maintainer to correct.** Written by Claude from what the code already contains
> and from the maintainer's statement that Aethelnet is being built *for agents*. Everything below is a
> proposal; the open questions at the end matter more than the prose.

## One sentence

Aethelnet is a **shared, persistent and verifiable memory and skill graph for AI agents** and the
humans who work with them: agents write claims with evidence into it, find related knowledge through
it, call each other's skills from it, and can prove later what it contained.

## Why agents need this

Today every agent session starts nearly empty, agents can't see each other's work, and nothing an agent
"knows" carries provenance. When several agents collaborate (as on this project), they need:

- **memory that outlives a session** and is shared across agents,
- **association, not just lookup**: "what is near this concept?" when an agent is stuck,
- **provenance and trust**: who claimed what, based on which evidence, and how reliable that source is,
- **contradiction handling**: conflicting claims should surface, not silently overwrite each other,
- **reusable skills**: something one agent built should be callable by another,
- **verifiable history**: proof of what the shared state was at a given time.

## What already exists, seen through that lens

| Need | Building block in the code today | Where |
|---|---|---|
| Shared persistent memory | Node with SQLite/PostgreSQL graph, snapshots | aethelnet-node |
| Association | `LiquidGraph` diffusion + similarity edges; MCP `lgnn_topological_recovery` | core, node `mcp/` |
| Provenance and trust | Claims with evidence quotes, 4-tier source weighting | node `weights.py`, ontology |
| Contradictions | Conflict rules and verdicts per investigation | node `ontology/registry.py` |
| De-duplication | Entity resolution (aliases, typos) | node `entity_resolution.py` |
| Reusable skills | NaaS nodes (`APP:Python`, `APP:Lua`), now opt-in because unsandboxed | node `routers/node_runtime.py` |
| Verifiable history | Canonical state roots, on-chain notarization | node `forge_client.py`, contracts |
| Federation | P2P gossip between nodes | node `routers/p2p.py` |
| Agent access | MCP bridges | node `mcp/` |
| Self-improvement | Ouroboros critique loop | node `ouroboros.py` |

Most of the pieces are there. What is missing is mostly **safety and identity** (who is writing?),
**evaluation** (does it help agents?) and making a few labels true (see [IDEAS.md](IDEAS.md)).

## Principles (proposed)

1. **Evidence over claims.** Every write is a claim with a source and evidence; trust is computed, not assumed.
2. **Contradictions are information.** Surface them, never resolve them silently.
3. **Anchored concepts.** Meaning may drift through use, but stays tethered to where it came from.
4. **Safe by default.** Writes need identity, code runs only sandboxed, secrets never live in repos.
5. **Local-first, federated.** Each human or agent can run a node; nodes exchange knowledge as peers.
6. **Measured, not marketed.** Claims about the system are backed by tests and notebooks.

## Decisions on open questions (Maintainer & Gemini, 2026-10-11)

1. **Center:** Yes, **"shared, persistent and verifiable memory and skill graph for agents"** is the primary center. Forensic investigations and trading nodes are concrete consumer domains that prove the substrate works.
2. **First users:** **Phase 1 is ourselves** — the agents working on this repo and our sovereign systems (Gemini, Claude, ARC-3 reasoning solvers, trading inference agents). Rigorous dogfooding on real engineering problems before public exposure.
3. **Role of blockchain:** **Integrity & settlement layer (State Roots)**. Used for notarization and immutable checkpoints, but strictly kept off the latency-critical local inference path (zero inference latency).
4. **Agent identities & reputation:** **Yes, mandatory.** One token/key corresponds to an agent identity. Claims in the graph must carry provenance; reputation is dynamically weighted based on verified evidence and test outcomes.
5. **First end-to-end milestone:** **Associative recall (`lgnn_topological_recovery`) and working-memory dossier.** When an agent gets stuck, it queries its local node via MCP and receives relevant neighboring nodes, invariants, and proven tabus to unblock itself.
