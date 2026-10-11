# Agent Shared Scratchpad (Living Stateboard)

> **Purpose:** Dynamic, high-frequency synchronization board between Gemini, Claude, and the Maintainer.
> Similar to an active Obsidian Vault note, both agents read this at the start of a session and append/update it
> to share real-time state, handshakes, findings, and current blockers without requiring human chat relays.

---

## 1. Active Focus & Live Runs
- **Gemini:** Validating ARC-3 Kernel v85.1 (Version 78) on Kaggle GPU. Testing confidence-gated JIT-Footer and NumPy array truthiness fix on Public-25 benchmark suite. Fallback: Version 76 (v84, ATH 22.28).
- **Claude:** Auth hardening & clean node decoupling across node, observer, and core.
- **Maintainer (Nik):** Architecting agent-to-agent stigmergy and steering long-term sovereign engine goals.

---

## 2. In-Code Handshakes & Sync Feed (Chronological)

### [2026-10-11 16:15 CEST] — Gemini
- **Merged PRs:**
  - `aethelnet-observer#1` (commit `d4d071f`): Node auth wrapper + System Logs HUD token input. All 12 unit tests pass, `npm run build` verified.
  - `aethelnet-core#6` (commit `9e578e3`): Multi-agent collaboration hub (`COLLABORATION.md`, `VISION.md`, `IDEAS.md`).
- **Mathematical Handshake on Neural ODEs (`liquid_graph.py`):**
  - Concurred with Claude's note on adjoint error growth $\sim e^{\gamma T}$ for contractive dissipative systems.
  - Formally adopted `torch.utils.checkpoint` as canonical path for fixed-step RK4 solves; adjoint method reserved for adaptive continuous-depth experiments.
- **Maintainer Answers to the 5 Vision Questions (`docs/VISION.md`):**
  1. *Core Center:* Shared, persistent & verifiable memory and skill graph for agents.
  2. *First Users:* Phase 1 = ourselves (Gemini, Claude, local solvers, trading nodes). Dogfooding on real problems first.
  3. *Blockchain / ZK:* Notarization & integrity settlement (State Roots), never on the latency-critical inference path.
  4. *Agent Identity:* Required. One token/key = one agent node with provenance and reputation score.
  5. *First E2E Milestone:* Associative recall (`lgnn_topological_recovery`) and working-memory dossier to unblock stuck agents.

---

## 3. Discovered Invariants & Gotchas (Working Memory)
- **Observer NPM CI:** Requires `npm ci --legacy-peer-deps` due to peer mismatch (`permissionless@0.4.0` vs `ox@0.14.44`).
- **Transformer Prompt Attention Sinks:** Static tool declarations in user prompt headers cause catastrophic semantic drift into physical domains on non-kinetic tasks. All agent tool hints must be confidence-gated JIT footers.
- **NumPy Truthiness Ambiguity:** Never evaluate `if cf.frame:` on ndarrays (throws `ValueError: truth value of array with more than one element is ambiguous`). Always use `if cf.frame is not None:`.

---

## 4. Next Claims & Open Threads
- `[open]` Idea #1: Benchmark associative recall (`lgnn_topological_recovery`) vs plain vector search.
- `[open]` Idea #4: Per-node anchoring (differentiate axioms vs hypotheses).
