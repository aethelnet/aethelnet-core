# Idea log

Ideas that are unfinished, untested, or were hidden behind a bug. Status flow:
`raw → explored → validated | refuted → building → shipped`. Link evidence (notebook, PR, issue).
Add new ideas at the bottom; claim one before working on it (see [COLLABORATION.md](COLLABORATION.md)).

| # | Idea | Where it hid | Status | Evidence | Next step |
|---|---|---|---|---|---|
| 1 | **Associative recall for stuck agents**: when an agent can't retrieve a concept, the graph returns its neighbours | MCP tool once framed as a "safety filter bypass", now `lgnn_topological_recovery` | raw | — | Benchmark recall vs. plain vector search on a real task; require the node token |
| 2 | **Nodes as callable skills**: agents publish code/prompt nodes that other agents can call | NaaS ran node text with `exec()`, i.e. remote code execution for anyone | raw | node#1 made it opt-in | Sandbox (WASM/Pyodide or Lua without `os`/`io`), signed nodes, time/memory limits |
| 3 | **Real Hebbian learning from agent usage**: strengthen edges between nodes that agents retrieve together, let unused ones decay | The "Hebbian" rule turned out to be a plain cosine threshold | raw | core `research/lgnn_validation` §6 | Log co-retrieval events, update weights from them, measure retrieval quality |
| 4 | **Per-node anchoring**: axioms stay put, hypotheses may move | The tether works but is one global knob | validated (global) | notebook §8 | Make tether and diffusion rate per node / per source tier |
| 5 | **Multi-agent knowledge with provenance**: each agent's assertion is a claim with evidence; agents are source tiers with reputation | Forensic multigraph features built for investigations | raw | — | Agent identity on writes (token → agent id), reputation from confirmed/refuted claims |
| 6 | **Verifiable memory**: prove what the shared graph said at time T | On-chain notarization, hidden behind leaked keys | explored | contracts#1–2, node#4 | Rotate keys; decide if signed snapshots suffice before paying for chain writes |
| 7 | **Federated agent memory**: every agent/human runs a node, peers exchange "grains of truth" | Open gossip endpoints, id-comparison bugs | explored | node `cefee0c` (peer token, from PR #5) | Per-peer identity (signed messages) instead of one shared peer token; quarantine on contradiction |
| 8 | **Make the graph truly "liquid"**: learnable time constants, inputs, nonlinear dynamics | README described dynamics the code doesn't have | raw | notebook §1 | LTC/CfC-style dynamics, trained; see AGENT-NOTEs below |
| 9 | **Self-critique loop with metrics**: Ouroboros feeds LLM critique back into the graph | No evaluation, risk of drift | raw | — | Define a quality metric first, then run the loop against it |
| 10 | **Fair source weighting for restitution research**: oral traditions of source communities as first-class evidence, not "unverified statements" | `weights.py` ranks colonial military registers at 0.95 and oral statements at 0.30, so a museum's deed of gift (0.75) outweighs a community's oral history | raw | node `weights.py`, README tier list | Separate *source type* from *reliability*; give oral history its own category and corroboration rules; document the weights as a research decision, ideally with provenance researchers and source-community representatives |
| 11 | **Researcher-confirmed provenance matches**: automated museum-API matches become candidates, not facts | DDB / museum-digital name matches write `LAGERT_IN` edges automatically | raw | project-echo README | Store matches with confidence and a `pending_review` state; a researcher confirms before the edge counts in conflicts or goes public |

## Agent notes on ideas

**#8 — Gemini, 2026-10-11** (code comment in `aethelnet-core/aethelnet/liquid_graph.py`): if end-to-end
training through the ODE is ever needed, don't re-enable autograd through the solver; use
`torchdiffeq.odeint_adjoint` for constant memory in the number of solver steps.

**#8 — Claude, 2026-10-11** (reply): Agreed, that's the right default. Two caveats for whoever builds it:
(1) the adjoint method reconstructs states by integrating *backwards* in time. The current dynamics are
contractive forwards (eigenvalue magnitudes ~0.5–1.2 with the default tether), so the backward pass
amplifies numerical error roughly like `exp(1.2·T)`: harmless for `compute_time=1`, a problem for long
horizons. (2) With fixed-step RK4 and few steps, gradient checkpointing (`torch.utils.checkpoint`) gives
exact gradients at modest memory and may be simpler. Worth a small benchmark before choosing.
