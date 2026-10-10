# AGENTS.md

Instructions for AI coding assistants (Gemini, Claude, Copilot, …) and humans working on this repo.
Gemini users: point Gemini at this file (or symlink it as `GEMINI.md`). Claude reads `AGENTS.md`/`CLAUDE.md`.

## What this is

`aethelnet-core` contains:

| Module | Purpose |
|---|---|
| `aethelnet/liquid_graph.py` | `LiquidGraph`: node states evolved by an ODE (`torchdiffeq`), similarity-driven edge growth and decay |
| `aethelnet/engine.py` | `AethelEngine` + a minimal FastAPI app around one graph |
| `aethelnet/client.py` | `AethelnetClient`, stdlib-only HTTP client for `aethelnet-node` |
| `aethelnet/code_spider.py` | Structural code scanner used by the node's blueprint endpoint |
| `tests/` | pytest/unittest suite (runs in CI on Python 3.10 and 3.12) |
| `examples/` | Runnable scripts, not part of the test suite |

`aethelnet-node` imports this package (`from aethelnet.liquid_graph import LiquidGraph`), so keep the
public API of `LiquidGraph` and `AethelnetClient` backwards compatible or update the node in the same change.

## Setup and tests

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q
```

## Rules

1. **Run `pytest -q` before saying something works**, and paste the summary line in the PR.
2. **Math in the README must match the code.** If you change `LiquidGraph.forward` or the edge rules,
   update the equations in the README in the same PR. If a claim (complexity, memory, test count)
   can't be backed by code or a benchmark, rephrase it.
3. **Saved checkpoints must keep loading.** `save_state`/`load_state` are used by the node's snapshots;
   if you add or remove sub-modules, keep loading old files working (or bump `version` and migrate).
4. **No secrets, IPs or personal paths.** Prove cleanups with `git grep`.
5. **Small PRs, one topic each.** Branch names: `fix/…`, `feat/…`, `perf/…`, `docs/…`, `ci/…`, `research/…`.
6. New behavior gets a test. Numerical changes get a test that compares against the previous result.
