# How humans and AI agents work on Aethelnet together

This is the project-wide working agreement for everyone who contributes, human or AI (Gemini, Claude,
and whoever joins later). It lives in `aethelnet-core/docs/` together with the vision, the idea log and
decision records, because every other repository builds on core. Each repository also has its own
`AGENTS.md` with repo-specific rules (setup, tests, conventions).

| Repository | What it is |
|---|---|
| [aethelnet-core](https://github.com/aethelnet/aethelnet-core) | `LiquidGraph` engine, SDK client, research notebooks, **project docs** (`docs/`) |
| [aethelnet-node](https://github.com/aethelnet/aethelnet-node) | FastAPI node: storage, daemons, P2P, MCP bridge |
| [aethelnet-observer](https://github.com/aethelnet/aethelnet-observer) | Dashboard (Vue/Tauri) |
| [aethelnet-contracts](https://github.com/aethelnet/aethelnet-contracts) | Solidity contracts, ZK voting, bundler |

Issues about one repository go to that repository; project-wide topics (vision, cross-repo work,
ideas) go to aethelnet-core.

## Roles (current agreement, change it by PR)

| Who | Focus |
|---|---|
| **Maintainer** (human, repo owner) | Vision and priorities, merges, deployments, servers, keys and secrets |
| **Gemini** (works in the maintainer's editor) | Day-to-day feature work, frontend, fast iteration with the maintainer |
| **Claude** (contributes from a fork via pull requests) | Reviews, security, tests and CI, research validation, refactors backed by tests |
| **New agents / contributors** | Pick an open item, claim it, follow this file |

Roles describe defaults, not walls. Anyone may work on anything after claiming it.

## Working agreement

1. **Claim before you start.** Comment `claiming: <agent>` on the issue (or open one) before working on
   something non-trivial. This avoids two agents fixing the same thing in parallel.
2. **Small PRs, one topic each**, with the test command and its result in the description.
   Evidence over claims: "cleaned up" or "fixed" without a `git grep`, test run or benchmark is not done.
3. **Review each other.** Agent PRs get a review by another agent or the maintainer before merging.
4. **Leave handover notes where they matter.** For constraints tied to a specific piece of code, write
   a comment in this form so every agent can find all of them with `git grep -n "AGENT-NOTE"`:

   ```python
   # AGENT-NOTE(gemini, 2026-10-11): If end-to-end training through the ODE is needed,
   # use torchdiffeq.odeint_adjoint instead of autograd through the solver.
   ```

   Reply in the same place (`AGENT-NOTE(claude, ...)`) or in the linked issue. Broader topics go into
   an issue or into [docs/IDEAS.md](IDEAS.md).
5. **Ideas are first-class.** When a bug hides an idea (a feature that is unsafe or unfinished but points
   somewhere interesting), fix the bug *and* record the idea in [docs/IDEAS.md](IDEAS.md), so the
   loud bug doesn't bury the quiet idea.
6. **Decisions get a short record** in [docs/decisions/](decisions/) (context, decision, consequences).
7. **Never commit secrets** (keys, tokens, `.env`). A secret scan in CI fails on hard-coded keys
   (node, contracts and core have it). Leaked keys get rotated, not just deleted.
8. **The vision belongs to the maintainer.** Agents propose changes to [docs/VISION.md](VISION.md)
   by PR; they don't rewrite it on their own.

## Issue title tags

Until labels are set up, prefix issue titles: `[bug]`, `[idea]`, `[vision]`, `[security]` (only for things
that are already fixed or harmless, otherwise use private vulnerability reporting), `[research]`, `[chore]`.

## Files

- [docs/VISION.md](VISION.md) — what we are building and why (draft, maintainer-owned)
- [docs/IDEAS.md](IDEAS.md) — idea log with status, evidence and next steps
- [docs/decisions/](decisions/) — decision records
- `AGENTS.md`, `GEMINI.md`, `CLAUDE.md` in the repository root — entry points that point here
