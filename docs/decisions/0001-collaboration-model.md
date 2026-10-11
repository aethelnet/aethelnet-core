# 0001 — Collaboration model for humans and AI agents

- **Date:** 2026-10-11
- **Status:** proposed

## Context

Several AI agents work on Aethelnet: Gemini in the maintainer's editor, Claude from a fork through pull
requests, and possibly more later. On 2026-10-10 two of them fixed the same CI problem in parallel, and
earlier an agent reported a cleanup as done while secrets were still in the repository. Ideas also got
buried under the bugs they came with (e.g. callable skill nodes behind a remote-code-execution hole).

The original `aethelnet/aethelnet` repository is the owner's archived profile repository, so it can't
host shared documents. Core is the repository every other part depends on.

## Decision

- `aethelnet-core/docs/` is the hub for vision, ideas, decisions and the shared working agreement ([COLLABORATION.md](../COLLABORATION.md)).
- Work is claimed on an issue before it starts; changes arrive as small, tested PRs; agents review each other.
- Code-local handovers use `AGENT-NOTE(<agent>, <date>)` comments; broader topics use issues or [IDEAS.md](../IDEAS.md).
- The maintainer owns the vision and merges; agents propose.

## Consequences

- Less duplicate work and fewer unverified "done" claims.
- Ideas get a status and evidence instead of living in chat histories.
- Small overhead: claiming, writing notes, keeping IDEAS.md current.
- Labels and assignment need the maintainer (or triage access for agents' accounts); until then, issue title tags are used.
