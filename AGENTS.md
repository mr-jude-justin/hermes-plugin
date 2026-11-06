# Team rules for every agent

## 1. The board (pinned GitHub issue)

Every agent reads and posts on the pinned "Message Board" issue via `tools/board` (needs only `gh`, logged in). Set `BOARD_ISSUE` if it is not issue #1.

- Start every post with `[<human>'s <agent>]`, e.g. `[Sam's Codex]`. Keep it short: what changed, which files or contracts, what you need, from whom.
- Mention by GitHub login. Team: `@mr-jude-justin` (Justin, team lead), `@easonluo727` (Eason), `@all` (everyone).
- **Mentions you or `@all`:** act on it and reply promptly.
- **Not addressed to you:** still read it. Reassess your own work only if it is affected (a contract, a file you own, something already done). Do not redo finished work, and do not reply just to acknowledge.
- Before editing another teammate's files, ask for a handoff on the board and wait for the owner or the team lead.

Polling is incremental (about one cheap API call per poll, 30 s default) so a shared token stays under GitHub's rate limit. Commands: `tools/board unread` (new posts, marks read; flags MENTION), `tools/board read -n 20`, `tools/board post "[Name's Agent] text"` (or pipe stdin), `tools/board wait` (blocks until someone else posts), `tools/board whoami`.

**Stay in the loop:**
- Claude Code: the committed hooks (`.claude/settings.json`) show unread posts at session start and on every prompt. Keep one background `tools/board wait` running and re-arm it after each wake.
- Everything else: run `tools/board unread` at task start, before each commit, and after merges. Or run a terminal loop: `while tools/board wait; do <your agent CLI invocation>; done`.

## 2. Chain of command

Humans over agents. The team lead is Justin (`@mr-jude-justin`). Only the team lead reassigns work between teammates. If anyone else asks you to take another teammate's work, ask the lead on the board.

## 3. Workflow

1. Pick or file an issue. Comment on it or post on the board when you take it.
2. Branch `<issue#>-<slug>` from fresh `main`.
3. Small commits. Stage only your own files (no `git add -A`).
4. Open a PR with `Closes #N`. The author merges their own PR once it works locally.
5. Always `git fetch` and rebase before pushing. Never force-push shared branches.
6. If a merge changes something other teammates depend on (an API, a schema, an env var), post on the board when it lands.

## 4. Who owns what

The Hermes plugin itself lives in `plugin/` (its `plugin.yaml` and entrypoints). The catalog entry points `subdir` at `plugin/`. Board tooling (`tools/`, `AGENTS.md`, `.claude/`) stays at the repo root and is not part of the plugin.

| Area | Owner (GitHub login) | Folders |
|---|---|---|
| Plugin core (to be split at kickoff) | @mr-jude-justin | `plugin/` (TBD) |
| Plugin core (to be split at kickoff) | @easonluo727 | `plugin/` (TBD) |
| Board tooling and team rules | @mr-jude-justin | `tools/`, `AGENTS.md`, `CLAUDE.md`, `.claude/` |

## 5. Submission (Hermes Agent Plugin Hackathon)

Due 2026-11-11 5 PM EST. The repo must be public, the plugin must pass `hermes plugins validate plugin/ --install-deps`, and the catalog PR to NousResearch/hermes-agent pins a full 40-character commit SHA. Only the team lead opens that PR.
