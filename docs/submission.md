# Catalog submission (draft)

Only the team lead (@mr-jude-justin) opens the catalog PR. Due 2026-11-11 5 PM EST.

## Checklist

1. All work merged to `main`; `uvx pytest plugin` passes.
2. Make the repo public: `gh repo edit mr-jude-justin/hermes-plugin --visibility public --accept-visibility-change-consequences`.
3. Pin the commit: `SHA=$(git rev-parse origin/main)` (full 40 characters).
4. Run what catalog CI runs, on a clean checkout at that SHA:
   ```bash
   git clone https://github.com/mr-jude-justin/hermes-plugin /tmp/cf && git -C /tmp/cf checkout $SHA
   hermes plugins validate /tmp/cf/plugin --install-deps
   ```
5. Fork `NousResearch/hermes-agent`, add `plugin-catalog/cost-forecast.yaml` (below) with the SHA filled in, and open the PR with the description below.
6. Wait for the catalog CI job to go green.
7. Post the demo video on X and fill in the hackathon Typeform.

## `plugin-catalog/cost-forecast.yaml`

```yaml
name: cost-forecast
repo: https://github.com/mr-jude-justin/hermes-plugin
sha: <40-hex commit sha>
subdir: plugin
title: Cost Forecast
description: "Quotes a task's cost before it runs and stops for your OK when it would go over budget. It estimates steps and tokens from rules, your own past tasks and calibration, prices each projected call from models.dev, then holds the turn for a quote above a threshold; mid-run it reforecasts after every model call, pauses at Hermes' approval gate at the soft cap and blocks at a hard per-task limit and daily cap that /yolo cannot skip. API-key users only. Disclosure — fetches public model prices from models.dev/api.json at most once a day in a background thread; stores daily spend, a finished-task history (features, tool names and file paths touched, token counts, costs, no message text) and hashed word counts in <HERMES_HOME>/plugin-data/cost-forecast/; holds a quoted request's text until it is answered or for 30 minutes; reads agent.reasoning_effort from Hermes config and checks whether the agent's working directory is a git repo; no credentials, telemetry, shell commands or self-updating."
maintainer: mr-jude-justin
tier: community
category: tools
requires_hermes: ">=0.23.0"
version: "0.1.0"
capabilities:
  provides_tools: []
  provides_hooks:
  - api_request_error
  - on_session_end
  - post_api_request
  - post_llm_call
  - post_tool_call
  - pre_llm_call
  - pre_tool_call
  - subagent_start
  - subagent_stop
  provides_middleware: []
  requires_env: []
```

Check `version` against `plugin/plugin.yaml` and `provides_hooks` against `HOOKS` in `plugin/__init__.py` before opening the PR.

## PR description

> **Add cost-forecast: quote a task's cost before it runs**
>
> Hermes and existing catalog plugins (hermes-telemetry, cost-sentinel, token-cost-meter) report what a task cost after the fact, or stop at a daily limit. None tells you what a task will cost before it starts. cost-forecast does:
>
> - **Before the task** (`pre_llm_call`): it estimates the number of model calls and the tokens per call from rules, the user's own past tasks and calibration, prices each projected call from models.dev (context-tier aware), and, above a threshold, blocks the turn with a quote. The user replies continue / cancel / cap $N. The held request is re-injected as `pre_llm_call` context on approval.
> - **During the task**: it costs every call in `post_api_request` (subagents attributed by child session id) and reforecasts. `pre_tool_call` returns `approve` at the soft cap (Hermes' own approval gate) and `block` at the hard per-task limit and the daily cap, so `/yolo` and `approvals.mode: off` cannot spend past them.
> - **After the task**: the real cost is recorded against the estimate, which feeds the next forecast.
> - **Unattended runs** (cron, webhook, api_server) never get a quote, since nobody can answer. Only the hard limit and the daily cap apply.
>
> Surfaces: hooks only (`pre_llm_call`, `post_api_request`, `api_request_error`, `pre_tool_call`, `post_tool_call`, `subagent_start`, `subagent_stop`, `post_llm_call`, `on_session_end`), `ctx.register_command("forecast")`, `ctx.get_config`, `plugin_db` / `plugin_data_dir`. No tools, no middleware, no core overrides, no Python dependencies.
>
> Disclosure (rule 13):
> - Network: models.dev/api.json, at most daily, background thread.
> - Stored data: `<HERMES_HOME>/plugin-data/cost-forecast/`, containing daily spend, task history (no message text; tool names and file paths), hashed word counts, and a quoted request's text until it is answered or for up to 30 minutes.
> - Reads: `agent.reasoning_effort` (via `hermes_cli.config.load_config_readonly`), and whether the working directory (via `agent.runtime_cwd.resolve_agent_cwd`) contains a `.git` folder.
> - No credentials, no telemetry, no shell commands, no self-updating.
>
> API-key users only. The README carries a risk notice against running Hermes on a Claude subscription.
>
> `hermes plugins validate plugin/ --install-deps` passes at the pinned SHA.
