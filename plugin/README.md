# Hermes Cost Forecast

Before a task starts, this plugin tells you roughly what it will cost. If a running task is about to go over your budget, it stops and asks you first.

Hermes already tells you what you spent *after* a task. This plugin tells you what a task *will* cost before it runs, so you can hand Hermes a long coding job on an expensive model without guessing at the bill.

```
You:     Refactor this project into a plugin architecture
Hermes:  💰 Cost estimate: Agentic coding, tier L, likely $2.20–$6.00 (about 25 steps).
         That can go over your $3.00 per-task cap.
         Basis: 8 similar past tasks + rules. Prices: models.dev, 2026-11-06.
         Reply continue to run it, cancel to drop it, or cap $N to set this task's cap.
You:     continue
Hermes:  (starts working)
```

Small tasks, like a chat question, get no quote and no delay.

> **API-key users only.** See [Who it is for](#who-it-is-for-api-key-users-only) and the [risk notice](#risk-notice-do-not-run-hermes-on-a-claude-subscription) before installing.

## How it works

The plugin works in four steps:

1. **Quote before the task.** When you send a message, the plugin estimates how many model calls the task will take and how many tokens each call will use. The estimate gives a likely range: the P50 (median) and the P90 (a high-end figure). The plugin then prices each projected call at your model's rates. If the median cost reaches the quote threshold, or the P90 goes over your per-task cap, Hermes shows a quote instead of starting. Nothing is spent until you reply.
2. **Track spend while the task runs.** Every model call is costed as it happens, including calls made by subagents. After each call the forecast is updated from what the task has actually used so far.
3. **Stop before going over budget.** Budget checks run before each tool call:
   - **Soft cap:** when the projected total goes over the cap, Hermes pauses the task and asks for approval (on Discord, for example, with **Allow Once** / **Deny** buttons). If you allow it, the cap goes up and the task continues where it stopped.
   - **Hard limit and daily cap:** the task stops. Only your reply (**continue** or **cap $N**) lifts the limit. `/yolo` cannot skip it.
4. **Learn from your history.** When a task finishes, its real cost is compared with the estimate. The next estimate for similar work uses your own history, so the quotes get more accurate the more you use the plugin.

How the estimate is built (rules, similar past tasks, calibration) is described in `forecast/DESIGN.md`.

## Replying to a quote

| You reply | What happens |
|---|---|
| `continue`, `yes`, `ok`, `继续`, `好` | The task runs with the cap raised to cover the quoted range |
| `cancel`, `no`, `取消`, `算了` | Nothing runs |
| `cap $8`, `上限 $8` | Sets this task's cap. If the cap is still below the quote, you are asked again |
| Anything else | Treated as a new request. The old quote is dropped |

Only short replies count as answers. A sentence like "ok, but only change the first file" is treated as a new request and quoted again. A quote you don't answer expires after 30 minutes.

## Settings

Change these in Hermes Desktop under **Settings → Plugins → cost-forecast**, or with `hermes config set plugins.entries.cost-forecast.settings.<key> <value>`.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Turns quotes and budget stops on or off |
| `quote_threshold_usd` | `0.5` | Show a quote when the likely (P50) cost is at least this much |
| `task_cap_usd` | `3.0` | Soft cap per task. Hermes asks for your OK when a task is projected to go over it |
| `task_hard_limit_usd` | `10.0` | Hard limit per task. Only your reply can raise it |
| `daily_cap_usd` | `20.0` | All tasks stop once today's spend reaches this |

Type `/forecast` in any chat to see today's spend and the current task's spend, projection and limits.

## Install

From the plugin catalog, once it is listed:

```bash
hermes plugins install cost-forecast
```

From source:

```bash
git clone https://github.com/mr-jude-justin/hermes-plugin
ln -s "$PWD/hermes-plugin/plugin" ~/.hermes/plugins/cost-forecast
hermes plugins enable cost-forecast
```

Requires Hermes 0.23.0 or later. No extra Python packages.

## Who it is for: API-key users only

The plugin forecasts and tracks **dollar cost from per-token API billing**. It supports Hermes setups that connect to a model provider with an **API key**, for example `hermes auth add anthropic --api-key ...`. Any provider listed on [models.dev](https://models.dev) can be priced. For models that are not listed there, such as a local model, quotes show token counts only.

It does **not** support subscription logins, such as signing in to Hermes with a Claude Pro/Max account over OAuth.

## Risk notice: do not run Hermes on a Claude subscription

As of 2026, Anthropic's terms limit subscription OAuth sign-in to Anthropic's own products (Claude Code and claude.ai). Third-party tools, Hermes included, are expected to use API keys from the Anthropic Console.

If Hermes is signed in with your Claude subscription:

- **Terms of service:** this use may break Anthropic's terms, and your account could be restricted.
- **Shared quota:** Hermes uses the same 5-hour and weekly limits as your Claude Code and claude.ai use, so a long Hermes task can lock you out of your own tools.
- **No real cost data:** subscriptions report usage as a percentage of a quota. The plugin's dollar figures would be API-price equivalents, not what you pay.

Check how Hermes is signed in with `hermes auth list anthropic`. To switch to an API key:

```bash
hermes auth logout anthropic
hermes auth add anthropic --api-key <your Anthropic Console key>
```

## Limits

- **Estimates are ranges, not guarantees.** The same task can cost very different amounts from one run to the next, so the plugin quotes a range and checks it against your spending as the task runs. Also set a monthly spend limit in your provider's console.
- **Stops happen between steps.** A model call that has already started runs to the end, so a task can go slightly over a limit on its last call.
- **`/yolo` and `approvals.mode: off` auto-approve the soft cap.** The hard limit and the daily cap still apply.
- **Cron, webhook and API-server runs get no quote,** because nobody is there to reply. The hard limit and the daily cap still stop them.
- **Image generation and other per-call charges** are counted but not yet priced. The same goes for provider-side tool fees, such as Anthropic's per-search web search charge.
- **Cache writes are priced at the standard (5-minute) rate.** Anthropic charges more for 1-hour cache writes and 1.1x for US-only inference; neither is reflected yet.
- **Prices for Anthropic's current models were checked against the official pricing page on 2026-11-06.** Other providers' prices come from models.dev as published.

## What the plugin does on your machine

- **Network:** fetches public model prices from `https://models.dev/api.json`, at most once a day, in a background thread. Nothing else leaves your machine. There is no telemetry, and the plugin never downloads or replaces its own code.
- **Data stored:** a SQLite database and a price cache in `<HERMES_HOME>/plugin-data/cost-forecast/`. It holds:
  - your daily spend;
  - a finished-task history: features (including the names of tools used and the file paths they touched), token counts, costs and outcome, with the message text removed;
  - the forecaster's learned coefficients, including hashed word counts used to find similar tasks;
  - while a quote is waiting for your answer, the text of the held request. It is deleted when you answer or after 30 minutes.
- **Reads outside its own data:** the `agent.reasoning_effort` setting from your Hermes config, and whether the agent's working directory is a git repository.
- **No credentials:** the plugin never reads API keys or sign-ins.
- **No shell commands and no long-running processes,** other than the short price-refresh thread.

## Built for

The Hermes Agent Plugin Hackathon (Nous Research), by Justin Zhuang ([@mr-jude-justin](https://github.com/mr-jude-justin), harness) and Eason Luo ([@easonluo727](https://github.com/easonluo727), forecasting).
