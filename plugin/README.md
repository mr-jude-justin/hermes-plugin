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
4. **Learn from your history.** When a task finishes, its real cost is compared with the estimate. The next estimate for similar work uses your own history, so the quotes get more accurate the more you use the plugin. See [It learns from your own tasks](#it-learns-from-your-own-tasks).

How the estimate is built (rules, similar past tasks, calibration) is described in `forecast/DESIGN.md`.

## It learns from your own tasks

Every finished task is recorded with its estimate and its real cost. The plugin learns from these records in three ways:

- **Similar past tasks.** For a new request, it finds the most similar tasks you have already run. It reuses how many model calls they really took and how large each call was. It reuses shapes, not dollar amounts, so most of the history still applies after you switch models (different models can take a different number of calls for the same task; that is not adjusted for yet). Text similarity uses hashed word and character counts; the message text itself is never stored.
- **Calibration.** After 3 finished tasks, it measures how far its estimates run high or low for *you*, corrects for that, and widens or narrows the range so that about 9 in 10 tasks land under the high-end (P90) figure. The quote shows the result, for example "calibrated x0.54 (n=13)": over 13 finished tasks, yours have been costing about 54% of what the plugin would otherwise have estimated.
- **The cheap model's bias.** If the cheap-model estimate is on (`l3_mode`), it learns how much that model tends to underestimate steps and corrects for it.

During a task it also corrects itself live: after every model call, it re-estimates the remaining cost from how fast this task is actually growing, how far through its to-do list it is, and whether it is stuck (repeated failures, re-editing the same file). A task that the budget stops counts only as a lower bound, so stopping never teaches the plugin to quote lower.

In testing, a 30-file task was first quoted at $0.01–$0.02 and actually cost $0.053. After that one run, the same request was quoted at $0.011–$0.14.

What it does **not** do:

- **It does not change its own code or rules.** The rules and default settings change only with a new plugin release, which you install through the catalog.
- **It does not learn across users.** Each user's records stay in their own `<HERMES_HOME>/plugin-data/cost-forecast/` and are never uploaded. A new user starts from the defaults, so the first few quotes are the least accurate.
- **It does not learn costs from declined quotes.** Those only help set when a quote should appear.

## Replying to a quote

| You reply | What happens |
|---|---|
| `continue`, `yes`, `ok`, `继续`, `好` | The task runs with the cap raised to cover the quoted range |
| `cancel`, `no`, `取消`, `算了` | Nothing runs |
| `cap $8`, `上限 $8` | Sets this task's cap. If the cap is still below the quote, you are asked again |
| Anything else | Treated as a new request. The old quote is dropped |

Only short replies count as answers. A sentence like "ok, but only change the first file" is treated as a new request and quoted again. A quote you don't answer expires after 30 minutes.

## Settings

Change these in Hermes Desktop under **Settings → Plugins → cost-forecast**, or with `hermes config set plugins.entries.cost-forecast.settings.<key> <value>`. From the command line, quote `off` (`'"off"'`), because YAML reads a bare `off` as false. The plugin treats false as off anyway.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Turns quotes and budget stops on or off |
| `quote_threshold_usd` | `0.5` | Show a quote when the likely (P50) cost is at least this much |
| `task_cap_usd` | `3.0` | Soft cap per task. Hermes asks for your OK when a task is projected to go over it |
| `task_hard_limit_usd` | `10.0` | Hard limit per task. Only your reply can raise it |
| `daily_cap_usd` | `20.0` | All tasks stop once today's spend reaches this |
| `l3_mode` | `off` | Cheap-model estimate: `auto` asks a cheap model to size tasks that the rules and your history can't size; `always` asks it for every message |
| `l3_model` | empty | The cheap model. If empty, one is picked for your provider: claude-haiku-5-5, gpt-5-mini, gemini-3.1-flash-lite or deepseek-v4-flash |

The cheap-model estimate needs the **model override** permission, which Hermes asks you for when you enable the plugin. Without it, the setting does nothing. Each cheap-model call costs a fraction of a cent, and that cost is counted in your spend.

Type `/forecast` in any chat to see today's spend and the current task's spend, projection and limits.

## Using it with hermes-telemetry

[hermes-telemetry](https://github.com/nujovich/hermes-telemetry) reports what you have already spent. This plugin estimates what a task will cost. The two work together: we tested both installed on Hermes 0.23.0, and quotes, approvals and both plugins' commands all worked side by side.

- **Commands:** `/budget` and `/stats` belong to hermes-telemetry; `/forecast` belongs to this plugin.
- **Two daily limits:** hermes-telemetry ships with its own global budget ($5 a day in `enforce` mode), separate from this plugin's `daily_cap_usd` ($20). Whichever is lower stops tasks first. Raise one or turn one off so you have a single daily limit.
- **Totals can differ:** each plugin prices calls from its own price table. In our test, the per-call costs matched except for the first call of a session, where the two plugins priced the cache write differently.

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
hermes config set auth.adopt_external_logins false   # stop Hermes from borrowing Claude Code's login
hermes auth logout anthropic                          # remove the subscription sign-in
hermes auth add anthropic --type api-key              # paste the key at the prompt
hermes config set model.provider anthropic            # "auto" does not pick up a key added this way
hermes auth list anthropic                            # should list one api_key and no oauth entries
```

Removing the subscription sign-in is not enough on its own. By default Hermes adopts your Claude Code login as a backup credential, and it switches to that login when the API key gets a 401 or 429 error, such as when your balance runs out. You get no notice when this happens. `auth.adopt_external_logins false` turns this off.

## Limits

- **Estimates are ranges, not guarantees.** The same task can cost very different amounts from one run to the next, so the plugin quotes a range and checks it against your spending as the task runs. Also set a monthly spend limit in your provider's console.
- **Stops happen between steps.** A model call that has already started runs to the end, so a task can go slightly over a limit on its last call.
- **`/yolo` and `approvals.mode: off` auto-approve the soft cap.** The hard limit and the daily cap still apply.
- **Cron, webhook and API-server runs get no quote,** because nobody is there to reply. The hard limit and the daily cap still stop them.
- **Image generation and other per-call charges** are counted but not yet priced. The same goes for provider-side tool fees, such as Anthropic's per-search web search charge.
- **Cache writes are priced at the standard (5-minute) rate.** Anthropic charges more for 1-hour cache writes and 1.1x for US-only inference; neither is reflected yet.
- **Prices for Anthropic's current models were checked against the official pricing page on 2026-11-06.** Other providers' prices come from models.dev as published.

## What the plugin does on your machine

- **Network:** fetches public model prices from `https://models.dev/api.json`, at most once a day, in a background thread. If you turn on the cheap-model estimate (`l3_mode`, off by default), your message is also sent through Hermes to your own model provider, in one extra short call to a cheap model. Nothing else leaves your machine. There is no telemetry, and the plugin never downloads or replaces its own code.
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
