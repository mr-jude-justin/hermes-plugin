# Hermes Cost Forecast

Before a task starts, this plugin tells you roughly what it will cost. While the task runs, it stops and asks you if the cost is about to go over your budget.

> Work in progress for the Hermes Agent Plugin Hackathon. Install and usage instructions come with the first release.

## Who it is for: API-key users only

The plugin forecasts and tracks **dollar cost from per-token API billing**. It supports Hermes setups that connect to a model provider with an **API key**, for example `hermes auth add anthropic --api-key ...`.

It does **not** support subscription logins, such as signing in to Hermes with a Claude Pro/Max account over OAuth.

## Risk notice: do not run Hermes on a Claude subscription

As of 2026, Anthropic's terms limit subscription OAuth sign-in to Anthropic's own products (Claude Code and claude.ai). Third-party tools, Hermes included, are expected to use API keys from the Anthropic Console.

If Hermes is signed in with your Claude subscription:

- **Terms of service:** that use may break Anthropic's terms, and your account could be restricted.
- **Shared quota:** Hermes draws on the same 5-hour and weekly limits as your Claude Code and claude.ai use, so a long Hermes task can lock you out of your own tools.
- **No cost data:** subscriptions report usage as a percentage of a quota, not as a cost per call, so this plugin has nothing to forecast or check against.

Check how Hermes is signed in with `hermes auth list anthropic`. To switch to an API key:

```bash
hermes auth logout anthropic
hermes auth add anthropic --api-key <your Anthropic Console key>
```

We also recommend setting a monthly spend limit in your provider's console. This plugin's forecasts are estimates, not a billing guarantee.
