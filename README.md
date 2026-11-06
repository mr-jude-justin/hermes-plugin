# Hermes cost forecast plugin

A Hermes Agent plugin that tells you roughly what a task will cost before it starts, and stops to ask when a running task is about to go over your budget.

Built for the Hermes Agent Plugin Hackathon (Nous Research).

**API-key users only.** The plugin works with per-token API billing. It does not support subscription logins such as Claude Pro/Max over OAuth. Anthropic's terms limit those logins to its own products, and subscriptions report quota percentages, not dollars. See `plugin/README.md` for the risk notice.

- `plugin/`: the plugin itself (not written yet)
- `docs/plan.md`: the plan, interface contract and timeline
- `AGENTS.md`: team rules and who owns which files
- `tools/board`: the team message board script (not part of the plugin)
