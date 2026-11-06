"""Hermes cost forecast plugin: quote a task's cost before it runs and stop for
confirmation when it would go over budget. See plugin/README.md and docs/plan.md.

Wiring only. The harness (hooks, ledger, budget) is in `harness/`, prices in
`pricing.py`, the forecaster in `forecast/`.
"""

from . import forecast
from .harness.core import Harness, Settings
from .harness.store import Store
from .pricing import PriceBook

PLUGIN_ID = "cost-forecast"

HOOKS = (
    "pre_llm_call", "post_api_request", "api_request_error", "pre_tool_call", "post_tool_call",
    "subagent_start", "subagent_stop", "post_llm_call", "on_session_end",
)


def register(ctx):
    # Local files only: register() must not touch the network.
    from plugins.plugin_storage import plugin_data_dir, plugin_db

    store = Store(plugin_db(PLUGIN_ID))
    prices = PriceBook(plugin_data_dir(PLUGIN_ID))
    forecast.configure(store, store)
    harness = Harness(store=store, pricebook=prices, forecast=forecast,
                      settings=lambda: Settings.from_getter(ctx.get_config))
    for name in HOOKS:
        ctx.register_hook(name, getattr(harness, name))
    ctx.register_command("budget", harness.budget_command,
                         description="Show today's spend and the current task's budget")
