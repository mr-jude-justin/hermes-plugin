"""Hermes cost forecast plugin: quote a task's cost before it runs and stop for
confirmation when it would go over budget. See plugin/README.md and docs/plan.md.

Wiring only. The harness (hooks, ledger, budget) is in `harness/`, prices in
`pricing.py`, the forecaster in `forecast/`.
"""

from . import forecast
from .harness.core import Harness, Settings
from .harness.l3 import L3Client
from .harness.store import Store
from .pricing import PriceBook

PLUGIN_ID = "cost-forecast"

HOOKS = (
    "pre_llm_call", "post_api_request", "api_request_error", "pre_tool_call", "post_tool_call",
    "subagent_start", "subagent_stop", "post_llm_call", "on_session_end",
)


def _configured_provider():
    try:
        from hermes_cli.config import load_config_readonly
        provider = ((load_config_readonly() or {}).get("model") or {}).get("provider")
        return None if provider in (None, "", "auto") else provider
    except Exception:
        return None


def register(ctx):
    # Local files only: register() must not touch the network.
    from plugins.plugin_storage import plugin_data_dir, plugin_db

    store = Store(plugin_db(PLUGIN_ID))
    prices = PriceBook(plugin_data_dir(PLUGIN_ID))
    settings = lambda: Settings.from_getter(ctx.get_config)  # noqa: E731
    harness = Harness(store=store, pricebook=prices, forecast=forecast, settings=settings)
    l3 = L3Client(ctx, settings, prices, provider_hint=lambda: harness.last_provider or _configured_provider())
    try:
        # l3.mode is also readable live, so a settings change applies without a restart.
        forecast.configure(store, store, llm=l3, l3_mode=l3.mode)
    except TypeError:  # a forecaster without L3 support
        forecast.configure(store, store)
    for name in HOOKS:
        ctx.register_hook(name, getattr(harness, name))
    ctx.register_command("forecast", harness.forecast_command,
                         description="Show today's spend and the current task's budget")
