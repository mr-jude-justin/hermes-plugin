"""The cheap-model call the forecaster's L3 layer uses (see forecast/DESIGN.md).

The forecaster gets a plain callable and never sees `ctx` or prices:

    llm(prompt, *, max_tokens, timeout_s) -> (reply_text, usd_cost) | None

It returns None whenever L3 should not run: `l3_mode` is off, no cheap model is known
for the user's provider, the user has not granted `llm.model_override`, or the call
fails or times out. The forecaster then falls back to L1/L2.

`ctx.llm` calls do not fire `post_api_request`, so the cost returned here is the only
record of them; the harness bills it through `Estimate.estimate_cost_usd`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

try:
    from ..contracts import Usage
except ImportError:
    from contracts import Usage

log = logging.getLogger("cost-forecast")

# A cheap model on each provider. ctx.llm runs `model=` on the user's active provider,
# so the model has to belong to it.
CHEAP_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-5-mini",
    "google": "gemini-3.1-flash-lite",
    "gemini": "gemini-3.1-flash-lite",
    "deepseek": "deepseek-v4-flash",
}


class L3Client:
    def __init__(self, ctx: Any, settings: Callable[[], Any], pricebook: Any,
                 provider_hint: Callable[[], Optional[str]]):
        self._ctx, self._settings, self._prices, self._provider_hint = ctx, settings, pricebook, provider_hint

    @property
    def mode(self) -> str:
        """off | auto | always, read live so a settings change applies without a restart."""
        try:
            return str(self._settings().l3_mode or "off")
        except Exception:
            return "off"

    def model(self) -> Optional[str]:
        explicit = (self._settings().l3_model or "").strip()
        if explicit:
            return explicit
        return CHEAP_MODELS.get((self._provider_hint() or "").lower())

    def __call__(self, prompt: str, *, max_tokens: int = 400, timeout_s: float = 4.0):
        if self.mode == "off":
            return None
        model = self.model()
        if not model:
            return None
        try:
            result = self._ctx.llm.complete(
                messages=[{"role": "user", "content": prompt}], model=model,
                max_tokens=max_tokens, timeout=timeout_s, temperature=0.0, purpose="cost-forecast.l3")
        except Exception as e:  # PluginLlmTrustError without consent, timeouts, provider errors
            log.info("L3 call skipped: %s", e)
            return None
        u = getattr(result, "usage", None)
        usage = Usage(input_tokens=int(getattr(u, "input_tokens", 0) or 0),
                      output_tokens=int(getattr(u, "output_tokens", 0) or 0),
                      cache_read_tokens=int(getattr(u, "cache_read_tokens", 0) or 0),
                      cache_write_tokens=int(getattr(u, "cache_write_tokens", 0) or 0))
        usage.prompt_tokens = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
        cost = getattr(u, "cost_usd", None)
        if cost is None:
            cost = self._prices.cost_of(usage, getattr(result, "provider", None), getattr(result, "model", model))
        return getattr(result, "text", "") or "", cost
