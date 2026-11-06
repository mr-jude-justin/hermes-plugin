"""Public forecast entry points (docs/plan.md 5.1 / 5.2, types in plugin/contracts.py).

`estimate()` currently runs L1 only. L2 (history) and L3 (cheap model) will be
combined here later without changing the signature.
"""

from __future__ import annotations

from typing import Any

try:  # imported as part of the plugin package (Hermes, harness)
    from ..contracts import Estimate, Tokens
except ImportError:  # imported as top-level `forecast` (forecast tests)
    from contracts import Estimate, Tokens  # type: ignore[no-redef]

from .rules import L1Result, classify
from .tokens import TokenSplit


def _tokens(split: TokenSplit) -> Tokens:
    d = split.as_dict()
    return Tokens(
        input=d["input"],
        cache_read=d["cache_read"],
        cache_write=d["cache_write"],
        output=d["output"],
    )


def to_estimate(r: L1Result) -> Estimate:
    return Estimate(
        tier=r.tier,  # type: ignore[arg-type]
        task_type=r.task_type,  # type: ignore[arg-type]
        tokens_p50=_tokens(r.tokens_p50),
        tokens_p90=_tokens(r.tokens_p90),
        per_call_units=dict(r.per_call_units),
        est_steps=r.steps_p50,
        est_steps_p90=r.steps_p90,
        confidence=r.confidence,
        basis=r.basis,
        layer_used="L1",
        estimate_cost_usd=0.0,  # L1 makes no model calls
    )


def estimate(features: Any) -> Estimate:
    """Pre-task forecast. `features` is a `contracts.TaskFeatures` (a dict also works)."""
    return to_estimate(classify(features))
