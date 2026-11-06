"""User-facing text: quotes, stops and the re-injected request.

Hermes shows plugin approvals under a generic "Hermes wants to run a command" header,
so every budget message starts with the 💰 marker to read as a budget notice, not a
security warning.
"""

from __future__ import annotations

from typing import Optional

try:
    from ..contracts import Estimate
except ImportError:
    from contracts import Estimate

_TYPE_LABEL = {
    "chat": "Chat", "coding_chat": "Coding question", "code_reasoning": "Code reasoning",
    "agentic_coding": "Agentic coding", "research": "Research", "generation": "Generation",
    "delegated": "Multi-agent", "browser": "Browser", "long_doc": "Long document",
}


def usd(x: Optional[float]) -> str:
    if x is None:
        return "?"
    if x < 0.095:
        return f"${x:.3f}"
    return f"${x:.2f}" if x < 10 else f"${x:.0f}"


def _range(e: Estimate) -> str:
    if e.cost_usd_p50 is None:
        total = e.tokens_p50.total
        return f"about {total / 1000:.0f}k tokens (no price data for this model)"
    low, high = usd(e.cost_usd_p50), usd(e.cost_usd_p90)
    return low if low == high else f"{low}–{high}"


def _steps(n: float) -> str:
    n = max(1, round(n or 0))
    return f"{n} step" if n == 1 else f"{n} steps"


def quote(e: Estimate, cap_usd: float, price_source: str) -> str:
    kind = _TYPE_LABEL.get(e.task_type, e.task_type)
    over = ""
    if e.cost_usd_p90 is not None and e.cost_usd_p90 > cap_usd:
        over = f" That can go over your {usd(cap_usd)} per-task cap."
    return (
        f"💰 Cost estimate: {kind}, tier {e.tier}, likely {_range(e)} (about {_steps(e.est_steps)}).{over}\n"
        f"Basis: {e.basis or 'rules'}. Prices: {price_source}.\n"
        f"Reply **continue** to run it, **cancel** to drop it, or **cap $N** to set this task's cap."
    )


def cancelled() -> str:
    return "💰 Cancelled. Nothing was run."


def cap_still_low(e: Estimate, cap_usd: float) -> str:
    return (f"💰 Cap set to {usd(cap_usd)}, but this task likely costs up to {usd(e.cost_usd_p90)}. "
            f"Reply **continue** to run it anyway, or **cap $N** with a higher cap.")


def reinject(message: str) -> str:
    return ("The user approved the cost estimate for their previous request, which was held back "
            "before you could see it. Carry out that request now. The request was:\n\n" + message)


def soft_stop(spent: float, projected: float, cap: float, new_cap: float) -> str:
    return (f"💰 Budget check: spent {usd(spent)} so far, now projected at {usd(projected)}, over the "
            f"{usd(cap)} cap. Allow to continue up to {usd(new_cap)}.")


def hard_stop(spent: float, ceiling: float) -> str:
    return (f"💰 Budget limit reached: this task has spent {usd(spent)} of its {usd(ceiling)} hard limit. "
            "Do not call any more tools. Tell the user what is finished, what is left, and that they can "
            "reply continue to resume with a new budget.")


def daily_stop(spent_today: float, daily_cap: float) -> str:
    return (f"💰 Daily budget reached: {usd(spent_today)} of {usd(daily_cap)} spent today. Do not call any "
            "more tools. Tell the user the daily budget is used up.")


def resume_context(cap: float) -> str:
    return (f"The user approved more budget (new cap {usd(cap)}). Continue the task you were working on "
            "before the budget stop, starting where you left off.")
