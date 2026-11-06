"""Public forecast entry points (docs/plan.md 5.1 / 5.2, types in plugin/contracts.py).

    configure(state, history=None)  once, from the plugin's register(); cheap,
                                    no I/O (state is read lazily on first use)
    estimate(features) -> Estimate  L1 rules, refined by L2 history when available
    record_outcome(features, estimate, actual, steps, outcome)
                                    feeds finished tasks into the L2 index
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

try:  # imported as part of the plugin package (Hermes, harness)
    from ..contracts import Estimate, Tokens
except ImportError:  # imported as top-level `forecast` (forecast tests)
    from contracts import Estimate, Tokens  # type: ignore[no-redef]

from . import defaults as D
from . import history as H
from . import midtask as RF
from .features import message_of
from .rules import L1Result, classify
from .tokens import TokenSplit, project, step_splits

_index: Optional[H.L2Index] = None
_history: Any = None


def configure(state: Any, history: Any = None) -> None:
    """Bind the harness-provided storage. Does no I/O."""
    global _index, _history
    _index = H.L2Index(state) if state is not None else None
    _history = history


def _resolve_index(state: Any) -> Optional[H.L2Index]:
    if state is not None:
        return state if isinstance(state, H.L2Index) else H.L2Index(state)
    return _index


def _tokens(split: TokenSplit) -> Tokens:
    d = split.as_dict()
    return Tokens(input=d["input"], cache_read=d["cache_read"], cache_write=d["cache_write"], output=d["output"])


def _shape_dict(r: L1Result, growth: float, output: float, cache_hit: float) -> dict:
    shape = {
        "start_prompt": round(r.start_prompt),
        "growth": round(growth, 1),
        "output_per_step": round(output, 1),
        "cache_hit": round(cache_hit, 4),
        "warm_cache": r.warm_cache,
    }
    if r.subagents_p50 or r.subagents_p90:
        sub = D.TYPE_SHAPES["research"]
        shape["subagents"] = {
            "count_p50": r.subagents_p50,
            "count_p90": r.subagents_p90,
            "steps_p50": D.SUBAGENT_STEPS[0],
            "steps_p90": D.SUBAGENT_STEPS[1],
            "start_prompt": D.SUBAGENT_FLOOR_TOKENS,
            "growth": sub.growth,
            "output_per_step": sub.output,
            "cache_hit": round(r.cache_hit, 4),
        }
    return shape


def projected_steps(estimate: Any, which: str = "p50") -> list[Tokens]:
    """Expand `estimate.shape` into one Tokens per projected call (main agent first,
    then each subagent's calls). The sum matches `tokens_p50` / `tokens_p90` up to
    rounding and the fractional-step approximation. Empty if there is no shape."""
    if which not in ("p50", "p90"):
        raise ValueError("which must be 'p50' or 'p90'")
    shape = getattr(estimate, "shape", None)
    if not shape:
        return []
    k = estimate.est_steps if which == "p50" else estimate.est_steps_p90
    # A reforecast's shape covers only the calls still to come.
    k = max(0.0, k - float(shape.get("steps_done", 0) or 0))
    splits = step_splits(k, shape["start_prompt"], shape["growth"], shape["output_per_step"],
                         shape["cache_hit"], bool(shape.get("warm_cache")))
    sub = shape.get("subagents")
    if sub:
        n = float(sub["count_" + which])
        one = step_splits(sub["steps_" + which], sub["start_prompt"], sub["growth"],
                          sub["output_per_step"], sub["cache_hit"])
        while n > 0:
            f = min(1.0, n)
            splits += [s.scaled(f) for s in one] if f < 1.0 else one
            n -= 1
    return [_tokens(s) for s in splits]


_ESTIMATE_FIELDS = {f.name for f in dataclasses.fields(Estimate)}


def _build(**kw: Any) -> Estimate:
    # Forward compatible: only pass fields the current contract has (e.g. `shape`).
    return Estimate(**{k: v for k, v in kw.items() if k in _ESTIMATE_FIELDS})


def to_estimate(r: L1Result) -> Estimate:
    return _build(
        tier=r.tier,
        task_type=r.task_type,
        tokens_p50=_tokens(r.tokens_p50),
        tokens_p90=_tokens(r.tokens_p90),
        per_call_units=dict(r.per_call_units),
        est_steps=r.steps_p50,
        est_steps_p90=r.steps_p90,
        confidence=r.confidence,
        basis=r.basis,
        layer_used="L1",
        estimate_cost_usd=0.0,
        shape=_shape_dict(r, r.growth, r.output_per_step, r.cache_hit),
    )


def estimate(features: Any, state: Any = None) -> Estimate:
    """Pre-task forecast. `features` is a `contracts.TaskFeatures` (a dict also works)."""
    r = classify(features)
    idx = _resolve_index(state)
    if idx is None:
        return to_estimate(r)

    neigh = idx.neighbours(message_of(features), r.task_type)
    b = H.blend(neigh, r.steps_p50, r.steps_p90, r.growth, r.output_per_step, r.cache_hit)
    if b is None:
        return to_estimate(r)

    p50, p90 = H.reproject(b, r.start_prompt, r.warm_cache)
    # Keep L1's subagent segments (delegated tasks) on top of the re-projected main agent.
    main50 = project(r.steps_p50, r.start_prompt, r.growth, r.output_per_step, r.cache_hit, r.warm_cache)
    main90 = project(r.steps_p90, r.start_prompt, r.growth, r.output_per_step, r.cache_hit, r.warm_cache)
    extra50 = TokenSplit(*(max(0.0, a - m) for a, m in zip(_parts(r.tokens_p50), _parts(main50))))
    extra90 = TokenSplit(*(max(0.0, a - m) for a, m in zip(_parts(r.tokens_p90), _parts(main90))))

    basis = f"{b.n} similar past task{'s' if b.n != 1 else ''} (top sim {b.top_sim:.2f}, weight {b.lam:.2f}) + {r.basis}"
    return _build(
        tier=H.tier_for_steps(b.steps_p50),
        task_type=r.task_type,
        tokens_p50=_tokens(p50 + extra50),
        tokens_p90=_tokens(p90 + extra90),
        per_call_units=dict(r.per_call_units),
        est_steps=round(b.steps_p50, 1),
        est_steps_p90=round(b.steps_p90, 1),
        confidence=round(min(0.8, max(r.confidence, 0.25 + 0.5 * b.lam)), 2),
        basis=basis,
        layer_used="L2",
        estimate_cost_usd=0.0,
        shape=_shape_dict(r, b.growth, b.output_per_step, b.cache_hit),
    )


def reforecast(features: Any, original: Any, spent: Any, steps_done: int,
               recent_steps: Optional[list] = None) -> Estimate:
    """Mid-task forecast of the final total (spent + remaining).

    `tokens_p50/p90` include what is already spent and never drop below it.
    `shape` and `projected_steps(...)` cover only the remaining calls, so the
    harness prices those and adds `spent.cost_usd`.
    """
    p = RF.reforecast_parts(features, original, spent, steps_done, recent_steps or [])
    k50 = p["steps_done"] + p["rem_p50"]
    k90 = p["steps_done"] + p["rem_p90"]
    conf0 = float(getattr(original, "confidence", 0.3) or 0.3)
    return _build(
        tier=H.tier_for_steps(k50),
        task_type=p["task_type"],
        tokens_p50=_tokens(p["tokens_p50"]),
        tokens_p90=_tokens(p["tokens_p90"]),
        per_call_units=dict(getattr(original, "per_call_units", {}) or {}),
        est_steps=round(k50, 1),
        est_steps_p90=round(k90, 1),
        confidence=round(min(0.9, conf0 + (0.9 - conf0) * p["progress"]), 2),
        basis="reforecast: " + ", ".join(p["signals"]),
        layer_used="reforecast",
        estimate_cost_usd=0.0,
        shape=p["shape"],
    )


def _parts(s: TokenSplit) -> tuple[float, float, float, float]:
    return (s.input, s.cache_read, s.cache_write, s.output)


def record_outcome(features: Any, estimate: Any, actual: Any, steps: int,
                   outcome: str = "completed", state: Any = None) -> None:
    """Add a finished task to the L2 index.

    `declined` (nothing ran) and missing `actual` are skipped: they feed the
    harness's quote threshold, not the forecaster. `stopped` is kept and treated
    as a lower bound (right-censored).
    """
    if outcome == "declined" or actual is None:
        return
    idx = _resolve_index(state)
    if idx is None:
        return
    task_type = getattr(estimate, "task_type", None) or classify(features).task_type
    rec = H.make_record(features, task_type, actual, steps, outcome)
    if rec is not None:
        idx.add(rec)
