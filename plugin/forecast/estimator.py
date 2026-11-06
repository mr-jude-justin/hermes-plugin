"""Public forecast entry points (docs/plan.md 5.1 / 5.2, types in plugin/contracts.py).

    configure(state, history=None)  once, from the plugin's register(); cheap,
                                    no I/O (state is read lazily on first use)
    estimate(features) -> Estimate  L1 rules, refined by L2 history when available
    record_outcome(features, estimate, actual, steps, outcome)
                                    feeds finished tasks into the L2 index
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

try:  # imported as part of the plugin package (Hermes, harness)
    from ..contracts import Estimate, Tokens
except ImportError:  # imported as top-level `forecast` (forecast tests)
    from contracts import Estimate, Tokens  # type: ignore[no-redef]

from . import defaults as D
from . import history as H
from . import l3 as L3
from . import midtask as RF
from .features import message_of
from .rules import L1Result, classify
from .tokens import TokenSplit, project, step_splits

_index: Optional[H.L2Index] = None
_history: Any = None
_llm: Any = None
_l3_mode: str = "auto"


def configure(state: Any, history: Any = None, llm: Any = None, l3_mode: str = "auto") -> None:
    """Bind the harness-provided storage and (optionally) the cheap-model callable.

    Does no I/O. `llm(prompt, *, max_tokens, timeout_s) -> (text, cost_usd) | None`;
    `llm=None` turns L3 off. `l3_mode`: "auto" (only when L1/L2 are unsure),
    "always" (every non-empty message, for demos) or "off".
    """
    global _index, _history, _llm, _l3_mode
    _index = H.L2Index(state) if state is not None else None
    _history = history
    _llm = llm
    _l3_mode = l3_mode if l3_mode in ("auto", "always", "off") else "auto"


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


def _solve_steps(target: float, start: float, g: float, o: float, h: float, warm: bool) -> float:
    """Step count K whose projected total equals `target` (total is increasing in K)."""
    lo, hi = 0.05, 5000.0
    if project(lo, start, g, o, h, warm).total >= target:
        return lo
    for _ in range(60):
        mid = (lo + hi) / 2
        if project(mid, start, g, o, h, warm).total < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def estimate(features: Any, state: Any = None, llm: Any = None, l3_mode: Optional[str] = None) -> Estimate:
    """Pre-task forecast. `features` is a `contracts.TaskFeatures` (a dict also works).

    L1 rules → L2 history (if similar past tasks exist) → L3 cheap model (if
    enabled and L1/L2 are unsure) → calibration (if enough finished tasks exist).
    `llm` / `l3_mode` override what `configure` set (used by tests).
    """
    r = classify(features)
    idx = _resolve_index(state)

    # Draft: main-agent steps and shape, from L1 or L2.
    k50, k90 = r.steps_p50, r.steps_p90
    g, o, h = r.growth, r.output_per_step, r.cache_hit
    task_type = r.task_type
    units = dict(r.per_call_units)
    layer, basis, confidence = "L1", r.basis, r.confidence
    has_neighbours = False
    if idx is not None:
        neigh = idx.neighbours(message_of(features), r.task_type)
        b = H.blend(neigh, r.steps_p50, r.steps_p90, r.growth, r.output_per_step, r.cache_hit)
        if b is not None:
            has_neighbours = True
            k50, k90 = b.steps_p50, b.steps_p90
            g, o, h = b.growth, b.output_per_step, b.cache_hit
            layer = "L2"
            basis = f"{b.n} similar past task{'s' if b.n != 1 else ''} (top sim {b.top_sim:.2f}, weight {b.lam:.2f}) + {r.basis}"
            confidence = round(min(0.8, max(r.confidence, 0.25 + 0.5 * b.lam)), 2)

    # L3: a cheap model's step guess, as one more signal (l3.py).
    l3_cost, l3_steps = 0.0, None
    use_llm = llm if llm is not None else _llm
    # The harness's llm carries a live `.mode`, so a settings change applies
    # without a restart; an explicit argument wins.
    mode = l3_mode or getattr(use_llm, "mode", None) or _l3_mode
    if use_llm is not None and L3.should_run(mode, message_of(features), r.confidence, has_neighbours):
        guess = L3.ask(use_llm, features)
        if guess is not None:
            l3_cost, l3_steps = guess.cost_usd, guess.steps_p50
            corr = L3.correction(idx.records if idx is not None else [])
            k50, k90 = L3.blend_steps(k50, k90, guess, corr)
            if guess.task_type and r.confidence <= L3.TYPE_OVERRIDE_MAX_L1_CONFIDENCE and guess.task_type != task_type:
                # L1 was unsure what kind of task this is; take the model's reading
                # and move halfway to that type's per-step shape.
                shp = D.TYPE_SHAPES[guess.task_type]
                g, o = (g + shp.growth) / 2, (o + shp.output) / 2
                task_type = guess.task_type
            if task_type == "generation" and guess.images > units.get("image_gen", 1):
                units["image_gen"] = guess.images
            if layer == "L1":
                layer = "L3"
            basis = f"cheap model ~{guess.steps_p50:g} steps (x{corr:.2f} corrected, weight {L3.W_L3}) + {basis}"

    # Subagent segments from L1 (delegated tasks) sit on top of the main agent.
    l1main50 = project(r.steps_p50, r.start_prompt, r.growth, r.output_per_step, r.cache_hit, r.warm_cache)
    l1main90 = project(r.steps_p90, r.start_prompt, r.growth, r.output_per_step, r.cache_hit, r.warm_cache)
    extra50 = TokenSplit(*(max(0.0, a - m) for a, m in zip(_parts(r.tokens_p50), _parts(l1main50))))
    extra90 = TokenSplit(*(max(0.0, a - m) for a, m in zip(_parts(r.tokens_p90), _parts(l1main90))))

    # Calibration: correct the P50 total for systematic bias and set the P90
    # spread from past coverage, by moving the step counts so that `shape` and
    # `projected_steps` (what the harness prices) stay consistent.
    args = (r.start_prompt, g, o, h, r.warm_cache)
    draft50 = project(k50, *args).total + extra50.total
    draft90 = project(k90, *args).total + extra90.total
    cal = idx.calibration() if idx is not None else None
    if cal is not None and cal.active:
        t50 = draft50 * math.exp(cal.bias(task_type))
        t90 = t50 * math.exp(cal.spread(math.log(max(draft90, draft50 * 1.01) / draft50)))
        k50 = _solve_steps(max(1.0, t50 - extra50.total), *args)
        k90 = max(k50 * 1.05, _solve_steps(max(1.0, t90 - extra90.total), *args))
        # layer_used keeps the contract's L1 | L2 | L3 values; calibration shows in basis.
        basis = f"{basis}; {cal.describe(task_type)}"

    p50 = project(k50, r.start_prompt, g, o, h, r.warm_cache) + extra50
    p90 = project(k90, r.start_prompt, g, o, h, r.warm_cache) + extra90
    shape = _shape_dict(r, g, o, h)
    # Pre-calibration totals: calibration measures its bias against these, not
    # against its own output (otherwise the correction would undo itself).
    shape["draft_p50"] = round(draft50)
    shape["draft_p90"] = round(draft90)
    if l3_steps is not None:
        shape["l3_steps"] = l3_steps  # raw guess, so its bias can be learned
    return _build(
        tier=H.tier_for_steps(k50),
        task_type=task_type,
        tokens_p50=_tokens(p50),
        tokens_p90=_tokens(p90),
        per_call_units=units,
        est_steps=round(k50, 1),
        est_steps_p90=round(k90, 1),
        confidence=confidence,
        basis=basis,
        layer_used=layer,
        estimate_cost_usd=round(l3_cost, 6),
        shape=shape,
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
    rec = H.make_record(features, task_type, actual, steps, outcome, estimate)
    if rec is not None:
        idx.add(rec)
