"""Mid-task reforecast (DESIGN.md sections 2.2, 5 and 6.5).

TokenCast's biggest gain is here, not at launch: Task Update MAE −27.8% and
91.2% coverage. Given what has already been spent and how the last calls
looked, predict how much more the task will consume.

    final = spent + remaining,  remaining = segment(K_rem, L_next, ĝ, ô, ĥ)

Remaining steps K_rem combine four signals:
  1. the original estimate as a log-normal prior on K, conditioned on K > k
  2. plan items from Hermes' `todo_list` (TokenCast: "remaining plan items" is a top feature)
  3. the agentic-coding stage (setup/explore/fix/validate/closeout, 2604.22750 Table 1)
  4. runaway signals: consecutive no-progress calls, tool errors and repeated
     reads/edits of the same file (TokenCast Fig. 5; 2604.22750 Fig. 4)

A discrete-time hazard model fitted on history is the planned upgrade for (1);
see DESIGN.md 6.5 item 2.
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import NormalDist, median
from typing import Any, Optional

from . import defaults as D
from .features import get
from .tokens import TokenSplit, project

_Z90 = NormalDist().inv_cdf(0.9)

# Cumulative share of rounds at the middle of each stage (2604.22750 Table 1:
# setup 9.98, explore 30.37, fix 33.53, validate 16.59, closeout 9.53).
_STAGE_MID_CUM = {
    "setup": 0.050,
    "explore": 0.252,
    "fix": 0.571,
    "validate": 0.822,
    "closeout": 0.952,
}
_EXPLORE_TOOLS = {"read_file", "search_files", "list_files", "grep", "glob"}
_FIX_TOOLS = {"write_file", "patch", "edit_file"}
_VALIDATE_TOOLS = {"terminal"}

W_PLAN_BASE = 0.5
W_PLAN_PROGRESS = 0.4
W_STAGE = 0.25
OBS_K0 = 3.0  # observed steps at which observed and prior shape weigh equally
MAX_RUNAWAY_MULT = 2.5


def _num(x: Any, name: str) -> float:
    return float(get(x, name, 0) or 0)


def _prompt(u: Any) -> float:
    p = _num(u, "prompt_tokens")
    return p or (_num(u, "input_tokens") + _num(u, "cache_read_tokens") + _num(u, "cache_write_tokens"))


def _slope(ys: list[float]) -> float:
    n = len(ys)
    if n < 2:
        return 0.0
    xs = range(n)
    mx, my = (n - 1) / 2, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


def observed_shape(steps: list[Any]) -> Optional[dict]:
    """Growth, output, cache hit and last prompt size from recent calls."""
    steps = [s for s in steps if _prompt(s) > 0][-10:]
    if not steps:
        return None
    prompts = [_prompt(s) for s in steps]
    cr = sum(_num(s, "cache_read_tokens") for s in steps)
    inp = sum(_num(s, "input_tokens") for s in steps)
    return {
        "n": len(steps),
        "growth": max(0.0, _slope(prompts)),
        "output": median(_num(s, "output_tokens") for s in steps),
        "cache_hit": cr / (cr + inp) if (cr + inp) > 0 else None,
        "last_prompt": prompts[-1],
    }


def conditional_remaining(k50: float, k90: float, done: float) -> tuple[float, float]:
    """P50/P90 of K − done under a log-normal K, given K > done."""
    k50 = max(k50, 1.0)
    sigma = max(0.05, math.log(max(k90, k50 * 1.05) / k50) / _Z90)
    dist = NormalDist(math.log(k50), sigma)
    f = dist.cdf(math.log(done)) if done > 0 else 0.0
    f = min(f, 0.999)

    def q(p: float) -> float:
        return math.exp(dist.inv_cdf(min(0.9999, f + p * (1 - f))))

    return max(1.0, q(0.5) - done), max(1.0, q(0.9) - done)


def detect_stage(tools: list[Any]) -> Optional[str]:
    recent = [str(get(t, "tool_name", "") or "") for t in tools[-6:]]
    if not recent:
        return None
    if len(tools) <= 2 and not any(t in _FIX_TOOLS for t in recent):
        return "setup"
    # Later calls weigh more; ties go to the later stage.
    order = {"explore": 0, "fix": 1, "validate": 2}
    c: Counter = Counter()
    for i, t in enumerate(recent):
        kind = ("explore" if t in _EXPLORE_TOOLS else "fix" if t in _FIX_TOOLS
                else "validate" if t in _VALIDATE_TOOLS else None)
        if kind:
            c[kind] += i + 1
    if not c:
        return None
    stage = max(c, key=lambda s: (c[s], order[s]))
    # Tests after edits mean validate, even when reads are interleaved.
    ever_fixed = any(str(get(t, "tool_name", "")) in _FIX_TOOLS for t in tools)
    if stage == "explore" and ever_fixed and c.get("validate"):
        stage = "validate"
    return stage


def repeated_file_touches(tools: list[Any], window: int = 12) -> int:
    paths = [get(t, "path", None) for t in tools[-window:]]
    counts = Counter(p for p in paths if p)
    return sum(n - 2 for n in counts.values() if n >= 3)


def runaway_multiplier(features: Any, tools: list[Any]) -> tuple[float, list[str]]:
    sigs: list[str] = []
    mult = 1.0
    nop = int(get(features, "no_progress_calls", 0) or 0)
    if nop:
        mult *= 1 + 0.15 * nop
        sigs.append(f"{nop} no-progress calls")
    recent_fail = sum(1 for t in tools[-8:] if get(t, "ok", None) is False)
    if recent_fail >= 2:
        mult *= 1 + 0.1 * recent_fail
        sigs.append(f"{recent_fail} recent tool failures")
    rep = repeated_file_touches(tools)
    if rep:
        mult *= 1 + 0.1 * rep
        sigs.append("repeated file touches")
    retries = int(get(features, "retries", 0) or 0)
    if retries >= 2:
        mult *= 1.1
        sigs.append(f"{retries} retries")
    return min(mult, MAX_RUNAWAY_MULT), sigs


def plan_remaining(features: Any, done_steps: float, orig_k50: float) -> Optional[float]:
    plan = get(features, "plan_items", None)
    total = int(get(plan, "total", 0) or 0) if plan is not None else 0
    done = int(get(plan, "done", 0) or 0) if plan is not None else 0
    if total <= 0 or done >= total:
        return None
    per_item = done_steps / done if done > 0 else orig_k50 / total
    return max(1.0, (total - done) * max(per_item, 1.0))


def remaining_steps(features: Any, original: Any, steps_done: float,
                    task_type: str) -> tuple[float, float, list[str]]:
    sigs: list[str] = []
    k50 = float(get(original, "est_steps", 0) or D.TIER_STEPS[get(original, "tier", "M") or "M"][0])
    k90 = float(get(original, "est_steps_p90", 0) or k50 * 2.5)
    r50, r90 = conditional_remaining(k50, k90, steps_done)

    log50 = math.log(r50)

    plan = plan_remaining(features, steps_done, k50)
    if plan is not None:
        p = get(features, "plan_items")
        done, tot = int(get(p, "done", 0) or 0), int(get(p, "total", 0) or 0)
        # The more items are done, the better the observed steps-per-item rate.
        w = min(0.9, W_PLAN_BASE + W_PLAN_PROGRESS * done / tot)
        log50 = (1 - w) * log50 + w * math.log(plan)
        r90 = math.exp((1 - w) * math.log(r90) + w * math.log(plan * 1.8))
        sigs.append(f"plan {done}/{tot}")

    tools = list(get(features, "step_tools", []) or [])
    if task_type in ("agentic_coding", "code_reasoning") and steps_done >= 3:
        stage = detect_stage(tools)
        if stage:
            cum = _STAGE_MID_CUM[stage]
            stage_rem = max(1.0, steps_done / cum - steps_done)
            log50 = (1 - W_STAGE) * log50 + W_STAGE * math.log(stage_rem)
            sigs.append(f"stage {stage}")
    r50 = math.exp(log50)

    mult, run_sigs = runaway_multiplier(features, tools)
    if mult > 1.0:
        r50 *= mult
        r90 *= mult * 1.3  # uncertainty grows faster than the centre
        sigs += run_sigs
    r90 = max(r90, r50 * 1.3)
    return r50, r90, sigs


def reforecast_parts(features: Any, original: Any, spent: Any, steps_done: int,
                     recent_steps: list[Any]) -> dict:
    """Everything `estimator.reforecast` needs to build an Estimate."""
    task_type = str(get(original, "task_type", "") or "chat")
    shape0 = dict(get(original, "shape", None) or {})
    g0 = float(shape0.get("growth", D.TYPE_SHAPES.get(task_type, D.TYPE_SHAPES["chat"]).growth))
    o0 = float(shape0.get("output_per_step", D.TYPE_SHAPES.get(task_type, D.TYPE_SHAPES["chat"]).output))
    h0 = float(shape0.get("cache_hit", D.cache_hit_for(get(features, "model", None))))
    l0 = float(shape0.get("start_prompt", D.HERMES_FLOOR_TOKENS))

    steps_in = list(recent_steps or get(features, "step_usages", []) or [])
    obs = observed_shape(steps_in)
    k = float(max(0, steps_done))
    sigs: list[str] = [f"step {int(k)}"]

    if obs:
        w = obs["n"] / (obs["n"] + OBS_K0)
        g = w * obs["growth"] + (1 - w) * g0
        o = w * obs["output"] + (1 - w) * o0
        h = h0 if obs["cache_hit"] is None else w * obs["cache_hit"] + (1 - w) * h0
        l_next = obs["last_prompt"] + g
        sigs.append(f"observed {obs['n']} calls")
    else:
        g, o, h = g0, o0, h0
        l_next = l0 + g0 * k

    r50, r90, more = remaining_steps(features, original, k, task_type)
    sigs += more

    rem50 = project(r50, l_next, g, o, h, warm_cache=True)
    rem90 = project(r90, l_next, g * D.REFORECAST_GROWTH_P90_MULT, o, h, warm_cache=True)

    sp = TokenSplit(_num(spent, "input_tokens"), _num(spent, "cache_read_tokens"),
                    _num(spent, "cache_write_tokens"), _num(spent, "output_tokens"))

    # Subagents the original expected but that have not started yet.
    sub = shape0.get("subagents")
    started = len(get(features, "subagents", []) or [])
    sub_shape = None
    sub50 = sub90 = TokenSplit(0, 0, 0, 0)
    if sub and started == 0:
        sub_shape = sub
        one50 = project(sub["steps_p50"], sub["start_prompt"], sub["growth"], sub["output_per_step"], sub["cache_hit"])
        one90 = project(sub["steps_p90"], sub["start_prompt"], sub["growth"], sub["output_per_step"], sub["cache_hit"])
        sub50, sub90 = one50.scaled(sub["count_p50"]), one90.scaled(sub["count_p90"])

    return {
        "task_type": task_type,
        "steps_done": k,
        "rem_p50": r50,
        "rem_p90": r90,
        "tokens_p50": sp + rem50 + sub50,
        "tokens_p90": sp + rem90 + sub90,
        "shape": {
            "start_prompt": round(l_next),
            "growth": round(g, 1),
            "growth_p90": round(g * D.REFORECAST_GROWTH_P90_MULT, 1),
            "output_per_step": round(o, 1),
            "cache_hit": round(h, 4),
            "warm_cache": True,
            "steps_done": k,
            **({"subagents": sub_shape} if sub_shape else {}),
        },
        "progress": k / (k + r50) if (k + r50) > 0 else 0.0,
        "signals": sigs,
    }
