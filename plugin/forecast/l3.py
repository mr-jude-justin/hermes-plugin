"""L3: ask a cheap model how many steps the task will take (DESIGN.md section 7).

Self-prediction is weak (2604.22750: Pearson r ≤ 0.39, systematic
under-estimation; TokenCast: its 90% intervals cover only 52.7%), but it reads
meaning that rules miss ("refactor this function" vs "refactor the project").
So it is only a signal:

- it is asked for *steps*, not tokens; tokens still come from the segment model;
- its step count is corrected by a learned under-estimation factor and enters
  the log-space P50 with weight W_L3;
- its interval is ignored; the P90 spread stays the L1/L2 one;
- any failure (no `llm`, timeout, bad JSON) falls back silently.

The harness builds `llm` from `ctx.llm.complete` with the user's provider's cheap
model and prices the call; we only pass the cost through to `estimate_cost_usd`.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from statistics import median
from typing import Any, Callable, Iterable, Optional

from . import defaults as D
from .features import get, message_of

W_L3 = 0.3
PRIOR_CORRECTION = 1.25  # models under-estimate (2604.22750); magnitude is a guess
CORRECTION_K0 = 5.0
AUTO_MAX_L1_CONFIDENCE = 0.4
TYPE_OVERRIDE_MAX_L1_CONFIDENCE = 0.3
MAX_TOKENS = 300
TIMEOUT_S = 4.0
MAX_MESSAGE_CHARS = 1500

LLM = Callable[..., Optional[tuple]]

PROMPT = """You estimate how much work an AI coding/research agent (Hermes) will do for one user request.
One "step" is one model call; the agent makes one call per tool use (read a file, run a command, search, edit) plus a final answer.
Typical: a greeting or simple question = 1-2 steps; a small script = 2-4; a single-file fix = 5-12; a multi-file bug fix = 15-30; a large refactor or new module = 40-150.

Context: {context}
User request:
<<<
{message}
>>>

Reply with JSON only, no prose:
{{"task_type": one of {types}, "tier": "S"|"M"|"L"|"XL", "steps_p50": int, "steps_p90": int, "images": int}}"""


@dataclass
class L3Guess:
    task_type: Optional[str]
    tier: Optional[str]
    steps_p50: float
    steps_p90: float
    images: int
    cost_usd: float


def build_prompt(features: Any) -> str:
    repo = get(features, "repo_size", None)
    lines = (repo.get("lines", 0) if isinstance(repo, dict) else getattr(repo, "lines", 0)) if repo else 0
    ctx = []
    ctx.append("inside a git repository" if get(features, "cwd_is_git_repo", False) else "not in a git repository")
    if lines:
        ctx.append(f"repository has ~{int(lines):,} lines")
    ctx.append(f"agent model: {get(features, 'model', '') or 'unknown'}")
    return PROMPT.format(
        context="; ".join(ctx),
        message=message_of(features)[:MAX_MESSAGE_CHARS],
        types=" | ".join(f'"{t}"' for t in D.TASK_TYPES),
    )


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse(text: str, cost: Optional[float]) -> Optional[L3Guess]:
    m = _JSON.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        p50 = float(d.get("steps_p50"))
        p90 = float(d.get("steps_p90", p50 * 2))
    except (ValueError, TypeError, AttributeError):
        return None
    if not (math.isfinite(p50) and math.isfinite(p90)) or p50 <= 0:
        return None
    p50 = min(max(p50, 1.0), 500.0)
    p90 = min(max(p90, p50), 1000.0)
    t = d.get("task_type")
    tier = d.get("tier")

    def _int(x: Any) -> int:
        try:
            return max(0, min(int(x), 50))
        except (TypeError, ValueError):
            return 0

    return L3Guess(
        task_type=t if t in D.TASK_TYPES else None,
        tier=tier if tier in D.TIERS else None,
        steps_p50=p50,
        steps_p90=p90,
        images=_int(d.get("images", 0)),
        cost_usd=float(cost or 0.0),
    )


def normalize_mode(mode: Any) -> str:
    """"auto" | "always" | "off". Anything else (None, False, "False", typos) is off,
    because L3 spends the user's money."""
    m = str(mode).strip().lower() if mode is not None else "off"
    return m if m in ("auto", "always") else "off"


def should_run(mode: str, message: str, l1_confidence: float, has_neighbours: bool) -> bool:
    mode = normalize_mode(mode)
    if not message.strip() or mode == "off":
        return False
    if mode == "always":
        return True
    return not has_neighbours and l1_confidence < AUTO_MAX_L1_CONFIDENCE


def ask(llm: Optional[LLM], features: Any) -> Optional[L3Guess]:
    if llm is None:
        return None
    try:
        reply = llm(build_prompt(features), max_tokens=MAX_TOKENS, timeout_s=TIMEOUT_S)
    except Exception:
        return None
    if not reply:
        return None
    text, cost = (reply[0], reply[1] if len(reply) > 1 else None) if isinstance(reply, tuple) else (reply, None)
    return parse(str(text), cost)


def correction(records: Iterable[dict]) -> float:
    """Multiplier for L3 step guesses: median actual/L3 steps, shrunk toward the prior."""
    ratios = [math.log(r["steps"] / r["l3k"]) for r in records
              if r.get("outcome") == "completed" and (r.get("l3k") or 0) > 0 and (r.get("steps") or 0) > 0]
    prior = math.log(PRIOR_CORRECTION)
    if not ratios:
        return PRIOR_CORRECTION
    lam = len(ratios) / (len(ratios) + CORRECTION_K0)
    return math.exp(lam * median(ratios) + (1 - lam) * prior)


def blend_steps(k50: float, k90: float, guess: L3Guess, corr: float) -> tuple[float, float]:
    """New (P50, P90) steps. The P90/P50 spread is kept from the draft."""
    spread = k90 / k50 if k50 > 0 else 2.5
    l3 = guess.steps_p50 * corr
    new50 = math.exp((1 - W_L3) * math.log(max(k50, 0.5)) + W_L3 * math.log(l3))
    return new50, new50 * spread
