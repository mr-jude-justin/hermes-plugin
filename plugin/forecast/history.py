"""L2: match the new task against the user's own finished tasks.

The index lives in `ForecastState["l2_records"]` as a JSON list. Each record
keeps only hashed n-gram counts of the message plus the realised cost shape:

    {"v": {bucket: count}, "type": str, "model": str, "steps": int,
     "l1": start prompt, "g": growth/step, "o": output/step, "h": cache hit,
     "total": total tokens, "outcome": "completed" | "stopped", "ts": float}

What transfers from neighbours is the *shape*: steps K and the per-step g, o, h.
Absolute tokens do not transfer, since they depend on the start prompt. The
shape is then re-projected with the current task's start prompt (tokens.project),
so a change of floor or model context carries over correctly.

Blending with the L1 prior happens in log space, with shrinkage
λ = S / (S + K0), where S is the sum of neighbour similarities. Few or weak
neighbours leave the estimate close to L1 (DESIGN.md 3 and 6.5 item 1).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional

from . import defaults as D
from .calibration import Calibration
from .features import approx_tokens, get, message_of
from .textvec import cosine, document_frequency, hashed_counts, tfidf
from .tokens import TokenSplit, project

KEY = "l2_records"
MAX_RECORDS = 500
TOP_K = 8
MIN_SIM = 0.2
SHRINK_K0 = 2.0  # similarity mass at which history and prior weigh equally
SPREAD_K0 = 5.0  # spread needs more data than the centre
SAME_TYPE_MIN = 3
CENSOR_INFLATE = 1.5  # stopped tasks ran at least this long; assume ~1.5x more
CENSOR_WEIGHT = 0.5
MIN_P90_RATIO = 1.3


def _start_prompt(features: Any) -> float:
    steps = get(features, "step_usages", []) or []
    if steps:
        first = steps[0]
        p = get(first, "prompt_tokens", 0) or 0
        if p:
            return float(p)
    ctx = int(get(features, "context_tokens_now", 0) or 0)
    return float((ctx or D.HERMES_FLOOR_TOKENS) + approx_tokens(message_of(features)))


def _usage_parts(actual: Any) -> tuple[float, float, float, float, float]:
    inp = float(get(actual, "input_tokens", 0) or 0)
    cr = float(get(actual, "cache_read_tokens", 0) or 0)
    cw = float(get(actual, "cache_write_tokens", 0) or 0)
    out = float(get(actual, "output_tokens", 0) or 0)
    prompt = float(get(actual, "prompt_tokens", 0) or 0) or (inp + cr + cw)
    return inp, cr, cw, out, prompt


def _est_total(estimate: Any, name: str) -> float:
    t = get(estimate, name, None) if estimate is not None else None
    if t is None:
        return 0.0
    total = getattr(t, "total", None)
    if total is None and isinstance(t, dict):
        total = sum(float(t.get(k, 0) or 0) for k in ("input", "cache_read", "cache_write", "output"))
    return float(total or 0)


def _draft(estimate: Any, key: str) -> float:
    shape = get(estimate, "shape", None) if estimate is not None else None
    return float((shape or {}).get(key, 0) or 0)


def make_record(features: Any, task_type: str, actual: Any, steps: int, outcome: str,
                estimate: Any = None) -> Optional[dict]:
    counts = hashed_counts(message_of(features))
    k = int(steps or get(actual, "calls", 0) or 0)
    if not counts or k <= 0:
        return None
    inp, cr, _cw, out, prompt = _usage_parts(actual)
    l1 = _start_prompt(features)
    g = 2.0 * (prompt - k * l1) / (k * (k - 1)) if k > 1 else 0.0
    return {
        "v": counts,
        "type": task_type,
        "model": str(get(features, "model", "") or ""),
        "steps": k,
        "l1": round(l1),
        "g": round(max(0.0, g), 1),
        "o": round(out / k, 1),
        "h": round(cr / (cr + inp), 4) if (cr + inp) > 0 else 0.0,
        "total": round(prompt + out),
        # Launch estimate, for calibration (calibration.py): what was issued
        # (coverage) and the pre-calibration draft (bias and spread).
        "p50": round(_est_total(estimate, "tokens_p50")),
        "p90": round(_est_total(estimate, "tokens_p90")),
        "d50": round(_draft(estimate, "draft_p50") or _est_total(estimate, "tokens_p50")),
        "d90": round(_draft(estimate, "draft_p90") or _est_total(estimate, "tokens_p90")),
        "l3k": _draft(estimate, "l3_steps"),  # cheap-model step guess, 0 if L3 did not run
        "outcome": outcome,
        "ts": time.time(),
    }


class L2Index:
    """In-memory view of `l2_records`, loaded lazily from ForecastState."""

    def __init__(self, state: Any = None):
        self.state = state
        self._records: Optional[list[dict]] = None
        self._vecs: Optional[list[dict]] = None
        self._df: Any = None
        self._cal: Optional[Calibration] = None

    @property
    def records(self) -> list[dict]:
        if self._records is None:
            raw = self.state.get(KEY, []) if self.state is not None else []
            self._records = list(raw or [])
        return self._records

    def _invalidate(self) -> None:
        self._vecs = None
        self._df = None
        self._cal = None

    def calibration(self) -> "Calibration":
        if self._cal is None:
            self._cal = Calibration(self.records)
        return self._cal

    def add(self, record: dict) -> None:
        recs = self.records
        recs.append(record)
        if len(recs) > MAX_RECORDS:
            del recs[: len(recs) - MAX_RECORDS]
        self._invalidate()
        if self.state is not None:
            self.state.set(KEY, recs)  # one write per finished task

    def _ensure_vectors(self) -> None:
        if self._vecs is not None:
            return
        df, n = document_frequency(r["v"] for r in self.records)
        self._df = (df, n)
        self._vecs = [tfidf(r["v"], df, n) for r in self.records]

    def neighbours(self, message: str, task_type: str, k: int = TOP_K) -> list[tuple[float, dict]]:
        if not self.records:
            return []
        self._ensure_vectors()
        df, n = self._df
        q = tfidf(hashed_counts(message), df, n)
        if not q:
            return []
        scored = [
            (cosine(q, v), r)
            for v, r in zip(self._vecs or [], self.records)
            if r.get("outcome") in ("completed", "stopped")
        ]
        scored = [(s, r) for s, r in scored if s >= MIN_SIM]
        same = [(s, r) for s, r in scored if r.get("type") == task_type]
        pool = same if len(same) >= SAME_TYPE_MIN else scored
        pool.sort(key=lambda sr: -sr[0])
        return pool[:k]


def _wquantile(pairs: list[tuple[float, float]], q: float) -> float:
    """Weighted quantile of (value, weight) pairs."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= q * total:
            return v
    return pairs[-1][0]


@dataclass
class L2Blend:
    n: int
    top_sim: float
    lam: float
    steps_p50: float
    steps_p90: float
    growth: float
    output_per_step: float
    cache_hit: float


def blend(neigh: list[tuple[float, dict]], prior_k50: float, prior_k90: float,
          prior_g: float, prior_o: float, prior_h: float) -> Optional[L2Blend]:
    if not neigh:
        return None
    logk: list[tuple[float, float]] = []
    gs, os_, hs = [], [], []
    mass = 0.0
    for sim, r in neigh:
        k = float(r["steps"])
        w = sim
        if r.get("outcome") == "stopped":
            k *= CENSOR_INFLATE
            w *= CENSOR_WEIGHT
        logk.append((math.log(max(k, 1.0)), w))
        gs.append((float(r.get("g", prior_g)), w))
        os_.append((float(r.get("o", prior_o)), w))
        hs.append((float(r.get("h", prior_h)), w))
        mass += w

    lam = mass / (mass + SHRINK_K0)
    lam_s = len(neigh) / (len(neigh) + SPREAD_K0)

    h50 = _wquantile(logk, 0.5)
    h90 = _wquantile(logk, 0.9)
    log50 = lam * h50 + (1 - lam) * math.log(prior_k50)
    spread = lam_s * (h90 - h50) + (1 - lam_s) * math.log(prior_k90 / prior_k50)
    spread = max(spread, math.log(MIN_P90_RATIO))

    def mix(pairs: list[tuple[float, float]], prior: float) -> float:
        return lam * _wquantile(pairs, 0.5) + (1 - lam) * prior

    return L2Blend(
        n=len(neigh),
        top_sim=neigh[0][0],
        lam=lam,
        steps_p50=math.exp(log50),
        steps_p90=math.exp(log50 + spread),
        growth=mix(gs, prior_g),
        output_per_step=mix(os_, prior_o),
        cache_hit=min(1.0, max(0.0, mix(hs, prior_h))),
    )


# Tier boundaries: geometric midpoints between tier step medians.
def tier_for_steps(k: float) -> str:
    meds = [D.TIER_STEPS[t][0] for t in D.TIERS]
    for t, a, b in zip(D.TIERS, meds, meds[1:]):
        if k < math.sqrt(a * b):
            return t
    return D.TIERS[-1]


def reproject(b: L2Blend, start_prompt: float, warm: bool) -> tuple[TokenSplit, TokenSplit]:
    p50 = project(b.steps_p50, start_prompt, b.growth, b.output_per_step, b.cache_hit, warm)
    p90 = project(b.steps_p90, start_prompt, b.growth, b.output_per_step, b.cache_hit, warm)
    return p50, p90
