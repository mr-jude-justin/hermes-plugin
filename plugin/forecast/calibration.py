"""Calibration of launch estimates against what actually happened (DESIGN.md 4, 6.5 item 3).

Derived from the L2 records, each of which stores the launch estimate's P50/P90
totals next to the realised total. Nothing extra is persisted.

Bias: r = log(actual / P50) over completed tasks. The correction is the median
of r, shrunk toward 0 by n/(n+K0). Each task type then gets its own offset from
the global median, shrunk the same way (a two-level hierarchy).

P90: a split-conformal quantile of the de-biased residuals, taken at a level
adapted online with ACI (Gibbs & Candès): α ← α + γ·(0.1 − miss), replayed
over the records in time order. While data is scarce, the spread is blended
with the prior spread (P90/P50 of the draft estimate).

`stopped` tasks are right-censored: they never enter the bias. They count as a
P90 miss only when they had already exceeded P90; otherwise they are skipped.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Iterable

MIN_N = 3
K0 = 10.0
GAMMA = 0.05
TARGET_MISS = 0.1
MIN_SPREAD = math.log(1.3)


def _quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    # Conformal finite-sample correction: ceil((n+1)q)-th smallest.
    i = min(len(xs) - 1, max(0, math.ceil((len(xs) + 1) * q) - 1))
    return xs[i]


class Calibration:
    def __init__(self, records: Iterable[dict]):
        recs = sorted((r for r in records if (r.get("p50") or 0) > 0 and (r.get("total") or 0) > 0),
                      key=lambda r: r.get("ts", 0))
        done = [r for r in recs if r.get("outcome") == "completed"]
        self.n = len(done)

        # Bias and spread are measured against the pre-calibration draft ("d50").
        # Records written before drafts were stored fall back to the issued P50.
        res = [(r.get("type", ""), math.log(r["total"] / (r.get("d50") or r["p50"]))) for r in done]
        all_r = [x for _, x in res]
        med = median(all_r) if all_r else 0.0
        lam = self.n / (self.n + K0)
        self.global_bias = lam * med

        by_type: dict[str, list[float]] = {}
        for t, x in res:
            by_type.setdefault(t, []).append(x)
        self.type_offset = {
            t: (len(xs) / (len(xs) + K0)) * (median(xs) - med)
            for t, xs in by_type.items() if len(xs) >= 2
        }
        self.residuals = [x - self.bias(t) for t, x in res]

        alpha = TARGET_MISS
        self.scored = 0
        for r in recs:
            p90 = r.get("p90") or 0
            if p90 <= 0:
                continue
            miss = r["total"] > p90
            if r.get("outcome") == "stopped" and not miss:
                continue  # censored: we don't know whether it would have ended inside
            alpha = min(0.5, max(0.005, alpha + GAMMA * (TARGET_MISS - float(miss))))
            self.scored += 1
        self.alpha = alpha

    @property
    def active(self) -> bool:
        return self.n >= MIN_N

    @property
    def level(self) -> float:
        return 1.0 - self.alpha

    def bias(self, task_type: str) -> float:
        """Log multiplier for the P50 total."""
        return self.global_bias + self.type_offset.get(task_type, 0.0)

    def spread(self, prior_spread: float) -> float:
        """Log ratio P90/P50 after calibration."""
        if not self.active:
            return max(MIN_SPREAD, prior_spread)
        lam = self.n / (self.n + K0)
        q = _quantile(self.residuals, self.level)
        # P90 sits above the median; residuals are already centred by the bias.
        return max(MIN_SPREAD, lam * q + (1 - lam) * prior_spread)

    def describe(self, task_type: str) -> str:
        return f"calibrated x{math.exp(self.bias(task_type)):.2f}, P90 level {self.level:.2f} (n={self.n})"
