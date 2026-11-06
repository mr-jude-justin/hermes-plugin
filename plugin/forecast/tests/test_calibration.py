import json
import math
import random

import pytest

from contracts import TaskFeatures, Usage
from forecast import estimate, projected_steps, record_outcome
from forecast import history as H
from forecast.calibration import Calibration
from forecast.tokens import project


class MemState:
    def __init__(self):
        self.d, self.writes = {}, 0

    def get(self, key, default=None):
        return json.loads(self.d[key]) if key in self.d else default

    def set(self, key, value):
        self.writes += 1
        self.d[key] = json.dumps(value)


def rec(total, p50, p90=None, outcome="completed", type_="agentic_coding", ts=0.0):
    return {"v": {"1": 1}, "type": type_, "steps": 5, "total": total, "p50": p50,
            "p90": p90 if p90 is not None else p50 * 3, "outcome": outcome, "ts": ts}


def usage_with_total(total):
    # Mostly cache reads, like real Hermes calls.
    return Usage(input_tokens=int(total * 0.08), cache_read_tokens=int(total * 0.8),
                 cache_write_tokens=int(total * 0.1), output_tokens=int(total * 0.02),
                 prompt_tokens=int(total * 0.98), calls=8)


# --- Calibration object --------------------------------------------------------

def test_inactive_below_min_records():
    c = Calibration([rec(50, 100), rec(50, 100)])
    assert not c.active
    assert c.spread(math.log(3)) == pytest.approx(math.log(3))


def test_bias_recovers_systematic_overestimate_with_shrinkage():
    c = Calibration([rec(50, 100, ts=i) for i in range(30)])
    # The true ratio is 0.5; shrinkage n/(n+10) = 0.75 gives exp(0.75·ln 0.5) ≈ 0.59.
    assert math.exp(c.bias("agentic_coding")) == pytest.approx(0.5 ** 0.75, rel=0.01)
    few = Calibration([rec(50, 100, ts=i) for i in range(3)])
    assert math.exp(few.bias("agentic_coding")) > math.exp(c.bias("agentic_coding"))


def test_per_type_offset():
    recs = [rec(50, 100, type_="agentic_coding", ts=i) for i in range(15)]
    recs += [rec(200, 100, type_="research", ts=100 + i) for i in range(15)]
    c = Calibration(recs)
    assert c.bias("agentic_coding") < 0 < c.bias("research")
    assert c.bias("chat") == pytest.approx(c.global_bias)


def test_stopped_never_biases_and_only_counts_as_miss_when_over_p90():
    base = [rec(100, 100, ts=i) for i in range(10)]
    with_stop = base + [rec(10, 100, p90=300, outcome="stopped", ts=50)]  # stopped early: censored
    assert Calibration(with_stop).global_bias == pytest.approx(Calibration(base).global_bias)
    assert Calibration(with_stop).scored == Calibration(base).scored
    over = base + [rec(500, 100, p90=300, outcome="stopped", ts=50)]  # already past P90: a real miss
    assert Calibration(over).scored == Calibration(base).scored + 1
    assert Calibration(over).alpha < Calibration(base).alpha


def test_aci_widens_after_misses_and_tightens_after_hits():
    misses = Calibration([rec(1000, 100, p90=300, ts=i) for i in range(10)])
    hits = Calibration([rec(100, 100, p90=300, ts=i) for i in range(10)])
    assert misses.level > 0.9 > hits.level


def test_records_without_launch_estimate_are_ignored():
    old = [{"v": {"1": 1}, "type": "chat", "steps": 2, "total": 100, "outcome": "completed", "ts": i}
           for i in range(10)]
    assert not Calibration(old).active


# --- End to end through estimate / record_outcome -----------------------------

def tf(message):
    return TaskFeatures(message=message, model="claude-opus-5-5", context_tokens_now=70_000,
                        cwd_is_git_repo=True)


TASKS = [
    "fix the failing test in tests/test_api.py",
    "refactor the payment service into modules",
    "add a --verbose flag to the cli",
    "migrate the config loader to toml",
    "implement retry with backoff in the http client",
    "write tests for the date parser",
]


def test_estimates_shrink_when_user_tasks_are_consistently_cheaper():
    state = MemState()
    probe = tf("implement caching for the search endpoint")
    before = estimate(probe, state=state)
    for i in range(12):
        f = tf(TASKS[i % len(TASKS)])
        e = estimate(f, state=state)
        record_outcome(f, e, usage_with_total(e.tokens_p50.total * 0.4), 8, state=state)
    after = estimate(probe, state=state)
    assert after.tokens_p50.total < before.tokens_p50.total * 0.85
    assert "calibrated x" in after.basis
    assert state.writes == 12  # still one write per finished task


def test_calibrated_estimate_stays_consistent_with_projected_steps():
    state = MemState()
    for i in range(8):
        f = tf(TASKS[i % len(TASKS)])
        e = estimate(f, state=state)
        record_outcome(f, e, usage_with_total(e.tokens_p50.total * 2.0), 30, state=state)
    e = estimate(tf("implement caching for the search endpoint"), state=state)
    assert "calibrated" in e.basis
    for which, tot in (("p50", e.tokens_p50.total), ("p90", e.tokens_p90.total)):
        assert sum(t.total for t in projected_steps(e, which)) == pytest.approx(tot, rel=0.03)
    assert e.tokens_p90.total > e.tokens_p50.total


def test_coverage_converges_toward_target_on_noisy_stream():
    """Actual = P50_true × lognormal(σ=1.0), wider than the prior assumes. Coverage
    of the issued P90 over the second half should end up near 90%."""
    rng = random.Random(7)
    state = MemState()
    hits = []
    for i in range(160):
        f = tf(TASKS[i % len(TASKS)] + f" #{i % 17}")
        e = estimate(f, state=state)
        true_median = project(12, 70_000, 3_000, 1_000, 0.9).total
        actual_total = true_median * math.exp(rng.gauss(0, 1.0))
        if i >= 80:
            hits.append(actual_total <= e.tokens_p90.total)
        record_outcome(f, e, usage_with_total(actual_total), 12, state=state)
    coverage = sum(hits) / len(hits)
    assert 0.8 <= coverage <= 0.97, coverage


def test_index_calibration_cache_invalidates_on_add():
    state = MemState()
    idx = H.L2Index(state)
    c1 = idx.calibration()
    idx.add(rec(50, 100, ts=1))
    assert idx.calibration() is not c1
