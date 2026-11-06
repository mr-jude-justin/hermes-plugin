import json
import time

import pytest

from contracts import Estimate, TaskFeatures, Usage
from forecast import configure, estimate, projected_steps, record_outcome
from forecast import estimator as E
from forecast import history as H
from forecast.textvec import cosine, document_frequency, hashed_counts, tfidf
from forecast.tokens import project, step_splits


class MemState:
    """Stand-in for the harness ForecastState (JSON values, counts writes)."""

    def __init__(self):
        self.d = {}
        self.writes = 0
        self.reads = 0

    def get(self, key, default=None):
        self.reads += 1
        return json.loads(self.d[key]) if key in self.d else default

    def set(self, key, value):
        self.writes += 1
        self.d[key] = json.dumps(value)


def tf(message, **kw):
    base = dict(model="claude-opus-5-5", context_tokens_now=70_000, cwd_is_git_repo=True,
                enabled_toolsets=["terminal", "file", "web"])
    return TaskFeatures(message=message, **{**base, **kw})


def actual_for(steps, l1=70_000, g=3_000, o=1_000, h=0.9):
    s = project(steps, l1, g, o, h)
    prompt = s.input + s.cache_read + s.cache_write
    return Usage(input_tokens=int(s.input), cache_read_tokens=int(s.cache_read),
                 cache_write_tokens=int(s.cache_write), output_tokens=int(s.output),
                 prompt_tokens=int(prompt), calls=int(steps), model="claude-opus-5-5")


def feed(state, message, steps, n=1, outcome="completed", **kw):
    for _ in range(n):
        f = tf(message)
        record_outcome(f, estimate(f, state=state), actual_for(steps, **kw), steps, outcome, state=state)


# --- text vectors -------------------------------------------------------------

def test_hashing_is_stable_and_textless():
    a = hashed_counts("修复 parser 里的 bug")
    assert a == hashed_counts("修复 parser 里的 bug")
    assert all(k.isdigit() for k in a)  # only bucket ids, no text
    assert "parser" not in json.dumps(a)


def test_similar_messages_score_higher():
    docs = [hashed_counts(m) for m in ["fix the parser bug", "fix the lexer bug", "draw a cat", "调研最新论文"]]
    df, n = document_frequency(docs)
    q = tfidf(hashed_counts("fix a bug in the parser"), df, n)
    sims = [cosine(q, tfidf(d, df, n)) for d in docs]
    assert sims[0] > sims[1] > sims[2]
    assert sims[3] < 0.1


# --- L2 behaviour -------------------------------------------------------------

def test_no_history_is_plain_l1():
    e = estimate(tf("fix the failing test in tests/test_api.py"), state=MemState())
    assert e.layer_used == "L1"


def test_history_pulls_steps_toward_user_reality():
    state = MemState()
    msg = "fix the failing test in tests/test_api.py"
    prior = estimate(tf(msg), state=state)  # L1 says L tier, about 20 steps
    feed(state, "fix the failing test in tests/test_api.py", steps=5, n=6)
    e = estimate(tf(msg), state=state)
    assert e.layer_used == "L2"
    assert e.est_steps < prior.est_steps
    assert 5 <= e.est_steps < 12
    assert e.tokens_p50.total < prior.tokens_p50.total
    assert "similar past task" in e.basis
    assert e.confidence > prior.confidence


def test_shrinkage_one_neighbour_moves_less_than_many():
    msg = "fix the failing test in tests/test_api.py"
    one, many = MemState(), MemState()
    feed(one, msg, steps=4, n=1)
    feed(many, msg, steps=4, n=8)
    e1, e8 = estimate(tf(msg), state=one), estimate(tf(msg), state=many)
    assert e8.est_steps < e1.est_steps < 20


def test_unrelated_history_is_ignored():
    state = MemState()
    feed(state, "帮我画三张猫的头像", steps=1, n=10)
    e = estimate(tf("refactor the payment service into modules"), state=state)
    assert e.layer_used == "L1"


def test_declined_is_not_recorded_and_stopped_is_lower_bound():
    state = MemState()
    f = tf("migrate the database layer to sqlalchemy")
    record_outcome(f, estimate(f, state=state), None, 0, "declined", state=state)
    record_outcome(f, estimate(f, state=state), actual_for(10), 10, "declined", state=state)
    assert state.writes == 0

    done, stopped = MemState(), MemState()
    feed(done, "migrate the database layer to sqlalchemy", steps=10, n=5)
    feed(stopped, "migrate the database layer to sqlalchemy", steps=10, n=5, outcome="stopped")
    assert estimate(f, state=stopped).est_steps > estimate(f, state=done).est_steps


def test_record_stores_shape_not_text():
    state = MemState()
    feed(state, "secret project codename BLUEFALCON refactor", steps=6)
    raw = state.d[H.KEY]
    assert "BLUEFALCON" not in raw and "bluefalcon" not in raw
    rec = json.loads(raw)[0]
    assert rec["steps"] == 6 and rec["outcome"] == "completed"
    assert rec["g"] == pytest.approx(3_000, rel=0.01)
    assert rec["h"] == pytest.approx(0.9, abs=0.01)


def test_one_write_per_outcome_and_lazy_load():
    state = MemState()
    configure(state)
    try:
        assert state.reads == 0  # configure does no I/O
        f = tf("add tests for the cli")
        record_outcome(f, estimate(f), actual_for(8), 8)
        assert state.writes == 1
        record_outcome(f, estimate(f), actual_for(8), 8)
        assert state.writes == 2
        assert estimate(f).layer_used == "L2"
    finally:
        configure(None)


def test_index_is_capped():
    state = MemState()
    idx = H.L2Index(state)
    for i in range(H.MAX_RECORDS + 20):
        idx.add({"v": {"1": 1}, "type": "chat", "steps": 1, "outcome": "completed", "ts": i})
    assert len(json.loads(state.d[H.KEY])) == H.MAX_RECORDS


def test_estimate_fast_with_full_index():
    state = MemState()
    idx = H.L2Index(state)
    words = ["parser", "lexer", "api", "cli", "db", "auth", "cache", "queue", "ui", "docs"]
    for i in range(H.MAX_RECORDS):
        idx.records.append(H.make_record(
            tf(f"fix the {words[i % 10]} bug number {i} in module {words[(i * 3) % 10]}"),
            "agentic_coding", actual_for(5 + i % 30), 5 + i % 30, "completed"))
    f = tf("fix the parser bug in module api")
    estimate(f, state=idx)  # warm the vector cache
    t0 = time.perf_counter()
    for _ in range(20):
        e = estimate(f, state=idx)
    assert (time.perf_counter() - t0) / 20 * 1000 < 50
    assert e.layer_used == "L2"


def test_tier_follows_steps():
    assert H.tier_for_steps(2) == "S"
    assert H.tier_for_steps(6) == "M"
    assert H.tier_for_steps(25) == "L"
    assert H.tier_for_steps(80) == "XL"


# --- per-step projection ------------------------------------------------------

def test_step_splits_sum_to_project():
    for k in (1, 2, 5, 20):
        s = step_splits(k, 70_000, 3_000, 1_000, 0.9)
        assert len(s) == k
        assert sum(x.total for x in s) == pytest.approx(project(k, 70_000, 3_000, 1_000, 0.9).total)
    # prompt grows each step, so later calls hit higher context tiers
    s = step_splits(5, 70_000, 30_000, 1_000, 0.9)
    prompts = [x.input + x.cache_read + x.cache_write for x in s]
    assert prompts == sorted(prompts) and prompts[-1] == pytest.approx(190_000)


@pytest.mark.parametrize("msg", [
    "fix the failing test in tests/test_api.py",
    "你好",
    "用 3 个子 agent 并行调研",
])
def test_projected_steps_match_estimate(msg):
    e = estimate(tf(msg))
    for which, tot, k in (("p50", e.tokens_p50.total, e.est_steps), ("p90", e.tokens_p90.total, e.est_steps_p90)):
        steps = projected_steps(e, which)
        assert len(steps) >= k
        assert sum(t.total for t in steps) == pytest.approx(tot, rel=0.02)


def test_projected_steps_usable_for_pricing():
    e = estimate(tf("fix the failing test in tests/test_api.py"))
    u = Usage.from_tokens(projected_steps(e)[0], "claude-opus-5-5")
    assert u.prompt_tokens == u.input_tokens + u.cache_read_tokens + u.cache_write_tokens > 0


def test_projected_steps_without_shape():
    assert projected_steps(Estimate()) == []
    with pytest.raises(ValueError):
        projected_steps(Estimate(), "p99")
