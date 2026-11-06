import json

import pytest

from contracts import TASK_TYPES, TIERS, Estimate, RepoSize, TaskFeatures, from_dict, to_dict
from forecast import classify, estimate
from forecast.defaults import SUBAGENT_FLOOR_TOKENS


def tf(message, **kw):
    base = dict(
        model="claude-opus-5-5",
        context_tokens_now=70_000,
        enabled_toolsets=["terminal", "file", "web", "image_gen", "todo", "delegation"],
    )
    return TaskFeatures(message=message, **{**base, **kw})


def test_returns_contract_estimate():
    e = estimate(tf("fix the failing test in tests/test_api.py", cwd_is_git_repo=True))
    assert isinstance(e, Estimate)
    assert e.tier in TIERS and e.task_type in TASK_TYPES
    assert e.layer_used == "L1"
    assert e.estimate_cost_usd == 0.0
    assert e.cost_usd_p50 is None and e.cost_usd_p90 is None  # harness fills these
    assert 0 < e.est_steps < e.est_steps_p90
    assert 0 < e.tokens_p50.total < e.tokens_p90.total
    assert e.basis.startswith("rules:")


def test_matches_classify():
    f = tf("调研一下最新的 agent 成本预测论文")
    r, e = classify(f), estimate(f)
    assert (e.tier, e.task_type, e.est_steps) == (r.tier, r.task_type, r.steps_p50)
    assert e.tokens_p50.total == pytest.approx(r.tokens_p50.total, abs=4)


def test_token_parts_are_ints_and_serialisable():
    e = estimate(tf("帮我画三张猫的头像"))
    for part in ("input", "cache_read", "cache_write", "output"):
        assert isinstance(getattr(e.tokens_p50, part), int)
    assert e.per_call_units == {"image_gen": 3}
    back = from_dict(Estimate, json.loads(json.dumps(to_dict(e))))
    assert back == e


def test_repo_size_dataclass_is_read():
    small = estimate(tf("explain how src/auth/login.py works", cwd_is_git_repo=True))
    big = estimate(
        tf("explain how src/auth/login.py works", cwd_is_git_repo=True, repo_size=RepoSize(files=900, lines=200_000))
    )
    assert TIERS.index(big.tier) >= TIERS.index(small.tier)
    assert "large repo" in big.basis


def test_delegated_adds_subagent_floors():
    two = estimate(tf("用 2 个子 agent 并行调研"))
    five = estimate(tf("用 5 个子 agent 并行调研"))
    assert two.task_type == five.task_type == "delegated"
    # Each extra subagent pays at least its own floor once (first call, cold cache).
    assert five.tokens_p50.total - two.tokens_p50.total > 3 * SUBAGENT_FLOOR_TOKENS


def test_delegated_chinese_count():
    r = classify(tf("分头并行调研这三个方向，用子 agent"))
    assert r.subagents_p50 == 3


def test_empty_features():
    e = estimate(TaskFeatures())
    assert e.tier == "S" and e.task_type == "chat"
