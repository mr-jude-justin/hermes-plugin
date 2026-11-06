from dataclasses import dataclass, field
import time

import pytest

from forecast import classify
from forecast.defaults import TIERS

BASE = {
    "model": "claude-opus-5-5",
    "session_turns_so_far": 0,
    "context_tokens_now": 70_000,
    "enabled_toolsets": ["terminal", "file", "web", "image_gen", "todo", "delegation"],
    "cwd_is_git_repo": False,
    "attachments": 0,
}


def feat(message, **kw):
    return {**BASE, "message": message, **kw}


def tier_ge(a, b):
    return TIERS.index(a) >= TIERS.index(b)


@pytest.mark.parametrize(
    "message,kw,expected_type",
    [
        ("你好", {}, "chat"),
        ("hi, how are you?", {}, "chat"),
        ("这段代码为什么报错？\n```python\nprint(x)\n```", {}, "coding_chat"),
        ("explain how the auth module works in src/auth/login.py", {"cwd_is_git_repo": True}, "code_reasoning"),
        ("帮我把这个项目重构成插件架构", {"cwd_is_git_repo": True}, "agentic_coding"),
        ("fix the failing tests and add a test for the parser", {"cwd_is_git_repo": True}, "agentic_coding"),
        # Justin's first live run was classified as chat:
        ("write a fib script, save and run it", {}, "agentic_coding"),
        ("write me a python program that renames photos", {}, "agentic_coding"),
        ("调研一下 2026 年最新的 agent 成本预测论文", {}, "research"),
        ("search for the latest news on hermes agent plugins", {}, "research"),
        ("帮我画三张猫的头像", {}, "generation"),
        ("generate 4 images of a red fox logo", {}, "generation"),
        ("分头并行调研这三个方向，用子 agent", {}, "delegated"),
        ("open the page and click the login button, then take a screenshot", {}, "browser"),
        ("总结这些文件的要点", {"attachments": 4}, "long_doc"),
    ],
)
def test_task_type(message, kw, expected_type):
    assert classify(feat(message, **kw)).task_type == expected_type


def test_chat_is_small_and_cheap():
    r = classify(feat("你好"))
    assert r.tier == "S"
    assert r.per_call_units == {}
    # Floor dominates: about two calls of ~70k context.
    assert 100_000 < r.tokens_p50.total < 250_000


def test_refactor_whole_repo_is_big():
    r = classify(
        feat(
            "把整个项目从零重构成插件架构，所有模块都要迁移并补测试",
            cwd_is_git_repo=True,
            repo_size={"files": 900, "lines": 120_000},
        )
    )
    assert r.task_type == "agentic_coding"
    assert r.tier == "XL"


def test_standalone_script_is_small():
    # Justin's live run: this took 2 API calls on Hermes.
    r = classify(feat("write a fib script, save and run it"))
    assert r.task_type == "agentic_coding" and r.tier == "S"
    # The same ask inside a repo stays bigger.
    assert TIERS.index(classify(feat("write a fib script, save and run it", cwd_is_git_repo=True)).tier) > 0


def test_small_fix_is_downgraded():
    r = classify(feat("quick typo fix in README.md", cwd_is_git_repo=True))
    assert r.task_type == "agentic_coding"
    assert r.tier == "M"  # default L, minus one for "quick/typo"


def test_long_context_bumps_tier():
    short = classify(feat("解释一下这个函数", context_tokens_now=60_000))
    long = classify(feat("解释一下这个函数", context_tokens_now=400_000))
    assert TIERS.index(long.tier) == TIERS.index(short.tier) + 1
    assert long.tokens_p50.total > short.tokens_p50.total


def test_image_count_units():
    assert classify(feat("帮我画三张猫的头像")).per_call_units == {"image_gen": 3}
    assert classify(feat("generate 4 images of a red fox logo")).per_call_units == {"image_gen": 4}
    assert classify(feat("draw a cat")).per_call_units == {"image_gen": 1}


def test_effort_scales_output_only():
    lo = classify(feat("implement a CLI tool", reasoning_effort="low", cwd_is_git_repo=True))
    hi = classify(feat("implement a CLI tool", reasoning_effort="high", cwd_is_git_repo=True))
    assert hi.tokens_p50.output > lo.tokens_p50.output
    assert hi.tokens_p50.cache_read == pytest.approx(lo.tokens_p50.cache_read)


def test_p90_above_p50():
    r = classify(feat("fix the bug in parser.py", cwd_is_git_repo=True))
    assert r.steps_p90 > r.steps_p50
    assert r.tokens_p90.total > r.tokens_p50.total


def test_unknown_provider_has_no_cache_split():
    r = classify(feat("你好", model="some-local-model"))
    assert r.tokens_p50.cache_read == 0 and r.tokens_p50.cache_write == 0


def test_mixed_research_then_code_has_two_weights():
    r = classify(feat("先调研一下最新的方案，然后实现到 src/app.py 里", cwd_is_git_repo=True))
    assert {"research", "agentic_coding"} <= set(r.type_weights)
    assert sum(r.type_weights.values()) == pytest.approx(1.0, abs=0.01)


def test_toolset_alone_does_not_decide_type():
    # web / image_gen are enabled for every message; a plain greeting stays chat.
    r = classify(feat("在吗"))
    assert r.task_type == "chat"


def test_accepts_dataclass_features():
    @dataclass
    class TF:
        message: str = "fix the failing test in tests/test_api.py"
        model: str = "claude-opus-5-5"
        cwd_is_git_repo: bool = True
        enabled_toolsets: list = field(default_factory=lambda: ["terminal", "file"])
        context_tokens_now: int = 80_000

    assert classify(TF()).task_type == "agentic_coding"


def test_missing_fields_use_floor():
    r = classify({"message": "hello"})
    assert r.start_prompt >= 70_000
    assert r.tier == "S"


def test_fast_enough():
    f = feat("把整个项目重构成插件架构，所有模块都要迁移并补测试 " * 20, cwd_is_git_repo=True)
    t0 = time.perf_counter()
    for _ in range(200):
        classify(f)
    per_call_ms = (time.perf_counter() - t0) / 200 * 1000
    assert per_call_ms < 5  # target < 1 ms, loose bound for slow CI
