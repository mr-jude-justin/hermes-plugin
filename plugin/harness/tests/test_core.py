import json
import sqlite3
import types

import pytest

from contracts import Estimate, Tokens, Usage
from harness.core import Harness, Settings
from harness.store import Store
from pricing import PriceBook

# Model "m" costs $10 per million tokens of every kind, so 100k tokens = $1.
CATALOGUE = {"acme": {"m": {"input": 10, "output": 10, "cache_read": 10, "cache_write": 10}}}


def make(tmp_path, estimate_usd=0.1, steps=1, **settings):
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps({"fetched_at": 9e12, "prices": CATALOGUE}))
    prices = PriceBook(None, snapshot=snap)
    store = Store(sqlite3.connect(":memory:", check_same_thread=False))
    recorded = []

    def estimate(features):
        tokens = Tokens(input=estimate_usd * 100_000 * steps)
        return Estimate(tier="L", task_type="agentic_coding", tokens_p50=tokens,
                        tokens_p90=Tokens(input=tokens.input * 2), est_steps=steps, est_steps_p90=steps)

    fc = types.SimpleNamespace(estimate=estimate, record_outcome=lambda *a: recorded.append(a))
    h = Harness(store=store, pricebook=prices, forecast=fc, settings=lambda: Settings(**settings))
    return h, store, recorded


def call(h, sid, usd, provider="acme", model="m"):
    tokens = int(usd * 100_000)
    h.post_api_request(session_id=sid, model=model, provider=provider,
                       usage={"input_tokens": tokens, "prompt_tokens": tokens, "request_count": 1})


def test_small_task_runs_without_a_quote_and_is_recorded(tmp_path):
    h, store, recorded = make(tmp_path, estimate_usd=0.1)
    assert h.pre_llm_call(session_id="s", user_message="hi", model="m") is None
    call(h, "s", 0.05)
    assert h.runs["s"].spent == pytest.approx(0.05)
    h.post_llm_call(session_id="s")
    assert "s" not in h.runs
    (task,) = store.load()
    assert task.outcome == "completed" and task.features.message == "" and task.actual.input_tokens == 5000
    assert recorded[0][4] == "completed"
    assert store.spent_today() == pytest.approx(0.05)


def test_big_task_is_quoted_then_reinjected_on_continue(tmp_path):
    h, store, _ = make(tmp_path, estimate_usd=2.0)
    r = h.pre_llm_call(session_id="s", user_message="refactor everything", model="m")
    assert r["action"] == "block" and "💰" in r["message"] and "$2.00–$4.00" in r["message"]
    assert "s" not in h.runs
    r = h.pre_llm_call(session_id="s", user_message="continue", model="m")
    assert "refactor everything" in r["context"]
    run = h.runs["s"]
    assert run.cap >= 4.0 * 1.2  # the user accepted a P90 of $4
    assert store.get_pending("s") is None


def test_cancel_records_a_declined_quote(tmp_path):
    h, store, recorded = make(tmp_path, estimate_usd=2.0)
    h.pre_llm_call(session_id="s", user_message="big job", model="m")
    r = h.pre_llm_call(session_id="s", user_message="取消", model="m")
    assert r["action"] == "block" and "Cancelled" in r["message"]
    assert store.load()[0].outcome == "declined" and store.load()[0].actual is None
    assert recorded[0][4] == "declined"


def test_new_message_drops_the_quote(tmp_path):
    h, store, _ = make(tmp_path, estimate_usd=2.0)
    h.pre_llm_call(session_id="s", user_message="big job", model="m")
    r = h.pre_llm_call(session_id="s", user_message="好的，不过先只改第一个文件然后告诉我结果", model="m")
    assert r["action"] == "block"  # re-quoted as a new task
    assert store.get_pending("s").message.startswith("好的")
    assert store.load()[0].outcome == "declined"


def test_cap_reply(tmp_path):
    h, _, _ = make(tmp_path, estimate_usd=2.0)
    h.pre_llm_call(session_id="s", user_message="big job", model="m")
    low = h.pre_llm_call(session_id="s", user_message="cap $1", model="m")
    assert low["action"] == "block" and "Cap set to $1.00" in low["message"]
    ok = h.pre_llm_call(session_id="s", user_message="上限 $50", model="m")
    assert "big job" in ok["context"]
    assert h.runs["s"].ceiling >= 50


def test_soft_stop_asks_once_per_cap_level(tmp_path):
    h, _, _ = make(tmp_path, estimate_usd=0.1, steps=4, task_cap_usd=1.0, task_hard_limit_usd=10.0)
    h.pre_llm_call(session_id="s", user_message="hi", model="m")
    call(h, "s", 0.9)  # run rate 0.9/step, 3 steps left -> projected 3.6 > cap 1
    r = h.pre_tool_call(tool_name="terminal", args={}, session_id="s", tool_call_id="t1")
    assert r["action"] == "approve" and r["rule_key"].startswith("cost-forecast:") and "💰" in r["message"]
    assert h.runs["s"].cap > 1.0
    assert h.pre_tool_call(tool_name="terminal", args={}, session_id="s", tool_call_id="t2") is None


def test_hard_stop_blocks_then_resumes_on_continue(tmp_path):
    h, store, _ = make(tmp_path, estimate_usd=0.1, task_hard_limit_usd=1.0, task_cap_usd=1.0)
    h.pre_llm_call(session_id="s", user_message="hi", model="m")
    call(h, "s", 1.2)
    r = h.pre_tool_call(tool_name="terminal", args={}, session_id="s", tool_call_id="t1")
    assert r["action"] == "block" and "hard limit" in r["message"]
    h.post_llm_call(session_id="s")
    assert "s" in h.stopped and store.load() == []
    r = h.pre_llm_call(session_id="s", user_message="continue", model="m")
    assert "Continue the task" in r["context"]
    assert h.runs["s"].ceiling > 1.2 and not h.runs["s"].stopped
    assert h.pre_tool_call(tool_name="terminal", args={}, session_id="s", tool_call_id="t2") is None


def test_hard_stop_then_new_message_records_stopped(tmp_path):
    h, store, _ = make(tmp_path, estimate_usd=0.1, task_hard_limit_usd=1.0, task_cap_usd=1.0)
    h.pre_llm_call(session_id="s", user_message="hi", model="m")
    call(h, "s", 1.2)
    h.pre_tool_call(tool_name="terminal", args={}, session_id="s")
    h.post_llm_call(session_id="s")
    h.pre_llm_call(session_id="s", user_message="something else entirely, please", model="m")
    assert store.load()[0].outcome == "stopped"


def test_daily_cap_blocks_new_tasks(tmp_path):
    h, store, _ = make(tmp_path, daily_cap_usd=1.0)
    store.add_spend(1.5)
    r = h.pre_llm_call(session_id="s", user_message="hi", model="m")
    assert r["action"] == "block" and "Daily budget" in r["message"]


def test_subagent_usage_is_attributed_to_the_parent(tmp_path):
    h, _, _ = make(tmp_path)
    h.pre_llm_call(session_id="p", user_message="hi", model="m")
    h.subagent_start(parent_session_id="p", child_session_id="c", child_role="leaf")
    call(h, "c", 0.3)
    h.subagent_stop(child_session_id="c", child_status="completed")
    run = h.runs["p"]
    assert run.spent == pytest.approx(0.3)
    (child,) = run.features.subagents
    assert child.usage.input_tokens == 30000 and child.status == "completed" and child.model == "m"


def test_tool_steps_plan_items_and_errors(tmp_path):
    h, _, _ = make(tmp_path)
    h.pre_llm_call(session_id="s", user_message="hi", model="m")
    for i in range(2):
        h.pre_tool_call(tool_name="read_file", args={"path": "a.py"}, session_id="s", tool_call_id=f"r{i}")
        h.post_tool_call(tool_name="read_file", result="ok", session_id="s", tool_call_id=f"r{i}")
    h.pre_tool_call(tool_name="todo_list", args={"todos": [{"status": "completed"}, {"status": "pending"}]},
                    session_id="s", tool_call_id="t")
    h.pre_tool_call(tool_name="terminal", args={"command": "x"}, session_id="s", tool_call_id="x")
    h.post_tool_call(tool_name="terminal", result="Error: boom", session_id="s", tool_call_id="x")
    f = h.runs["s"].features
    assert [s.path for s in f.step_tools[:2]] == ["a.py", "a.py"]
    assert f.plan_items.total == 2 and f.plan_items.done == 1
    assert f.tool_errors == 1 and f.step_tools[-1].ok is False


def test_hooks_never_raise(tmp_path):
    h, _, _ = make(tmp_path)
    h.forecast.estimate = lambda f: 1 / 0
    assert h.pre_llm_call(session_id="s", user_message="hi", model="m") is None
    assert h.pre_tool_call(tool_name="x", args=None, session_id="nobody") is None
    h.post_api_request(session_id="s", usage={"input_tokens": "garbage"})


def test_unpriced_model_never_quotes(tmp_path):
    h, _, _ = make(tmp_path, estimate_usd=50)
    assert h.pre_llm_call(session_id="s", user_message="hi", model="local-llama") is None


def test_forecast_command(tmp_path):
    h, _, _ = make(tmp_path)
    h.pre_llm_call(session_id="s", user_message="hi", model="m")
    text = h.forecast_command("", session_id="s")
    assert "Spent today" in text and "Current task (running)" in text


@pytest.mark.parametrize("platform", ["cron", "webhook", "api_server"])
def test_unattended_runs_skip_the_quote_but_keep_the_hard_limit(tmp_path, platform):
    h, _, _ = make(tmp_path, estimate_usd=2.0, task_hard_limit_usd=1.0, task_cap_usd=1.0)
    assert h.pre_llm_call(session_id="s", user_message="big job", model="m", platform=platform) is None
    assert h.runs["s"].ceiling == 1.0  # nobody accepted the $4 P90, so the limit stays put
    call(h, "s", 1.2)
    r = h.pre_tool_call(tool_name="terminal", args={}, session_id="s")
    assert r["action"] == "block"


def test_cron_env_counts_as_unattended(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    h, _, _ = make(tmp_path, estimate_usd=2.0)
    assert h.pre_llm_call(session_id="s", user_message="big job", model="m", platform="discord") is None


def test_estimate_cost_is_billed_once_and_counts_toward_the_task(tmp_path):
    h, store, _ = make(tmp_path, estimate_usd=0.1)
    real = h.forecast.estimate
    h.forecast.estimate = lambda f: (lambda e: (setattr(e, "estimate_cost_usd", 0.002), e)[1])(real(f))
    h.pre_llm_call(session_id="s", user_message="hi", model="m")
    assert store.spent_today() == pytest.approx(0.002)
    assert h.runs["s"].spent == pytest.approx(0.002)


def test_subagent_turns_belong_to_the_parent_task(tmp_path):
    h, store, _ = make(tmp_path, estimate_usd=2.0, quote_threshold_usd=0.5)
    h.pre_llm_call(session_id="p", user_message="big job", model="m")
    h.pre_llm_call(session_id="p", user_message="continue", model="m")
    h.subagent_start(parent_session_id="p", child_session_id="c")
    # The child's own turn is neither quoted nor tracked as its own task.
    assert h.pre_llm_call(session_id="c", user_message="write a haiku", model="m") is None
    assert "c" not in h.runs
    call(h, "c", 0.3)
    h.post_llm_call(session_id="c")
    assert "p" in h.runs and h.runs["p"].spent == pytest.approx(0.3)
    h.post_llm_call(session_id="p")
    (task,) = [t for t in store.load() if t.outcome == "completed"]
    assert task.features.subagents[0].usage.input_tokens == 30000


@pytest.mark.parametrize("x,expected", [(0.0204, 0.021), (4.83, 4.9), (13.2, 14), (0.1, 0.1), (1.0, 1.0)])
def test_round_up_keeps_two_significant_digits(x, expected):
    from harness.core import _round_up
    assert _round_up(x) == pytest.approx(expected)


def test_small_amounts_display():
    from harness.messages import usd
    assert (usd(0.005), usd(0.0204), usd(0.5), usd(12.4)) == ("$0.005", "$0.020", "$0.50", "$12")
