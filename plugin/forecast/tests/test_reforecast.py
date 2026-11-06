import pytest

from contracts import PlanItems, SubagentUsage, TaskFeatures, ToolStep, Usage
from forecast import estimate, projected_steps, reforecast
from forecast import midtask as RF
from forecast.tokens import step_splits


def tf(message="fix the failing test in tests/test_api.py", **kw):
    base = dict(model="claude-opus-5-5", context_tokens_now=70_000, cwd_is_git_repo=True,
                enabled_toolsets=["terminal", "file"])
    return TaskFeatures(message=message, **{**base, **kw})


def run(steps, l1=70_000, g=3_000, o=1_000, h=0.9):
    """Per-call Usage of a simulated run."""
    out = []
    for s in step_splits(steps, l1, g, o, h):
        prompt = s.input + s.cache_read + s.cache_write
        out.append(Usage(input_tokens=int(s.input), cache_read_tokens=int(s.cache_read),
                         cache_write_tokens=int(s.cache_write), output_tokens=int(s.output),
                         prompt_tokens=int(prompt), calls=1))
    return out


def total(us):
    acc = Usage()
    for u in us:
        acc = acc + u
    return acc


def at(features, original, calls, k, **kw):
    done = calls[:k]
    f = features
    for name, value in kw.items():
        setattr(f, name, value)
    f.step_usages = done
    return reforecast(f, original, total(done), k, done[-5:])


def test_basic_contract_and_floor_at_spent():
    f = tf()
    orig = estimate(f)
    calls = run(30)
    e = at(f, orig, calls, 10)
    spent = total(calls[:10])
    assert e.layer_used == "reforecast"
    assert e.tokens_p50.total >= spent.tokens.total
    assert e.tokens_p90.total >= e.tokens_p50.total
    assert e.est_steps > 10 and e.est_steps_p90 > e.est_steps
    assert e.shape["steps_done"] == 10
    assert e.basis.startswith("reforecast: step 10")


def test_past_the_original_median_still_expects_more():
    f = tf()
    orig = estimate(f)  # P50 about 20 steps
    calls = run(60)
    e = at(f, orig, calls, 40)
    assert e.est_steps > 41


def test_hot_run_raises_the_forecast():
    f = tf()
    orig = estimate(f)
    hot = run(30, g=12_000)  # context grows 4x faster than the prior assumes
    cool = run(30, g=1_000)
    e_hot, e_cool = at(tf(), orig, hot, 8), at(tf(), orig, cool, 8)
    assert e_hot.tokens_p50.total > e_cool.tokens_p50.total
    assert e_hot.shape["growth"] > e_cool.shape["growth"]


def test_plan_items_steer_remaining_steps():
    orig = estimate(tf())
    calls = run(40)
    early = at(tf(), orig, calls, 4, plan_items=PlanItems(total=10, done=2))
    late = at(tf(), orig, calls, 4, plan_items=PlanItems(total=10, done=9))
    assert early.est_steps > late.est_steps
    assert "plan 2/10" in early.basis


def test_runaway_signals_raise_and_widen():
    orig = estimate(tf())
    calls = run(40)
    calm = at(tf(), orig, calls, 10)
    loops = [ToolStep("read_file", True, "src/a.py"), ToolStep("patch", True, "src/a.py"),
             ToolStep("terminal", False), ToolStep("read_file", True, "src/a.py"),
             ToolStep("patch", True, "src/a.py"), ToolStep("terminal", False)]
    stuck = at(tf(), orig, calls, 10, no_progress_calls=4, step_tools=loops)
    assert stuck.est_steps > calm.est_steps
    assert (stuck.est_steps_p90 - stuck.est_steps) > (calm.est_steps_p90 - calm.est_steps)
    assert "no-progress" in stuck.basis and "repeated file touches" in stuck.basis


def test_stage_detection():
    reads = [ToolStep("read_file"), ToolStep("search_files"), ToolStep("read_file"), ToolStep("read_file")]
    edits = reads + [ToolStep("patch"), ToolStep("write_file"), ToolStep("patch")]
    tests = edits + [ToolStep("terminal"), ToolStep("read_file"), ToolStep("terminal")]
    assert RF.detect_stage(reads) == "explore"
    assert RF.detect_stage(edits) == "fix"
    assert RF.detect_stage(tests) == "validate"
    assert RF.detect_stage([ToolStep("read_file")]) == "setup"
    assert RF.detect_stage([]) is None


def test_later_stage_means_fewer_remaining():
    orig = estimate(tf())
    calls = run(40)
    explore = [ToolStep("read_file")] * 6
    validate = [ToolStep("patch")] * 3 + [ToolStep("terminal"), ToolStep("read_file"), ToolStep("terminal")]
    e1 = at(tf(), orig, calls, 8, step_tools=explore)
    e2 = at(tf(), orig, calls, 8, step_tools=validate)
    assert e2.est_steps < e1.est_steps


def test_error_shrinks_as_task_progresses():
    # True run: 14 calls, 7 plan items done evenly. The original says L (~20).
    true_k = 14
    calls = run(true_k, g=4_000)
    true_total = total(calls).tokens.total
    orig = estimate(tf())
    errs = []
    for k in (3, 7, 11):
        e = at(tf(), orig, calls, k, plan_items=PlanItems(total=7, done=k // 2))
        errs.append(abs(e.tokens_p50.total - true_total) / true_total)
    launch_err = abs(orig.tokens_p50.total - true_total) / true_total
    # Every mid-run forecast beats the launch quote and stays within 25%.
    assert all(err < launch_err for err in errs), (launch_err, errs)
    assert max(errs) < 0.25, errs


def test_projected_steps_cover_only_remaining():
    orig = estimate(tf())
    calls = run(30)
    e = at(tf(), orig, calls, 10)
    spent = total(calls[:10]).tokens.total
    rem = projected_steps(e)
    assert sum(t.total for t in rem) == pytest.approx(e.tokens_p50.total - spent, rel=0.03)
    assert len(rem) == pytest.approx(e.est_steps - 10, abs=1)
    # first remaining call is bigger than the last observed one (context keeps growing)
    last = calls[9]
    assert rem[0].input + rem[0].cache_read + rem[0].cache_write > last.prompt_tokens


def test_pending_subagents_are_kept_until_they_start():
    f = tf("用 3 个子 agent 并行调研")
    orig = estimate(f)
    calls = run(10)
    pending = at(tf("用 3 个子 agent 并行调研"), orig, calls, 3)
    started = at(tf("用 3 个子 agent 并行调研"), orig, calls, 3,
                 subagents=[SubagentUsage("c1"), SubagentUsage("c2"), SubagentUsage("c3")])
    assert "subagents" in pending.shape and "subagents" not in started.shape
    assert pending.tokens_p50.total > started.tokens_p50.total


def test_no_observed_steps_falls_back_to_original_shape():
    orig = estimate(tf())
    e = reforecast(tf(), orig, Usage(), 0, [])
    assert e.tokens_p50.total > 0
    assert e.shape["growth"] == pytest.approx(orig.shape["growth"], rel=0.01)


def test_confidence_rises_with_progress():
    orig = estimate(tf())
    calls = run(30)
    c = [at(tf(), orig, calls, k, plan_items=PlanItems(total=6, done=min(5, k // 4))).confidence for k in (2, 10, 20)]
    assert c[0] <= c[1] <= c[2] and c[2] > orig.confidence


def test_conditional_remaining_monotone():
    a = RF.conditional_remaining(20, 50, 5)
    b = RF.conditional_remaining(20, 50, 30)
    assert a[0] >= 1 and b[0] >= 1
    assert 5 + a[0] < 30 + b[0]
    assert a[1] > a[0]
