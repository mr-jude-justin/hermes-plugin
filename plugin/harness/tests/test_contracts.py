import typing

from contracts import (
    OUTCOMES, TASK_TYPES, TIERS, Estimate, Outcome, PastTask, PlanItems, SubagentUsage, TaskFeatures,
    TaskType, Tier, Tokens, ToolStep, Usage, from_dict, to_dict,
)

# A real `usage` dict from post_api_request (Hermes v0.23.0, probe run on 2026-11-06).
HERMES_USAGE = {
    "input_tokens": 4, "output_tokens": 316, "cache_read_tokens": 83182, "cache_write_tokens": 910,
    "reasoning_tokens": 0, "request_count": 1, "prompt_tokens": 84096, "total_tokens": 84412,
}


def test_literals_match_tuples():
    assert typing.get_args(TaskType) == TASK_TYPES
    assert typing.get_args(Tier) == TIERS
    assert typing.get_args(Outcome) == OUTCOMES


def test_usage_from_hermes():
    u = Usage.from_hermes(HERMES_USAGE, model="claude-haiku-5-5")
    assert u.tokens == Tokens(input=4, cache_read=83182, cache_write=910, output=316)
    assert u.prompt_tokens == 84096
    assert u.calls == 1
    assert u.cost_usd is None
    assert u.model == "claude-haiku-5-5"


def test_usage_from_hermes_missing_fields():
    assert Usage.from_hermes({}).tokens.total == 0


def test_usage_add():
    a = Usage.from_hermes(HERMES_USAGE, model="m")
    b = Usage(output_tokens=10, calls=1, cost_usd=0.5, model="m")
    s = a + b
    assert s.output_tokens == 326
    assert s.calls == 2
    assert s.cost_usd == 0.5
    assert s.model == "m"
    assert (a + Usage(model="other")).model is None
    assert (a + Usage()).cost_usd is None


def test_tokens_add_and_total():
    t = Tokens(1, 2, 3, 4) + Tokens(1, 1, 1, 1)
    assert t == Tokens(2, 3, 4, 5)
    assert t.total == 14


def test_past_task_round_trip():
    features = TaskFeatures(
        message="", model="claude-opus-5-5", enabled_toolsets=["terminal", "todo"],
        step_usages=[Usage.from_hermes(HERMES_USAGE)],
        step_tools=[ToolStep("read_file", ok=True, path="plugin/contracts.py")],
        plan_items=PlanItems(total=5, done=2),
        subagents=[SubagentUsage("child-1", role="leaf", usage=Usage(output_tokens=7))],
    )
    estimate = Estimate(tier="L", task_type="agentic_coding", tokens_p50=Tokens(1, 2, 3, 4),
                        per_call_units={"image_gen": 2}, cost_usd_p50=2.2)
    task = PastTask(features, estimate, actual=None, steps=0, model="claude-opus-5-5", outcome="declined")
    again = from_dict(PastTask, to_dict(task))
    assert again == task
    assert isinstance(again.features.step_tools[0], ToolStep)
    assert isinstance(again.features.subagents[0].usage, Usage)


def test_from_dict_ignores_unknown_fields():
    data = to_dict(Estimate(tier="M"))
    data["field_from_the_future"] = 1
    assert from_dict(Estimate, data).tier == "M"


def test_usage_from_tokens_sets_prompt_size():
    u = Usage.from_tokens(Tokens(input=10.4, cache_read=80_000, cache_write=900, output=300.6), model="m")
    assert (u.input_tokens, u.output_tokens, u.prompt_tokens, u.calls) == (10, 301, 80_910, 1)


def test_estimate_shape_round_trips():
    e = Estimate(shape={"start_prompt": 68000, "growth": 1200, "output_per_step": 300,
                        "cache_hit": 0.95, "warm_cache": False})
    assert from_dict(Estimate, to_dict(e)) == e
