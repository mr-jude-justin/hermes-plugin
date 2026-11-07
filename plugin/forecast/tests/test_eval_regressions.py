"""Regressions from Justin's first real-money eval (Sonnet, Anthropic API key, 2026-11-06).

Real outcomes are noted per case. These pin the L1 reading, not exact numbers.
"""

import json

import pytest

from contracts import TaskFeatures, Usage
from forecast import classify, estimate, projected_steps, record_outcome


def feat(message, repo=False):
    return {"message": message, "model": "claude-sonnet-5", "context_tokens_now": 65_000, "cwd_is_git_repo": repo}


@pytest.mark.parametrize("message,repo,want_type,want_tier", [
    ("Explain TCP vs UDP in 3 bullets", False, "chat", "S"),              # real: 1 call
    ("read inventory.py and explain the bug", True, "code_reasoning", "S"),  # real: 2 calls
    ("write a palindrome function, code only", False, "coding_chat", "S"),  # real: 2 calls
    ("Write a Python function is_palindrome(s). Code only.", True, "coding_chat", "S"),  # same, asked in a repo
    ("Explain TCP vs UDP in 3 bullets ", True, "chat", "S"),  # being in a repo is not code context
    ("add in_stock() and a one-line check", True, "agentic_coding", "M"),  # real: 6 calls (was chat/S)
    ("use 2 subagents to each write a haiku", False, "delegated", "M"),   # main agent no longer L
])
def test_eval_cases(message, repo, want_type, want_tier):
    r = classify(feat(message, repo))
    assert (r.task_type, r.tier) == (want_type, want_tier)


def test_generic_explain_needs_code_context():
    assert classify(feat("解释一下量子纠缠")).task_type == "chat"
    assert classify(feat("why does getUser return None here?", repo=True)).task_type == "code_reasoning"
    assert classify(feat("explain how the auth module works in src/auth/login.py", repo=True)).task_type == "code_reasoning"


def test_identifiers_count_as_code():
    assert "code identifier" in classify(feat("rename fetch_rows to load_rows")).signals
    assert "code identifier" not in classify(feat("what is the capital of France")).signals


class MemState:
    def __init__(self):
        self.d = {}

    def get(self, key, default=None):
        return json.loads(self.d[key]) if key in self.d else default

    def set(self, key, value):
        self.d[key] = json.dumps(value)


def usage_with_total(total):
    return Usage(input_tokens=int(total * 0.08), cache_read_tokens=int(total * 0.8),
                 cache_write_tokens=int(total * 0.1), output_tokens=int(total * 0.02),
                 prompt_tokens=int(total * 0.98), calls=6)


def test_calibration_scales_subagents_too_and_keeps_steps_sane():
    """Justin's eval: delegated est_steps came out as 0.1 after calibration."""
    st = MemState()
    msgs = ["use 2 subagents to each write a haiku about rain", "use 2 subagents to write haikus about the sea",
            "use 3 subagents to research cats in parallel", "use 2 subagents to summarise two files in parallel",
            "split this up between 2 subagents and write limericks"]
    for m in msgs * 2:
        f = TaskFeatures(message=m, model="claude-sonnet-5", context_tokens_now=65_000)
        e = estimate(f, state=st)
        record_outcome(f, e, usage_with_total(e.tokens_p50.total * 0.15), 6, state=st)  # tasks ~7x cheaper
    e = estimate(TaskFeatures(message="use 2 subagents to each write a haiku", model="claude-sonnet-5",
                              context_tokens_now=65_000), state=st)
    assert "calibrated" in e.basis
    assert e.est_steps >= 1.0
    assert e.est_steps_p90 > e.est_steps
    # The interval did not collapse. With a big per-call floor, tokens grow almost linearly
    # in steps, so check the spread in tokens, not steps.
    assert e.tokens_p90.total / e.tokens_p50.total > 1.25
    for which, tot in (("p50", e.tokens_p50.total), ("p90", e.tokens_p90.total)):
        assert sum(t.total for t in projected_steps(e, which)) == pytest.approx(tot, rel=0.03)


# --- Fresh-task eval, 2026-11-07 (eval/live/2026-11-07-fresh-tasks.json), exact prompts ---

SPANISH = ('In plugin/harness/replies.py, add the Spanish approval words "sí" and "continuar" and the cancel '
           'word "cancelar". Add tests for them in plugin/harness/tests/test_replies.py, then run '
           '`uvx pytest -q plugin/harness/tests` and make sure everything passes.')


def test_everything_passes_is_not_broad_scope_and_command_paths_are_not_files():
    # Real: 7 calls. L1 used to read it as XL / 60 steps.
    r = classify(feat(SPANISH, repo=True))
    assert r.task_type == "agentic_coding" and r.tier == "M"
    assert "broad scope" not in r.signals and "many files" not in r.signals
    assert "2 path(s)" in r.signals  # replies.py + test_replies.py; not the dir inside `uvx pytest ...`


def test_paths_inside_code_are_not_counted():
    from forecast.features import count_paths
    assert count_paths("edit src/a.py, then run `pytest tests/unit` and ```\ncat docs/x.md\n```") == 1


def test_tool_using_task_never_drops_below_two_steps():
    """Real: explain_pricing read pricing.py and answered (2 calls); calibration had shrunk it to 1.3."""
    st = MemState()
    for m in ["explain how cost_of works in plugin/pricing.py", "explain the parser in src/parse.py",
              "read src/cli.py and explain the flags", "explain what store.py does"]:
        f = TaskFeatures(message=m, model="claude-sonnet-5", context_tokens_now=65_000, cwd_is_git_repo=True)
        e = estimate(f, state=st)
        record_outcome(f, e, usage_with_total(e.tokens_p50.total * 0.2), 2, state=st)  # pushes calibration down hard
    e = estimate(TaskFeatures(message="Explain how unknown providers are priced in plugin/pricing.py",
                              model="claude-sonnet-5", context_tokens_now=65_000, cwd_is_git_repo=True), state=st)
    assert "calibrated" in e.basis
    assert e.est_steps >= 2.0
    assert e.est_steps_p90 >= e.est_steps + 1.0
    for which, tot in (("p50", e.tokens_p50.total), ("p90", e.tokens_p90.total)):
        assert sum(t.total for t in projected_steps(e, which)) == pytest.approx(tot, rel=0.03)


def test_chat_can_still_be_one_step():
    st = MemState()
    for m in ["hi", "hello there", "thanks!", "good morning"]:
        f = TaskFeatures(message=m, model="claude-sonnet-5", context_tokens_now=65_000)
        e = estimate(f, state=st)
        record_outcome(f, e, usage_with_total(e.tokens_p50.total * 0.3), 1, state=st)
    assert estimate(TaskFeatures(message="hey", model="claude-sonnet-5"), state=st).est_steps < 2.0
