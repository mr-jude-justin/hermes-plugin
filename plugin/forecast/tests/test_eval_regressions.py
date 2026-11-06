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
    ("write a palindrome function, code only", False, "agentic_coding", "S"),  # real: 2 calls
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
