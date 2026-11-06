import json

import pytest

from contracts import TaskFeatures, Usage
from forecast import configure, estimate, projected_steps, record_outcome
from forecast import l3 as L3
from forecast.tokens import project


class MemState:
    def __init__(self):
        self.d = {}

    def get(self, key, default=None):
        return json.loads(self.d[key]) if key in self.d else default

    def set(self, key, value):
        self.d[key] = json.dumps(value)


class FakeLLM:
    def __init__(self, reply=None, cost=0.0007, raises=False):
        self.reply = reply if reply is not None else {"task_type": "agentic_coding", "tier": "L",
                                                      "steps_p50": 30, "steps_p90": 45, "images": 0}
        self.cost, self.raises, self.calls, self.prompts = cost, raises, 0, []

    def __call__(self, prompt, *, max_tokens, timeout_s):
        self.calls += 1
        self.prompts.append(prompt)
        assert max_tokens == L3.MAX_TOKENS and timeout_s == L3.TIMEOUT_S
        if self.raises:
            raise RuntimeError("PluginLlmTrustError")
        text = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
        return text, self.cost


def tf(message, **kw):
    base = dict(model="claude-opus-5-5", context_tokens_now=70_000, cwd_is_git_repo=True)
    return TaskFeatures(message=message, **{**base, **kw})


AMBIGUOUS = "can you take care of the payment thing we talked about"


# --- parsing --------------------------------------------------------------------

def test_parse_tolerates_prose_and_clamps():
    g = L3.parse('Sure! {"task_type":"research","tier":"M","steps_p50":0.2,"steps_p90":-3,"images":99} done', 0.001)
    assert g.task_type == "research" and g.tier == "M"
    assert g.steps_p50 == 1.0 and g.steps_p90 == 1.0 and g.images == 50
    assert g.cost_usd == 0.001


@pytest.mark.parametrize("text", ["", "no json here", '{"steps_p50": "many"}', '{"task_type": "chat"}',
                                  '{"steps_p50": NaN}', "{broken"])
def test_parse_rejects_garbage(text):
    assert L3.parse(text, 0.0) is None


def test_unknown_type_and_tier_are_dropped():
    g = L3.parse('{"task_type":"poetry","tier":"XXL","steps_p50":3,"steps_p90":5}', None)
    assert g.task_type is None and g.tier is None and g.cost_usd == 0.0


def test_prompt_has_message_context_and_no_secrets_beyond_it():
    p = L3.build_prompt(tf("fix the parser bug", repo_size={"files": 10, "lines": 52_000}))
    assert "fix the parser bug" in p and "52,000 lines" in p and "inside a git repository" in p
    assert "steps_p50" in p and "agentic_coding" in p


# --- when it runs ---------------------------------------------------------------

def test_auto_skips_confident_and_chat_and_runs_when_unsure():
    llm = FakeLLM()
    estimate(tf("你好"), llm=llm, l3_mode="auto")  # confident short chat
    assert llm.calls == 0
    e = estimate(tf(AMBIGUOUS), llm=llm, l3_mode="auto")
    assert llm.calls == 1
    assert e.layer_used == "L3" and "cheap model" in e.basis


def test_always_runs_and_off_never_runs():
    llm = FakeLLM()
    estimate(tf("你好"), llm=llm, l3_mode="always")
    assert llm.calls == 1
    estimate(tf(AMBIGUOUS), llm=llm, l3_mode="off")
    assert llm.calls == 1


def test_auto_skips_when_history_has_neighbours():
    state, llm = MemState(), FakeLLM()
    f = tf(AMBIGUOUS)
    for _ in range(3):
        record_outcome(f, estimate(f, state=state), Usage(cache_read_tokens=500_000, prompt_tokens=500_000, calls=6), 6,
                       state=state)
    llm.calls = 0
    e = estimate(f, state=state, llm=llm, l3_mode="auto")
    assert llm.calls == 0 and e.layer_used == "L2"


def test_failures_fall_back_silently():
    base = estimate(tf(AMBIGUOUS))
    for bad in (FakeLLM(raises=True), FakeLLM(reply="I cannot help"), lambda *a, **k: None):
        e = estimate(tf(AMBIGUOUS), llm=bad, l3_mode="always")
        assert e.layer_used == "L1" and e.est_steps == base.est_steps and e.estimate_cost_usd == 0


# --- how it combines ------------------------------------------------------------

def test_moves_p50_partly_and_keeps_draft_spread():
    base = estimate(tf(AMBIGUOUS))
    e = estimate(tf(AMBIGUOUS), llm=FakeLLM({"task_type": "agentic_coding", "tier": "XL",
                                             "steps_p50": 100, "steps_p90": 101}), l3_mode="always")
    assert base.est_steps < e.est_steps < 100 * L3.PRIOR_CORRECTION
    # The model's narrow interval (100..101) is ignored; the draft's P90/P50 ratio is kept.
    assert e.est_steps_p90 / e.est_steps == pytest.approx(base.est_steps_p90 / base.est_steps, rel=0.02)


def test_cost_and_raw_guess_are_reported():
    e = estimate(tf(AMBIGUOUS), llm=FakeLLM(cost=0.0012), l3_mode="always")
    assert e.estimate_cost_usd == pytest.approx(0.0012)
    assert e.shape["l3_steps"] == 30
    for which, tot in (("p50", e.tokens_p50.total), ("p90", e.tokens_p90.total)):
        assert sum(t.total for t in projected_steps(e, which)) == pytest.approx(tot, rel=0.03)


def test_type_override_only_when_l1_is_unsure():
    research = {"task_type": "research", "tier": "M", "steps_p50": 8, "steps_p90": 16, "images": 0}
    unsure = estimate(tf(AMBIGUOUS), llm=FakeLLM(research), l3_mode="always")
    assert unsure.task_type == "research"
    sure = estimate(tf("fix the failing test in tests/test_api.py and add a test for the parser"),
                    llm=FakeLLM(research), l3_mode="always")
    assert sure.task_type == "agentic_coding"


def test_image_count_from_model_for_generation():
    gen = {"task_type": "generation", "tier": "S", "steps_p50": 2, "steps_p90": 3, "images": 6}
    e = estimate(tf("make some pictures of cats for the slides", cwd_is_git_repo=False),
                 llm=FakeLLM(gen), l3_mode="always")
    assert e.task_type == "generation" and e.per_call_units == {"image_gen": 6}


def test_correction_is_learned_from_records():
    # The model keeps saying 10 steps; tasks actually take 30.
    recs = [{"outcome": "completed", "steps": 30, "l3k": 10} for _ in range(20)]
    c = L3.correction(recs)
    assert L3.PRIOR_CORRECTION < c < 3.0
    assert c > L3.correction(recs[:2])
    assert L3.correction([]) == L3.PRIOR_CORRECTION


def test_record_keeps_l3_guess_for_learning():
    state = MemState()
    f = tf(AMBIGUOUS)
    e = estimate(f, state=state, llm=FakeLLM(), l3_mode="always")
    record_outcome(f, e, Usage(cache_read_tokens=900_000, prompt_tokens=900_000, calls=12), 12, state=state)
    rec = json.loads(state.d["l2_records"])[0]
    assert rec["l3k"] == 30 and rec["steps"] == 12


def test_live_mode_attribute_wins_over_configured_mode():
    llm = FakeLLM()
    llm.mode = "off"
    configure(None, llm=llm, l3_mode="always")
    try:
        estimate(tf(AMBIGUOUS))
        assert llm.calls == 0
        llm.mode = "always"  # user flips the setting; no restart
        assert estimate(tf("你好")).layer_used == "L3"
    finally:
        configure(None)


def test_configure_wires_llm_and_mode():
    llm = FakeLLM()
    configure(None, llm=llm, l3_mode="always")
    try:
        assert estimate(tf("你好")).layer_used == "L3"
        configure(None, llm=None)
        assert estimate(tf(AMBIGUOUS)).layer_used == "L1"
    finally:
        configure(None)
    assert llm.calls == 1
