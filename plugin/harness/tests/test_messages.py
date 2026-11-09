from contracts import Estimate, Tokens
from harness import messages


def test_usd_formatting():
    assert messages.usd(None) == "?"
    assert messages.usd(0.0123) == "$0.012"
    assert messages.usd(0.5) == "$0.50"
    assert messages.usd(12.4) == "$12"


def test_quote_warns_when_p90_is_over_the_cap():
    e = Estimate(tier="L", task_type="agentic_coding", est_steps=24.6, cost_usd_p50=2.2, cost_usd_p90=6.0,
                 basis="8 similar past tasks")
    text = messages.quote(e, cap_usd=3.0, price_source="models.dev")
    assert text.startswith("💰 Cost estimate: Agentic coding, tier L, likely $2.20–$6.00 (about 25 steps).")
    assert "over your $3.00 per-task cap" in text and "Basis: 8 similar past tasks" in text


def test_quote_without_prices_shows_tokens_and_no_cap_warning():
    e = Estimate(tier="M", task_type="research", est_steps=1, tokens_p50=Tokens(input=40_000, output=2_000))
    text = messages.quote(e, cap_usd=3.0, price_source="none")
    assert "about 42k tokens (no price data for this model)" in text and "about 1 step)" in text
    assert "cap." not in text.split("\n")[0] and "Basis: rules" in text


def test_quote_collapses_equal_range():
    e = Estimate(task_type="chat", est_steps=1, cost_usd_p50=0.5, cost_usd_p90=0.501)
    assert "likely $0.50 (" in messages.quote(e, 3.0, "x")


def test_every_budget_message_has_the_marker():
    e = Estimate(cost_usd_p90=5.0)
    texts = [messages.cancelled(), messages.cap_still_low(e, 2.0), messages.soft_stop(1, 4, 3, 6),
             messages.hard_stop(10, 10), messages.daily_stop(20, 20)]
    assert all(t.startswith("💰") for t in texts)


def test_reinject_carries_the_request():
    assert messages.reinject("do the thing").endswith("\n\ndo the thing")
