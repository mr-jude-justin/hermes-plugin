import json
import types

from harness.core import Settings
from harness.l3 import L3Client
from pricing import PriceBook

CAT = {"anthropic": {"claude-haiku-5-5": {"input": 0.1, "output": 0.5}}}


def prices(tmp_path):
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps({"fetched_at": 9e12, "prices": CAT}))
    return PriceBook(None, snapshot=snap)


def ctx_with(complete):
    return types.SimpleNamespace(llm=types.SimpleNamespace(complete=complete))


def result(text="{}", inp=1000, out=200):
    usage = types.SimpleNamespace(input_tokens=inp, output_tokens=out, cache_read_tokens=0,
                                  cache_write_tokens=0, cost_usd=None)
    return types.SimpleNamespace(text=text, provider="anthropic", model="claude-haiku-5-5", usage=usage)


def client(tmp_path, complete, provider="anthropic", **settings):
    return L3Client(ctx_with(complete), lambda: Settings(**settings), prices(tmp_path), lambda: provider)


def test_off_by_default_and_never_calls(tmp_path):
    calls = []
    c = client(tmp_path, lambda **k: calls.append(k))
    assert c.mode == "off" and c("size this") is None and calls == []


def test_auto_picks_the_providers_cheap_model_and_prices_the_call(tmp_path):
    seen = {}

    def complete(**kw):
        seen.update(kw)
        return result('{"steps_p50": 3}')

    text, cost = client(tmp_path, complete, l3_mode="auto")("size this", max_tokens=300, timeout_s=2.5)
    assert text == '{"steps_p50": 3}'
    assert cost == (1000 * 0.1 + 200 * 0.5) / 1e6
    assert seen["model"] == "claude-haiku-5-5" and seen["timeout"] == 2.5 and seen["max_tokens"] == 300


def test_explicit_model_wins(tmp_path):
    seen = {}
    client(tmp_path, lambda **kw: seen.update(kw) or result(), l3_mode="always", l3_model="claude-haiku-4-5")("x")
    assert seen["model"] == "claude-haiku-4-5"


def test_unknown_provider_without_a_model_is_off(tmp_path):
    assert client(tmp_path, lambda **kw: result(), provider="nous", l3_mode="auto")("x") is None


def test_missing_consent_or_errors_return_none(tmp_path):
    class PluginLlmTrustError(Exception):
        pass

    def deny(**kw):
        raise PluginLlmTrustError("cannot override the model")

    assert client(tmp_path, deny, l3_mode="auto")("x") is None
