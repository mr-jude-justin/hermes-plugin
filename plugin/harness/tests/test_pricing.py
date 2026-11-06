import json
import time

import pytest

from contracts import Usage
from pricing import SNAPSHOT, PriceBook, compact

CATALOGUE = {
    "anthropic": {"models": {
        "claude-haiku-5-5": {"cost": {
            "input": 0.1, "output": 0.5, "cache_read": 0.01, "cache_write": 0.125,
            "tiers": [{"input": 0.5, "output": 2.5, "cache_read": 0.05, "cache_write": 0.625,
                       "tier": {"type": "context", "size": 100000}}]}},
    }},
    "openai": {"models": {"gpt-5": {"cost": {"input": 1.25, "output": 10, "cache_read": 0.125}}}},
    "reseller": {"models": {"gpt-5": {"cost": {"input": 9, "output": 90}}}},
    "local": {"models": {"free-model": {}}},
}


def book(tmp_path, catalogue=CATALOGUE, fetched_at=None):
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps({"fetched_at": fetched_at or time.time(), "prices": compact(catalogue)}))
    return PriceBook(tmp_path / "data", snapshot=snap, fetch=lambda: catalogue)


def test_compact_drops_unpriced_models():
    assert "local" not in compact(CATALOGUE)
    assert compact(CATALOGUE, ("openai",)).keys() == {"openai"}


def test_cost_below_context_tier(tmp_path):
    # Real usage shape from the probe run: 84k prompt, mostly cache reads.
    u = Usage(input_tokens=4, cache_read_tokens=83182, cache_write_tokens=910, output_tokens=316, prompt_tokens=84096)
    cost = book(tmp_path).cost_of(u, "anthropic", "claude-haiku-5-5")
    assert cost == pytest.approx((4 * 0.1 + 83182 * 0.01 + 910 * 0.125 + 316 * 0.5) / 1e6)


def test_cost_above_context_tier_uses_tier_price(tmp_path):
    u = Usage(input_tokens=1000, output_tokens=1000, prompt_tokens=150_000)
    cost = book(tmp_path).cost_of(u, "anthropic", "claude-haiku-5-5")
    assert cost == pytest.approx((1000 * 0.5 + 1000 * 2.5) / 1e6)


def test_missing_cache_write_price_falls_back_to_input(tmp_path):
    u = Usage(cache_write_tokens=1_000_000, prompt_tokens=1_000_000)
    assert book(tmp_path).cost_of(u, "openai", "gpt-5") == pytest.approx(1.25)


def test_lookup_strips_vendor_prefix_and_prefers_first_party(tmp_path):
    b = book(tmp_path)
    assert b.price("openrouter", "openai/gpt-5").input_per_mtok == 1.25
    assert b.price("some-custom-endpoint", "gpt-5").input_per_mtok == 1.25


def test_unknown_model_is_none(tmp_path):
    b = book(tmp_path)
    assert b.price("anthropic", "nope") is None
    assert b.cost_of(Usage(output_tokens=10), "local", "free-model") is None


def test_stale_data_refreshes_in_background_and_persists(tmp_path):
    newer = {"openai": {"models": {"gpt-5": {"cost": {"input": 2, "output": 20}}}}}
    b = book(tmp_path, fetched_at=1.0)
    b._fetch = lambda: newer
    assert b.price("openai", "gpt-5").input_per_mtok == 1.25  # answers from old data right away
    for _ in range(100):
        if b.source == "models.dev":
            break
        time.sleep(0.01)
    assert b.price("openai", "gpt-5").input_per_mtok == 2
    reloaded = PriceBook(tmp_path / "data", snapshot=tmp_path / "missing.json", fetch=lambda: {})
    assert reloaded.source == "models.dev" and reloaded.price("openai", "gpt-5").input_per_mtok == 2


def test_failed_refresh_keeps_old_data(tmp_path):
    def boom():
        raise OSError("offline")
    b = book(tmp_path, fetched_at=1.0)
    b._fetch = boom
    b.refresh_if_stale()
    time.sleep(0.05)
    assert b.price("openai", "gpt-5").input_per_mtok == 1.25
    assert b.source == "snapshot"


def test_bundled_snapshot_has_main_providers(tmp_path):
    b = PriceBook(None, snapshot=SNAPSHOT)
    assert b.source == "snapshot"
    assert b.price("anthropic", "claude-opus-5-5") is not None
    assert {"anthropic", "openai", "google"} <= b.prices.keys()
