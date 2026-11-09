import sqlite3

import pytest

from contracts import Estimate, PastTask, TaskFeatures, Usage
from harness import store as store_mod
from harness.store import PENDING_TTL, Store


@pytest.fixture
def store():
    return Store(sqlite3.connect(":memory:", check_same_thread=False))


def test_kv_round_trip_and_default(store):
    assert store.get("missing", {"a": 1}) == {"a": 1}
    store.set("coef", {"x": [1.5, 2]})
    assert store.get("coef") == {"x": [1.5, 2]}
    store.set("coef", 3)
    assert store.get("coef") == 3


def test_history_drops_message_text(store):
    task = PastTask(features=TaskFeatures(message="secret plan", message_len=11), estimate=Estimate(),
                    actual=Usage(input_tokens=10), steps=1, model="m", finished_at=1.0)
    store.append(task)
    (loaded,) = store.load()
    assert loaded.features.message == "" and loaded.features.message_len == 11
    assert loaded.actual.input_tokens == 10


def test_history_keeps_only_the_newest_rows(store, monkeypatch):
    monkeypatch.setattr(store_mod, "HISTORY_LIMIT", 3)
    for i in range(5):
        store.append(PastTask(features=TaskFeatures(), estimate=None, actual=None, steps=i, model="m"))
    assert [t.steps for t in store.load()] == [2, 3, 4]


def test_pending_round_trip_and_expiry(store):
    e = Estimate(tier="L", cost_usd_p50=2.0, cost_usd_p90=4.0)
    store.put_pending("s", "refactor", TaskFeatures(message="refactor"), e, turn_id="t1", now=1000.0)
    p = store.get_pending("s", now=1000.0 + PENDING_TTL - 1)
    assert (p.message, p.turn_id, p.estimate.cost_usd_p90, p.features.message) == ("refactor", "t1", 4.0, "refactor")
    assert store.get_pending("s", now=1000.0 + PENDING_TTL + 1) is None


def test_drop_pending(store):
    store.put_pending("s", "m", TaskFeatures(), Estimate())
    store.drop_pending("s")
    assert store.get_pending("s") is None


def test_daily_spend_adds_up_per_day(store):
    day1, day2 = 1_700_000_000.0, 1_700_000_000.0 + 86_400 * 2
    assert store.add_spend(0.25, now=day1) == pytest.approx(0.25)
    assert store.add_spend(0.5, now=day1) == pytest.approx(0.75)
    assert store.spent_today(now=day1) == pytest.approx(0.75)
    assert store.spent_today(now=day2) == 0.0
