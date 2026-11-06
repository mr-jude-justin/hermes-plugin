"""Price table: turns token usage into USD (owner: @mr-jude-justin).

Prices come from models.dev (https://models.dev/api.json, MIT licensed, updated
daily), never from Hermes' own cache. The plugin lives as long as the Hermes
process, which for the gateway can be days, so prices refresh lazily: a lookup
that finds the data older than `max_age` starts a background refresh and answers
from what it already has. Nothing here touches the network during `register()`.

Load order: the refreshed copy in the plugin's data dir, else the snapshot that
ships with the plugin (`data/prices_snapshot.json`), else nothing (every price is
unknown and quotes show tokens only).

All prices are USD per million tokens. Some models charge more once a call's
prompt passes a size (models.dev `tiers`); `cost_of` picks the tier per call.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

try:
    from .contracts import Price, Usage
except ImportError:  # imported as a top-level module in tests
    from contracts import Price, Usage

SOURCE_URL = "https://models.dev/api.json"
SNAPSHOT = Path(__file__).parent / "data" / "prices_snapshot.json"
CACHE_NAME = "prices.json"
DAY = 24 * 3600

# Providers kept in the bundled snapshot (the full catalogue is about 6 MB).
SNAPSHOT_PROVIDERS = (
    "anthropic", "openai", "google", "deepseek", "xai", "mistral", "openrouter",
    "moonshotai", "zhipuai", "alibaba",
)


def compact(catalogue: dict[str, Any], providers: Optional[tuple[str, ...]] = None) -> dict[str, Any]:
    """Keep only `{provider: {model: cost}}` from a models.dev catalogue."""
    out: dict[str, Any] = {}
    for pid, p in catalogue.items():
        if providers is not None and pid not in providers:
            continue
        models = {mid: m["cost"] for mid, m in (p.get("models") or {}).items() if m.get("cost")}
        if models:
            out[pid] = models
    return out


def _price(cost: dict[str, Any]) -> Price:
    return Price(
        input_per_mtok=cost.get("input"),
        output_per_mtok=cost.get("output"),
        cache_read_per_mtok=cost.get("cache_read"),
        cache_write_per_mtok=cost.get("cache_write"),
    )


class PriceBook:
    """Look up prices and cost usage. `data_dir` is the plugin's `plugin_data_dir()`."""

    def __init__(self, data_dir: Optional[Path] = None, *, max_age: float = DAY,
                 fetch=None, snapshot: Path = SNAPSHOT):
        self._data_dir = Path(data_dir) if data_dir else None
        self._max_age = max_age
        self._fetch = fetch or _fetch_catalogue
        self._lock = threading.Lock()
        self._refreshing = False
        self.prices: dict[str, dict[str, Any]] = {}
        self.fetched_at: float = 0.0
        self.source: str = "none"
        self._load(snapshot)

    # --- loading and refresh -------------------------------------------------------------

    def _cache_path(self) -> Optional[Path]:
        return self._data_dir / CACHE_NAME if self._data_dir else None

    def _load(self, snapshot: Path) -> None:
        for path, source in ((self._cache_path(), "models.dev"), (snapshot, "snapshot")):
            if path and path.exists():
                try:
                    blob = json.loads(path.read_text())
                    self.prices, self.fetched_at, self.source = blob["prices"], blob["fetched_at"], source
                    return
                except (ValueError, KeyError, OSError):
                    continue

    def _stale(self) -> bool:
        return time.time() - self.fetched_at > self._max_age

    def refresh_if_stale(self) -> None:
        """Start one background refresh when the data is old. Never blocks."""
        with self._lock:
            if self._refreshing or not self._stale() or not self._data_dir:
                return
            self._refreshing = True
        threading.Thread(target=self._refresh, name="cost-forecast-prices", daemon=True).start()

    def _refresh(self) -> None:
        try:
            prices = compact(self._fetch())
            fetched_at = time.time()
            path = self._cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"fetched_at": fetched_at, "prices": prices}, separators=(",", ":")))
            tmp.replace(path)
            self.prices, self.fetched_at, self.source = prices, fetched_at, "models.dev"
        except Exception:
            pass  # keep the old data; the next stale lookup tries again
        finally:
            with self._lock:
                self._refreshing = False

    # --- lookup ----------------------------------------------------------------------------

    def _cost_entry(self, provider: Optional[str], model: str) -> Optional[dict[str, Any]]:
        if not model:
            return None
        provider = (provider or "").lower()
        vendor, _, bare = model.rpartition("/")
        candidates = [(provider, model), (provider, bare)]
        if vendor:
            candidates.append((vendor.lower(), bare))
        for pid, mid in candidates:
            entry = self.prices.get(pid, {}).get(mid)
            if entry:
                return entry
        # Unknown provider name (a custom endpoint, a router, a Hermes id that differs from
        # models.dev's): match the bare model id, first-party providers first, since
        # resellers often mark prices up.
        order = [p for p in SNAPSHOT_PROVIDERS if p in self.prices] + \
                [p for p in self.prices if p not in SNAPSHOT_PROVIDERS]
        for pid in order:
            entry = self.prices[pid].get(bare)
            if entry:
                return entry
        return None

    def price(self, provider: Optional[str], model: str, prompt_tokens: int = 0) -> Optional[Price]:
        """Price of one call with `prompt_tokens` of prompt, or None when unknown."""
        self.refresh_if_stale()
        entry = self._cost_entry(provider, model)
        if entry is None:
            return None
        chosen = entry
        for tier in entry.get("tiers") or []:
            size = (tier.get("tier") or {}).get("size")
            if (tier.get("tier") or {}).get("type") == "context" and size is not None and prompt_tokens > size:
                chosen = {**entry, **{k: v for k, v in tier.items() if k != "tier"}}
        return _price(chosen)

    def cost_of(self, usage: Usage, provider: Optional[str], model: Optional[str] = None) -> Optional[float]:
        """USD cost of one model call. None when the model has no price.

        Pass one call at a time: the context tier depends on that call's prompt size.
        Missing cache prices fall back to the input price; reasoning tokens are already
        counted in `output_tokens`.
        """
        p = self.price(provider, model or usage.model or "", usage.prompt_tokens)
        if p is None or p.input_per_mtok is None or p.output_per_mtok is None:
            return None
        cache_read = p.cache_read_per_mtok if p.cache_read_per_mtok is not None else p.input_per_mtok
        cache_write = p.cache_write_per_mtok if p.cache_write_per_mtok is not None else p.input_per_mtok
        return (usage.input_tokens * p.input_per_mtok
                + usage.cache_read_tokens * cache_read
                + usage.cache_write_tokens * cache_write
                + usage.output_tokens * p.output_per_mtok) / 1_000_000

    def describe_source(self) -> str:
        """For quote messages, e.g. 'models.dev, 2026-11-06'."""
        if not self.fetched_at:
            return "no price data"
        return f"{self.source}, {time.strftime('%Y-%m-%d', time.localtime(self.fetched_at))}"


def _fetch_catalogue(timeout: float = 15.0) -> dict[str, Any]:
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "hermes-cost-forecast"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def write_snapshot(path: Path = SNAPSHOT) -> None:
    """Regenerate the bundled snapshot: `python plugin/pricing.py`."""
    prices = compact(_fetch_catalogue(), SNAPSHOT_PROVIDERS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fetched_at": time.time(), "prices": prices}, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    write_snapshot()
    print(f"wrote {SNAPSHOT}")
