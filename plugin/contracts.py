"""Shared types between the harness and the forecaster (owner: @mr-jude-justin).

The contract is docs/plan.md sections 5.1 and 5.2. It only grows: add fields with
defaults, never rename or remove one. Changing a field needs a board post first.

Stdlib only, so both sides can import it without extra dependencies.

Forecast entry points (implemented in `plugin/forecast/`):

    estimate(features: TaskFeatures) -> Estimate
    reforecast(features, original: Estimate, spent: Usage, steps_done: int,
               recent_steps: list[Usage]) -> Estimate
    record_outcome(features, estimate, actual: Usage | None, steps: int,
                   outcome: Outcome) -> None

Storage the harness provides to the forecaster: `History` and `ForecastState` below.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

Tier = Literal["S", "M", "L", "XL"]
TIERS: tuple[str, ...] = ("S", "M", "L", "XL")

TaskType = Literal[
    "chat", "coding_chat", "code_reasoning", "agentic_coding", "research",
    "generation", "delegated", "browser", "long_doc",
]
TASK_TYPES: tuple[str, ...] = (
    "chat", "coding_chat", "code_reasoning", "agentic_coding", "research",
    "generation", "delegated", "browser", "long_doc",
)

# completed: the task ran to the end; calibrate on it.
# declined: the user cancelled the quote, nothing ran; feeds the quote threshold only.
# stopped: the budget stopped it mid-run; `actual` is a lower bound (right-censored).
Outcome = Literal["completed", "declined", "stopped"]
OUTCOMES: tuple[str, ...] = ("completed", "declined", "stopped")

Mode = Literal["safe", "fast"]


@dataclass
class Tokens:
    """A token split. Each part is billed at a different rate."""

    input: float = 0
    cache_read: float = 0
    cache_write: float = 0
    output: float = 0

    @property
    def total(self) -> float:
        return self.input + self.cache_read + self.cache_write + self.output

    def __add__(self, other: Tokens) -> Tokens:
        return Tokens(self.input + other.input, self.cache_read + other.cache_read,
                      self.cache_write + other.cache_write, self.output + other.output)


@dataclass
class Usage:
    """Token usage of one model call, or a sum of calls.

    `from_hermes` reads the `usage` dict Hermes passes to `post_api_request`.
    `cost_usd` is filled by the harness from `pricing.py`; the forecaster never prices.
    """

    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    prompt_tokens: int = 0  # input + cache_read + cache_write as Hermes reports it
    calls: int = 0
    cost_usd: Optional[float] = None
    model: Optional[str] = None

    @classmethod
    def from_hermes(cls, usage: dict[str, Any], model: Optional[str] = None) -> Usage:
        u = usage or {}
        return cls(
            input_tokens=int(u.get("input_tokens") or 0),
            cache_read_tokens=int(u.get("cache_read_tokens") or 0),
            cache_write_tokens=int(u.get("cache_write_tokens") or 0),
            output_tokens=int(u.get("output_tokens") or 0),
            reasoning_tokens=int(u.get("reasoning_tokens") or 0),
            prompt_tokens=int(u.get("prompt_tokens") or 0),
            calls=int(u.get("request_count") or 1),
            model=model,
        )

    @classmethod
    def from_tokens(cls, t: Tokens, model: Optional[str] = None) -> Usage:
        """One projected call, for pricing a forecast step with `PriceBook.cost_of`."""
        i, r, w, o = (int(round(x)) for x in (t.input, t.cache_read, t.cache_write, t.output))
        return cls(input_tokens=i, cache_read_tokens=r, cache_write_tokens=w, output_tokens=o,
                   prompt_tokens=i + r + w, calls=1, model=model)

    @property
    def tokens(self) -> Tokens:
        return Tokens(self.input_tokens, self.cache_read_tokens, self.cache_write_tokens, self.output_tokens)

    def __add__(self, other: Usage) -> Usage:
        cost = None if self.cost_usd is None and other.cost_usd is None else (self.cost_usd or 0) + (other.cost_usd or 0)
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            calls=self.calls + other.calls,
            cost_usd=cost,
            model=self.model if self.model == other.model else None,
        )


@dataclass
class Price:
    """USD per million tokens for one model. None when the price table has no entry."""

    input_per_mtok: Optional[float] = None
    output_per_mtok: Optional[float] = None
    cache_read_per_mtok: Optional[float] = None
    cache_write_per_mtok: Optional[float] = None


@dataclass
class RepoSize:
    files: int = 0
    lines: int = 0


@dataclass
class ToolStep:
    """One tool call. `path` is set for file tools (read_file, write_file, patch, search_files)."""

    tool_name: str
    ok: Optional[bool] = None  # None until post_tool_call reports back
    path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PlanItems:
    """Progress of Hermes' `todo_list` tool."""

    total: int = 0
    done: int = 0


@dataclass
class SubagentUsage:
    """Usage of one `delegate_task` child, attributed by its session id."""

    session_id: str
    role: Optional[str] = None
    model: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    status: Optional[str] = None  # completed | interrupted | failed, from subagent_stop


@dataclass
class TaskFeatures:
    """Everything the harness knows about a task.

    The first block is known before the task starts. The second block is filled in
    while it runs (empty at estimate time) and is what `reforecast` reads.
    History rows store features with `message` blanked; the forecaster keeps its own
    hashed n-grams through `ForecastState` if it needs text similarity.
    """

    # Known before the task starts.
    message: str = ""
    message_len: int = 0
    has_code_block: bool = False
    has_paths_or_urls: bool = False
    attachments: int = 0
    model: str = ""
    price: Price = field(default_factory=Price)
    session_turns_so_far: int = 0
    context_tokens_now: int = 0
    recent_tools: list[str] = field(default_factory=list)
    enabled_toolsets: list[str] = field(default_factory=list)
    cwd_is_git_repo: bool = False
    repo_size: Optional[RepoSize] = None
    mode: Mode = "safe"
    reasoning_effort: Optional[str] = None
    platform: str = ""
    session_id: str = ""

    # Filled in while the task runs.
    step_usages: list[Usage] = field(default_factory=list)
    step_tools: list[ToolStep] = field(default_factory=list)
    tool_errors: int = 0
    retries: int = 0
    no_progress_calls: int = 0  # consecutive
    plan_items: Optional[PlanItems] = None
    subagents: list[SubagentUsage] = field(default_factory=list)


@dataclass
class Estimate:
    """What the forecaster returns. The harness fills the `cost_usd_*` fields."""

    tier: Tier = "S"
    task_type: TaskType = "chat"
    tokens_p50: Tokens = field(default_factory=Tokens)
    tokens_p90: Tokens = field(default_factory=Tokens)
    per_call_units: dict[str, int] = field(default_factory=dict)  # e.g. {"image_gen": 3}
    est_steps: float = 0  # P50
    est_steps_p90: float = 0
    confidence: float = 0.0
    basis: str = ""
    layer_used: str = "L1"  # L1 | L2 | L3
    estimate_cost_usd: float = 0.0  # what producing this estimate cost (L3 calls)
    cost_usd_p50: Optional[float] = None
    cost_usd_p90: Optional[float] = None
    # Per-step shape the forecaster used: {start_prompt, growth, output_per_step,
    # cache_hit, warm_cache}. `forecast.projected_steps(estimate, which)` expands it into
    # one Tokens per step so the harness can price each call at its context tier.
    shape: Optional[dict[str, Any]] = None


@dataclass
class PastTask:
    features: TaskFeatures
    estimate: Optional[Estimate]
    actual: Optional[Usage]  # None when declined
    steps: int
    model: str
    outcome: Outcome = "completed"
    finished_at: float = 0.0


class History(Protocol):
    def load(self) -> list[PastTask]: ...


class ForecastState(Protocol):
    """Small key-value store the forecaster owns (calibration coefficients, n-gram tables).
    Values must be JSON-serialisable."""

    def get(self, key: str, default: Any = None) -> Any: ...

    def set(self, key: str, value: Any) -> None: ...


# --- Serialisation (the harness persists these as JSON) -----------------------------------

def to_dict(obj: Any) -> dict[str, Any]:
    return dataclasses.asdict(obj)


def _build(cls: type, data: Any) -> Any:
    if data is None:
        return None
    nested = {
        Usage: {},
        Tokens: {},
        Price: {},
        RepoSize: {},
        ToolStep: {},
        PlanItems: {},
        SubagentUsage: {"usage": Usage},
        TaskFeatures: {"price": Price, "repo_size": RepoSize, "plan_items": PlanItems},
        Estimate: {"tokens_p50": Tokens, "tokens_p90": Tokens},
        PastTask: {"features": TaskFeatures, "estimate": Estimate, "actual": Usage},
    }[cls]
    lists = {
        TaskFeatures: {"step_usages": Usage, "step_tools": ToolStep, "subagents": SubagentUsage},
    }.get(cls, {})
    known = {f.name for f in dataclasses.fields(cls)}
    kwargs = {}
    for k, v in data.items():
        if k not in known:
            continue  # tolerate rows written by a newer contract
        if k in nested:
            v = _build(nested[k], v)
        elif k in lists:
            v = [_build(lists[k], x) for x in v or []]
        kwargs[k] = v
    return cls(**kwargs)


def from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Rebuild a contract dataclass (and its nested ones) from `to_dict` output."""
    return _build(cls, data)
