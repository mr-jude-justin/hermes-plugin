"""Cold-start priors for the forecaster.

Every number here is a starting guess, taken from published benchmarks
(TokenCast arXiv 2609.35760, arXiv 2604.22750) and Justin's Hermes v0.23.0
probe. None of it has been measured on real Hermes trajectories yet. Calibration
(L2 history + record_outcome) is meant to replace these values per user. See
DESIGN.md sections 2 and 2.1.
"""

from __future__ import annotations

from dataclasses import dataclass

# Tiers are defined by the number of LLM calls (steps) K, not by tokens: the
# Hermes fixed floor makes absolute token counts harness-specific.
TIERS: tuple[str, ...] = ("S", "M", "L", "XL")

# (K_p50, K_p90) per tier. P90 is about 2.5x P50 on purpose. Same-task runs
# vary up to 30x, and launch-time coverage in TokenCast was only 82%.
TIER_STEPS: dict[str, tuple[float, float]] = {
    "S": (2.0, 4.0),
    "M": (8.0, 20.0),
    "L": (20.0, 50.0),
    "XL": (60.0, 150.0),
}

# Hermes v0.23.0: a bare "reply ok" turn sends about 68k prompt tokens (system
# prompt + tool schemas). Used when the harness cannot tell us the context size.
HERMES_FLOOR_TOKENS: int = 70_000

# Hermes v0.23.0: every `delegate_task` child starts with its own prompt of
# about 52.6k tokens (Justin's probe), so each subagent pays its own floor.
SUBAGENT_FLOOR_TOKENS: int = 52_600

# Even a new session's first call reads ~20k from cache: the tool-schema prefix is
# shared across sessions within the provider's cache TTL (Justin's probe). Only the
# very first task of a run pays a fully cold first call.
SHARED_PREFIX_TOKENS: int = 20_000

# Delegated tasks: how many subagents to expect when the message gives no count
# (P50, P90), how many steps each one runs, and its per-step shape. Justin's
# eval: each child made ~2 calls from its ~52k floor. Guesses beyond that.
SUBAGENTS_DEFAULT: tuple[float, float] = (2.0, 4.0)
SUBAGENT_STEPS: tuple[float, float] = (3.0, 8.0)
SUBAGENT_GROWTH: float = 1_000
SUBAGENT_OUTPUT: float = 400

TASK_TYPES: tuple[str, ...] = (
    "chat",
    "coding_chat",
    "code_reasoning",
    "agentic_coding",
    "research",
    "generation",
    "delegated",
    "browser",
    "long_doc",
)


@dataclass(frozen=True)
class TypeShape:
    """Per-step cost shape of a task type."""

    growth: float  # g: net prompt growth per step (tokens)
    output: float  # o: output tokens per step
    default_tier: str


# Refit to Justin's real Hermes + Sonnet runs (11/6, cold start, 7 tasks): the
# prompt grew ~250-700 tokens/step (more on file reads; +7.6k when a long doc was
# read); output was 64-190 tokens on tool steps and ~550-610 on answers. The
# first guesses (g 3-8k, o ~1k) came from SWE-bench-style harnesses and made
# each step ~3x too expensive. Research/browser keep larger g (pages, screenshots).
TYPE_SHAPES: dict[str, TypeShape] = {
    "chat": TypeShape(growth=300, output=550, default_tier="S"),
    "coding_chat": TypeShape(growth=400, output=600, default_tier="S"),
    "code_reasoning": TypeShape(growth=2_000, output=350, default_tier="M"),
    "agentic_coding": TypeShape(growth=700, output=200, default_tier="M"),
    "research": TypeShape(growth=4_000, output=400, default_tier="M"),
    "generation": TypeShape(growth=300, output=200, default_tier="S"),
    # The main agent mostly coordinates; subagents are added as their own segments.
    "delegated": TypeShape(growth=600, output=250, default_tier="M"),
    "browser": TypeShape(growth=1_500, output=150, default_tier="M"),
    "long_doc": TypeShape(growth=4_000, output=550, default_tier="M"),
}

# P90 must cover growth uncertainty too, not only more steps: a realistic
# multi-file task had the exact step count but grew ~3.9k/step from real file
# reads and went just over P90 (Justin's fresh-task test, 11/6).
GROWTH_P90_MULT: float = 2.0
# Mid-run the growth is observed, so the margin is smaller.
REFORECAST_GROWTH_P90_MULT: float = 1.3
# Inside a git repo, coding tasks read real files. The tiny per-step growth
# seen on the 17-line toy repo does not carry over (fresh test: ~3.9k/step;
# a two-file explain added ~11k in one step).
REPO_GROWTH_FLOOR: dict[str, float] = {"agentic_coding": 2_000, "code_reasoning": 3_000}

# Fraction of the reused prefix that is billed as a cache read. Real Hermes calls
# after the first write only ~400 new tokens of a ~67k prompt: effective ~0.99.
CACHE_HIT_ANTHROPIC: float = 0.98
CACHE_HIT_UNKNOWN: float = 0.0

# Multiplier on per-step output tokens. 2604.22750 has no effort sweep, so these
# are pure guesses until our eval measures them.
EFFORT_OUTPUT_MULT: dict[str, float] = {
    "minimal": 0.4,
    "min": 0.4,
    "low": 0.6,
    "medium": 1.0,
    "high": 1.6,
    "max": 2.5,
    "xhigh": 2.0,
    "ultra": 2.5,
}

# Above this many tokens already in context, every step is expensive, so bump a tier.
LONG_CONTEXT_TOKENS: int = 150_000

# Large-repo threshold for agentic/code-reasoning work.
LARGE_REPO_LINES: int = 50_000


def cache_hit_for(model: str | None) -> float:
    """Default cache-hit rate by provider, guessed from the model id."""
    m = (model or "").lower()
    if "claude" in m or "anthropic" in m:
        return CACHE_HIT_ANTHROPIC
    return CACHE_HIT_UNKNOWN
