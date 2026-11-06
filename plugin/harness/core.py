"""The harness: hook handlers that feed the forecaster and enforce the budget.

One task is one user turn: it starts in `pre_llm_call` (or when the user approves a
quote) and finishes in `post_llm_call` / `on_session_end`. See docs/plan.md section 3.

Budget levels, per task:
- quote: before the task, when the P50 cost reaches `quote_threshold_usd` or the P90
  goes over the cap, `pre_llm_call` blocks the turn and stores the prompt.
- soft cap: when the projected total goes over the cap mid-run, `pre_tool_call`
  returns `approve`, which pauses at Hermes' approval gate. Approving raises the cap.
  `/yolo` and `approvals.mode: off` auto-approve, so this level alone is not enough.
- hard limit: when spending reaches the hard limit, `pre_tool_call` returns `block`,
  which approvals cannot skip. Only the user's reply ("continue" / "cap $N") lifts it.
- daily cap: same as the hard limit, across all tasks of the day.

Every hook catches its own errors: a crash in a budget plugin must never take the
user's turn down with it.
"""

from __future__ import annotations

import copy
import logging
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

try:
    from ..contracts import Estimate, PastTask, SubagentUsage, TaskFeatures, ToolStep, Usage
    from . import collect, messages, replies
except ImportError:
    from contracts import Estimate, PastTask, SubagentUsage, TaskFeatures, ToolStep, Usage
    from harness import collect, messages, replies

log = logging.getLogger("cost-forecast")

STOPPED_TTL = 30 * 60


@dataclass
class Settings:
    enabled: bool = True
    quote_threshold_usd: float = 0.50
    task_cap_usd: float = 3.00
    task_hard_limit_usd: float = 10.00
    daily_cap_usd: float = 20.00
    l3_mode: str = "off"   # off | auto | always: cheap-model estimate (needs llm.model_override)
    l3_model: str = ""     # empty: a cheap model on the user's provider

    @classmethod
    def from_getter(cls, get: Callable[..., Any]) -> Settings:
        s = cls()
        for f in ("enabled", "quote_threshold_usd", "task_cap_usd", "task_hard_limit_usd", "daily_cap_usd",
                  "l3_mode", "l3_model"):
            try:
                value = get(f, default=getattr(s, f))
                setattr(s, f, type(getattr(s, f))(value))
            except Exception:
                pass
        return s


@dataclass
class Run:
    session_id: str
    features: TaskFeatures
    estimate: Estimate
    cap: float
    ceiling: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    spent: float = 0.0
    actual: Usage = field(default_factory=Usage)
    steps: int = 0
    projected: float = 0.0
    stopped: bool = False
    stopped_at: float = 0.0
    open_tools: dict[str, ToolStep] = field(default_factory=dict)


def _round_up(x: float) -> float:
    return float(math.ceil(x)) if x >= 1 else math.ceil(x * 10) / 10


class Harness:
    def __init__(self, *, store, pricebook, forecast, settings: Callable[[], Settings]):
        self.store, self.prices, self.forecast, self.settings = store, pricebook, forecast, settings
        self._lock = threading.RLock()
        self.runs: dict[str, Run] = {}          # active task per session
        self.stopped: dict[str, Run] = {}       # hard-stopped tasks waiting for "continue"
        self.parent_of: dict[str, str] = {}     # subagent session -> parent session
        self.context_tokens: dict[str, int] = {}
        self.provider_of: dict[str, str] = {}
        self.last_provider: Optional[str] = None
        self.recent_tools: dict[str, list[str]] = {}

    # --- helpers -------------------------------------------------------------------------------

    def _run_for(self, session_id: Optional[str]) -> Optional[Run]:
        if not session_id:
            return None
        with self._lock:
            run = self.runs.get(session_id)
            if run is None and session_id in self.parent_of:
                run = self.runs.get(self.parent_of[session_id])
            return run

    def _cost(self, usage: Usage, provider: Optional[str], model: str) -> Optional[float]:
        return self.prices.cost_of(usage, provider, model)

    def price_estimate(self, e: Estimate, model: str, provider: Optional[str] = None) -> Estimate:
        """Fill `cost_usd_p50/p90`, pricing each projected step at its context tier."""
        projected = getattr(self.forecast, "projected_steps", None)
        for which in ("p50", "p90"):
            total: Optional[float] = None
            if projected and e.shape:
                try:
                    steps = projected(e, which)
                    costs = [self._cost(Usage.from_tokens(t, model), provider, model) for t in steps]
                    total = None if not costs or any(c is None for c in costs) else sum(costs)
                except Exception:
                    log.exception("projected_steps failed")
            if total is None:
                tokens = e.tokens_p50 if which == "p50" else e.tokens_p90
                n = max(1.0, e.est_steps if which == "p50" else (e.est_steps_p90 or e.est_steps))
                avg = type(tokens)(tokens.input / n, tokens.cache_read / n, tokens.cache_write / n, tokens.output / n)
                one = self._cost(Usage.from_tokens(avg, model), provider, model)
                total = None if one is None else one * n
            if total is not None:
                total += e.estimate_cost_usd or 0.0
            setattr(e, f"cost_usd_{which}", total)
        return e

    def _start(self, session_id: str, features: TaskFeatures, estimate: Estimate,
               cap: Optional[float] = None, accepted: bool = False) -> Run:
        """Start a task. `accepted`: the user saw the quote and said yes, so the cap (and,
        through it, the hard limit) may rise to cover the quoted P90."""
        s = self.settings()
        cap = cap if cap is not None else s.task_cap_usd
        if accepted and estimate.cost_usd_p90 is not None:
            cap = max(cap, _round_up(estimate.cost_usd_p90 * 1.2))
        run = Run(session_id, features, estimate, cap=cap, ceiling=max(s.task_hard_limit_usd, cap),
                  projected=estimate.cost_usd_p50 or 0.0, spent=estimate.estimate_cost_usd or 0.0)
        with self._lock:
            self.runs[session_id] = run
        return run

    def _finish(self, run: Run, outcome: str) -> None:
        actual = None if outcome == "declined" else run.actual
        self._record(run.features, run.estimate, actual, run.steps, outcome)

    def _record(self, features: TaskFeatures, estimate: Optional[Estimate], actual: Optional[Usage],
                steps: int, outcome: str) -> None:
        record = getattr(self.forecast, "record_outcome", None)
        if record:
            try:
                record(features, estimate, actual, steps, outcome)
            except Exception:
                log.exception("record_outcome failed")
        try:
            self.store.append(PastTask(copy.deepcopy(features), estimate, actual, steps,
                                       features.model, outcome, time.time()))
        except Exception:
            log.exception("history append failed")

    # --- pre_llm_call: quote, re-inject, resume ------------------------------------------------

    def pre_llm_call(self, session_id: str = "", user_message: str = "", conversation_history=None,
                     model: str = "", platform: str = "", turn_id: str = "", **_: Any):
        try:
            return self._pre_llm_call(session_id, user_message or "", conversation_history, model, platform, turn_id)
        except Exception:
            log.exception("pre_llm_call failed; letting the turn run")
            return None

    def _pre_llm_call(self, sid, message, history, model, platform, turn_id):
        s = self.settings()
        if not s.enabled or not sid:
            return None
        with self._lock:
            if sid in self.parent_of:
                # A subagent's own turn: it belongs to the parent's task, which was already
                # quoted. Never quote it (nobody can answer) and never start a separate run.
                return None
        reply = replies.parse(message)

        # 1. A hard-stopped task waiting for the user.
        with self._lock:
            stopped = self.stopped.pop(sid, None)
        if stopped and time.time() - stopped.stopped_at < STOPPED_TTL:
            if reply.kind in ("approve", "cap"):
                new_ceiling = reply.cap_usd if reply.kind == "cap" else stopped.spent + stopped.ceiling
                stopped.ceiling = max(new_ceiling, stopped.spent + 0.01)
                stopped.cap = max(stopped.cap, stopped.ceiling)
                stopped.stopped = False
                with self._lock:
                    self.runs[sid] = stopped
                return {"context": messages.resume_context(stopped.ceiling)}
            self._finish(stopped, "stopped")
            if reply.kind == "cancel":
                return {"action": "block", "message": messages.cancelled()}
        elif stopped:
            self._finish(stopped, "stopped")

        # 2. A quote waiting for the user.
        pending = self.store.get_pending(sid)
        if pending:
            if reply.kind == "approve":
                self.store.drop_pending(sid)
                self._start(sid, pending.features, pending.estimate, accepted=True)
                return {"context": messages.reinject(pending.message)}
            if reply.kind == "cancel":
                self.store.drop_pending(sid)
                self._record(pending.features, pending.estimate, None, 0, "declined")
                return {"action": "block", "message": messages.cancelled()}
            if reply.kind == "cap":
                p90 = pending.estimate.cost_usd_p90
                if p90 is not None and p90 > reply.cap_usd:
                    return {"action": "block", "message": messages.cap_still_low(pending.estimate, reply.cap_usd)}
                self.store.drop_pending(sid)
                run = self._start(sid, pending.features, pending.estimate, cap=reply.cap_usd, accepted=True)
                run.ceiling = max(run.ceiling, reply.cap_usd)
                return {"context": messages.reinject(pending.message)}
            # Anything else is a new request: the quote lapses.
            self.store.drop_pending(sid)
            self._record(pending.features, pending.estimate, None, 0, "declined")

        # 3. A new request.
        if self.store.spent_today() >= s.daily_cap_usd:
            return {"action": "block", "message": messages.daily_stop(self.store.spent_today(), s.daily_cap_usd)}
        provider = self.provider_of.get(sid)
        features = collect.task_features(
            message=message, model=model, platform=platform, session_id=sid, conversation_history=history,
            context_tokens_now=self.context_tokens.get(sid, 0), recent_tools=self.recent_tools.get(sid),
            price=self.prices.price(provider, model) if model else None)
        try:
            estimate = self.price_estimate(self.forecast.estimate(features), model, provider)
            if estimate.estimate_cost_usd:  # the L3 call; ctx.llm calls never reach post_api_request
                self.store.add_spend(estimate.estimate_cost_usd)
        except Exception:
            log.exception("estimate failed; running without a quote")
            self._start(sid, features, Estimate(basis="estimate failed"))
            return None

        p50, p90 = estimate.cost_usd_p50, estimate.cost_usd_p90
        # Nobody can answer a quote in cron or webhook runs: skip it and rely on the hard
        # limit and daily cap, which stop the task without waiting for anyone.
        if collect.unattended(platform):
            self._start(sid, features, estimate)
            return None
        if p50 is not None and (p50 >= s.quote_threshold_usd or (p90 or 0) > s.task_cap_usd):
            self.store.put_pending(sid, message, features, estimate, turn_id)
            return {"action": "block",
                    "message": messages.quote(estimate, s.task_cap_usd, self.prices.describe_source())}
        self._start(sid, features, estimate)
        return None

    # --- post_api_request: ledger and reforecast -----------------------------------------------

    def post_api_request(self, session_id: str = "", model: str = "", provider: str = "",
                         usage: Optional[dict] = None, **_: Any):
        try:
            self._post_api_request(session_id, model, provider, usage or {})
        except Exception:
            log.exception("post_api_request failed")

    def _post_api_request(self, sid, model, provider, usage):
        u = Usage.from_hermes(usage, model)
        u.cost_usd = self._cost(u, provider, model)
        if u.cost_usd:
            self.store.add_spend(u.cost_usd)
        with self._lock:
            if provider:
                self.provider_of[sid] = provider
                self.last_provider = provider
            if u.prompt_tokens:
                self.context_tokens[sid] = u.prompt_tokens
        run = self._run_for(sid)
        if run is None:
            return
        with self._lock:
            run.features.step_usages.append(u)
            run.actual = run.actual + u
            run.spent += u.cost_usd or 0.0
            run.steps += 1
            if sid != run.session_id:
                for child in run.features.subagents:
                    if child.session_id == sid:
                        child.usage = child.usage + u
                        child.model = model or child.model
        run.projected = self._reforecast(run, model, provider)

    def _reforecast(self, run: Run, model: str, provider: str) -> float:
        fn = getattr(self.forecast, "reforecast", None)
        projected_steps = getattr(self.forecast, "projected_steps", None)
        if fn:
            try:
                e = fn(run.features, run.estimate, run.actual, run.steps, run.features.step_usages[-5:])
                # A reforecast's shape covers only the remaining calls (shape.steps_done is set):
                # price those and add what is already spent.
                if projected_steps and e.shape and e.shape.get("steps_done") is not None:
                    costs = [self._cost(Usage.from_tokens(t, model), provider, model)
                             for t in projected_steps(e, "p50")]
                    if costs and all(c is not None for c in costs):
                        return run.spent + sum(costs)
                e = self.price_estimate(e, model, provider)
                if e.cost_usd_p50 is not None:
                    return max(run.spent, e.cost_usd_p50)
            except Exception:
                log.exception("reforecast failed; using the run rate")
        # Fallback: the run rate times the steps the estimate expected.
        avg = run.spent / max(1, run.steps)
        remaining = max(1.0, (run.estimate.est_steps or 0) - run.steps)
        return run.spent + avg * remaining

    # --- pre_tool_call / post_tool_call: budget stops and step features --------------------------

    def pre_tool_call(self, tool_name: str = "", args: Any = None, task_id: str = "", session_id: str = "",
                      tool_call_id: str = "", **_: Any):
        try:
            return self._pre_tool_call(tool_name, args, session_id, tool_call_id)
        except Exception:
            log.exception("pre_tool_call failed")
            return None

    def _pre_tool_call(self, tool_name, args, sid, tool_call_id):
        with self._lock:
            tools = self.recent_tools.setdefault(sid, [])
            tools.append(tool_name)
            del tools[:-10]
        run = self._run_for(sid)
        if run is None:
            return None
        step = ToolStep(tool_name, path=collect.tool_path(tool_name, args))
        with self._lock:
            f = run.features
            prev = f.step_tools[-1] if f.step_tools else None
            f.no_progress_calls = f.no_progress_calls + 1 if (
                prev and step.path and (prev.tool_name, prev.path) == (step.tool_name, step.path)) else 0
            f.step_tools.append(step)
            run.open_tools[tool_call_id or tool_name] = step
            if tool_name == "todo_list":
                f.plan_items = collect.plan_items(args, f.plan_items)

        s = self.settings()
        if not s.enabled:
            return None
        spent_today = self.store.spent_today()
        if spent_today >= s.daily_cap_usd:
            self._stop(run)
            return {"action": "block", "message": messages.daily_stop(spent_today, s.daily_cap_usd)}
        if run.spent >= run.ceiling:
            self._stop(run)
            return {"action": "block", "message": messages.hard_stop(run.spent, run.ceiling)}
        if run.projected > run.cap:
            old = run.cap
            new_cap = min(run.ceiling, _round_up(max(run.projected * 1.25, old * 1.5)))
            if new_cap > old:
                run.cap = new_cap  # if the user denies, the tool fails and the model stops
                return {"action": "approve", "rule_key": f"cost-forecast:{run.id}:{old:.2f}",
                        "message": messages.soft_stop(run.spent, run.projected, old, new_cap)}
        return None

    def _stop(self, run: Run) -> None:
        run.stopped = True
        run.stopped_at = time.time()

    def post_tool_call(self, tool_name: str = "", result: Any = None, session_id: str = "",
                       tool_call_id: str = "", **_: Any):
        try:
            run = self._run_for(session_id)
            if run is None:
                return
            with self._lock:
                step = run.open_tools.pop(tool_call_id or tool_name, None)
                if step is None:
                    return
                step.ok = not collect.tool_failed(result)
                if not step.ok:
                    run.features.tool_errors += 1
                    step.error = (result if isinstance(result, str) else str(result))[:200]
        except Exception:
            log.exception("post_tool_call failed")

    def api_request_error(self, session_id: str = "", **_: Any):
        run = self._run_for(session_id)
        if run is not None:
            run.features.retries += 1

    # --- subagents ---------------------------------------------------------------------------

    def subagent_start(self, parent_session_id: str = "", child_session_id: str = "", child_role: str = "",
                       **_: Any):
        try:
            with self._lock:
                root = self.parent_of.get(parent_session_id, parent_session_id)
                if not child_session_id:
                    return
                self.parent_of[child_session_id] = root
                run = self.runs.get(root)
                if run is not None:
                    run.features.subagents.append(SubagentUsage(child_session_id, role=child_role or None))
        except Exception:
            log.exception("subagent_start failed")

    def subagent_stop(self, child_session_id: str = "", child_status: str = "", **_: Any):
        run = self._run_for(child_session_id)
        if run is None:
            return
        for child in run.features.subagents:
            if child.session_id == child_session_id:
                child.status = child_status or child.status

    # --- end of turn ---------------------------------------------------------------------------

    def post_llm_call(self, session_id: str = "", **_: Any):
        self._end_turn(session_id, interrupted=False)

    def on_session_end(self, session_id: str = "", interrupted: bool = False, **_: Any):
        self._end_turn(session_id, interrupted=bool(interrupted))

    def _end_turn(self, sid: str, interrupted: bool) -> None:
        try:
            with self._lock:
                if sid in self.parent_of:  # a subagent finishing is not the end of the task
                    return
                run = self.runs.pop(sid, None)
                if run is None:
                    return
                if run.stopped:
                    self.stopped[sid] = run  # waits for "continue"
                    return
            self._finish(run, "stopped" if interrupted else "completed")
        except Exception:
            log.exception("end of turn failed")

    # --- /forecast ---------------------------------------------------------------------------

    def forecast_command(self, raw_args: str = "", *, session_id: str = "", **_: Any) -> str:
        s = self.settings()
        lines = [f"💰 Spent today: {messages.usd(self.store.spent_today())} of {messages.usd(s.daily_cap_usd)} daily cap."]
        run = self._run_for(session_id) or self.stopped.get(session_id)
        if run is not None:
            state = "stopped, reply continue to resume" if run.stopped else "running"
            lines.append(f"Current task ({state}): spent {messages.usd(run.spent)}, projected "
                         f"{messages.usd(run.projected)}, cap {messages.usd(run.cap)}, hard limit {messages.usd(run.ceiling)}.")
        lines.append(f"Quote above {messages.usd(s.quote_threshold_usd)}; per-task cap {messages.usd(s.task_cap_usd)}; "
                     f"hard limit {messages.usd(s.task_hard_limit_usd)}. Prices: {self.prices.describe_source()}.")
        return "\n".join(lines)
