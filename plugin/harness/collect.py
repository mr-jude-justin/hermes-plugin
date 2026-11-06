"""Feature collection: turn hook arguments into `TaskFeatures` fields.

Everything that reaches into Hermes internals is wrapped so a Hermes upgrade that moves
a helper costs us one feature, never the hook.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

try:
    from ..contracts import PlanItems, Price, TaskFeatures
except ImportError:
    from contracts import PlanItems, Price, TaskFeatures

FILE_TOOLS = {"read_file", "write_file", "patch", "search_files"}
_PATH_ARGS = ("path", "file_path", "file", "filename", "target", "directory", "dir")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_PATHLIKE = re.compile(r"(?:~|\.{1,2})?/[\w.\-]+(?:/[\w.\-]+)+|\b[\w\-]+\.(?:py|js|ts|tsx|go|rs|java|md|json|ya?ml|toml|sh|sql|html|css)\b")


def _hermes_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly
        return load_config_readonly() or {}
    except Exception:
        return {}


# Surfaces where nobody can answer a quote (Hermes' own list for approvals, plus cron).
UNATTENDED_PLATFORMS = {"webhook", "msgraph_webhook", "api_server", "cron"}


def unattended(platform: str) -> bool:
    cron = os.environ.get("HERMES_CRON_SESSION", "").strip().lower() in {"1", "true", "yes", "on"}
    return cron or (platform or "").lower() in UNATTENDED_PLATFORMS


def reasoning_effort() -> Optional[str]:
    agent = _hermes_config().get("agent") or {}
    value = agent.get("reasoning_effort")
    return str(value) if value else None


def cwd_is_git_repo() -> bool:
    try:
        from agent.runtime_cwd import resolve_agent_cwd
        cwd = Path(resolve_agent_cwd())
    except Exception:
        return False
    return any((p / ".git").exists() for p in (cwd, *cwd.parents))


def task_features(*, message: str, model: str, platform: str, session_id: str,
                  conversation_history: Optional[list] = None, context_tokens_now: int = 0,
                  recent_tools: Optional[list[str]] = None, price: Optional[Price] = None) -> TaskFeatures:
    history = conversation_history or []
    prior_user_turns = sum(1 for m in history if isinstance(m, dict) and m.get("role") == "user") - 1
    return TaskFeatures(
        message=message,
        message_len=len(message),
        has_code_block="```" in message or "~~~" in message,
        has_paths_or_urls=bool(_URL.search(message) or _PATHLIKE.search(message)),
        attachments=0,  # Hermes does not pass attachments to pre_llm_call
        model=model or "",
        price=price or Price(),
        session_turns_so_far=max(0, prior_user_turns),
        context_tokens_now=context_tokens_now,
        recent_tools=list(recent_tools or [])[-10:],
        cwd_is_git_repo=cwd_is_git_repo(),
        reasoning_effort=reasoning_effort(),
        platform=platform or "",
        session_id=session_id or "",
    )


def tool_path(tool_name: str, args: Any) -> Optional[str]:
    if tool_name not in FILE_TOOLS or not isinstance(args, dict):
        return None
    for key in _PATH_ARGS:
        if isinstance(args.get(key), str) and args[key]:
            return args[key]
    return None


def plan_items(args: Any, previous: Optional[PlanItems]) -> Optional[PlanItems]:
    """Progress from a `todo_list` call. A merge call updates only the items it names."""
    if not isinstance(args, dict) or not isinstance(args.get("todos"), list):
        return previous
    todos = [t for t in args["todos"] if isinstance(t, dict)]
    done = sum(1 for t in todos if t.get("status") == "completed")
    if args.get("merge") and previous:
        return PlanItems(total=max(previous.total, len(todos)), done=max(previous.done, done))
    return PlanItems(total=len(todos), done=done)


def tool_failed(result: Any) -> bool:
    text = result if isinstance(result, str) else str(result)
    head = text[:300].lower()
    return head.startswith(("error", "blocked", "[cost-forecast]")) or '"error"' in head or "traceback" in head
