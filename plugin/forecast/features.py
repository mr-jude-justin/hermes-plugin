"""Feature access and cheap text helpers.

Until `plugin/contracts.py` lands, features arrive either as a dict or as an
object with attributes (dataclass), using the field names from docs/plan.md
sections 5, 5.1 and 5.2. `get` hides the difference.
"""

from __future__ import annotations

import re
from typing import Any

_MISSING = object()


def get(features: Any, name: str, default: Any = None) -> Any:
    if isinstance(features, dict):
        value = features.get(name, default)
    else:
        value = getattr(features, name, _MISSING)
        if value is _MISSING:
            value = default
    return default if value is None else value


def message_of(features: Any) -> str:
    return str(get(features, "message", "") or "")


_CJK = re.compile(r"[㐀-鿿豈-﫿぀-ヿ가-힯]")


def approx_tokens(text: str) -> int:
    """Rough token count. CJK chars are about 1 token each, other text about 4 chars per token."""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    return cjk + max(0, len(text) - cjk) // 4


_CODE_FENCE = re.compile(r"```|~~~")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
# Things that look like file paths: a/b/c, ./x, ~/x, or name.ext with a short
# alphabetic extension (avoids matching numbers like 3.5).
_PATH = re.compile(
    r"(?:(?:~|\.{1,2})?/[\w.\-]+(?:/[\w.\-]+)*)"
    r"|(?:\b[\w\-]+(?:/[\w.\-]+)+)"
    r"|(?:\b[\w\-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|kt|c|cc|cpp|h|hpp|rb|php|swift|"
    r"md|txt|json|yaml|yml|toml|ini|cfg|sh|sql|html|css|scss|vue|ipynb|csv|pdf|docx)\b)",
    re.IGNORECASE,
)


def has_code_block(features: Any) -> bool:
    flag = get(features, "has_code_block", None)
    if flag is not None:
        return bool(flag)
    return bool(_CODE_FENCE.search(message_of(features)))


def count_urls(text: str) -> int:
    return len(_URL.findall(text))


_FENCED = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")


def count_paths(text: str) -> int:
    """Distinct file-like paths the message mentions, ignoring URLs and anything
    inside backticks or code fences: a path in `uvx pytest -q plugin/tests` is a
    command to run, not a file to change (fresh-task eval)."""
    stripped = _URL.sub(" ", text)
    stripped = _FENCED.sub(" ", stripped)
    stripped = _INLINE_CODE.sub(" ", stripped)
    return len(set(_PATH.findall(stripped)))


def toolsets(features: Any) -> set[str]:
    return {str(t).lower() for t in (get(features, "enabled_toolsets", []) or [])}
