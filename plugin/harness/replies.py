"""Read the user's answer to a quote without calling a model.

Only a short reply counts as an answer: "好的，不过先只改第一个文件" is a new task,
not an approval, so anything longer than `MAX_REPLY_CHARS` is treated as a new message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

MAX_REPLY_CHARS = 24

APPROVE = {
    "continue", "go", "go ahead", "yes", "y", "ok", "okay", "proceed", "run it", "do it", "approve", "sure",
    "继续", "好", "好的", "可以", "开始", "确认", "行", "同意", "跑吧", "继续吧",
}
CANCEL = {
    "cancel", "stop", "no", "n", "abort", "never mind", "nevermind", "skip",
    "取消", "不要", "算了", "停", "不用了", "停止", "不",
}

_CAP = re.compile(r"(?:cap|limit|budget|上限|预算)?\s*(?:to|改成|改为|=|:|：)?\s*\$\s*(\d+(?:\.\d+)?)"
                  r"|(?:cap|limit|budget|上限|预算)\s*(?:to|改成|改为|=|:|：)?\s*(\d+(?:\.\d+)?)",
                  re.IGNORECASE)
_TRIM = re.compile(r"^[\s,.!?。！？，~]+|[\s,.!?。！？，~]+$")


@dataclass
class Reply:
    kind: Literal["approve", "cancel", "cap", "other"]
    cap_usd: Optional[float] = None


def parse(text: str) -> Reply:
    t = _TRIM.sub("", (text or "").strip().lower())
    if not t or len(t) > MAX_REPLY_CHARS:
        return Reply("other")
    if t in APPROVE:
        return Reply("approve")
    if t in CANCEL:
        return Reply("cancel")
    m = _CAP.search(t)
    if m:
        return Reply("cap", float(m.group(1) or m.group(2)))
    return Reply("other")
