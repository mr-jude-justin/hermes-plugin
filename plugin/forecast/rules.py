"""L1: rule-based task typing and tiering (< 1 ms, no I/O, no model calls).

The rules give (a) a task type, which sets the per-step cost shape, (b) a tier,
which sets how many steps to expect, and (c) per-call billed units such as images.
The output is a cold-start prior. L2 history and calibration refine it. See
DESIGN.md sections 2, 2.1 and 3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import defaults as D
from .features import (
    approx_tokens,
    count_paths,
    count_urls,
    get,
    has_code_block,
    message_of,
    toolsets,
)
from .tokens import TokenSplit, project


def _rx(*parts: str) -> re.Pattern[str]:
    return re.compile("|".join(parts), re.IGNORECASE)


# Keyword patterns per task type, Chinese and English. English uses word
# boundaries. Chinese has none, so its patterns are short phrases.
_TYPE_PATTERNS: dict[str, re.Pattern[str]] = {
    "agentic_coding": _rx(
        r"修复|修一下|修个|修掉|改一下|改成|实现|重构|迁移|加个功能|新增功能|加功能",
        r"写测试|加测试|补测试|跑测试|跑一下测试|部署|升级依赖|搭建|脚手架|开发一个",
        r"写一个.{0,8}(脚本|程序|工具|插件|服务|接口|网站|应用)",
        r"\b(fix|implement|refactor|migrate|rewrite|scaffold|deploy|port)\b",
        r"\badd (a |an |the )?(feature|tests?|endpoint|command|flag|option)\b",
        r"\bwrite (unit |integration )?tests\b",
        r"\b(build|create|make) (a |an )?(script|cli|app|tool|plugin|service|website|api)\b",
    ),
    "code_reasoning": _rx(
        r"解释|为什么|怎么工作|原理|读一下|读读|审查|代码审查|梳理|定位|看看.{0,6}(代码|模块|函数|逻辑)",
        r"分析.{0,6}(代码|模块|仓库|项目)|找出.{0,6}(bug|问题)",
        r"\b(explain|review|walk me through|understand|analy[sz]e|trace)\b",
        r"\b(why does|how does|where is|what does)\b",
        r"\bfind the (bug|issue|cause)\b",
    ),
    "coding_chat": _rx(
        r"这段代码|这个函数|这个报错|报错信息|写一个函数|正则",
        r"\b(this code|this function|snippet|regex|stack ?trace|error message|one-liner)\b",
    ),
    "research": _rx(
        r"搜索|搜一下|查一下|查查|查资料|调研|最新|新闻|论文|文献|市场|竞品|对比.{0,6}(方案|产品|工具)",
        r"\b(search|look up|research|latest|news|papers?|literature|survey|find sources)\b",
        r"\bcompare\b",
    ),
    "generation": _rx(
        r"画一|画个|画张|画幅|帮我画|生成.{0,6}(图|图片|图像|海报|logo|头像|插画|壁纸)|做一张|配图|出图",
        r"\bdraw\b",
        r"\b(generate|create|make|render) (an? |some |\d+ )?(images?|pictures?|logos?|posters?|illustrations?|icons?|wallpapers?)\b",
    ),
    "delegated": _rx(
        r"分头|并行|子任务|子agent|子 agent|多个agent|多个 agent|派.{0,4}agent|分别调研|同时.{0,4}(几个|多个)",
        r"\b(in parallel|parallelize|subagents?|sub-agents?|delegate|fan out|multiple agents)\b",
        r"\bsplit (it|this|the work) up\b",
    ),
    "browser": _rx(
        r"打开网页|浏览器|点击|填表|填写表单|截图|网页上|登录.{0,6}(网站|页面|后台)|操作.{0,4}(网页|页面|电脑|桌面)",
        r"\b(browser|click|screenshot|navigate to|fill (out|in) (the |a )?form)\b",
        r"\bopen (the |this )?(page|site|website|url)\b",
        r"\blog ?in to\b",
    ),
    "long_doc": _rx(
        r"读完|通读|全文|整本|总结.{0,6}(文件|文档|这些|报告|论文|pdf|PDF)",
        r"\bsummari[sz]e (these|the|this|all) (files|documents|docs|reports?|papers?|pdfs?|book)\b",
        r"\bread (all|through|these)\b",
        r"\b(entire|whole) (document|book|report)\b",
    ),
}

# A toolset only counts as a weak supporting signal, because it is enabled for
# every message. It never decides a type on its own.
_TYPE_TOOLSETS: dict[str, tuple[str, ...]] = {
    "agentic_coding": ("terminal", "file"),
    "research": ("web",),
    "generation": ("image_gen",),
    "delegated": ("delegation",),
    "browser": ("browser", "computer_use", "vision"),
}

_SCOPE_UP = _rx(
    r"整个|全部|所有|整套|从零|从头|端到端|完整|全面|大规模|全量|每个文件",
    r"\b(entire|whole|from scratch|end[- ]to[- ]end|full|complete|everything)\b",
    r"\ball (the )?files\b|\bacross the (repo|codebase|project)\b",
)
_SCOPE_DOWN = _rx(
    r"简单|小改|一行|顺手|快速|小问题|改个名|就一个|一个小|稍微",
    r"\b(quick|quickly|small|simple|tiny|one[- ]line|typo|minor|just)\b",
)

_IMAGE_COUNT = re.compile(
    r"(\d+)\s*(张|幅|个|images?|pictures?|pics?|logos?|posters?|icons?|variations?)",
    re.IGNORECASE,
)
_CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "几": 3}
_IMAGE_COUNT_CN = re.compile(r"([一两二三四五六几])\s*(张|幅)")

_SUBAGENT_COUNT = re.compile(
    r"(\d+)\s*(个|路)?\s*(方向|子任务|子 ?agent|agents?|subagents?|sub-agents?|parts?|directions?|topics?)",
    re.IGNORECASE,
)
_SUBAGENT_COUNT_CN = re.compile(r"([两二三四五六几])\s*个\s*(方向|子任务|子 ?agent|部分|问题)", re.IGNORECASE)

_CODE_TYPES = {"agentic_coding", "code_reasoning", "coding_chat"}


@dataclass
class L1Result:
    task_type: str
    tier: str
    type_weights: dict[str, float]
    steps_p50: float
    steps_p90: float
    growth: float
    output_per_step: float
    cache_hit: float
    start_prompt: float
    warm_cache: bool
    tokens_p50: TokenSplit
    tokens_p90: TokenSplit
    per_call_units: dict[str, int]
    confidence: float
    signals: list[str] = field(default_factory=list)
    subagents_p50: float = 0.0
    subagents_p90: float = 0.0

    @property
    def basis(self) -> str:
        return "rules: " + (", ".join(self.signals) if self.signals else "no strong signal")


def _type_scores(features: Any, msg: str, sigs: list[str]) -> dict[str, float]:
    scores = {t: 0.0 for t in D.TASK_TYPES}
    scores["chat"] = 0.5  # fallback

    for t, pat in _TYPE_PATTERNS.items():
        hits = {m.group(0).lower() for m in pat.finditer(msg)}
        if hits:
            scores[t] += 1.0 + 0.5 * (len(hits) - 1)
            sigs.append(f"{t} keywords")

    tools = toolsets(features)
    for t, names in _TYPE_TOOLSETS.items():
        if scores[t] > 0 and any(n in tools for n in names):
            scores[t] += 0.3

    code_block = has_code_block(features)
    paths = count_paths(msg)
    urls = count_urls(msg)
    in_repo = bool(get(features, "cwd_is_git_repo", False))
    attachments = int(get(features, "attachments", 0) or 0)

    if code_block:
        scores["coding_chat"] += 1.5
        sigs.append("code block")
    if paths:
        scores["code_reasoning"] += 0.3
        scores["agentic_coding"] += 0.3
        sigs.append(f"{paths} path(s)")
    if in_repo and (scores["agentic_coding"] >= 1 or scores["code_reasoning"] >= 1):
        # Being in a repo makes "do it in the repo" more likely than "talk about a snippet".
        scores["agentic_coding"] += 0.3
        scores["code_reasoning"] += 0.3
        if code_block:
            scores["coding_chat"] -= 0.5
    if urls:
        scores["research"] += 0.5 * min(urls, 3)
        sigs.append(f"{urls} url(s)")
    if attachments:
        scores["long_doc"] += 0.7 if attachments == 1 else 1.5
        sigs.append(f"{attachments} attachment(s)")

    return scores


def _weights(scores: dict[str, float]) -> dict[str, float]:
    top = max(scores.values())
    kept = {t: s for t, s in scores.items() if s > 0 and s >= 0.5 * top}
    total = sum(kept.values())
    return {t: round(s / total, 3) for t, s in sorted(kept.items(), key=lambda kv: -kv[1])}


def _image_units(msg: str) -> int:
    m = _IMAGE_COUNT.search(msg)
    if m:
        return max(1, min(int(m.group(1)), 50))
    m = _IMAGE_COUNT_CN.search(msg)
    if m:
        return _CN_NUM[m.group(1)]
    return 1


def _subagent_counts(msg: str) -> tuple[float, float]:
    m = _SUBAGENT_COUNT.search(msg)
    if m:
        n = float(max(1, min(int(m.group(1)), 20)))
        return n, n
    m = _SUBAGENT_COUNT_CN.search(msg)
    if m:
        n = float(_CN_NUM[m.group(1)])
        return n, n
    return D.SUBAGENTS_DEFAULT


def classify(features: Any) -> L1Result:
    """Run the L1 rules on one task's features."""
    msg = message_of(features)
    sigs: list[str] = []

    scores = _type_scores(features, msg, sigs)
    weights = _weights(scores)
    task_type = next(iter(weights))
    ordered = sorted(scores.values(), reverse=True)
    margin = (ordered[0] - ordered[1]) / ordered[0] if ordered[0] > 0 else 0.0

    # Tier: start at the type's default and nudge it with scope signals.
    shift = 0.0
    if _SCOPE_UP.search(msg):
        shift += 1
        sigs.append("broad scope")
    if _SCOPE_DOWN.search(msg):
        shift -= 1
        sigs.append("small scope")
    if task_type in _CODE_TYPES and count_paths(msg) >= 3:
        shift += 1
        sigs.append("many files")
    repo = get(features, "repo_size", {}) or {}
    repo_lines = repo.get("lines", 0) if isinstance(repo, dict) else getattr(repo, "lines", 0)
    if task_type in ("agentic_coding", "code_reasoning") and (repo_lines or 0) > D.LARGE_REPO_LINES:
        shift += 0.5
        sigs.append("large repo")
    msg_tokens = approx_tokens(msg)
    if msg_tokens > 1500:
        shift += 0.5
        sigs.append("long spec")
    if task_type == "long_doc" and int(get(features, "attachments", 0) or 0) >= 3:
        shift += 1
    context_now = int(get(features, "context_tokens_now", 0) or 0)
    if context_now > D.LONG_CONTEXT_TOKENS:
        shift += 1
        sigs.append("long context")

    base = D.TIERS.index(D.TYPE_SHAPES[task_type].default_tier)
    tier_idx = int(min(len(D.TIERS) - 1, max(0, base + int(shift + 0.5 if shift >= 0 else shift - 0.5))))
    tier = D.TIERS[tier_idx]
    k50, k90 = D.TIER_STEPS[tier]

    # Shape: blend g and o over the candidate types.
    growth = sum(w * D.TYPE_SHAPES[t].growth for t, w in weights.items())
    output = sum(w * D.TYPE_SHAPES[t].output for t, w in weights.items())
    effort = str(get(features, "reasoning_effort", "") or "").lower()
    if effort in D.EFFORT_OUTPUT_MULT:
        output *= D.EFFORT_OUTPUT_MULT[effort]
        sigs.append(f"effort {effort}")

    model = get(features, "model", None)
    cache_hit = D.cache_hit_for(model)
    start_prompt = float((context_now or D.HERMES_FLOOR_TOKENS) + msg_tokens)
    warm = int(get(features, "session_turns_so_far", 0) or 0) > 0

    tokens_p50 = project(k50, start_prompt, growth, output, cache_hit, warm)
    tokens_p90 = project(k90, start_prompt, growth, output, cache_hit, warm)

    # Delegated work: each subagent is its own segment that starts at its own
    # floor with a cold cache (DESIGN.md 6.5 item 4).
    n50 = n90 = 0.0
    if task_type == "delegated":
        n50, n90 = _subagent_counts(msg)
        s50, s90 = D.SUBAGENT_STEPS
        sub_g, sub_o = D.TYPE_SHAPES["research"].growth, D.TYPE_SHAPES["research"].output
        one50 = project(s50, D.SUBAGENT_FLOOR_TOKENS, sub_g, sub_o, cache_hit)
        one90 = project(s90, D.SUBAGENT_FLOOR_TOKENS, sub_g, sub_o, cache_hit)
        tokens_p50 = tokens_p50 + one50.scaled(n50)
        tokens_p90 = tokens_p90 + one90.scaled(n90)
        sigs.append(f"~{n50:g} subagents")

    units: dict[str, int] = {}
    if task_type == "generation" or ("generation" in weights and "image_gen" in toolsets(features)):
        units["image_gen"] = _image_units(msg)

    # L1 alone is a weak predictor (DESIGN.md section 0), so confidence stays low.
    if task_type == "chat" and len(sigs) == 0 and msg_tokens < 40:
        confidence = 0.5
    else:
        confidence = round(min(0.45, 0.25 + 0.2 * margin), 2)

    return L1Result(
        task_type=task_type,
        tier=tier,
        type_weights=weights,
        steps_p50=k50,
        steps_p90=k90,
        growth=growth,
        output_per_step=output,
        cache_hit=cache_hit,
        start_prompt=start_prompt,
        warm_cache=warm,
        tokens_p50=tokens_p50,
        tokens_p90=tokens_p90,
        per_call_units=units,
        subagents_p50=n50,
        subagents_p90=n90,
        confidence=confidence,
        signals=sigs,
    )
