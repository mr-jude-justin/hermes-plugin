"""Segment cost model (TokenCast arXiv 2609.35760, Eq. 1, linear-growth case).

A task is K LLM calls. Call k sends a prompt of length L_k = L1 + g·(k−1) and
emits o output tokens. Summed over the task:

    total prompt = K·L1 + g·K(K−1)/2        (quadratic in K)

With prompt caching, the part of each prompt that repeats the previous call's
prompt is the cached prefix. A fraction h of it is billed as cache_read, and the
rest is uncached input. The newly appended part is written to the cache. Without
caching (h == 0) every prompt token is plain input.

K may be fractional (it is a P50/P90 of a distribution). The closed forms below
are smooth in K.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenSplit:
    input: float
    cache_read: float
    cache_write: float
    output: float

    @property
    def total(self) -> float:
        return self.input + self.cache_read + self.cache_write + self.output

    def __add__(self, other: TokenSplit) -> TokenSplit:
        return TokenSplit(
            self.input + other.input,
            self.cache_read + other.cache_read,
            self.cache_write + other.cache_write,
            self.output + other.output,
        )

    def scaled(self, factor: float) -> TokenSplit:
        f = float(factor)
        return TokenSplit(self.input * f, self.cache_read * f, self.cache_write * f, self.output * f)

    def as_dict(self) -> dict[str, int]:
        return {
            "input": round(self.input),
            "cache_read": round(self.cache_read),
            "cache_write": round(self.cache_write),
            "output": round(self.output),
        }


def project(
    steps: float,
    start_prompt: float,
    growth: float,
    output_per_step: float,
    cache_hit: float,
    warm_cache: bool = False,
) -> TokenSplit:
    """Token split for `steps` calls starting at prompt length `start_prompt`.

    `warm_cache`: the first call's prompt prefix is already cached (later turn
    of a session). If False, the first call writes the whole prompt to the cache.
    """
    k = max(0.0, float(steps))
    if k == 0:
        return TokenSplit(0.0, 0.0, 0.0, 0.0)
    l1 = max(0.0, float(start_prompt))
    g = max(0.0, float(growth))
    h = min(1.0, max(0.0, float(cache_hit)))

    total_prompt = k * l1 + g * k * (k - 1) / 2
    output = k * max(0.0, float(output_per_step))

    if h == 0.0:
        return TokenSplit(total_prompt, 0.0, 0.0, output)

    # Prefix of call 1 is L1 if warm, else nothing. Prefix of call j ≥ 2 is
    # L_{j−1}. The sum over j = 2..K of L_{j−1} is (K−1)·L1 + g·(K−1)(K−2)/2.
    km1 = max(0.0, k - 1)
    prefix = (l1 if warm_cache else 0.0) + km1 * l1 + g * km1 * max(0.0, k - 2) / 2
    prefix = min(prefix, total_prompt)
    new = total_prompt - prefix

    return TokenSplit(
        input=(1 - h) * prefix,
        cache_read=h * prefix,
        cache_write=new,
        output=output,
    )
