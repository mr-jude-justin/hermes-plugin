"""Hashed n-gram vectors for L2 similarity (TF-IDF + cosine, no raw text kept).

TokenCast represents tasks with TF-IDF over word 1–2 grams and char 3–5 grams.
We use the same features but hash them into buckets with a stable hash (crc32),
so only bucket counts are stored, never the message text (privacy rule agreed in
docs/plan.md 5.2). Chinese has no spaces, so each CJK character counts as a "word".
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from typing import Iterable, Mapping

BUCKETS = 1 << 18
MAX_CHARS = 2000

_WORD = re.compile(r"[a-z0-9_]+|[㐀-鿿豈-﫿぀-ヿ가-힯]")
_SPACE = re.compile(r"\s+")

Vec = dict[str, float]  # bucket (as str, JSON-friendly) -> weight


def _bucket(token: str) -> str:
    return str(zlib.crc32(token.encode("utf-8")) % BUCKETS)


def hashed_counts(text: str) -> dict[str, int]:
    """Raw hashed n-gram counts of a message. This is what gets stored."""
    t = _SPACE.sub(" ", (text or "").lower()).strip()[:MAX_CHARS]
    if not t:
        return {}
    grams: list[str] = []
    words = _WORD.findall(t)
    grams += ["w:" + w for w in words]
    grams += ["b:" + a + " " + b for a, b in zip(words, words[1:])]
    for n in (3, 4, 5):
        grams += ["c:" + t[i : i + n] for i in range(len(t) - n + 1)]
    return dict(Counter(_bucket(g) for g in grams))


def document_frequency(docs: Iterable[Mapping[str, int]]) -> tuple[Counter, int]:
    df: Counter = Counter()
    n = 0
    for d in docs:
        df.update(d.keys())
        n += 1
    return df, n


def tfidf(counts: Mapping[str, int], df: Mapping[str, int], n_docs: int) -> Vec:
    """Sublinear TF × smoothed IDF, L2-normalised."""
    vec: Vec = {}
    for k, c in counts.items():
        idf = math.log((1 + n_docs) / (1 + df.get(k, 0))) + 1.0
        vec[k] = (1.0 + math.log(c)) * idf
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm == 0:
        return {}
    return {k: v / norm for k, v in vec.items()}


def cosine(a: Vec, b: Vec) -> float:
    """Cosine of two L2-normalised sparse vectors."""
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())
