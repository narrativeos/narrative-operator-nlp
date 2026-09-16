"""
Negation Detection — Identify negation scope in text.

Detects negation words and their scope, allowing downstream components
to understand that a relation/attribute is negated.

Negation words supported:
- 不 (not)
- 没/没有 (did not / do not have)
- 未 (not yet)
- 非 (non-)
- 无 (without)
- 勿 (do not)
- 别 (don't)
- 莫 (not)
- 未尝 (never)
- 不曾 (never)
"""

from __future__ import annotations

import re
from typing import Optional


# Negation words (sorted by length descending for longest-match-first)
# Includes both modern Chinese and classical Chinese negation words
_NEGATION_WORDS = [
    # Multi-char negation words (modern + classical)
    "没有", "不曾", "未尝", "未始", "未能",
    # Single-char negation words (modern + classical)
    "不", "没", "未", "非", "无", "勿", "别", "莫",
    # Classical Chinese specific negation words
    "弗", "毋", "罔", "微",
]

# Build regex pattern: match any negation word
_NEGATION_RE = re.compile(
    r"(?:{})".format("|".join(re.escape(w) for w in sorted(_NEGATION_WORDS, key=len, reverse=True)))
)

# Sentence-final punctuation: a negation word's scope never crosses
# a sentence boundary (北京没有X。Y生产Z — 没有 does not negate Z).
_SENT_END_CHARS = frozenset("。！？；.!?;")


def find_negation_spans(text: str) -> list[tuple[int, int]]:
    """Find all negation words in text.

    Returns list of (start, end) spans for each negation word.
    """
    spans = []
    for match in _NEGATION_RE.finditer(text):
        spans.append((match.start(), match.end()))
    return spans


def is_in_negation_scope(
    span: tuple[int, int],
    negation_spans: list[tuple[int, int]],
    max_distance: int = 20,
    text: str = "",
) -> bool:
    """Check if a span falls within the scope of a negation word.

    A span is considered negated if:
    1. A negation word falls INSIDE the span (the common Chinese case:
       北京[没有]生产碳纤维 — the evidence span covers subject..object
       and the negation sits between them), OR
    2. The span starts after a negation word, within max_distance
       characters of it, with no sentence boundary in between
       (a negation's scope never crosses 。/！/？/；).

    This is a simplified scope model. Full scope resolution requires
    syntactic parsing (which we avoid for performance).

    Args:
        span: (start, end) of the entity/relation
        negation_spans: list of negation word spans
        max_distance: maximum characters between negation and target
        text: original text (enables the sentence-boundary guard)
    """
    s, e = span
    for ns, ne in negation_spans:
        # Negation word inside the span
        if ns >= s and ne <= e:
            return True
        # Target appears after negation, within scope
        if s >= ne and s - ne <= max_distance:
            # A sentence boundary between negation and target ends
            # the negation's scope.
            if text and any(c in _SENT_END_CHARS for c in text[ne:s]):
                continue
            return True
    return False


def compute_negation_aware_confidence(
    base_confidence: float,
    span: tuple[int, int],
    negation_spans: list[tuple[int, int]],
    text: str = "",
) -> tuple[float, bool]:
    """Adjust confidence based on negation scope.

    Returns:
        (adjusted_confidence, is_negated)
    """
    is_neg = is_in_negation_scope(span, negation_spans, text=text)
    if is_neg:
        # Reduce confidence for negated relations
        # (downstream can choose to filter or flag these)
        return (base_confidence * 0.7, True)
    return (base_confidence, False)