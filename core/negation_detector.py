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
_NEGATION_WORDS = [
    "没有", "不曾", "未尝",
    "不", "没", "未", "非", "无", "勿", "别", "莫",
]

# Build regex pattern: match any negation word
_NEGATION_RE = re.compile(
    r"(?:{})".format("|".join(re.escape(w) for w in sorted(_NEGATION_WORDS, key=len, reverse=True)))
)


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
) -> bool:
    """Check if a span falls within the scope of a negation word.

    A span is considered negated if:
    1. It appears after a negation word
    2. Within max_distance characters of the negation word

    This is a simplified scope model. Full scope resolution requires
    syntactic parsing (which we avoid for performance).

    Args:
        span: (start, end) of the entity/relation
        negation_spans: list of negation word spans
        max_distance: maximum characters between negation and target
    """
    s, e = span
    for ns, ne in negation_spans:
        # Target appears after negation, within scope
        if s >= ne and s - ne <= max_distance:
            return True
    return False


def compute_negation_aware_confidence(
    base_confidence: float,
    span: tuple[int, int],
    negation_spans: list[tuple[int, int]],
) -> tuple[float, bool]:
    """Adjust confidence based on negation scope.

    Returns:
        (adjusted_confidence, is_negated)
    """
    is_neg = is_in_negation_scope(span, negation_spans)
    if is_neg:
        # Reduce confidence for negated relations
        # (downstream can choose to filter or flag these)
        return (base_confidence * 0.7, True)
    return (base_confidence, False)