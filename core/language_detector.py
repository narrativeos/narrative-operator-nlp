"""
Language Detector — Multi-Feature Confidence Scoring.

Classifies Chinese text as modern or classical using a weighted
multi-feature scoring approach. Returns confidence scores used by
the two-stage pipeline in analyzer.py to decide which NLP model
to apply first (and whether to fall back).
"""

from __future__ import annotations

import re
from typing import Literal

# ---------------------------------------------------------------------------
# Feature Sets
# ---------------------------------------------------------------------------

# Classical Chinese marker characters (虚词、助词、语气词)
_CLASSICAL_CHARS: frozenset[str] = frozenset(
    "之乎者也焉矣其乃若夫盖兮耳哉欤耶欤欤"
)

# Classical Chinese personal pronouns
_CLASSICAL_PRONOUNS: frozenset[str] = frozenset(
    "吾余予汝尔卿寡人朕孤臣妾"
)

# Classical Chinese negation patterns
_CLASSICAL_NEGATION = re.compile(r"[不未弗毋勿非微]\w?")

# Classical copula / judgement pattern
_CLASSICAL_COPULA = re.compile(r"者[^。！？\n]{1,20}也")

# Classical interrogative
_CLASSICAL_INTERROG = re.compile(r"[何胡奚曷安焉恶孰]\w?")

# Modern Chinese markers (strong negative signal for classical)
_MODERN_PARTICLES = frozenset("的了着们")
_MODERN_WORDS: tuple[str, ...] = (
    "我们", "你们", "他们", "她们", "这个", "那个", "哪个",
    "因为", "所以", "而且", "但是", "虽然", "如果", "可以",
    "应该", "已经", "正在", "什么", "怎么", "为什么",
)

# Classical sentence-ending particles
_CLASSICAL_FINAL = frozenset("也矣焉耳乎哉欤耶")

# Minimum sentence length to apply classification
_MIN_SENTENCE_LENGTH = 3


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def classical_confidence(text: str) -> float:
    """
    Compute a confidence score [0.0, 1.0] that ``text`` is classical Chinese.

    0.0 = definitively modern
    0.5 = ambiguous
    1.0 = definitively classical

    The score is used to decide:
    - Which model to try FIRST (modern if < 0.5, classical if >= 0.5)
    - Whether to fall back to the other model on poor-quality output
    """
    text = text.strip()
    n = len(text)
    if n < _MIN_SENTENCE_LENGTH:
        return 0.0  # Too short to classify reliably

    score = 0.0

    # ---- Positive signals (classical) ----

    # 1. Classical marker density (weak: 0.0–0.30)
    marker_count = sum(1 for c in text if c in _CLASSICAL_CHARS)
    score += min(marker_count / max(n, 1) * 6, 0.30)

    # 2. Classical personal pronouns (strong: 0.0–0.15)
    pronoun_hits = sum(1 for c in text if c in _CLASSICAL_PRONOUNS)
    score += min(pronoun_hits * 0.08, 0.15)

    # 3. Classical negation pattern (medium: 0.0–0.15)
    if _CLASSICAL_NEGATION.search(text):
        score += 0.15

    # 4. 者...也 copula (very strong: 0.0–0.15)
    if _CLASSICAL_COPULA.search(text):
        score += 0.15

    # 5. Classical interrogatives (medium: 0.0–0.10)
    interrog_hits = len(_CLASSICAL_INTERROG.findall(text))
    score += min(interrog_hits * 0.08, 0.10)

    # 6. Classical sentence-final particles (strong: 0.0–0.15)
    if text and text[-1] in _CLASSICAL_FINAL:
        score += 0.15

    # 7. Short, dense sentences favor classical (weak: 0.0–0.05)
    if n < 20 and marker_count >= 1:
        score += 0.05

    # 8. Classical quoting pattern: 曰 / 云 (medium: 0.0–0.10)
    if "曰" in text or "云" in text:
        score += 0.10

    # ---- Negative signals (modern) ----

    # 8. Modern particles ("的","了","着","们")
    modern_count = sum(1 for c in text if c in _MODERN_PARTICLES)
    score -= min(modern_count * 0.04, 0.20)

    # 9. Modern function words
    word_penalty = 0.0
    for w in _MODERN_WORDS:
        if w in text:
            word_penalty += 0.05
    score -= min(word_penalty, 0.25)

    # 10. 的 + noun pattern (strong modern signal)
    if re.search(r"的[\u4e00-\u9fff]", text):
        score -= 0.08

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

LanguageClass = Literal["modern", "classical"]


def classify(text: str) -> tuple[LanguageClass, float]:
    """
    Classify text and return (primary_language, confidence).

    The primary language determines which model to try FIRST.
    Confidence drives the fallback decision in the two-stage pipeline.

    Thresholds:
        confidence < 0.25  →  modern-first, no fallback
        0.25 <= c < 0.45   →  modern-first, with fallback
        0.45 <= c < 0.65   →  classical-first, with fallback
        c >= 0.65          →  classical-first, no fallback
    """
    conf = classical_confidence(text)
    if conf >= 0.40:
        return ("classical", conf)
    return ("modern", conf)


def should_fallback(confidence: float) -> bool:
    """Whether to try the alternative model on poor-quality output."""
    return 0.25 <= confidence <= 0.65
