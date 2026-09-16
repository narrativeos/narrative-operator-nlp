"""
Tests: Language Detector (core/language_detector.py).

Focus: the three-way detector must return a *label-relative* confidence,
i.e. P(label | text). A clearly modern sentence must therefore carry a
HIGH modern confidence, not a low one. This guards against the historical
bug where the raw classical score was surfaced as the modern confidence,
making obviously-modern sentences appear "low confidence" in the UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from core.language_detector import (
    CLASSICAL_THRESHOLD,
    classical_confidence,
    detect_language,
)


# ---------------------------------------------------------------------------
# Fixtures — representative samples per language
# ---------------------------------------------------------------------------

MODERN_SAMPLES = [
    "阿里巴巴集团成立于1999年，由马云等人在杭州创立。",
    "公司最初专注于B2B电子商务，为中小企业提供在线交易平台。",
    "人工智能技术正在改变世界。",
    "深度学习是人工智能的重要分支。",
]

CLASSICAL_SAMPLES = [
    "陈胜者，阳城人也，字涉。",
    "吾日三省吾身：为人谋而不忠乎？",
    "学而时习之，不亦说乎？",
    "夫君子之行，静以修身，俭以养德。",
]

ENGLISH_SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "This is a simple English sentence for testing.",
]


# ---------------------------------------------------------------------------
# Label-relative confidence semantics
# ---------------------------------------------------------------------------

class TestConfidenceSemantics:
    def test_confidence_always_in_unit_interval(self):
        for sample in MODERN_SAMPLES + CLASSICAL_SAMPLES + ENGLISH_SAMPLES:
            label, conf = detect_language(sample)
            assert 0.0 <= conf <= 1.0, f"{sample!r} -> conf {conf} out of range"

    def test_modern_sentence_has_high_modern_confidence(self):
        """Regression: modern text must NOT be reported with a low confidence.

        Before the fix, detect_language returned the raw classical score for
        the "modern" label, so these sentences showed ~0.05–0.37 in the UI.
        """
        for sample in MODERN_SAMPLES:
            label, conf = detect_language(sample)
            assert label == "modern", f"{sample!r} mislabeled as {label}"
            assert conf > 0.5, (
                f"Modern sentence should carry high modern confidence, "
                f"got {conf:.3f} for {sample!r}"
            )

    def test_classical_sentence_has_classical_confidence(self):
        for sample in CLASSICAL_SAMPLES:
            label, conf = detect_language(sample)
            assert label == "classical", f"{sample!r} mislabeled as {label}"
            assert conf >= CLASSICAL_THRESHOLD, (
                f"Classical confidence {conf:.3f} below threshold for {sample!r}"
            )

    def test_english_detected(self):
        for sample in ENGLISH_SAMPLES:
            label, conf = detect_language(sample)
            assert label == "english", f"{sample!r} mislabeled as {label}"
            assert conf > 0.5

    def test_modern_confidence_is_complement_of_classical(self):
        """For a non-classical Chinese sentence, modern confidence == 1 - classical."""
        sample = "人工智能技术正在改变世界。"
        label, conf = detect_language(sample)
        assert label == "modern"
        # classical_confidence is the raw (pre-mixed-penalty) score; the
        # returned modern confidence must be strictly higher than the raw
        # classical score (the old bug returned the raw classical score).
        assert conf > classical_confidence(sample)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_very_short_text_defaults_to_modern(self):
        label, conf = detect_language("好")
        assert label == "modern"
        assert 0.0 <= conf <= 1.0

    def test_latin_heavy_text_is_modern_not_classical(self):
        """Text with heavy Latin content is never classical."""
        label, conf = detect_language("B2B C2C IPO 1+6+N 电子商务")
        assert label != "classical"
        assert label == "modern"
        assert conf > 0.5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
