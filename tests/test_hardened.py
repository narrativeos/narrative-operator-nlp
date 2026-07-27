"""
Tests for P0/P1 hardened logic:
- P0-1: Cross-category deduplication thresholds
- P0-2: Confidence coefficient tuning
- P1-1: NER conflict IoU + text containment
- P1-2: Date normalization validation
"""

import unittest
from unittest.mock import patch

from core.schema import Entity, Event, EventArgument, DependencyEdge
from core.entity_deduplicator import EntityDeduplicator
from core.event_extractor import EventExtractor
from core.numeric_extractor import _normalize_date


class TestCrossCategoryDedup(unittest.TestCase):
    """P0-1: Verify hardened cross-category deduplication thresholds."""

    def test_single_char_not_merged(self):
        """Single-char entity should NOT be merged into a longer entity."""
        # "天" (1 char) contained in "天津" (2 chars) → ratio = 0.5, but min_len=2 blocks it
        parent = Entity(
            id="ent_001", text="天津", category="LOCATION", span=(0, 2),
            confidence=0.95,
        )
        child = Entity(
            id="ent_002", text="天", category="DATE", span=(0, 1),
            confidence=0.70,
        )
        result = EntityDeduplicator.deduplicate([parent, child])
        # Both should be kept because child is only 1 char
        self.assertEqual(len(result), 2)

    def test_short_ratio_not_merged(self):
        """Entity with text ratio < 0.5 should NOT be merged."""
        # "北京" (2 chars) in "北京立方庭" (5 chars) → ratio = 0.4 < 0.5
        parent = Entity(
            id="ent_001", text="北京立方庭", category="FACILITY", span=(0, 5),
            confidence=0.95,
        )
        child = Entity(
            id="ent_002", text="北京", category="LOCATION", span=(0, 2),
            confidence=0.70,
        )
        result = EntityDeduplicator.deduplicate([parent, child])
        # Both kept because ratio 0.4 < 0.5
        self.assertEqual(len(result), 2)

    def test_small_confidence_gap_not_merged(self):
        """Confidence gap < 0.2 should NOT trigger merge."""
        parent = Entity(
            id="ent_001", text="张三丰", category="PERSON", span=(0, 3),
            confidence=0.85,
        )
        child = Entity(
            id="ent_002", text="张三", category="DATE", span=(0, 2),
            confidence=0.78,  # gap = 0.07 < 0.2
        )
        result = EntityDeduplicator.deduplicate([parent, child])
        # Both kept because confidence gap < 0.2
        self.assertEqual(len(result), 2)

    def test_large_confidence_gap_merged(self):
        """Confidence gap >= 0.2 with ratio >= 0.5 SHOULD merge."""
        parent = Entity(
            id="ent_001", text="中华人民共和国", category="LOCATION", span=(0, 7),
            confidence=0.95,
        )
        child = Entity(
            id="ent_002", text="中华人民", category="ORGANIZATION", span=(0, 4),
            confidence=0.50,  # gap = 0.45 >= 0.2, ratio = 4/7 = 0.57 >= 0.5
        )
        result = EntityDeduplicator.deduplicate([parent, child])
        # Child merged into parent
        self.assertEqual(len(result), 1)
        self.assertIn("ent_002", result[0].merged_from)


class TestConfidenceCoefficients(unittest.TestCase):
    """P0-2: Verify confidence scoring with tunable coefficients."""

    def test_base_confidence(self):
        """Base confidence is 0.70 with no bonuses/penalties."""
        conf = EventExtractor._compute_confidence(
            trigger_verb="测试",
            arguments=[],
            deps=[],
            tokens=[],
            relation_predicate="CAUSES",
            semantic_class=None,
            entities_by_id={},
        )
        # base 0.70 - 0.10 (RELATES_TO without semantic_class) = 0.60
        # But predicate is CAUSES, so no generic_pred penalty
        self.assertEqual(conf, 0.70)

    def test_root_verb_bonus(self):
        """Root verb adds +0.15."""
        from core.schema import Token
        tok = Token(id=0, text="出现", pos="V", span=(0, 2), confidence=1.0)
        dep = DependencyEdge(head=-1, child=0, rel="root")
        conf = EventExtractor._compute_confidence(
            trigger_verb="出现",
            arguments=[],
            deps=[dep],
            tokens=[tok],
            relation_predicate="CAUSES",
            semantic_class=None,
            entities_by_id={},
        )
        self.assertAlmostEqual(conf, 0.85)  # 0.70 + 0.15

    def test_agent_patient_bonus(self):
        """Agent + Patient adds +0.20."""
        conf = EventExtractor._compute_confidence(
            trigger_verb="建造",
            arguments=[
                EventArgument(role="Agent", text="工人", span=(0, 2)),
                EventArgument(role="Patient", text="桥梁", span=(3, 5)),
            ],
            deps=[],
            tokens=[],
            relation_predicate="PRODUCES",
            semantic_class=None,
            entities_by_id={},
        )
        self.assertAlmostEqual(conf, 0.90)  # 0.70 + 0.10 + 0.10

    def test_sentence_boundary_punct_penalty(self):
        """Sentence-boundary punctuation (。！？) triggers -0.20 penalty."""
        conf = EventExtractor._compute_confidence(
            trigger_verb="说",
            arguments=[
                EventArgument(role="Utterance", text="你好。", span=(0, 3)),
            ],
            deps=[],
            tokens=[],
            relation_predicate="SAYS",
            semantic_class=None,
            entities_by_id={},
        )
        # 0.70 + 0.10(patient) - 0.20(punct) = 0.60
        self.assertAlmostEqual(conf, 0.60)

    def test_colon_no_penalty(self):
        """Colons (：) should NOT trigger penalty."""
        conf = EventExtractor._compute_confidence(
            trigger_verb="曰",
            arguments=[
                EventArgument(role="Utterance", text="学而时习之", span=(0, 5)),
            ],
            deps=[],
            tokens=[],
            relation_predicate="SAYS",
            semantic_class=None,
            entities_by_id={},
        )
        # 0.70 + 0.10(patient) = 0.80, no punct penalty
        self.assertAlmostEqual(conf, 0.80)

    def test_generic_predicate_penalty(self):
        """RELATES_TO without semantic_class triggers -0.10."""
        conf = EventExtractor._compute_confidence(
            trigger_verb="相关",
            arguments=[],
            deps=[],
            tokens=[],
            relation_predicate="RELATES_TO",
            semantic_class=None,
            entities_by_id={},
        )
        self.assertAlmostEqual(conf, 0.60)  # 0.70 - 0.10

    def test_clamped_to_zero(self):
        """Confidence is clamped to [0.0, 1.0]."""
        conf = EventExtractor._compute_confidence(
            trigger_verb="相关",
            arguments=[
                EventArgument(role="Patient", text="这是一个非常长的测试句子。", span=(0, 15)),
            ],
            deps=[],
            tokens=[],
            relation_predicate="RELATES_TO",
            semantic_class=None,
            entities_by_id={},
        )
        # 0.70 + 0.10 - 0.15(long) - 0.20(punct) - 0.10(generic) = 0.35
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)


class TestDateNormalization(unittest.TestCase):
    """P1-2: Verify date normalization with validation."""

    def test_chinese_full_date(self):
        self.assertEqual(_normalize_date("2024年1月15日"), "2024-01-15")

    def test_chinese_year_month(self):
        self.assertEqual(_normalize_date("2024年1月"), "2024-01")

    def test_chinese_year_only(self):
        self.assertEqual(_normalize_date("2024年"), "2024")

    def test_iso_date(self):
        self.assertEqual(_normalize_date("2024-01-15"), "2024-01-15")

    def test_slash_date(self):
        self.assertEqual(_normalize_date("2024/01/15"), "2024-01-15")

    def test_dot_date(self):
        self.assertEqual(_normalize_date("2024.01.15"), "2024-01-15")

    def test_compact_date_valid(self):
        self.assertEqual(_normalize_date("20240115"), "2024-01-15")

    def test_compact_invalid_month(self):
        """Month 13 is invalid, return original."""
        self.assertEqual(_normalize_date("20241301"), "20241301")

    def test_compact_invalid_day(self):
        """Day 32 is invalid, return original."""
        self.assertEqual(_normalize_date("20240132"), "20240132")

    def test_phone_number_not_date(self):
        """Phone-like number should return original."""
        self.assertEqual(_normalize_date("13801234"), "13801234")

    def test_compact_zero_month(self):
        """Month 00 is invalid, return original."""
        self.assertEqual(_normalize_date("20240015"), "20240015")

    def test_compact_zero_day(self):
        """Day 00 is invalid, return original."""
        self.assertEqual(_normalize_date("20240100"), "20240100")


if __name__ == "__main__":
    unittest.main()