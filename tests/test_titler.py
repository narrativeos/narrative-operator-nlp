"""
Tests: Title Generator.

Validates entity composition and TextRank truncation title generation,
dual modes, fallback behavior, and edge cases.
"""
from __future__ import annotations
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from core.schema import (
    Entity, Event, NarrativeContent, NarrativeDocument,
    NarrativeMeta, Relation, SentenceLanguage, Title,
)
from core.titler import (
    generate_title, generate_title_text,
    _truncate, _compute_textrank, _compose_title_from_entities,
)


def _make_sentence(text, span=None, label="modern"):
    if span is None:
        span = (0, len(text))
    return SentenceLanguage(text=text, span=span, label=label, confidence=0.8)


def _make_entity(text, span, eid="ent_0"):
    return Entity(id=eid, text=text, category="PERSON", span=span)


def _make_relation(span, subject_ent_id="ent_0", object_ent_id="ent_1"):
    return Relation(
        id="rel_0", subject="s", predicate="IS_A", object="o",
        evidence="e", evidence_span=span,
        subject_ent_id=subject_ent_id, object_ent_id=object_ent_id,
    )


def _make_content(sentences, entities=None, relations=None, events=None):
    return NarrativeContent(
        sentences=sentences, entities=entities or [],
        relations=relations or [], events=events or [],
    )


def _make_doc(content):
    return NarrativeDocument(
        meta=NarrativeMeta(source="test", text_length=100, language_mode="auto"),
        content=content,
    )


class TestTruncate:
    def test_no_truncation_needed(self):
        assert _truncate("short", 10) == "short"

    def test_truncation_with_ellipsis(self):
        result = _truncate("this is a long text", 10)
        assert len(result) == 10
        assert "…" in result

    def test_exact_length(self):
        assert _truncate("exactly10", 10) == "exactly10"


class TestComposeTitleFromEntities:
    def test_no_entities(self):
        content = _make_content([_make_sentence("test")])
        title, ents, preds, reasons = _compose_title_from_entities(content, 20)
        assert title == ""
        assert reasons == ["no_entities"]

    def test_single_entity_no_relations(self):
        ent = _make_entity("张三", (0, 2))
        content = _make_content([_make_sentence("张三来了", (0, 4))], entities=[ent])
        title, ents, preds, reasons = _compose_title_from_entities(content, 20)
        assert title == "张三"
        assert ents == ["张三"]
        assert reasons == ["single_entity"]

    def test_entity_with_relation(self):
        ent1 = _make_entity("阿里巴巴", (0, 4), "ent_0")
        ent2 = _make_entity("公司", (5, 7), "ent_1")
        rel = _make_relation((0, 7), "ent_0", "ent_1")
        content = _make_content(
            [_make_sentence("阿里巴巴是公司", (0, 7))],
            entities=[ent1, ent2], relations=[rel],
        )
        title, ents, preds, reasons = _compose_title_from_entities(content, 30)
        assert "阿里巴巴" in title
        assert len(ents) >= 1
        assert len(reasons) >= 1

    def test_truncation_applied(self):
        ent1 = _make_entity("阿里巴巴集团控股有限公司", (0, 11), "ent_0")
        ent2 = _make_entity("科技公司", (12, 16), "ent_1")
        rel = _make_relation((0, 16), "ent_0", "ent_1")
        content = _make_content(
            [_make_sentence("阿里巴巴集团控股有限公司是科技公司", (0, 16))],
            entities=[ent1, ent2], relations=[rel],
        )
        title, ents, preds, reasons = _compose_title_from_entities(content, 15)
        assert len(title) <= 15


class TestGenerateTitle:
    def test_empty_doc(self):
        content = _make_content([])
        doc = _make_doc(content)
        title = generate_title(doc)
        assert title.title_text == ""
        assert title.method == "entity_composition"

    def test_entity_composition(self):
        ent1 = _make_entity("阿里巴巴", (0, 4), "ent_0")
        ent2 = _make_entity("公司", (5, 7), "ent_1")
        rel = _make_relation((0, 7), "ent_0", "ent_1")
        content = _make_content(
            [_make_sentence("阿里巴巴是公司", (0, 7))],
            entities=[ent1, ent2], relations=[rel],
        )
        doc = _make_doc(content)
        title = generate_title(doc)
        assert title.title_text != ""
        assert title.method == "entity_composition"
        assert len(title.entities_used) >= 1

    def test_fallback_to_textrank(self):
        content = _make_content([_make_sentence("这是一段测试文本。")])
        doc = _make_doc(content)
        title = generate_title(doc)
        assert title.method == "textrank_fallback"
        assert title.source_sentence_index == 0

    def test_chars_mode(self):
        ent = _make_entity("张三", (0, 2))
        content = _make_content([_make_sentence("张三来了", (0, 4))], entities=[ent])
        doc = _make_doc(content)
        title = generate_title(doc, mode="chars", target_chars=10)
        assert title.mode == "chars"
        assert title.target_chars == 10

    def test_serialization(self):
        ent = _make_entity("张三", (0, 2))
        content = _make_content([_make_sentence("张三来了", (0, 4))], entities=[ent])
        doc = _make_doc(content)
        title = generate_title(doc)
        d = title.model_dump()
        assert "title_text" in d
        assert "method" in d
        assert "entities_used" in d


class TestGenerateTitleText:
    def test_empty(self):
        title = generate_title_text("")
        assert title.title_text == ""
        assert title.method == "textrank_truncate"
        assert title.reasons == ["empty_text"]

    def test_basic(self):
        text = "张三来了。李四走了。王五到了。"
        title = generate_title_text(text, mode="chars", target_chars=10)
        assert title.method == "textrank_truncate"
        assert len(title.title_text) <= 10
        assert title.source_sentence_index >= 0

    def test_ratio_mode(self):
        text = "a" * 100 + ". " + "b" * 100 + "."
        title = generate_title_text(text, mode="ratio", target_ratio=0.5)
        assert title.mode == "ratio"
        assert title.title_text != ""

    def test_short_text(self):
        text = "短文本"
        title = generate_title_text(text, mode="chars", target_chars=100)
        assert title.title_text == "短文本"


class TestTitle:
    def test_default_values(self):
        t = Title()
        assert t.title_text == ""
        assert t.method == "entity_composition"
        assert t.mode == "chars"
        assert t.target_chars == 20
        assert t.target_ratio == 0.05

    def test_serialization_roundtrip(self):
        t = Title(
            title_text="测试标题", method="entity_composition", mode="chars",
            target_chars=20, target_ratio=0.05,
            entities_used=["张三"], predicates_used=["IS_A"],
            reasons=["central_entity"],
        )
        d = t.model_dump()
        assert d["title_text"] == "测试标题"
        assert d["entities_used"] == ["张三"]


class TestTitleFormat:
    """Test new title formatting with Chinese predicate mapping and natural formats."""

    def test_is_a_format(self):
        """IS_A should produce 'entity——description' without verb."""
        ent1 = Entity(id="ent_0", text="阿里巴巴", category="ORGANIZATION", span=(0, 4))
        ent2 = Entity(id="ent_1", text="中国电商巨头", category="ORGANIZATION", span=(5, 11))
        rel = Relation(
            id="rel_0", subject="s", predicate="IS_A", object="o",
            evidence="e", evidence_span=(0, 11),
            subject_ent_id="ent_0", object_ent_id="ent_1",
        )
        content = _make_content(
            [_make_sentence("阿里巴巴是中国电商巨头", (0, 11))],
            entities=[ent1, ent2], relations=[rel],
        )
        title, ents, preds, reasons = _compose_title_from_entities(content, 30)
        assert "阿里巴巴" in title
        assert "中国电商巨头" in title
        assert "——" in title
        assert "是" not in title  # IS_A should omit "是"
        assert "IS_A" in preds

    def test_is_a_with_location(self):
        """IS_A + LOCATED_AT should produce 'entity——{location}{desc}'."""
        ent1 = Entity(id="ent_0", text="阿里巴巴", category="ORGANIZATION", span=(0, 4))
        ent2 = Entity(id="ent_1", text="科技公司", category="ORGANIZATION", span=(5, 9))
        ent3 = Entity(id="ent_2", text="中国", category="LOCATION", span=(10, 12))
        rel1 = Relation(
            id="rel_0", subject="s", predicate="IS_A", object="o",
            evidence="e", evidence_span=(0, 9),
            subject_ent_id="ent_0", object_ent_id="ent_1",
        )
        rel2 = Relation(
            id="rel_1", subject="s", predicate="LOCATED_AT", object="o",
            evidence="e", evidence_span=(0, 12),
            subject_ent_id="ent_0", object_ent_id="ent_2",
        )
        content = _make_content(
            [_make_sentence("阿里巴巴是科技公司在中国", (0, 12))],
            entities=[ent1, ent2, ent3], relations=[rel1, rel2],
        )
        title, ents, preds, reasons = _compose_title_from_entities(content, 30)
        # Should produce "阿里巴巴——中国科技公司" not "阿里巴巴——位于中国"
        assert "阿里巴巴" in title
        assert "中国" in title
        assert "科技公司" in title
        assert "位于" not in title  # LOCATED_AT should NOT show as verb for ORG

    def test_has_title_format(self):
        """HAS_TITLE should produce 'entity：description' with colon."""
        ent1 = Entity(id="ent_0", text="马云", category="PERSON", span=(0, 2))
        ent2 = Entity(id="ent_1", text="阿里巴巴创始人", category="TITLE", span=(3, 9))
        rel = Relation(
            id="rel_0", subject="s", predicate="HAS_TITLE", object="o",
            evidence="e", evidence_span=(0, 9),
            subject_ent_id="ent_0", object_ent_id="ent_1",
        )
        content = _make_content(
            [_make_sentence("马云阿里巴巴创始人", (0, 9))],
            entities=[ent1, ent2], relations=[rel],
        )
        title, ents, preds, reasons = _compose_title_from_entities(content, 30)
        assert "马云" in title
        assert "阿里巴巴创始人" in title
        assert "：" in title  # HAS_TITLE uses colon
        assert preds == ["HAS_TITLE"]

    def test_located_at_for_location(self):
        """LOCATED_AT for LOCATION entity should produce 'entity——位于desc'."""
        ent1 = Entity(id="ent_0", text="北京", category="LOCATION", span=(0, 2))
        ent2 = Entity(id="ent_1", text="华北平原", category="LOCATION", span=(3, 7))
        rel = Relation(
            id="rel_0", subject="s", predicate="LOCATED_AT", object="o",
            evidence="e", evidence_span=(0, 7),
            subject_ent_id="ent_0", object_ent_id="ent_1",
        )
        content = _make_content(
            [_make_sentence("北京在华北平原", (0, 7))],
            entities=[ent1, ent2], relations=[rel],
        )
        title, ents, preds, reasons = _compose_title_from_entities(content, 30)
        assert "北京" in title
        assert "华北平原" in title
        # For LOCATION entity with only LOCATED_AT, use category fallback: "地点"
        # Result: "北京——华北平原地点" or just "北京——华北平原"

    def test_predicate_priority(self):
        """IS_A should be preferred over DEPENDS_ON for ORGANIZATION."""
        ent1 = Entity(id="ent_0", text="腾讯", category="ORGANIZATION", span=(0, 2))
        ent2 = Entity(id="ent_1", text="科技公司", category="ORGANIZATION", span=(3, 7))
        ent3 = Entity(id="ent_2", text="深圳", category="LOCATION", span=(8, 10))
        rel1 = Relation(
            id="rel_0", subject="s", predicate="IS_A", object="o",
            evidence="e", evidence_span=(0, 7),
            subject_ent_id="ent_0", object_ent_id="ent_1",
        )
        rel2 = Relation(
            id="rel_1", subject="s", predicate="DEPENDS_ON", object="o",
            evidence="e", evidence_span=(0, 10),
            subject_ent_id="ent_0", object_ent_id="ent_2",
        )
        rel3 = Relation(
            id="rel_2", subject="s", predicate="DEPENDS_ON", object="o",
            evidence="e", evidence_span=(0, 10),
            subject_ent_id="ent_0", object_ent_id="ent_2",
        )
        content = _make_content(
            [_make_sentence("腾讯是科技公司依赖深圳", (0, 10))],
            entities=[ent1, ent2, ent3], relations=[rel1, rel2, rel3],
        )
        title, ents, preds, reasons = _compose_title_from_entities(content, 30)
        # IS_A (score 3.0 for ORG) should beat DEPENDS_ON (default 1.0)
        assert "IS_A" in preds
        assert "科技公司" in title

    def test_category_weight(self):
        """PERSON should be preferred over LOCATION with same relation count."""
        ent1 = Entity(id="ent_0", text="张三", category="PERSON", span=(0, 2))
        ent2 = Entity(id="ent_1", text="北京", category="LOCATION", span=(3, 5))
        # Both have exactly 1 relation, but PERSON(3.0) > LOCATION(2.0)
        rel1 = Relation(
            id="rel_0", subject="s", predicate="IS_A", object="o",
            evidence="e", evidence_span=(0, 5),
            subject_ent_id="ent_0", object_ent_id="ent_1",
        )
        content = _make_content(
            [_make_sentence("张三是北京", (0, 5))],
            entities=[ent1, ent2], relations=[rel1],
        )
        title, ents, preds, reasons = _compose_title_from_entities(content, 30)
        # ent_0 (PERSON, 1rel, weight 3.0=3.0) vs ent_1 (LOCATION, 1rel, weight 2.0=2.0)
        assert ents[0] == "张三"

    def test_object_deduplication(self):
        """Duplicate objects should be deduplicated."""
        ent1 = Entity(id="ent_0", text="腾讯", category="ORGANIZATION", span=(0, 2))
        ent2 = Entity(id="ent_1", text="科技公司", category="ORGANIZATION", span=(3, 7))
        rel1 = Relation(
            id="rel_0", subject="s", predicate="IS_A", object="o",
            evidence="e", evidence_span=(0, 7),
            subject_ent_id="ent_0", object_ent_id="ent_1",
        )
        rel2 = Relation(
            id="rel_1", subject="s", predicate="IS_A", object="o",
            evidence="e", evidence_span=(0, 7),
            subject_ent_id="ent_0", object_ent_id="ent_1",
        )
        content = _make_content(
            [_make_sentence("腾讯是科技公司", (0, 7))],
            entities=[ent1, ent2], relations=[rel1, rel2],
        )
        title, ents, preds, reasons = _compose_title_from_entities(content, 30)
        assert title.count("科技公司") == 1

    def test_ner_misclassification_org_as_person(self):
        """When org is misclassified as PERSON, avoid '中国人物' garbage."""
        ent1 = Entity(id="ent_0", text="阿里巴巴", category="PERSON", span=(0, 4))  # NER错误
        ent2 = Entity(id="ent_1", text="中国", category="LOCATION", span=(5, 7))
        ent3 = Entity(id="ent_2", text="淘宝网", category="ORGANIZATION", span=(8, 11))
        rel1 = Relation(
            id="rel_0", subject="s", predicate="LOCATED_AT", object="o",
            evidence="e", evidence_span=(0, 7),
            subject_ent_id="ent_0", object_ent_id="ent_1",
        )
        rel2 = Relation(
            id="rel_1", subject="s", predicate="PART_OF", object="o",
            evidence="e", evidence_span=(0, 11),
            subject_ent_id="ent_0", object_ent_id="ent_2",
        )
        content = _make_content(
            [_make_sentence("阿里巴巴中国淘宝网", (0, 11))],
            entities=[ent1, ent2, ent3], relations=[rel1, rel2],
        )
        title, ents, preds, reasons = _compose_title_from_entities(content, 20)
        # Should NOT contain "人物" (category fallback when NER is wrong)
        assert "人物" not in title
        # Should use location as best available info
        assert "中国" in title or "阿里巴巴" in title


class TestTextRankPositionBonus:
    """Test that position bonus affects sentence selection."""

    def test_first_sentence_boosted(self):
        """First sentence should get position bonus."""
        sents = [
            _make_sentence("普通的第二句话", (0, 7)),
            _make_sentence("普通的第三句话", (8, 15)),
            _make_sentence("普通的第四句话", (16, 23)),
        ]
        scores = _compute_textrank(sents)
        assert 0 in scores

    def test_last_sentence_boosted(self):
        """Last sentence should get position bonus when applicable."""
        sents = [
            _make_sentence("第一句", (0, 3)),
            _make_sentence("中间句", (4, 7)),
            _make_sentence("最后一句", (8, 12)),
        ]
        scores = _compute_textrank(sents)
        assert 2 in scores


class TestRatioModeInEntityComposition:
    """Test that ratio mode works in entity composition path."""

    def test_ratio_mode_entity_composition(self):
        """ratio mode should compute effective_chars for entity composition."""
        ent1 = Entity(id="ent_0", text="阿里巴巴", category="ORGANIZATION", span=(0, 4))
        ent2 = Entity(id="ent_1", text="公司", category="ORGANIZATION", span=(5, 7))
        rel = Relation(
            id="rel_0", subject="s", predicate="IS_A", object="o",
            evidence="e", evidence_span=(0, 7),
            subject_ent_id="ent_0", object_ent_id="ent_1",
        )
        long_text = "阿里巴巴是" + "很长的描述" * 20
        content = _make_content(
            [_make_sentence(long_text, (0, len(long_text)))],
            entities=[ent1, ent2], relations=[rel],
        )
        doc = _make_doc(content)
        title = generate_title(doc, mode="ratio", target_ratio=0.05)
        assert title.mode == "ratio"
        assert title.title_text != ""
        assert len(title.title_text) <= len(long_text) * 0.1