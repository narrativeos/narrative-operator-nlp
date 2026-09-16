"""Regression tests for relation pipeline audit fixes.

Covers:
- R1:  type-based predicate inference is a RELATES_TO fallback only
- R2:  conjunction splitting does not corrupt proper nouns (颐和园)
- R3:  negation scope detects a negation word INSIDE the evidence span
- R4:  EntityHierarchyBuilder.from_relations continues IDs past max suffix
- R5:  classical function-word filter is scoped to classical text
- R6:  coref Level-3 expansion requires the full entity name in evidence
- R7:  subject_ent_id/object_ent_id are re-resolved after rewrites
- R8:  duplicate (subject, predicate, object) triples are collapsed
- R9:  copula bridge resolves 为/乃/即/系 → IS_A
- R11: mapper.map(id_base=...) continues numbering across segments
- R12: bridge never pairs arguments across a sentence boundary
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from core.analyzer import (
    _deduplicate_relations,
    _normalize_relations_by_coref,
)
from core.entity_hierarchy import EntityHierarchyBuilder
from core.mapper import HanlpSchemaMapper
from core.negation_detector import find_negation_spans, is_in_negation_scope
from core.relation_mapper import RelationExtractionRules
from core.schema import Entity, Relation, Token


def _ent(eid: str, text: str, category: str, span: tuple[int, int] = (0, 1)) -> Entity:
    return Entity(id=eid, text=text, category=category, span=span)


def _rel(rid: str, subject: str, predicate: str, obj: str,
         evidence: str = "x", confidence: float = 0.6, **kw) -> Relation:
    return Relation(
        id=rid, subject=subject, predicate=predicate, object=obj,
        evidence=evidence, evidence_span=(0, len(evidence)),
        confidence=confidence, source="test", **kw,
    )


# ── R1: type inference is fallback-only ──────────────────────────────

class TestTypeInferenceFallback:
    def test_verb_predicate_not_overridden(self):
        """导致→CAUSES must survive (PERSON, LOCATION) type inference."""
        rules = RelationExtractionRules()
        entities = [_ent("ent_001", "张三", "PERSON"),
                    _ent("ent_002", "北京", "LOCATION")]
        rel = _rel("rel_001", "张三", "CAUSES", "北京",
                   predicate_verb="导致")
        out = rules._entity_normalize(rel, {"张三", "北京"},
                                      strict=False, entities=entities)
        assert out is not None
        assert out.predicate == "CAUSES"

    def test_produces_not_overridden_by_location(self):
        """生产→PRODUCES must survive (PERSON, LOCATION) type inference."""
        rules = RelationExtractionRules()
        entities = [_ent("ent_001", "张三", "PERSON"),
                    _ent("ent_002", "北京", "LOCATION")]
        rel = _rel("rel_001", "张三", "PRODUCES", "北京",
                   predicate_verb="生产")
        out = rules._entity_normalize(rel, {"张三", "北京"},
                                      strict=False, entities=entities)
        assert out is not None
        assert out.predicate == "PRODUCES"

    def test_relates_to_fallback_still_infers(self):
        """RELATES_TO + (PERSON, LOCATION) → LOCATED_AT (fallback works)."""
        rules = RelationExtractionRules()
        entities = [_ent("ent_001", "张三", "PERSON"),
                    _ent("ent_002", "北京", "LOCATION")]
        rel = _rel("rel_001", "张三", "RELATES_TO", "北京",
                   predicate_verb="待")
        out = rules._entity_normalize(rel, {"张三", "北京"},
                                      strict=False, entities=entities)
        assert out is not None
        assert out.predicate == "LOCATED_AT"


# ── R2: conjunction splitting guards ─────────────────────────────────

class TestConjunctionSplitting:
    def test_proper_noun_with_inner_conjunction_not_split(self):
        """颐和园 contains 和 but must not split into 颐+园."""
        rules = RelationExtractionRules()
        rel = _rel("rel_001", "颐和园", "LOCATED_AT", "北京")
        out = rules._split_conjunction_relations([rel])
        assert len(out) == 1
        assert out[0].subject == "颐和园"

    def test_real_conjunction_still_splits(self):
        rules = RelationExtractionRules()
        rel = _rel("rel_001", "李四", "RELATES_TO", "王五和赵六")
        out = rules._split_conjunction_relations([rel])
        assert len(out) == 2
        assert {r.object for r in out} == {"王五", "赵六"}

    def test_semantic_class_preserved_on_split(self):
        rules = RelationExtractionRules()
        rel = _rel("rel_001", "张三", "RELATES_TO", "李四和王五",
                   predicate_verb="说", semantic_class="COMMUNICATION")
        out = rules._split_conjunction_relations([rel])
        assert len(out) == 2
        assert all(r.semantic_class == "COMMUNICATION" for r in out)


# ── R3: negation scope ───────────────────────────────────────────────

class TestNegationScope:
    def test_negation_inside_evidence_span(self):
        """北京[没有]生产碳纤维 — negation between subject and object."""
        text = "北京没有生产碳纤维。"
        neg = find_negation_spans(text)
        assert neg == [(2, 4)]
        # evidence span covers subject..object
        assert is_in_negation_scope((0, 9), neg) is True

    def test_span_before_negation_not_negated(self):
        text = "北京没有生产碳纤维。"
        neg = find_negation_spans(text)
        assert is_in_negation_scope((0, 1), neg) is False

    def test_negation_scope_does_not_cross_sentence(self):
        """北京没有X。Y生产Z — 没有 must not negate the next sentence."""
        text = "北京没有生产碳纤维。上海生产碳纤维。"
        neg = find_negation_spans(text)
        # 上海's evidence span starts after 没有 but across 。
        assert is_in_negation_scope((10, 17), neg, text=text) is False
        # Same sentence, after the negation → still in scope
        assert is_in_negation_scope((4, 9), neg, text=text) is True

    def test_mapper_sets_negated_flag(self):
        text = "北京没有生产碳纤维。"
        raw = {
            "tok/fine": ["北京", "没有", "生产", "碳纤维", "。"],
            "pos/ctb": ["NR", "AD", "VV", "NN", "PU"],
            "ner/pku": [("北京", "ns", 0, 2)],
            "ner/msra": [("北京", "LOCATION", 0, 2)],
            "ner/ontonotes": [("北京", "LOC", 0, 2)],
            "dep": [(3, "nsubj"), (0, "root"), (3, "dobj"),
                    (3, "punct"), (3, "neg")],
            "srl": [[("北京", "ARG0", 0, 1), ("生产", "PRED", 2, 3),
                     ("碳纤维", "ARG1", 3, 4)]],
        }
        mapper = HanlpSchemaMapper()
        doc = mapper.map(text, raw)
        rels = [r for r in doc.content.relations if r.predicate_verb == "生产"]
        assert rels, "expected a 生产 relation"
        assert rels[0].negated is True
        assert rels[0].predicate == "PRODUCES"

    def test_affirmative_relation_not_negated(self):
        text = "北京生产碳纤维。"
        raw = {
            "tok/fine": ["北京", "生产", "碳纤维", "。"],
            "pos/ctb": ["NR", "VV", "NN", "PU"],
            "ner/pku": [("北京", "ns", 0, 2)],
            "ner/msra": [("北京", "LOCATION", 0, 2)],
            "ner/ontonotes": [("北京", "LOC", 0, 2)],
            "dep": [(2, "nsubj"), (0, "root"), (2, "dobj"), (2, "punct")],
            "srl": [[("北京", "ARG0", 0, 1), ("生产", "PRED", 1, 2),
                     ("碳纤维", "ARG1", 2, 3)]],
        }
        mapper = HanlpSchemaMapper()
        doc = mapper.map(text, raw)
        rels = [r for r in doc.content.relations if r.predicate_verb == "生产"]
        assert rels
        assert rels[0].negated is False


# ── R4: hierarchy ID continuation ────────────────────────────────────

class TestHierarchyIdContinuation:
    def test_ids_continue_past_max_suffix(self):
        existing = [
            _rel("rel_005", "A", "RELATES_TO", "B"),
            _rel("rel_006", "C", "RELATES_TO", "D"),
            _rel("rel_007", "E", "RELATES_TO", "F"),
        ]
        hb = EntityHierarchyBuilder.from_relations(existing)
        parent = _ent("ent_001", "北京", "LOCATION", (0, 10))
        child = _ent("ent_002", "海淀", "LOCATION", (2, 4))
        rels = hb.build([parent, child], "北京海淀测试文本")
        assert len(rels) == 1
        assert rels[0].id == "rel_008"
        assert rels[0].id not in {r.id for r in existing}

    def test_empty_relations_starts_at_one(self):
        hb = EntityHierarchyBuilder.from_relations([])
        parent = _ent("ent_001", "北京", "LOCATION", (0, 10))
        child = _ent("ent_002", "海淀", "LOCATION", (2, 4))
        rels = hb.build([parent, child], "北京海淀测试文本")
        assert rels[0].id == "rel_001"


# ── R5: classical filter scoping ─────────────────────────────────────

class TestClassicalFilterScoping:
    def test_modern_entity_yun_not_blocked(self):
        """云 is a classical function word but a legitimate modern name."""
        rules = RelationExtractionRules()
        entities = [_ent("ent_001", "云", "PRODUCT"),
                    _ent("ent_002", "数据中心", "FACILITY")]
        rel = _rel("rel_001", "云", "LOCATED_AT", "数据中心")
        out = rules._entity_normalize(rel, {"云", "数据中心"},
                                      strict=False, entities=entities,
                                      classical=False)
        assert out is not None
        assert out.subject == "云"

    def test_classical_function_word_blocked(self):
        rules = RelationExtractionRules()
        entities = [_ent("ent_001", "之", "UNKNOWN"),
                    _ent("ent_002", "数据中心", "FACILITY")]
        rel = _rel("rel_001", "之", "RELATES_TO", "数据中心")
        out = rules._entity_normalize(rel, {"之", "数据中心"},
                                      strict=False, entities=entities,
                                      classical=True)
        assert out is None


# ── R6/R7: coref normalization guards ────────────────────────────────

class TestCorefNormalizationGuards:
    def test_complete_name_not_expanded_without_evidence(self):
        """北京 (not in entities) must not expand to 北京立方庭 when the
        evidence does not contain the full name."""
        entities = [_ent("ent_001", "北京立方庭", "LOCATION", (0, 5)),
                    _ent("ent_002", "海淀", "LOCATION", (6, 8))]
        rel = _rel("rel_001", "北京", "LOCATED_AT", "海淀",
                   subject_raw="北京", object_raw="海淀",
                   evidence="北京位于海淀区")
        _normalize_relations_by_coref([rel], [], entities)
        assert rel.subject == "北京"

    def test_fragment_expanded_with_evidence(self):
        """立方庭 (fragment) expands to 北京立方庭 when the evidence
        contains the full name."""
        entities = [_ent("ent_001", "北京立方庭", "LOCATION", (0, 5)),
                    _ent("ent_002", "海淀", "LOCATION", (6, 8))]
        rel = _rel("rel_001", "立方庭", "LOCATED_AT", "海淀",
                   subject_raw="立方庭", object_raw="海淀",
                   evidence="北京立方庭位于海淀")
        _normalize_relations_by_coref([rel], [], entities)
        assert rel.subject == "北京立方庭"

    def test_ent_ids_reresolved_after_rewrite(self):
        entities = [_ent("ent_001", "北京立方庭", "LOCATION", (0, 5)),
                    _ent("ent_002", "海淀", "LOCATION", (6, 8))]
        rel = _rel("rel_001", "立方庭", "LOCATED_AT", "海淀",
                   subject_raw="立方庭", object_raw="海淀",
                   subject_ent_id="ent_999",  # stale ID
                   evidence="北京立方庭位于海淀")
        _normalize_relations_by_coref([rel], [], entities)
        assert rel.subject == "北京立方庭"
        assert rel.subject_ent_id == "ent_001"
        assert rel.object_ent_id == "ent_002"


# ── R8: relation deduplication ───────────────────────────────────────

class TestRelationDedup:
    def test_duplicates_collapsed_keep_highest_confidence(self):
        r1 = _rel("rel_001", "A", "LOCATED_AT", "B", confidence=0.5)
        r2 = _rel("rel_002", "A", "LOCATED_AT", "B", confidence=0.8)
        r3 = _rel("rel_003", "A", "PRODUCES", "B", confidence=0.5)
        out = _deduplicate_relations([r1, r2, r3])
        assert [r.id for r in out] == ["rel_002", "rel_003"]

    def test_different_predicates_kept(self):
        r1 = _rel("rel_001", "A", "LOCATED_AT", "B")
        r2 = _rel("rel_002", "A", "PRODUCES", "B")
        out = _deduplicate_relations([r1, r2])
        assert len(out) == 2


# ── R9: copula bridge predicate ──────────────────────────────────────

class TestCopulaBridge:
    def test_copula_wei_maps_to_is_a(self):
        """名→鲲 via copula 为 must be IS_A, not RELATES_TO."""
        rules = RelationExtractionRules()
        text = "名鲲为知"
        tokens = [
            Token(id=0, text="名", pos="NOUN", span=(0, 1)),
            Token(id=1, text="鲲", pos="NOUN", span=(1, 2)),
            Token(id=2, text="为", pos="VERB", span=(2, 3)),
            Token(id=3, text="知", pos="NOUN", span=(3, 4)),
        ]
        # 名: obj→知 | 鲲: root→知 | 为: cop→鲲 | 知: root
        dep = [(4, "obj"), (4, "root"), (2, "cop"), (0, "root")]
        entities = [_ent("ent_001", "名", "PERSON"),
                    _ent("ent_002", "鲲", "PRODUCT")]
        rels = rules.extract_all(text, {"dep": dep, "srl": []},
                                 tokens, entities)
        cop_rels = [r for r in rels if r.predicate_verb == "为"]
        assert cop_rels, "expected a copula-bridged relation"
        assert cop_rels[0].predicate == "IS_A"
        assert cop_rels[0].subject == "名"
        assert cop_rels[0].object == "鲲"


# ── R11: cross-segment ID continuation ───────────────────────────────

def _raw_with_entity(ent_text: str) -> dict:
    return {
        "tok/fine": [ent_text, "。"],
        "pos/ctb": ["NR", "PU"],
        "ner/pku": [(ent_text, "ns", 0, len(ent_text))],
        "ner/msra": [(ent_text, "LOCATION", 0, len(ent_text))],
        "ner/ontonotes": [(ent_text, "LOC", 0, len(ent_text))],
        "dep": [(0, "root"), (1, "punct")],
        "srl": [],
    }


class TestIdBaseContinuation:
    def test_entity_ids_continue_across_segments(self):
        mapper = HanlpSchemaMapper()
        doc1 = mapper.map("北京。", _raw_with_entity("北京"))
        doc2 = mapper.map("上海。", _raw_with_entity("上海"),
                          id_base=(len(doc1.content.entities), 0))
        ids = [e.id for e in doc1.content.entities] + \
              [e.id for e in doc2.content.entities]
        assert len(ids) == len(set(ids)), f"ID collision: {ids}"
        assert ids[0] == "ent_001"
        assert ids[1] == "ent_002"

    def test_relation_ids_continue_across_segments(self):
        mapper = HanlpSchemaMapper()
        raw1 = {
            "tok/fine": ["北京", "生产", "碳纤维", "。"],
            "pos/ctb": ["NR", "VV", "NN", "PU"],
            "ner/pku": [("北京", "ns", 0, 2)],
            "ner/msra": [("北京", "LOCATION", 0, 2)],
            "ner/ontonotes": [("北京", "LOC", 0, 2)],
            "dep": [(2, "nsubj"), (0, "root"), (2, "dobj"), (2, "punct")],
            "srl": [[("北京", "ARG0", 0, 1), ("生产", "PRED", 1, 2),
                     ("碳纤维", "ARG1", 2, 3)]],
        }
        raw2 = {
            "tok/fine": ["上海", "生产", "钢材", "。"],
            "pos/ctb": ["NR", "VV", "NN", "PU"],
            "ner/pku": [("上海", "ns", 0, 2)],
            "ner/msra": [("上海", "LOCATION", 0, 2)],
            "ner/ontonotes": [("上海", "LOC", 0, 2)],
            "dep": [(2, "nsubj"), (0, "root"), (2, "dobj"), (2, "punct")],
            "srl": [[("上海", "ARG0", 0, 1), ("生产", "PRED", 1, 2),
                     ("钢材", "ARG1", 2, 3)]],
        }
        doc1 = mapper.map("北京生产碳纤维。", raw1)
        doc2 = mapper.map("上海生产钢材。", raw2,
                          id_base=(0, len(doc1.content.relations)))
        ids = [r.id for r in doc1.content.relations] + \
              [r.id for r in doc2.content.relations]
        assert len(ids) == len(set(ids)), f"ID collision: {ids}"


# ── R12: bridge sentence-boundary guard ──────────────────────────────

class TestBridgeSentenceBoundary:
    def test_no_cross_sentence_bridge(self):
        """The dep parser linked 人 (sentence 1) as obj of 立
        (sentence 2). The bridge must not pair 北京→人 across 。,
        but must still bridge the same-sentence 北京→海淀."""
        rules = RelationExtractionRules()
        text = "陈胜人。北京立于海淀。"
        tokens = [
            Token(id=0, text="陈胜", pos="PROPN", span=(0, 2)),
            Token(id=1, text="人", pos="NOUN", span=(2, 3)),
            Token(id=2, text="。", pos="PUNCT", span=(3, 4)),
            Token(id=3, text="北京", pos="PROPN", span=(4, 6)),
            Token(id=4, text="立", pos="VERB", span=(6, 7)),
            Token(id=5, text="于", pos="ADP", span=(7, 8)),
            Token(id=6, text="海淀", pos="PROPN", span=(8, 10)),
            Token(id=7, text="。", pos="PUNCT", span=(10, 11)),
        ]
        # 陈胜: nsubj→人 | 人: obj→立 (CROSS-SENTENCE) | 。: punct→人
        # 北京: nsubj→立 | 立: root | 于: prep→立 | 海淀: dobj→立 | 。: punct→立
        dep = [(2, "nsubj"), (5, "obj"), (2, "punct"),
               (5, "nsubj"), (0, "root"), (5, "prep"),
               (5, "dobj"), (5, "punct")]
        entities = [_ent("ent_001", "北京", "LOCATION", (4, 6)),
                    _ent("ent_002", "海淀", "LOCATION", (8, 10))]
        rels = rules.extract_all(text, {"dep": dep, "srl": []},
                                 tokens, entities)
        assert any(r.subject == "北京" and r.object == "海淀" for r in rels), \
            "same-sentence bridge should survive"
        assert not any(r.object == "人" for r in rels), \
            "cross-sentence bridge must be blocked"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))



