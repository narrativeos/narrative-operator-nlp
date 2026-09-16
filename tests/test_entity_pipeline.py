"""Regression tests for entity pipeline false-positive fixes.

Covers:
- _parse_xpos_category: major/minor class parsing (CTB "大类,小类")
- merge_same_category / merge_cross_category: punctuation guard
- _resolve_span: repeated-mention ambiguity
- extract_numeric_entities: ordinal/floor designators are not quantities
- EntityDeduplicator: identical span with different categories
- KeywordExtractor: parameter suffix match guard
- _map_classical_entities: function words never become entities
- _guess_entity_category: single-char suffix over-triggering
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from core.entity_mapper import EntityMappingRules
from core.entity_merger import EntityMerger
from core.entity_deduplicator import EntityDeduplicator
from core.entity_id_generator import EntityIdGenerator
from core.keyword_extractor import KeywordExtractor
from core.numeric_extractor import extract_numeric_entities
from core.schema import Entity, Token


def _ent(eid: str, text: str, category: str, span: tuple[int, int],
         confidence: float = 0.9) -> Entity:
    return Entity(
        id=eid,
        text=text,
        category=category,
        span=span,
        normalized=text,
        source="test",
        confidence=confidence,
    )


class TestParseXposCategory:
    """CTB xpos format is '大类,小类'; category checks must use the minor class."""

    f = staticmethod(EntityMappingRules._parse_xpos_category)

    def test_place_name(self):
        assert self.f("名詞,地名") == "LOCATION"

    def test_person_name(self):
        assert self.f("名詞,人名") == "PERSON"

    def test_proper_noun_is_location(self):
        # Previously returned None (major/minor class check was inverted)
        assert self.f("名詞,固有名詞") == "LOCATION"

    def test_general_noun_is_not_entity(self):
        assert self.f("名詞,一般名詞") is None

    def test_time_noun_is_not_entity(self):
        assert self.f("名詞,時間") is None

    def test_classical_title(self):
        assert self.f("名詞,官職名") == "TITLE"

    def test_classical_era(self):
        assert self.f("名詞,朝代名") == "ERA"

    def test_classical_institution(self):
        assert self.f("名詞,典章制度") == "INSTITUTION"

    def test_classical_astronomy(self):
        assert self.f("名詞,天文名") == "ASTRONOMY"

    def test_three_part_xpos_uses_minor_classes(self):
        # "名詞,一般名詞,動物" → minor classes "一般名詞,動物" → UNKNOWN
        assert self.f("名詞,一般名詞,動物") == "UNKNOWN"

    def test_organization(self):
        assert self.f("名詞,組織") == "ORGANIZATION"

    def test_work(self):
        assert self.f("名詞,作品") == "PRODUCT"


class TestMergePunctuationGuard:
    """Adjacent entities separated by punctuation must not be merged."""

    def setup_method(self):
        self.merger = EntityMerger()

    def test_no_merge_across_period(self):
        text = "海淀区。中关村"
        e1 = _ent("ent_001", "海淀区", "LOCATION", (0, 3))
        e2 = _ent("ent_002", "中关村", "LOCATION", (4, 7))
        out = self.merger.merge_same_category([e1, e2], text)
        assert [e.text for e in out] == ["海淀区", "中关村"]

    def test_no_merge_across_comma(self):
        text = "海淀区，中关村"
        e1 = _ent("ent_001", "海淀区", "LOCATION", (0, 3))
        e2 = _ent("ent_002", "中关村", "LOCATION", (4, 7))
        out = self.merger.merge_same_category([e1, e2], text)
        assert [e.text for e in out] == ["海淀区", "中关村"]

    def test_merge_without_punctuation(self):
        # Backward compatible: adjacent entities with no separator still merge
        text = "北京立方庭"
        e1 = _ent("ent_001", "北京", "LOCATION", (0, 2))
        e2 = _ent("ent_002", "立方庭", "LOCATION", (2, 5))
        out = self.merger.merge_same_category([e1, e2], text)
        assert len(out) == 1
        assert out[0].text == "北京立方庭"

    def test_no_text_argument_preserves_old_behavior(self):
        # Without text the guard is inactive (backward compatibility)
        e1 = _ent("ent_001", "海淀区", "LOCATION", (0, 3))
        e2 = _ent("ent_002", "中关村", "LOCATION", (3, 6))
        out = self.merger.merge_same_category([e1, e2])
        assert len(out) == 1
        assert out[0].text == "海淀区中关村"

    def test_cross_category_no_merge_across_punctuation(self):
        text = "北京。大学"
        e1 = _ent("ent_001", "北京", "LOCATION", (0, 2))
        e2 = _ent("ent_002", "大学", "ORGANIZATION", (3, 5))
        out = self.merger.merge_cross_category([e1, e2], text)
        assert [e.text for e in out] == ["北京", "大学"]


class TestResolveSpan:
    """Span resolution must not silently map a mention to the wrong occurrence."""

    f = staticmethod(EntityMappingRules._resolve_span)

    def test_valid_token_indices(self):
        tokens = [
            Token(id=0, text="他", pos="PN", span=(0, 1), confidence=1.0),
            Token(id=1, text="去了", pos="VV", span=(1, 3), confidence=1.0),
            Token(id=2, text="北京", pos="NR", span=(3, 5), confidence=1.0),
        ]
        assert self.f("北京", 2, 3, tokens, "他去了北京") == (3, 5)

    def test_single_occurrence_fallback(self):
        assert self.f("北京", 99, 100, [], "他去了北京") == (3, 5)

    def test_repeated_mention_returns_none(self):
        # Token indices out of range AND text occurs twice → ambiguous
        assert self.f("北京", 99, 100, [], "他去了北京，又去了北京") is None

    def test_not_found_returns_none(self):
        assert self.f("上海", 99, 100, [], "他去了北京") is None

    def test_map_drops_unresolvable_entity(self):
        rules = EntityMappingRules()
        # Token indices out of range + text occurs twice → ambiguous span
        e = rules.map(
            ("北京", "ns", 99, 100), "ner/pku",
            [Token(id=0, text="x", pos="NN", span=(0, 1), confidence=1.0)],
            "他去了北京，又去了北京",
        )
        assert e is None


class TestOrdinalDesignatorFilter:
    """NER models mislabel ordinal designators (3号/5层) as place names
    when they modify a facility noun (3号航站楼)."""

    def _rules(self):
        return EntityMappingRules()

    def test_ordinal_before_terminal_is_dropped(self):
        rules = self._rules()
        text = "旅客从3号航站楼出发"
        e = rules.map(("3号", "ns", 3, 4), "ner/pku", [], text)
        assert e is None

    def test_ordinal_floor_before_ward_is_dropped(self):
        rules = self._rules()
        text = "他在5层病房休息"
        e = rules.map(("5层", "ns", 2, 3), "ner/pku", [], text)
        assert e is None

    def test_ordinal_before_punctuation_kept(self):
        # No facility noun follows — not provably an ordinal modifier,
        # so the (suspicious) entity is kept for downstream handling.
        rules = self._rules()
        text = "请前往3号。"
        e = rules.map(("3号", "ns", 3, 4), "ner/pku", [], text)
        assert e is not None

    def test_ordinal_at_text_end_kept(self):
        rules = self._rules()
        text = "请前往3号"
        e = rules.map(("3号", "ns", 3, 4), "ner/pku", [], text)
        assert e is not None

    def test_real_place_name_unaffected(self):
        rules = self._rules()
        text = "北京立方庭位于海淀区"
        e = rules.map(("海淀区", "ns", 7, 8), "ner/pku", [], text)
        assert e is not None
        assert e.text == "海淀区"


class TestNumericExtractor:
    """Ordinals and floor designators are not quantities."""

    def setup_method(self):
        self.id_gen = EntityIdGenerator()

    def test_ordinal_not_quantity(self):
        ents = extract_numeric_entities("3号航站楼", set(), self.id_gen)
        assert ents == []

    def test_floor_not_quantity(self):
        ents = extract_numeric_entities("5层楼", set(), self.id_gen)
        assert ents == []

    def test_date_still_extracted(self):
        ents = extract_numeric_entities("2024年1月", set(), self.id_gen)
        assert len(ents) == 1
        assert ents[0].text == "2024年1月"
        assert ents[0].normalized == "2024-01"

    def test_quantity_still_extracted(self):
        ents = extract_numeric_entities("100个人", set(), self.id_gen)
        assert len(ents) == 1
        assert ents[0].text == "100个"

    def test_eight_digit_id_not_date(self):
        # Regression: bare 8-digit IDs must not be treated as dates
        ents = extract_numeric_entities("编号63906433", set(), self.id_gen)
        assert ents == []


class TestDeduplicatorSameSpan:
    """Identical span with different categories is a labeling conflict."""

    def test_keeps_higher_confidence(self):
        a = _ent("ent_001", "北京", "LOCATION", (0, 2), confidence=0.9)
        b = _ent("ent_002", "北京", "ORGANIZATION", (0, 2), confidence=0.7)
        out = EntityDeduplicator.deduplicate([a, b])
        assert len(out) == 1
        assert out[0].category == "LOCATION"
        assert "ent_002" in out[0].merged_from

    def test_keeps_higher_confidence_regardless_of_order(self):
        a = _ent("ent_001", "北京", "LOCATION", (0, 2), confidence=0.6)
        b = _ent("ent_002", "北京", "ORGANIZATION", (0, 2), confidence=0.95)
        out = EntityDeduplicator.deduplicate([a, b])
        assert len(out) == 1
        assert out[0].category == "ORGANIZATION"
        assert "ent_001" in out[0].merged_from

    def test_same_span_same_category_deduped_by_confidence(self):
        # Same category + identical span: existing behavior keeps the
        # higher-confidence one (exact-duplicate dedup)
        a = _ent("ent_001", "北京", "LOCATION", (0, 2), confidence=0.9)
        b = _ent("ent_002", "北京", "LOCATION", (0, 2), confidence=0.8)
        out = EntityDeduplicator.deduplicate([a, b])
        assert len(out) == 1
        assert out[0].id == "ent_001"

    def test_containment_still_preserved(self):
        # Different spans (containment) must NOT be collapsed by the
        # same-span rule
        parent = _ent("ent_001", "北京立方庭", "LOCATION", (0, 5), confidence=0.9)
        child = _ent("ent_002", "北京", "LOCATION", (0, 2), confidence=0.8)
        out = EntityDeduplicator.deduplicate([parent, child])
        assert {e.id for e in out} == {"ent_001", "ent_002"}


class TestKeywordSuffixGuard:
    """Parameter suffix match must not fire on unrelated compound words."""

    def setup_method(self):
        self.ke = KeywordExtractor()

    def test_unrelated_compound_not_parameter(self):
        # Ends with 强度 but is not a parameter keyword
        assert self.ke.lookup_category("材料强度") is None

    def test_unrelated_compound_toughness(self):
        assert self.ke.lookup_category("冲击韧性") is None

    def test_exact_keyword_still_parameter(self):
        assert self.ke.lookup_category("强度") == "PARAMETER"

    def test_dict_parameter_still_parameter(self):
        assert self.ke.lookup_category("抗拉强度") == "PARAMETER"


class TestClassicalFunctionWords:
    """Function words must never become entities, even when tagged PROPN."""

    def _fake_rules(self):
        class _FakeRules:
            _merger = EntityMerger()
            _id_gen = EntityIdGenerator()

        return _FakeRules()

    def test_single_char_function_words_filtered(self):
        toks = [
            Token(id=0, text="之", pos="PROPN", span=(0, 1), confidence=1.0),
            Token(id=1, text="其", pos="PROPN", span=(1, 2), confidence=1.0),
        ]
        res = EntityMappingRules._map_classical_entities(
            self._fake_rules(), toks, "之其", {"pos/xpos": ["", ""]}
        )
        assert res == []

    def test_multi_char_function_word_filtered(self):
        # 於是 is a classical function word (连词); must not become an entity
        toks = [Token(id=0, text="於是", pos="PROPN", span=(0, 2), confidence=1.0)]
        res = EntityMappingRules._map_classical_entities(
            self._fake_rules(), toks, "於是", {"pos/xpos": [""]}
        )
        assert all(e.text != "於是" for e in res)

    def test_real_place_name_still_extracted(self):
        toks = [Token(id=0, text="陽城", pos="PROPN", span=(0, 2), confidence=1.0)]
        res = EntityMappingRules._map_classical_entities(
            self._fake_rules(), toks, "陽城", {"pos/xpos": [""]}
        )
        assert any(e.text == "陽城" for e in res)


class TestPosSuffixCategory:
    """Single-char suffixes must not over-trigger on common words."""

    f = staticmethod(EntityMappingRules._guess_entity_category)

    ORG = {"公司", "集团", "有限", "股份", "中心",
           "大学", "学院", "学校", "医院", "银行",
           "委员会", "协会", "学会"}
    LOC = {"省", "市", "区", "县", "公路", "大街", "省城", "市区"}
    PROD = {"手机", "电脑", "汽车", "系统", "平台", "软件",
            "服务", "产品", "技术", "芯片"}

    def test_two_char_word_not_classified_by_suffix(self):
        assert self.f("社会", self.ORG, self.LOC, self.PROD) == "UNKNOWN"
        assert self.f("黄山", self.ORG, self.LOC, self.PROD) == "UNKNOWN"

    def test_org_suffix(self):
        assert self.f("北京大学", self.ORG, self.LOC, self.PROD) == "ORGANIZATION"

    def test_loc_suffix(self):
        assert self.f("海淀区", self.ORG, self.LOC, self.PROD) == "LOCATION"

    def test_product_suffix(self):
        assert self.f("智能手机", self.ORG, self.LOC, self.PROD) == "PRODUCT"

    def test_no_suffix(self):
        assert self.f("立方庭", self.ORG, self.LOC, self.PROD) == "UNKNOWN"


class TestDeduplicatorSameText:
    """Same text + same category at different spans collapses to one."""

    def test_repeated_mention_collapsed(self):
        a = _ent("ent_001", "北京", "LOCATION", (0, 2), confidence=0.9)
        b = _ent("ent_002", "北京", "LOCATION", (10, 12), confidence=0.8)
        out = EntityDeduplicator.deduplicate([a, b])
        assert len(out) == 1
        assert out[0].id == "ent_001"
        assert "ent_002" in out[0].merged_from

    def test_higher_confidence_wins_regardless_of_position(self):
        a = _ent("ent_001", "北京", "LOCATION", (0, 2), confidence=0.5)
        b = _ent("ent_002", "北京", "LOCATION", (10, 12), confidence=0.95)
        out = EntityDeduplicator.deduplicate([a, b])
        assert len(out) == 1
        assert out[0].id == "ent_002"
        assert "ent_001" in out[0].merged_from

    def test_tie_keeps_earlier_span(self):
        a = _ent("ent_001", "北京", "LOCATION", (0, 2), confidence=0.9)
        b = _ent("ent_002", "北京", "LOCATION", (10, 12), confidence=0.9)
        out = EntityDeduplicator.deduplicate([a, b])
        assert len(out) == 1
        assert out[0].id == "ent_001"
        assert "ent_002" in out[0].merged_from

    def test_three_mentions_one_winner(self):
        a = _ent("ent_001", "北京", "LOCATION", (0, 2), confidence=0.7)
        b = _ent("ent_002", "北京", "LOCATION", (10, 12), confidence=0.9)
        c = _ent("ent_003", "北京", "LOCATION", (20, 22), confidence=0.8)
        out = EntityDeduplicator.deduplicate([a, b, c])
        assert len(out) == 1
        assert out[0].id == "ent_002"
        assert set(out[0].merged_from) == {"ent_001", "ent_003"}

    def test_same_text_different_category_not_collapsed(self):
        # Different categories are a labeling question, not a repeat —
        # only identical-span conflicts are resolved, distinct spans stay.
        a = _ent("ent_001", "北京", "LOCATION", (0, 2), confidence=0.9)
        b = _ent("ent_002", "北京", "ORGANIZATION", (10, 12), confidence=0.9)
        out = EntityDeduplicator.deduplicate([a, b])
        assert {e.id for e in out} == {"ent_001", "ent_002"}

    def test_distinct_texts_not_collapsed(self):
        a = _ent("ent_001", "北京", "LOCATION", (0, 2), confidence=0.9)
        b = _ent("ent_002", "海淀", "LOCATION", (10, 12), confidence=0.9)
        out = EntityDeduplicator.deduplicate([a, b])
        assert {e.id for e in out} == {"ent_001", "ent_002"}


class TestNerConflictUnionFind:
    """Conflict grouping must be the transitive closure of pairwise
    conflicts (union-find), not seed-only comparison."""

    def _resolve(self, candidates):
        from core.entity_mapper import _resolve_ner_conflicts
        _resolve_ner_conflicts(candidates)
        return candidates

    def test_transitive_chain_collapses_to_one(self):
        # A~B (IoU 2/3 + containment), B~C (IoU 3/4 + containment),
        # but A!~C (IoU exactly 0.5). Seed-only grouping on A would
        # leave C in its own group; union-find merges all three.
        a = _ent("ent_001", "北京", "LOCATION", (0, 2),
                 confidence=0.5, )
        a.source = "ner/pku"
        b = _ent("ent_002", "北京市", "FACILITY", (0, 3), confidence=0.9)
        b.source = "ner/msra"
        c = _ent("ent_003", "北京市东", "ORGANIZATION", (0, 4),
                 confidence=0.7)
        c.source = "ner/ontonotes"
        out = self._resolve([a, b, c])
        assert len(out) == 1
        assert out[0].id == "ent_002"  # highest confidence wins
        assert out[0].ner_disputed is True
        assert out[0].ner_labels == {
            "ner/pku": "LOCATION",
            "ner/msra": "FACILITY",
            "ner/ontonotes": "ORGANIZATION",
        }

    def test_agreeing_group_untouched(self):
        # Overlapping candidates with the SAME category are not a
        # conflict — all are kept.
        a = _ent("ent_001", "北京", "LOCATION", (0, 2), confidence=0.9)
        a.source = "ner/pku"
        b = _ent("ent_002", "北京市", "LOCATION", (0, 3), confidence=0.8)
        b.source = "ner/msra"
        out = self._resolve([a, b])
        assert {e.id for e in out} == {"ent_001", "ent_002"}

    def test_two_independent_conflicts_resolved_separately(self):
        a = _ent("ent_001", "北京", "LOCATION", (0, 2), confidence=0.9)
        a.source = "ner/pku"
        b = _ent("ent_002", "北京市", "FACILITY", (0, 3), confidence=0.5)
        b.source = "ner/msra"
        c = _ent("ent_003", "上海", "LOCATION", (20, 22), confidence=0.9)
        c.source = "ner/pku"
        d = _ent("ent_004", "上海市", "FACILITY", (20, 23), confidence=0.5)
        d.source = "ner/msra"
        out = self._resolve([a, b, c, d])
        assert {e.id for e in out} == {"ent_001", "ent_003"}

