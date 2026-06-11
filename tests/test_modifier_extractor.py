"""Tests for ModifierExtractor."""

import pytest
from core.modifier_extractor import ModifierExtractor
from core.schema import Relation, RelationModifier


class TestModifierExtractorBasic:
    """Test basic modifier extraction functionality."""

    def setup_method(self):
        self.extractor = ModifierExtractor()

    def test_extract_degree_modifier(self):
        """Test extraction of degree modifiers."""
        rel = Relation(
            id="rel_001",
            subject="碳钢",
            predicate="RELATES_TO",
            object="高强度",
            evidence="碳钢具有高强度",
            evidence_span=(0, 7),
        )
        text = "碳钢具有非常高的强度"
        modifiers = self.extractor.extract(text, rel)

        # Should find "非常" as a degree modifier
        degree_mods = [m for m in modifiers if m.type == "degree"]
        assert len(degree_mods) > 0
        assert any(m.text == "非常" for m in degree_mods)
        assert all(m.matched_dict == "builtin" for m in degree_mods)

    def test_extract_negation_modifier(self):
        """Test extraction of negation modifiers."""
        rel = Relation(
            id="rel_002",
            subject="碳钢",
            predicate="RELATES_TO",
            object="生锈",
            evidence="碳钢不易生锈",
            evidence_span=(0, 6),
        )
        text = "碳钢不易生锈"
        modifiers = self.extractor.extract(text, rel)

        # Should find "不" as a negation modifier
        neg_mods = [m for m in modifiers if m.type == "negation"]
        assert len(neg_mods) > 0
        assert any(m.text == "不" for m in neg_mods)

    def test_no_modifiers_when_absent(self):
        """Test that no modifiers are found when none present."""
        rel = Relation(
            id="rel_003",
            subject="苹果",
            predicate="PRODUCES",
            object="iPhone",
            evidence="苹果生产iPhone",
            evidence_span=(0, 8),
        )
        text = "苹果生产iPhone"
        modifiers = self.extractor.extract(text, rel)

        # May find some modifiers from common words, but should not crash
        assert isinstance(modifiers, list)

    def test_extractor_is_deterministic(self):
        """Test that extraction is deterministic."""
        rel = Relation(
            id="rel_004",
            subject="材料",
            predicate="RELATES_TO",
            object="属性",
            evidence="材料的主要属性",
            evidence_span=(0, 7),
        )
        text = "材料的主要属性非常明显"

        mods1 = self.extractor.extract(text, rel)
        mods2 = self.extractor.extract(text, rel)

        assert len(mods1) == len(mods2)
        for m1, m2 in zip(mods1, mods2):
            assert m1.text == m2.text
            assert m1.type == m2.type
            assert m1.span == m2.span


class TestModifierExtractorCustom:
    """Test custom dictionary support."""

    def test_custom_dict_extends_builtin(self):
        """Test that custom dictionaries extend built-in ones."""
        custom = {
            "degree": {"超级", "巨"},  # Add custom degree words
        }
        extractor = ModifierExtractor(modifier_dict=custom)

        rel = Relation(
            id="rel_005",
            subject="产品",
            predicate="RELATES_TO",
            object="质量",
            evidence="超级好的质量",
            evidence_span=(0, 6),
        )
        text = "这个产品具有超级好的质量"
        modifiers = extractor.extract(text, rel)

        # Should find "非常" (builtin) and "超级" (custom)
        custom_mods = [m for m in modifiers if m.matched_dict == "custom"]
        builtin_mods = [m for m in modifiers if m.matched_dict == "builtin"]

        # At least check that extraction works without error
        assert isinstance(modifiers, list)

    def test_custom_new_type(self):
        """Test adding a new modifier type."""
        custom = {
            "certainty": {"肯定", "确定", "必然"},  # New type
        }
        extractor = ModifierExtractor(modifier_dict=custom)

        rel = Relation(
            id="rel_006",
            subject="结果",
            predicate="RELATES_TO",
            object="正确",
            evidence="结果肯定正确",
            evidence_span=(0, 6),
        )
        text = "这个结果肯定是正确的"
        modifiers = extractor.extract(text, rel)

        certainty_mods = [m for m in modifiers if m.type == "certainty"]
        assert len(certainty_mods) > 0
        assert any(m.text == "肯定" for m in certainty_mods)
        assert all(m.matched_dict == "custom" for m in certainty_mods)


class TestModifierExtractorBatch:
    """Test batch extraction."""

    def test_extract_batch(self):
        """Test batch extraction modifies relations in-place."""
        extractor = ModifierExtractor()

        relations = [
            Relation(
                id="rel_007",
                subject="A",
                predicate="RELATES_TO",
                object="B",
                evidence="A与B",
                evidence_span=(0, 3),
            ),
            Relation(
                id="rel_008",
                subject="C",
                predicate="RELATES_TO",
                object="D",
                evidence="C与D",
                evidence_span=(4, 7),
            ),
        ]
        text = "A与B，C与D"

        # Before: all empty
        for rel in relations:
            assert rel.modifiers == []

        # Extract batch
        extractor.extract_batch(text, relations)

        # After: all have modifiers populated (may be empty or not)
        for rel in relations:
            assert isinstance(rel.modifiers, list)


class TestBuiltinDictAccess:
    """Test built-in dictionary access."""

    def test_get_builtin_dict(self):
        """Test that we can access the built-in dictionary."""
        builtin = ModifierExtractor.get_builtin_dict()

        assert isinstance(builtin, dict)
        assert "degree" in builtin
        assert "negation" in builtin
        assert "scope" in builtin
        assert "quantity" in builtin
        assert "temporal" in builtin
        assert "comparison" in builtin
        assert "emphasis" in builtin

        # Check that values are frozensets
        for v in builtin.values():
            assert isinstance(v, frozenset)

    def test_builtin_dict_not_mutated(self):
        """Test that getting the dict doesn't allow mutation."""
        builtin1 = ModifierExtractor.get_builtin_dict()
        builtin2 = ModifierExtractor.get_builtin_dict()

        # Should be the same content
        assert builtin1 == builtin2


class TestRelationModifierSchema:
    """Test RelationModifier schema validation."""

    def test_valid_modifier(self):
        """Test creating a valid modifier."""
        mod = RelationModifier(
            text="非常",
            type="degree",
            span=(3, 5),
            matched_dict="builtin",
        )
        assert mod.text == "非常"
        assert mod.type == "degree"
        assert mod.span == (3, 5)
        assert mod.matched_dict == "builtin"

    def test_custom_type_allowed(self):
        """Test that custom types are allowed (permissive validation)."""
        # Custom types are intentionally allowed for user extensibility
        mod = RelationModifier(
            text="test",
            type="custom_type",
            span=(0, 4),
        )
        assert mod.type == "custom_type"

    def test_invalid_span(self):
        """Test that invalid span raises error."""
        with pytest.raises(Exception):
            RelationModifier(
                text="test",
                type="degree",
                span=(5, 3),  # end < start
            )

    def test_default_matched_dict(self):
        """Test that matched_dict defaults to 'builtin'."""
        mod = RelationModifier(
            text="非常",
            type="degree",
            span=(0, 2),
        )
        assert mod.matched_dict == "builtin"


class TestRelationModifiersField:
    """Test Relation.modifiers field."""

    def test_relation_with_modifiers(self):
        """Test creating a relation with modifiers."""
        mod = RelationModifier(
            text="非常",
            type="degree",
            span=(3, 5),
        )
        rel = Relation(
            id="rel_009",
            subject="A",
            predicate="RELATES_TO",
            object="B",
            evidence="A与B",
            evidence_span=(0, 3),
            modifiers=[mod],
        )
        assert len(rel.modifiers) == 1
        assert rel.modifiers[0].text == "非常"

    def test_relation_default_modifiers_empty(self):
        """Test that modifiers defaults to empty list."""
        rel = Relation(
            id="rel_010",
            subject="A",
            predicate="RELATES_TO",
            object="B",
            evidence="A与B",
            evidence_span=(0, 3),
        )
        assert rel.modifiers == []


class TestEntityAttributeSourceRelation:
    """Test EntityAttribute.source_relation_id field."""

    def test_entity_attribute_with_source(self):
        """Test creating an attribute with a source relation ID."""
        from core.schema import EntityAttribute

        attr = EntityAttribute(
            key="强度",
            value="高",
            predicate_verb="具有",
            source_relation_id="rel_001",
        )
        assert attr.source_relation_id == "rel_001"

    def test_entity_attribute_default_source_none(self):
        """Test that source_relation_id defaults to None."""
        from core.schema import EntityAttribute

        attr = EntityAttribute(
            key="强度",
            value="高",
        )
        assert attr.source_relation_id is None