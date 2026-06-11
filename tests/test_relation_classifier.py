"""Tests for RelationClassifier."""

import pytest
from core.relation_classifier import RelationClassifier
from core.schema import Entity, EntityAttribute, Relation


class TestRelationClassifierBasic:
    """Test basic relation classification."""

    def test_has_property_with_non_entity_object(self):
        """HAS_PROPERTY with non-entity object → attribute."""
        classifier = RelationClassifier()
        entities = [
            Entity(id="ent_001", text="碳钢", category="MATERIAL", span=(0, 2)),
        ]
        relations = [
            Relation(
                id="rel_001",
                subject="碳钢",
                predicate="HAS_PROPERTY",
                object="高强度",
                evidence="碳钢具有高强度",
                evidence_span=(0, 7),
            ),
        ]

        classifier.classify(relations, entities)

        # Should have attached an attribute
        assert len(entities[0].attributes) == 1
        attr = entities[0].attributes[0]
        assert attr.key == "强度"
        assert attr.value == "高"
        assert attr.source_relation_id == "rel_001"

    def test_relates_to_stays_as_relation(self):
        """RELATES_TO → stays as relation, not attribute."""
        classifier = RelationClassifier()
        entities = [
            Entity(id="ent_001", text="碳钢", category="MATERIAL", span=(0, 2)),
            Entity(id="ent_002", text="钢", category="MATERIAL", span=(5, 6)),
        ]
        relations = [
            Relation(
                id="rel_002",
                subject="碳钢",
                predicate="RELATES_TO",
                object="钢",
                evidence="碳钢是一种钢",
                evidence_span=(0, 6),
            ),
        ]

        classifier.classify(relations, entities)

        # Should NOT attach an attribute (RELATES_TO is not a property predicate)
        assert len(entities[0].attributes) == 0

    def test_has_property_with_entity_object_stays_relation(self):
        """HAS_PROPERTY with entity object → stays as relation."""
        classifier = RelationClassifier()
        entities = [
            Entity(id="ent_001", text="碳钢", category="MATERIAL", span=(0, 2)),
            Entity(id="ent_002", text="合金钢", category="MATERIAL", span=(5, 8)),
        ]
        relations = [
            Relation(
                id="rel_003",
                subject="碳钢",
                predicate="HAS_PROPERTY",
                object="合金钢",
                evidence="碳钢具有合金钢特性",
                evidence_span=(0, 9),
            ),
        ]

        classifier.classify(relations, entities)

        # Object is a known entity (not PARAMETER/NUMBER/DATE) → stays as relation
        assert len(entities[0].attributes) == 0

    def test_has_property_with_parameter_entity_becomes_attribute(self):
        """HAS_PROPERTY with PARAMETER entity → becomes attribute."""
        classifier = RelationClassifier()
        entities = [
            Entity(id="ent_001", text="碳钢", category="MATERIAL", span=(0, 2)),
            Entity(id="ent_002", text="强度", category="PARAMETER", span=(5, 7)),
        ]
        relations = [
            Relation(
                id="rel_004",
                subject="碳钢",
                predicate="HAS_PROPERTY",
                object="强度",
                evidence="碳钢的强度",
                evidence_span=(0, 5),
            ),
        ]

        classifier.classify(relations, entities)

        # PARAMETER entity → becomes attribute
        assert len(entities[0].attributes) == 1
        assert entities[0].attributes[0].key == "强度"
        assert entities[0].attributes[0].source_relation_id == "rel_004"


class TestRelationClassifierDedup:
    """Test attribute deduplication."""

    def test_duplicate_attributes_not_added(self):
        """Duplicate attributes are not added."""
        classifier = RelationClassifier()
        entities = [
            Entity(id="ent_001", text="碳钢", category="MATERIAL", span=(0, 2)),
        ]
        relations = [
            Relation(
                id="rel_005",
                subject="碳钢",
                predicate="HAS_PROPERTY",
                object="高强度",
                evidence="碳钢具有高强度",
                evidence_span=(0, 7),
            ),
            Relation(
                id="rel_006",
                subject="碳钢",
                predicate="HAS_PROPERTY",
                object="高强度",
                evidence="碳钢具有高强度",
                evidence_span=(0, 7),
            ),
        ]

        classifier.classify(relations, entities)

        # Should only have one attribute
        assert len(entities[0].attributes) == 1


class TestRelationClassifierParseAttribute:
    """Test attribute value parsing."""

    def test_parse_simple_value(self):
        """Simple value → key is full text."""
        classifier = RelationClassifier()
        entities = [
            Entity(id="ent_001", text="材料", category="MATERIAL", span=(0, 2)),
        ]
        relations = [
            Relation(
                id="rel_007",
                subject="材料",
                predicate="HAS_PROPERTY",
                object="红色",
                evidence="材料是红色的",
                evidence_span=(0, 6),
            ),
        ]

        classifier.classify(relations, entities)

        assert len(entities[0].attributes) == 1
        assert entities[0].attributes[0].key == "红色"
        assert entities[0].attributes[0].value == ""

    def test_parse_compound_value(self):
        """Compound value like '高强度' → key='强度', value='高'."""
        classifier = RelationClassifier()
        entities = [
            Entity(id="ent_001", text="材料", category="MATERIAL", span=(0, 2)),
        ]
        relations = [
            Relation(
                id="rel_008",
                subject="材料",
                predicate="HAS_PROPERTY",
                object="高韧性",
                evidence="材料具有高韧性",
                evidence_span=(0, 7),
            ),
        ]

        classifier.classify(relations, entities)

        assert len(entities[0].attributes) == 1
        assert entities[0].attributes[0].key == "韧性"
        assert entities[0].attributes[0].value == "高"


class TestRelationClassifierNoSubjectEntity:
    """Test when subject entity is not found."""

    def test_no_crash_when_subject_not_found(self):
        """No crash when subject entity is not in the entity list."""
        classifier = RelationClassifier()
        entities = [
            Entity(id="ent_001", text="钢", category="MATERIAL", span=(5, 6)),
        ]
        relations = [
            Relation(
                id="rel_009",
                subject="未知材料",  # Not in entity list
                predicate="HAS_PROPERTY",
                object="高强度",
                evidence="未知材料具有高强度",
                evidence_span=(0, 8),
            ),
        ]

        # Should not crash
        classifier.classify(relations, entities)

        # No attributes added (subject not found)
        assert len(entities[0].attributes) == 0


class TestRelationClassifierCustomCategories:
    """Test custom attribute categories."""

    def test_custom_attribute_categories(self):
        """Custom attribute categories work correctly."""
        classifier = RelationClassifier(attribute_categories={"PRODUCT"})
        entities = [
            Entity(id="ent_001", text="工厂", category="ORGANIZATION", span=(0, 2)),
            Entity(id="ent_002", text="iPhone", category="PRODUCT", span=(5, 11)),
        ]
        relations = [
            Relation(
                id="rel_010",
                subject="工厂",
                predicate="HAS_PROPERTY",
                object="iPhone",
                evidence="工厂生产iPhone",
                evidence_span=(0, 8),
            ),
        ]

        classifier.classify(relations, entities)

        # PRODUCT entity → becomes attribute
        assert len(entities[0].attributes) == 1
        assert entities[0].attributes[0].source_relation_id == "rel_010"