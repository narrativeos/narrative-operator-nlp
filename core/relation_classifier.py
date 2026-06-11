"""
Relation Classifier — Classify atomic relations into entity relations / entity attributes.

After relation extraction and conjunction splitting, this module:
1. Identifies which relations are entity-entity relations (keep as relations)
2. Identifies which relations are entity-attribute relations (attach to entities)
3. Sets up traceability between attributes and source relations

Design principles:
- Relations are the single source of truth
- Attributes are an indexed view of relations for quick access
- Every attribute has a source_relation_id for traceability
"""

from __future__ import annotations

from typing import Optional

from .schema import Entity, EntityAttribute, Relation


# Predicates that indicate property/attribute relations
PROPERTY_PREDICATES = frozenset({
    "HAS_PROPERTY",
    "PROPERTY_OF",
})

# Predicates that indicate entity-entity relations (never attributes)
ENTITY_RELATION_PREDICATES = frozenset({
    "IS_A",
    "PART_OF",
    "LOCATED_AT",
    "TEMPORAL_AT",
    "CAUSES",
    "AFFECTS",
    "PRODUCES",
    "COMPOSED_OF",
    "TRANSFERS_TO",
    "DEPENDS_ON",
    "MOVED_TO",
    "DEPARTED_FROM",
    "INTERACTS_WITH",
    "CONTROLS",
    "BENEFITS",
    "EQUIVALENT_TO",
    "REFERENCE_OF",
    "CONSTRAINT_OF",
    "RELATES_TO",
})


class RelationClassifier:
    """Classify relations into entity relations and entity attributes.

    After relation extraction, this classifier:
    1. Keeps entity-entity relations in the relations list
    2. Moves entity-attribute relations to entity.attributes
    3. Maintains traceability via source_relation_id
    """

    def __init__(
        self,
        attribute_categories: Optional[set[str]] = None,
    ):
        """Initialize the classifier.

        Args:
            attribute_categories: Entity categories that can be attribute values.
                If None, defaults to {PARAMETER, NUMBER, DATE}.
        """
        if attribute_categories is None:
            self.attribute_categories: set[str] = {"PARAMETER", "NUMBER", "DATE"}
        else:
            self.attribute_categories = attribute_categories

    def classify(
        self,
        relations: list[Relation],
        entities: list[Entity],
    ) -> list[Relation]:
        """Classify relations and attach attributes to entities.

        This method modifies entities in-place to add attributes.
        Relations that are classified as entity-entity relations are kept.
        Relations classified as attributes are also kept (not removed) to
        maintain the single source of truth principle.

        Args:
            relations: List of relations to classify.
            entities: List of entities (modified in-place).

        Returns:
            The same relations list (all relations are kept).
        """
        # Build entity lookup by text
        entity_by_text: dict[str, Entity] = {}
        for e in entities:
            entity_by_text[e.text] = e

        # Also build lookup by category for attribute value detection
        entity_by_category: dict[str, set[str]] = {}
        for e in entities:
            entity_by_category.setdefault(e.category, set()).add(e.text)

        attribute_values = set()
        for cat in self.attribute_categories:
            attribute_values.update(entity_by_category.get(cat, set()))

        for rel in relations:
            # Check if this is an attribute relation
            if self._is_attribute_relation(rel, entity_by_text, attribute_values):
                # Attach attribute to the subject entity
                self._attach_attribute(rel, entity_by_text)

        return relations

    def _is_attribute_relation(
        self,
        rel: Relation,
        entity_by_text: dict[str, Entity],
        attribute_values: set[str],
    ) -> bool:
        """Determine if a relation should be classified as an attribute.

        A relation is an attribute if:
        1. The predicate is HAS_PROPERTY or PROPERTY_OF, AND
        2. The object is NOT a known entity (i.e., it's a value, not an entity)

        Args:
            rel: The relation to classify.
            entity_by_text: Entity lookup by text.
            attribute_values: Set of texts that are attribute-type entities.

        Returns:
            True if the relation should be an attribute.
        """
        # Only HAS_PROPERTY / PROPERTY_OF are candidates
        if rel.predicate not in PROPERTY_PREDICATES:
            return False

        # If the object IS a known entity, it's a relation not an attribute
        obj_entity = entity_by_text.get(rel.object)
        if obj_entity is not None:
            # Exception: if the entity is a PARAMETER/NUMBER/DATE, it's still an attribute
            if obj_entity.category in self.attribute_categories:
                return True
            return False

        # Object is not a known entity → it's an attribute value
        return True

    def _attach_attribute(
        self,
        rel: Relation,
        entity_by_text: dict[str, Entity],
    ) -> None:
        """Attach a relation as an attribute to the subject entity.

        Args:
            rel: The relation to attach as an attribute.
            entity_by_text: Entity lookup by text.
        """
        # Find the subject entity
        subject_entity = entity_by_text.get(rel.subject)
        if subject_entity is None:
            # Try with raw text
            subject_entity = entity_by_text.get(rel.subject_raw)
        if subject_entity is None:
            return

        # Parse the object value into key/value
        key, value = self._parse_attribute_value(rel)

        # Check for duplicate
        if any(a.key == key and a.value == value for a in subject_entity.attributes):
            return

        # Create attribute with traceability
        attr = EntityAttribute(
            key=key,
            value=value,
            predicate_verb=rel.predicate_verb or "",
            confidence=rel.confidence,
            source_relation_id=rel.id,
        )
        subject_entity.attributes.append(attr)

    @staticmethod
    def _parse_attribute_value(rel: Relation) -> tuple[str, str]:
        """Parse the relation object into key/value for an attribute.

        For simple values, the key is the object text and value is empty.
        For compound values like "高强度", try to split into value+key.

        Args:
            rel: The relation to parse.

        Returns:
            (key, value) tuple.
        """
        obj = rel.object
        obj_raw = rel.object_raw or obj

        # Try to split compound values: "高强度" → key="强度", value="高"
        # This is a heuristic: if the last 2+ chars are a common parameter name
        common_suffixes = {
            "强度": "高",
            "韧性": "高",
            "硬度": "高",
            "温度": "高",
            "压力": "高",
            "浓度": "高",
            "速度": "高",
            "精度": "高",
            "密度": "高",
        }

        for suffix in common_suffixes:
            if obj_raw.endswith(suffix) and len(obj_raw) > len(suffix):
                return suffix, obj_raw[:-len(suffix)]

        # Default: use the full text as key
        return obj, ""