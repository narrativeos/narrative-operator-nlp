"""
Entity Hierarchy Builder — Detect containment relationships between entities.

After entity merging and deduplication, this module:
1. Detects containment relationships (span A contains span B, same category)
2. Sets parent_entity_id on child entities
3. Generates IS_PART_OF relations for containment pairs

Note: PART_OF already exists in RelationPredicate, so we reuse it.
"""

from __future__ import annotations

from .schema import Entity, Relation


class EntityHierarchyBuilder:
    """Detect and build entity containment hierarchies.

    For same-category entities where one span contains another:
    - Set parent_entity_id on the child
    - Generate a PART_OF relation
    """

    def __init__(self, relation_counter: int = 0):
        self._counter = relation_counter

    def build(
        self,
        entities: list[Entity],
        text: str,
    ) -> list[Relation]:
        """Build entity hierarchy and generate PART_OF relations.

        Modifies entities in-place to set parent_entity_id.

        Args:
            entities: List of entities (modified in-place).
            text: Original text for evidence extraction.

        Returns:
            List of PART_OF relations for containment pairs.
        """
        part_of_relations: list[Relation] = []

        # Sort by span length descending (longest first = most specific = parent)
        sorted_entities = sorted(entities, key=lambda e: e.span[1] - e.span[0], reverse=True)

        for i, child in enumerate(entities):
            # Find the smallest entity that fully contains this one
            # (most specific parent)
            best_parent: Entity | None = None
            best_parent_len = float('inf')

            for other in sorted_entities:
                if other.id == child.id:
                    continue
                if other.category != child.category:
                    continue

                # Check if 'other' fully contains 'child'
                if (other.span[0] <= child.span[0]
                        and other.span[1] >= child.span[1]
                        and (other.span[0] < child.span[0] or other.span[1] > child.span[1])):
                    # 'other' contains 'child'
                    other_len = other.span[1] - other.span[0]
                    child_len = child.span[1] - child.span[0]
                    if other_len < best_parent_len and other_len > child_len:
                        best_parent = other
                        best_parent_len = other_len

            if best_parent is None:
                continue

            # Set parent relationship
            child.parent_entity_id = best_parent.id

            # Generate PART_OF relation
            self._counter += 1
            evidence = text[child.span[0]:child.span[1]] if child.span[1] <= len(text) else child.text
            part_of_relations.append(Relation(
                id=f"rel_{self._counter:03d}",
                subject=child.text,
                subject_raw=child.text,
                predicate="PART_OF",
                predicate_verb=None,
                object=best_parent.text,
                object_raw=best_parent.text,
                evidence=evidence,
                evidence_span=child.span,
                confidence=min(child.confidence, best_parent.confidence),
                source="hierarchy/containment",
            ))

        return part_of_relations

    @staticmethod
    def detect_containment(
        child: Entity,
        potential_parents: list[Entity],
    ) -> Entity | None:
        """Find the most specific (smallest) parent that contains the child.

        Args:
            child: The entity to find a parent for.
            potential_parents: List of potential parent entities.

        Returns:
            The most specific parent entity, or None.
        """
        best: Entity | None = None
        best_len = float('inf')
        child_len = child.span[1] - child.span[0]

        for parent in potential_parents:
            if parent.id == child.id:
                continue
            if parent.category != child.category:
                continue

            parent_len = parent.span[1] - parent.span[0]
            if parent_len <= child_len:
                continue

            # Check full containment
            if (parent.span[0] <= child.span[0]
                    and parent.span[1] >= child.span[1]):
                if parent_len < best_len:
                    best = parent
                    best_len = parent_len

        return best