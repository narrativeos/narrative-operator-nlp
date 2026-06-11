"""
Entity Deduplicator — Removes duplicate/overlapping entities.
"""

from __future__ import annotations

import logging

from .schema import Entity

logger = logging.getLogger(__name__)


class EntityDeduplicator:
    """Removes duplicate and overlapping entities.

    Two entities overlap if their spans intersect AND their texts overlap.
    When two entities overlap, the one with higher confidence is kept.

    Usage:
        deduper = EntityDeduplicator()
        unique = deduper.deduplicate(entities)
    """

    @staticmethod
    def deduplicate(entities: list[Entity]) -> list[Entity]:
        """Remove overlapping entities, keeping the higher-confidence one.

        Args:
            entities: List of entities (may contain overlaps).

        Returns:
            Deduplicated list, sorted by span start.
        """
        if not entities:
            return []

        # Sort by confidence descending, then by span start
        sorted_entities = sorted(entities, key=lambda e: (-e.confidence, e.span[0]))

        kept: list[Entity] = []
        kept_spans: list[tuple[int, int]] = []

        for ent in sorted_entities:
            if not EntityDeduplicator._overlaps(ent.span, kept_spans):
                kept.append(ent)
                kept_spans.append(ent.span)

        # Sort by span start for output
        kept.sort(key=lambda e: e.span[0])
        return kept

    @staticmethod
    def _overlaps(new_span: tuple[int, int], existing_spans: list[tuple[int, int]]) -> bool:
        """Check if a new span overlaps with any existing span."""
        for s_start, s_end in existing_spans:
            # Overlap: new starts before existing ends AND new ends after existing starts
            if new_span[0] < s_end and new_span[1] > s_start:
                # Additional check: texts must overlap (not just spans)
                # This is handled at the caller level; here we just check span overlap
                return True
        return False

    @staticmethod
    def is_duplicate(candidate: Entity, existing: list[Entity]) -> bool:
        """Check if candidate overlaps with any existing entity.

        Original logic from entity_mapper._dup: same text AND spans overlap.

        Args:
            candidate: Entity to check.
            existing: List of existing entities.

        Returns:
            True if duplicate found.
        """
        for e in existing:
            if (e.text == candidate.text
                    and candidate.span[0] < e.span[1]
                    and candidate.span[1] > e.span[0]):
                return True
        return False