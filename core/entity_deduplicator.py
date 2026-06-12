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

        Improved: preserves containment relationships.
        - If A fully contains B (same category) → keep both (hierarchy will handle)
        - If A crosses B (partial overlap) → keep higher-confidence one
        """
        if not entities:
            return []

        # Sort by span length descending (longer = more specific = parent candidate)
        sorted_entities = sorted(entities, key=lambda e: (
            -(e.span[1] - e.span[0]),
            e.span[0],
        ))

        kept: list[Entity] = []
        kept_spans: list[tuple[int, int]] = []

        for ent in sorted_entities:
            # Check if this entity is fully contained by any kept entity (same category)
            is_contained = False
            for k in kept:
                if k.category != ent.category:
                    continue
                if (k.span[0] <= ent.span[0]
                        and k.span[1] >= ent.span[1]
                        and (k.span[0] < ent.span[0] or k.span[1] > ent.span[1])):
                    is_contained = True
                    break

            if is_contained:
                # Contained by a kept entity → keep both (hierarchy will link them)
                kept.append(ent)
                kept_spans.append(ent.span)
                continue

            # Check for crossing overlap (not full containment)
            should_add = True
            for k in kept:
                if k.category != ent.category:
                    continue
                # Overlap check
                if (ent.span[0] < k.span[1] and ent.span[1] > k.span[0]):
                    # Check if it's full containment (already handled above as is_contained)
                    # Also check if ent fully contains k (reverse containment)
                    if (ent.span[0] <= k.span[0] and ent.span[1] >= k.span[1]
                            and (ent.span[0] < k.span[0] or ent.span[1] > k.span[1])):
                        # ent contains k → keep both (hierarchy will handle)
                        continue
                    # Crossing overlap (partial, not full containment)
                    if ent.confidence >= k.confidence:
                        kept.remove(k)
                        kept_spans.remove(k.span)
                    else:
                        should_add = False
                    break

            if should_add:
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