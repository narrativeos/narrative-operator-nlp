"""
Entity ID Generator — Deterministic entity ID generation.

Generates stable, deterministic entity IDs based on text content and span,
ensuring the same entity gets the same ID across runs.
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)


class EntityIdGenerator:
    """Generates deterministic entity IDs.

    Uses a counter-based approach (ent_001, ent_002, ...) for backward
    compatibility, but tracks seen (text, span) pairs to avoid duplicate IDs.

    For fully deterministic IDs (e.g., for caching/comparison), use
    hash_mode=True to generate hash-based IDs.

    Usage:
        generator = EntityIdGenerator()
        ent_id = generator.generate("北京", (0, 2), "LOCATION")
        # -> "ent_001"
    """

    def __init__(self, hash_mode: bool = False):
        self._counter = 0
        self._seen: set[tuple[str, int, int]] = set()
        self._hash_mode = hash_mode

    def generate(self, text: str, span: tuple[int, int], category: str) -> str | None:
        """Generate a unique entity ID.

        Returns None if the entity at this span has already been assigned an ID.

        Args:
            text: Entity text.
            span: (start, end) character offset.
            category: Entity category (included for hash-based IDs).

        Returns:
            Entity ID string, or None if duplicate.
        """
        span_key = (text, span[0], span[1])
        if span_key in self._seen:
            return None
        self._seen.add(span_key)

        if self._hash_mode:
            key = f"{text}|{span[0]}:{span[1]}|{category}"
            hash_val = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
            return f"ent_{hash_val}"
        else:
            self._counter += 1
            return f"ent_{self._counter:03d}"

    @property
    def counter(self) -> int:
        """Current counter value (for debugging)."""
        return self._counter

    def reset(self):
        """Reset the generator state."""
        self._counter = 0
        self._seen = set()