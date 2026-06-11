"""
Entity Merger — Merges adjacent entities into compound entities.

Supports both same-category merging (original behavior) and cross-category
merging configured via YAML rules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from .schema import Entity, Token

logger = logging.getLogger(__name__)


class EntityMerger:
    """Merges adjacent entities using configurable rules.

    Supports:
    - Same-category merging (always enabled, original behavior)
    - Cross-category merging (configured via merge_rules.yaml)
    - Determiner merging (其+名→其名, configured via det_merge)

    Usage:
        merger = EntityMerger()
        merged = merger.merge_same_category(entities)
        merged = merger.merge_cross_category(entities)
        merged = merger.merge_det_entities(entities, tokens, raw)
    """

    def __init__(self, config_path: Optional[str] = None):
        self._merge_rules: list[dict] = []
        self._det_enabled = True
        self._det_words: set[str] = {"其", "该", "此", "彼", "上述", "下列"}

        if config_path is None:
            config_path = self._default_config_path()

        self._load_config(config_path)

    @staticmethod
    def _default_config_path() -> str:
        config_dir = Path(__file__).parent.parent / "config"
        return str(config_dir / "merge_rules.yaml")

    @staticmethod
    def _from_dir(config_dir: str) -> "EntityMerger":
        """Create instance from a config directory path."""
        return EntityMerger(str(Path(config_dir) / "merge_rules.yaml"))

    def _load_config(self, config_path: str):
        """Load merge rules from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning("Merge rules config not found: %s, using defaults", config_path)
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.error("Failed to load merge rules config: %s", exc)
            return

        self._merge_rules = config.get("merge_rules", [])

        # Sort by priority
        self._merge_rules.sort(key=lambda r: r.get("priority", 999))

        det_config = config.get("det_merge", {})
        self._det_enabled = det_config.get("enabled", True)
        det_words = det_config.get("det_words", [])
        if det_words:
            self._det_words = set(det_words)

        logger.info("Loaded %d merge rules", len(self._merge_rules))

    def merge_same_category(self, entities: list[Entity]) -> list[Entity]:
        """Merge adjacent same-category entities.

        '北京'(LOC) + '立方庭'(LOC) → '北京立方庭'(LOC)
        '碳'(MATERIAL) + '钢'(MATERIAL) → '碳钢'(MATERIAL)

        This is the original behavior, always enabled.
        """
        if len(entities) < 2:
            return entities

        merged: list[Entity] = []
        i = 0
        while i < len(entities):
            cur = entities[i]
            j = i + 1
            while j < len(entities):
                nxt = entities[j]
                if (cur.category == nxt.category
                        and cur.span[1] == nxt.span[0]):
                    # Merge attributes from both entities
                    merged_attrs = list(cur.attributes) + list(nxt.attributes)
                    # Deduplicate by key
                    seen_keys: set[str] = set()
                    deduped_attrs = []
                    for attr in merged_attrs:
                        if attr.key not in seen_keys:
                            seen_keys.add(attr.key)
                            deduped_attrs.append(attr)

                    cur = Entity(
                        id=cur.id,
                        text=cur.text + nxt.text,
                        category=cur.category,
                        span=(cur.span[0], nxt.span[1]),
                        normalized=cur.normalized + nxt.normalized,
                        source=cur.source,
                        confidence=min(cur.confidence, nxt.confidence),
                        attributes=deduped_attrs,
                    )
                    j += 1
                else:
                    break
            merged.append(cur)
            i = j
        return merged

    def merge_cross_category(self, entities: list[Entity]) -> list[Entity]:
        """Merge adjacent entities using configured cross-category rules.

        Applies rules in priority order. Each rule defines a from→to pattern.
        Only adjacent entities (no gap) are merged.
        """
        if not self._merge_rules or len(entities) < 2:
            return entities

        result: list[Entity] = list(entities)

        for rule in self._merge_rules:
            from_cats = rule.get("from", [])
            to_cat = rule.get("to", "")
            if len(from_cats) != 2 or not to_cat:
                continue

            new_result: list[Entity] = []
            i = 0
            while i < len(result):
                if i + 1 < len(result):
                    cur, nxt = result[i], result[i + 1]
                    if (cur.category == from_cats[0]
                            and nxt.category == from_cats[1]
                            and cur.span[1] == nxt.span[0]):
                        # Merge
                        merged_attrs = list(cur.attributes) + list(nxt.attributes)
                        seen_keys: set[str] = set()
                        deduped_attrs = []
                        for attr in merged_attrs:
                            if attr.key not in seen_keys:
                                seen_keys.add(attr.key)
                                deduped_attrs.append(attr)

                        new_result.append(Entity(
                            id=cur.id,
                            text=cur.text + nxt.text,
                            category=to_cat,
                            span=(cur.span[0], nxt.span[1]),
                            normalized=cur.normalized + nxt.normalized,
                            source=cur.source,
                            confidence=min(cur.confidence, nxt.confidence),
                            attributes=deduped_attrs,
                        ))
                        i += 2
                        continue
                new_result.append(result[i])
                i += 1
            result = new_result

        return result

    def merge_det_entities(
        self,
        entities: list[Entity],
        tokens: list[Token],
        raw: dict | None,
    ) -> list[Entity]:
        """Merge determiners into their entity head (其+名→其名).

        Uses dependency parse to find det→head relationships.
        """
        if not self._det_enabled:
            return entities

        dep = raw.get("dep", []) if raw else []
        if not dep:
            return entities

        # Build entity index by token index
        ent_by_idx: dict[int, Entity] = {}
        for e in entities:
            for i, t in enumerate(tokens):
                if t.span == e.span and t.text == e.text:
                    ent_by_idx[i] = e
                    break

        # Find det tokens
        det_tokens: set[int] = set()
        for i, t in enumerate(tokens):
            if t.text in self._det_words:
                # Check if this token has a det relationship to an entity
                for d in dep:
                    if not isinstance(d, (list, tuple)) or len(d) < 2:
                        continue
                    head_idx = int(d[0]) - 1  # 1-based to 0-based
                    rel = str(d[1]).strip().lower()
                    if i == head_idx - 1 and rel == "det":  # This token is the det
                        # Find the head
                        pass
                    if int(d[0]) - 1 == i and rel == "det":
                        # This token IS a det, find its head
                        pass

        # Simpler approach: find det tokens that immediately precede entities
        merged: list[Entity] = []
        merged_indices: set[int] = set()

        for i, e in enumerate(entities):
            # Find which token index this entity corresponds to
            e_idx = None
            for idx, ent in ent_by_idx.items():
                if ent is e:
                    e_idx = idx
                    break

            # If entity doesn't match a single token (multi-token entity),
            # keep it as-is without det merging
            if e_idx is None:
                merged.append(Entity(
                    id=e.id,
                    text=e.text,
                    category=e.category,
                    span=e.span,
                    normalized=e.normalized,
                    source=e.source,
                    confidence=e.confidence,
                    attributes=list(e.attributes),
                ))
                continue

            if e_idx in merged_indices:
                continue

            # Check if previous token is a determiner
            det_children: list[int] = []
            for ci, d in enumerate(dep):
                if not isinstance(d, (list, tuple)) or len(d) < 2:
                    continue
                if int(d[0]) - 1 == e_idx and str(d[1]).strip().lower() == "det":
                    det_children.append(ci)

            if not det_children:
                merged.append(e)
                continue

            # Merge: det child + entity → compound
            all_idx = sorted([e_idx] + det_children)
            merged_text = "".join(tokens[ii].text for ii in all_idx)
            merged_span = (
                tokens[all_idx[0]].span[0],
                tokens[all_idx[-1]].span[1],
            )
            merged.append(Entity(
                id=e.id,
                text=merged_text,
                category=e.category,
                span=merged_span,
                normalized=merged_text,
                source=e.source,
                confidence=e.confidence,
                attributes=list(e.attributes),
            ))
            merged_indices.add(e_idx)
            merged_indices.update(det_children)

        return merged