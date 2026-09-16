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
from .entity_id_generator import EntityIdGenerator

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

    @staticmethod
    def _mergeable_in_text(start: int, end: int, text: str) -> bool:
        """Return True if text[start:end] contains no punctuation.

        Used to guard same-category merging: entities separated by
        punctuation (。，、；：！？ etc.) are distinct mentions and must
        not be merged into a compound.
        """
        segment = text[start:end]
        for ch in segment:
            if not ch.isalnum() and not ch.isspace():
                # Any non-alphanumeric, non-space character (punctuation,
                # CJK or Latin) blocks the merge.
                return False
        return True

    def merge_same_category(
        self,
        entities: list[Entity],
        text: str = "",
    ) -> list[Entity]:
        """Merge adjacent same-category entities.

        '北京'(LOC) + '立方庭'(LOC) → '北京立方庭'(LOC)
        '碳'(MATERIAL) + '钢'(MATERIAL) → '碳钢'(MATERIAL)

        This is the original behavior, always enabled.

        Guard: when ``text`` is provided, two entities are only merged if
        no punctuation separates them in the original text. This prevents
        merging distinct entities that merely sit next to each other
        across a sentence boundary (e.g. '海淀区' + '中关村' must NOT
        become '海淀区中关村').
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
                    # Guard: refuse to merge across punctuation
                    if text and not self._mergeable_in_text(
                        cur.span[0], nxt.span[1], text
                    ):
                        break
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

    def merge_cross_category(
        self,
        entities: list[Entity],
        text: str = "",
    ) -> list[Entity]:
        """Merge adjacent entities using configured cross-category rules.

        Applies rules in priority order. Each rule defines a from→to pattern.
        Only adjacent entities (no gap) are merged. When ``text`` is
        provided, merges across punctuation are refused (same guard as
        merge_same_category).
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
                            and cur.span[1] == nxt.span[0]
                            and (not text or self._mergeable_in_text(
                                cur.span[0], nxt.span[1], text))):
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
        id_generator: Optional[EntityIdGenerator] = None,
    ) -> list[Entity]:
        """Merge determiners into their entity head, creating compound entities.

        Strategy:
        - Keep the original entity (e.g., "世界杯") intact
        - Create a NEW compound entity (e.g., "这届世界杯") with a new ID
        - EntityHierarchyBuilder will then detect the containment relationship
          and generate a PART_OF relation between them

        Uses dependency parse to find det→head relationships.

        Args:
            entities: List of entities to process.
            tokens: Token list for text reconstruction.
            raw: Raw NLP output containing dependency parse.
            id_generator: Optional ID generator for new compound entities.
                         If None, uses a simple counter-based fallback.
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

        # Simpler approach: find det tokens that immediately precede entities
        result: list[Entity] = []
        merged_indices: set[int] = set()
        _fallback_counter = 0

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
                result.append(e)
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
                result.append(e)
                continue

            # Build compound text: include ALL tokens between det and entity head
            # (e.g., "这" + "届" + "世界杯" → "这届世界杯")
            all_idx = sorted(set([e_idx] + det_children))
            min_idx, max_idx = all_idx[0], all_idx[-1]
            for ii in range(min_idx, max_idx + 1):
                all_idx.append(ii)
            all_idx = sorted(set(all_idx))
            merged_text = "".join(tokens[ii].text for ii in all_idx)
            merged_span = (
                tokens[all_idx[0]].span[0],
                tokens[all_idx[-1]].span[1],
            )

            # Keep the original entity intact (it's a sub-entity of the compound)
            result.append(e)

            # Generate a new ID for the compound entity
            if id_generator is not None:
                new_id = id_generator.generate(merged_text, merged_span, e.category)
                if new_id is None:
                    # Duplicate span already exists; skip compound creation
                    merged_indices.add(e_idx)
                    merged_indices.update(det_children)
                    continue
            else:
                _fallback_counter += 1
                new_id = f"ent_det_{_fallback_counter:03d}"

            # Create a NEW compound entity with the expanded text
            result.append(Entity(
                id=new_id,
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

        return result
