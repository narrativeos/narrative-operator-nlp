"""
Entity Mapping — Orchestrator for entity extraction.

Three-layer extraction pipeline:
  Layer 1: NER models (PKU/MSRA/OntoNotes) → PERSON/LOCATION/ORGANIZATION
  Layer 2: New word discovery (PMI+MTL) → UNKNOWN (optional, user-controlled)
  Layer 3: User custom dictionary → user-defined categories (optional)

Then: merge → deduplicate → sort.

This module is a thin orchestrator that delegates to specialized components.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .schema import Entity, EntityCategory, Token
from .ner_label_mapper import NerLabelMapper
from .keyword_extractor import KeywordExtractor
from .entity_merger import EntityMerger
from .entity_deduplicator import EntityDeduplicator
from .entity_id_generator import EntityIdGenerator

logger = logging.getLogger(__name__)

# Backward compatibility: mapper.py imports _PARAMETER from this module.
_PARAMETER: frozenset = frozenset()


def _load_default_parameter_keywords() -> frozenset:
    """Load parameter keywords from the default config for backward compat."""
    global _PARAMETER
    if _PARAMETER:
        return _PARAMETER
    try:
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "domain_keywords.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            _PARAMETER = frozenset(config.get("parameter", []))
    except Exception:
        pass
    return _PARAMETER


# Minimum score threshold for discovered words to become entities
# PMI-based scores: 3.0+ is strong, 2.0+ is moderate
_DISCOVER_ENTITY_THRESHOLD = 3.0


class EntityMappingRules:
    """Orchestrates entity extraction from HanLP output.

    Three-layer pipeline:
      1. NER models (PERSON/LOCATION/ORGANIZATION/FACILITY)
      2. New word discovery (UNKNOWN — optional, user-controlled)
      3. User custom dictionary (optional, passed via entity_categories)

    Usage:
        rules = EntityMappingRules()
        entities = rules.map_all(text, raw, tokens)

        # With custom dictionary:
        rules = EntityMappingRules(entity_categories={
            "MATERIAL": ["石墨烯"],
        })
    """

    def __init__(
        self,
        config_dir: Optional[str] = None,
        entity_categories: Optional[dict[str, list[str]]] = None,
    ):
        # Each component loads its own config file from the config directory
        if config_dir is None:
            config_dir = str(Path(__file__).parent.parent / "config")

        self._label_mapper = NerLabelMapper._from_dir(config_dir)
        self._keyword_extractor = KeywordExtractor._from_dir(config_dir)
        self._merger = EntityMerger._from_dir(config_dir)
        self._id_gen = EntityIdGenerator()

        # Layer 3: Inject user-defined custom keywords
        if entity_categories:
            self._keyword_extractor.add_keywords(entity_categories)
            logger.info(
                "Injected %d custom keyword categories",
                len(entity_categories),
            )

        # Sync _PARAMETER for backward compatibility with mapper.py
        _load_default_parameter_keywords()

    @property
    def label_mapper(self) -> NerLabelMapper:
        return self._label_mapper

    @property
    def keyword_extractor(self) -> KeywordExtractor:
        return self._keyword_extractor

    @property
    def merger(self) -> EntityMerger:
        return self._merger

    def reset(self):
        """Reset per-request state. Must be called before each new analysis."""
        self._id_gen.reset()

    # ── Public API (backward compatible) ──

    def map(self, raw_tuple: tuple, source: str, tokens: list[Token],
            text: str = "") -> Optional[Entity]:
        """Map a single NER tuple to an Entity."""
        if len(raw_tuple) < 4:
            return None

        ent_text = str(raw_tuple[0]).strip()
        label = str(raw_tuple[1]).strip()
        tok_s, tok_e = int(raw_tuple[2]), int(raw_tuple[3])
        confidence = float(raw_tuple[4]) if len(raw_tuple) >= 5 else 1.0

        if not ent_text or not label:
            return None

        category = self._label_mapper.map_label(label, source)
        if category is None:
            category = self._keyword_extractor._keyword_category(ent_text)
        if category is None:
            return None

        cs, ce = self._resolve_span(ent_text, tok_s, tok_e, tokens, text)

        ent_id = self._id_gen.generate(ent_text, (cs, ce), category)
        if ent_id is None:
            return None

        return Entity(
            id=ent_id,
            text=ent_text,
            category=category,
            span=(cs, ce),
            normalized=ent_text,
            source=source,
            confidence=confidence,
        )

    def map_all(
        self,
        text: str,
        raw: dict,
        tokens: list[Token],
        entity_dict: dict[str, str] | None = None,
        auto_discover_entities: bool = False,
    ) -> list[Entity]:
        """Extract all entities from HanLP output.

        Pipeline:
          Layer 1: NER models → known categories
          Layer 2: New word discovery → UNKNOWN (optional, user-controlled)
          Layer 3: Built-in keywords (material/standard/parameter)
          Layer 3: User custom dictionary (if provided via __init__)
          Merge: same-category merge → cross-category merge → dedup → sort

        Args:
            text: Original text.
            raw: HanLP output dict.
            tokens: Token list.
            entity_dict: Optional user-provided entity dictionary (backward compat).
            auto_discover_entities: If True, run new word discovery (PMI+MTL)
                and promote high-score candidates to UNKNOWN entities.
                Default False — discovery is opt-in so users control recall/precision.

        Returns:
            Sorted list of Entity objects.
        """
        entities: list[Entity] = []

        # ── Layer 1: NER models ──
        for ner_key in self._label_mapper.ner_sources:
            for raw_ent in raw.get(ner_key, []):
                mapped = self.map(raw_ent, ner_key, tokens, text)
                if mapped and not EntityDeduplicator.is_duplicate(mapped, entities):
                    entities.append(mapped)

        # ── Layer 2: New word discovery (optional, user-controlled) ──
        if auto_discover_entities:
            discover_entities = self._extract_from_discover(tokens, entities, text)
            entities.extend(discover_entities)

        # ── Layer 3a: Built-in keyword extraction (material/standard/parameter) ──
        keyword_entities = self._keyword_extractor.extract(
            tokens, entities, text, self._id_gen
        )
        entities.extend(keyword_entities)

        # ── Layer 3b: Classical Chinese fallback (when no NER/SRL) ──
        has_ner = any(
            raw.get(k) for k in self._label_mapper.ner_sources
        )
        has_srl = bool(raw.get("srl"))
        if not has_ner and not has_srl:
            classical_entities = self._map_classical_entities(
                tokens, text, raw, entity_dict,
            )
            for ce in classical_entities:
                if not EntityDeduplicator.is_duplicate(ce, entities):
                    entities.append(ce)

        # ── Merge & Dedup ──
        entities.sort(key=lambda e: e.span[0])
        entities = self._merger.merge_same_category(entities)
        entities.sort(key=lambda e: e.span[0])
        entities = self._merger.merge_cross_category(entities)

        if not has_ner and not has_srl:
            entities = self._merger.merge_det_entities(entities, tokens, raw)

        entities = EntityDeduplicator.deduplicate(entities)

        return entities

    # ── Layer 2: New word discovery → entities ──

    def _extract_from_discover(
        self,
        tokens: list[Token],
        existing_entities: list[Entity],
        text: str,
    ) -> list[Entity]:
        """Extract entities from new word discovery (PMI+MTL).

        Runs the discoverer module, filters candidates by score threshold,
        and promotes high-score candidates to UNKNOWN entities.

        This is the industry-standard approach for unsupervised entity
        discovery, replacing the previous POS-based extraction.

        All extracted entities have category=UNKNOWN and source="discover".
        """
        from .discoverer import discover as run_discover

        result: list[Entity] = []
        entity_spans = {(e.span[0], e.span[1]) for e in existing_entities}

        try:
            candidates = run_discover(
                text,
                min_len=2,
                max_len=4,
                min_freq=1,
                min_score=_DISCOVER_ENTITY_THRESHOLD,
                max_candidates=30,
                mtl_tokens=tokens,
            )

            for c in candidates.candidates:
                # Find the word in text to get span
                idx = text.find(c.word)
                if idx < 0:
                    continue
                span = (idx, idx + len(c.word))
                if span in entity_spans:
                    continue

                ent_id = self._id_gen.generate(c.word, span, "UNKNOWN")
                if ent_id is None:
                    continue

                # Convert PMI score to confidence [0, 1]
                # PMI scores typically range 0-10; normalize to 0.5-0.85
                confidence = min(0.85, 0.5 + c.score * 0.05)

                result.append(Entity(
                    id=ent_id,
                    text=c.word,
                    category="UNKNOWN",
                    span=span,
                    normalized=c.word,
                    source="discover",
                    confidence=confidence,
                ))
                entity_spans.add(span)

        except Exception as exc:
            logger.warning("New word discovery failed: %s", exc)

        result.sort(key=lambda e: e.span[0])
        return result

    # ── Helper methods ──

    @staticmethod
    def _resolve_span(
        ent_text: str,
        tok_s: int,
        tok_e: int,
        tokens: list[Token],
        text: str,
    ) -> tuple[int, int]:
        if 0 <= tok_s < len(tokens) and 0 < tok_e <= len(tokens):
            return tokens[tok_s].span[0], tokens[tok_e - 1].span[1]
        if text and ent_text:
            idx = text.find(ent_text)
            if idx >= 0:
                return idx, idx + len(ent_text)
        return 0, len(ent_text)

    # ── Classical Chinese entity extraction ──

    def _map_classical_entities(
        self,
        tokens: list[Token],
        text: str,
        raw: dict | None = None,
        entity_dict: dict[str, str] | None = None,
    ) -> list[Entity]:
        """Extract entities from classical Chinese using model-native signals."""
        entities: list[Entity] = []
        entity_spans: set[tuple[int, int]] = set()
        xpos_tags: list[str] = (raw or {}).get("pos/xpos", [])

        # Strategy 1: User-provided entity dictionary
        if entity_dict:
            sorted_dict = sorted(entity_dict.items(), key=lambda x: -len(x[0]))
            for term, cat in sorted_dict:
                if cat not in EntityCategory.ALL:
                    continue
                start = 0
                while True:
                    idx = text.find(term, start)
                    if idx < 0:
                        break
                    span_key = (idx, idx + len(term))
                    if span_key not in entity_spans:
                        ent_id = self._id_gen.generate(term, span_key, cat)
                        if ent_id:
                            entities.append(Entity(
                                id=ent_id,
                                text=term,
                                category=cat,
                                span=span_key,
                                normalized=term,
                                source="entity_dict",
                                confidence=0.95,
                            ))
                            entity_spans.add(span_key)
                    start = idx + 1

        # Strategy 2: PROPN/NR
        for i, t in enumerate(tokens):
            span_key = (t.span[0], t.span[1])
            if span_key in entity_spans:
                continue
            if t.pos in ("PROPN", "NR"):
                xpos = xpos_tags[i] if i < len(xpos_tags) else ""
                xcat = self._parse_xpos_category(xpos) if xpos else None
                cat = xcat or ("LOCATION" if len(t.text) >= 2 else None)
                if cat is None:
                    continue
                ent_id = self._id_gen.generate(t.text, t.span, cat)
                if ent_id:
                    entities.append(Entity(
                        id=ent_id,
                        text=t.text,
                        category=cat,
                        span=t.span,
                        normalized=t.text,
                        source=f"classical_{'xpos' if xcat else 'propn'}",
                        confidence=0.80,
                    ))
                    entity_spans.add(span_key)

        # Strategy 3: xpos-only
        for i, t in enumerate(tokens):
            span_key = (t.span[0], t.span[1])
            if span_key in entity_spans:
                continue
            xpos = xpos_tags[i] if i < len(xpos_tags) else ""
            if not xpos:
                continue
            xcat = self._parse_xpos_category(xpos)
            if "助数詞" in xpos or "度量衡" in xpos:
                continue
            if xcat is None and t.pos == "NOUN":
                xcat = "UNKNOWN"
            if xcat is None:
                continue
            ent_id = self._id_gen.generate(t.text, t.span, xcat)
            if ent_id:
                entities.append(Entity(
                    id=ent_id,
                    text=t.text,
                    category=xcat,
                    span=t.span,
                    normalized=t.text,
                    source="classical_xpos",
                    confidence=0.65,
                ))
                entity_spans.add(span_key)

        entities.sort(key=lambda e: e.span[0])
        entities = self._merger.merge_same_category(entities)

        return entities

    @staticmethod
    def _parse_xpos_category(xpos: str) -> Optional[str]:
        """Parse LZH xpos semantic categories into NSP entity categories."""
        if not xpos or "," not in xpos:
            return None
        parts = xpos.split(",")
        if len(parts) < 2:
            return None
        pos_class = parts[1].strip()
        if pos_class != "名詞":
            return None
        rest = ",".join(parts[2:]) if len(parts) > 2 else ""
        if "地名" in rest or "地形" in rest:
            return "LOCATION"
        if "人" in rest and "名" in rest:
            return "PERSON"
        if "組織" in rest:
            return "ORGANIZATION"
        if "作品" in rest:
            return "PRODUCT"
        if "固有名詞" in rest:
            return "LOCATION"
        if "固定物" in rest:
            return "LOCATION"
        if "主体" in rest or "動物" in rest or "植物" in rest:
            return "UNKNOWN"
        return None