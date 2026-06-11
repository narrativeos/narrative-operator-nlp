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
_DISCOVER_ENTITY_THRESHOLD = 3.0


def _spans_overlap(
    span: tuple[int, int],
    existing: set[tuple[int, int]],
) -> bool:
    """Check if a span overlaps with any existing span."""
    s, e = span
    for es, ee in existing:
        if s < ee and e > es:
            return True
    return False


class EntityMappingRules:
    """Orchestrates entity extraction from HanLP output.

    Five-layer pipeline:
      1. NER models (PERSON/LOCATION/ORGANIZATION/FACILITY)
      2. New word discovery (UNKNOWN — optional, user-controlled)
      3a. Built-in keywords (material/standard/parameter)
      3b. Numeric entities (dates/amounts/percentages)
      3c. POS-based fallback (NNP/PROPN for modern Chinese)
      3d. Classical Chinese fallback (when no NER/SRL)
    """

    def __init__(
        self,
        config_dir: Optional[str] = None,
        entity_categories: Optional[dict[str, list[str]]] = None,
    ):
        if config_dir is None:
            config_dir = str(Path(__file__).parent.parent / "config")

        self._label_mapper = NerLabelMapper._from_dir(config_dir)
        self._keyword_extractor = KeywordExtractor._from_dir(config_dir)
        self._merger = EntityMerger._from_dir(config_dir)
        self._id_gen = EntityIdGenerator()

        if entity_categories:
            self._keyword_extractor.add_keywords(entity_categories)
            logger.info(
                "Injected %d custom keyword categories",
                len(entity_categories),
            )

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
        self._id_gen.reset()

    # ── Public API (backward compatible) ──

    def map(self, raw_tuple: tuple, source: str, tokens: list[Token],
            text: str = "") -> Optional[Entity]:
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
        entities: list[Entity] = []

        # ── Layer 1: NER models ──
        for ner_key in self._label_mapper.ner_sources:
            for raw_ent in raw.get(ner_key, []):
                mapped = self.map(raw_ent, ner_key, tokens, text)
                if mapped and not EntityDeduplicator.is_duplicate(mapped, entities):
                    entities.append(mapped)

        # ── Layer 2: New word discovery ──
        if auto_discover_entities:
            discover_entities = self._extract_from_discover(tokens, entities, text)
            entities.extend(discover_entities)

        # ── Layer 3a: Built-in keyword extraction ──
        keyword_entities = self._keyword_extractor.extract(
            tokens, entities, text, self._id_gen
        )
        entities.extend(keyword_entities)

        # ── Layer 3b: Numeric entity extraction ──
        from .numeric_extractor import extract_numeric_entities
        entity_spans = {(e.span[0], e.span[1]) for e in entities}
        numeric_entities = extract_numeric_entities(text, entity_spans, self._id_gen)
        entities.extend(numeric_entities)

        # ── Layer 3c: POS-based fallback for modern Chinese ──
        pos_entities = self._extract_pos_based_entities(tokens, entities, text)
        entities.extend(pos_entities)

        # ── Layer 3d: Classical Chinese fallback (when no NER/SRL) ──
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
                idx = text.find(c.word)
                if idx < 0:
                    continue
                span = (idx, idx + len(c.word))
                if span in entity_spans:
                    continue

                ent_id = self._id_gen.generate(c.word, span, "UNKNOWN")
                if ent_id is None:
                    continue

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

    # ── POS-based entity extraction (modern Chinese fallback) ──

    def _extract_pos_based_entities(
        self,
        tokens: list[Token],
        existing_entities: list[Entity],
        text: str,
    ) -> list[Entity]:
        """Extract entities from POS tags when NER misses them."""
        result: list[Entity] = []
        entity_spans = {(e.span[0], e.span[1]) for e in existing_entities}

        NON_ENTITY_WORDS = {
            "的", "了", "在", "是", "我", "有", "和", "就",
            "不", "人", "都", "把", "被", "对", "从", "到",
        }

        ORG_SUFFIXES = {"公司", "集团", "有限", "股份", "中心", "院", "所",
                       "大学", "学院", "学校", "医院", "银行", "部", "局",
                       "委员会", "协会", "学会", "会", "社", "馆"}
        LOC_SUFFIXES = {"省", "市", "区", "县", "路", "街", "镇", "村",
                       "国", "州", "岛", "山", "河", "湖", "海", "港"}
        PRODUCT_SUFFIXES = {"手机", "电脑", "汽车", "系统", "平台", "软件",
                          "服务", "产品", "技术", "芯片"}

        i = 0
        while i < len(tokens):
            t = tokens[i]

            span_key = (t.span[0], t.span[1])
            if span_key in entity_spans:
                i += 1
                continue

            if t.pos not in ("NNP", "PROPN", "NR"):
                i += 1
                continue

            if len(t.text) < 2:
                i += 1
                continue

            merged_text = t.text
            merged_end = t.span[1]
            j = i + 1
            while j < len(tokens):
                nxt = tokens[j]
                if nxt.pos in ("NNP", "PROPN", "NR") and nxt.span[0] == merged_end:
                    if nxt.text in NON_ENTITY_WORDS:
                        break
                    merged_text += nxt.text
                    merged_end = nxt.span[1]
                    j += 1
                else:
                    break

            merged_span = (t.span[0], merged_end)
            if _spans_overlap(merged_span, entity_spans):
                i = j
                continue

            category = self._guess_entity_category(merged_text, ORG_SUFFIXES,
                                                    LOC_SUFFIXES, PRODUCT_SUFFIXES)

            ent_id = self._id_gen.generate(merged_text, merged_span, category)
            if ent_id is None:
                i = j
                continue

            result.append(Entity(
                id=ent_id,
                text=merged_text,
                category=category,
                span=merged_span,
                normalized=merged_text,
                source="pos/nnp",
                confidence=0.55,
            ))
            entity_spans.add(merged_span)
            i = j

        result.sort(key=lambda e: e.span[0])
        return result

    @staticmethod
    def _guess_entity_category(
        text: str,
        org_suffixes: set[str],
        loc_suffixes: set[str],
        product_suffixes: set[str],
    ) -> str:
        for suffix in sorted(org_suffixes, key=len, reverse=True):
            if text.endswith(suffix):
                return "ORGANIZATION"
        for suffix in sorted(loc_suffixes, key=len, reverse=True):
            if text.endswith(suffix):
                return "LOCATION"
        for suffix in sorted(product_suffixes, key=len, reverse=True):
            if text.endswith(suffix):
                return "PRODUCT"
        return "UNKNOWN"

    # ── Classical Chinese entity extraction ──

    def _map_classical_entities(
        self,
        tokens: list[Token],
        text: str,
        raw: dict | None = None,
        entity_dict: dict[str, str] | None = None,
    ) -> list[Entity]:
        entities: list[Entity] = []
        entity_spans: set[tuple[int, int]] = set()
        xpos_tags: list[str] = (raw or {}).get("pos/xpos", [])

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