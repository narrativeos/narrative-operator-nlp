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
from .dictionary_loader import DictionaryLoader

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

# Function words that should NEVER be extracted as entities (classical Chinese).
# Applied to both the PROPN loop and the xpos loop in _map_classical_entities.
_CLASSICAL_FUNCTION_WORDS = frozenset({
    # 虚词 - 代词
    "之", "其", "者", "所", "何", "安", "孰", "胡", "奚", "焉",
    # 虚词 - 助词
    "也", "矣", "乎", "哉", "耳", "焉", "兮", "夫", "盖", "惟",
    # 虚词 - 介词
    "于", "以", "而", "则", "若", "如", "使", "令", "况",
    # 虚词 - 连词
    "与", "及", "且", "或", "虽", "然", "故", "因", "是",
    "於是", "于是", "而况", "虽然", "既而", "遂",
    # 常见非实体词
    "人", "字", "时", "少", "多", "大", "小", "上", "下",
    "父", "母", "子", "女", "兄", "弟", "妻", "夫",
    "君", "臣", "官", "民", "兵", "将", "军",
})


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


def _compute_iou(span_a: tuple[int, int], span_b: tuple[int, int]) -> float:
    """Compute Intersection-over-Union of two spans."""
    inter_start = max(span_a[0], span_b[0])
    inter_end = min(span_a[1], span_b[1])
    if inter_end <= inter_start:
        return 0.0
    intersection = inter_end - inter_start
    union = (span_a[1] - span_a[0]) + (span_b[1] - span_b[0]) - intersection
    return intersection / union if union > 0 else 0.0


def _resolve_ner_conflicts(candidates: list[Entity]) -> None:
    """Resolve NER multi-model conflicts in-place.

    Groups candidates by span overlap (IoU > 0.8). When different NER models
    assign different categories to the same span, marks the winner as
    ner_disputed=True and records all labels in ner_labels.

    Conflict resolution: keep the highest-confidence entity, discard others.
    Non-conflicting groups (all labels agree) are left unchanged.

    Modifies candidates list in-place (removes discarded entities).
    """
    if len(candidates) <= 1:
        return

    # Group by span overlap
    groups: list[list[int]] = []  # list of lists of candidate indices
    assigned: set[int] = set()

    for i in range(len(candidates)):
        if i in assigned:
            continue
        group = [i]
        assigned.add(i)
        for j in range(i + 1, len(candidates)):
            if j in assigned:
                continue
            iou = _compute_iou(candidates[i].span, candidates[j].span)
            # P1: supplement IoU with text containment for partial overlaps
            # e.g. "北京市"[0,3] vs "北京"[0,2] has IoU=0.5 but clearly related
            text_contained = (
                candidates[i].text in candidates[j].text
                or candidates[j].text in candidates[i].text
            )
            if iou > 0.8 or (iou > 0.5 and text_contained):
                group.append(j)
                assigned.add(j)
        groups.append(group)

    # Resolve each group
    for group in groups:
        if len(group) == 1:
            continue

        # Collect unique categories
        categories: set[str] = set()
        ner_labels: dict[str, str] = {}
        for idx in group:
            c = candidates[idx]
            categories.add(c.category)
            ner_labels[c.source] = c.category

        if len(categories) == 1:
            continue  # All models agree, no conflict

        # Conflict: pick highest confidence, mark the rest for removal
        best_idx = max(group, key=lambda idx: candidates[idx].confidence)
        winner = candidates[best_idx]
        winner.ner_disputed = True
        winner.ner_labels = ner_labels

        # Remove losers (iterate in reverse to preserve indices)
        for idx in sorted(group, reverse=True):
            if idx != best_idx:
                candidates.pop(idx)


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

        # Load classical Chinese dictionaries using DictionaryLoader
        # Do NOT load CBDB to memory here - it's too slow (533k entries into keyword_extractor)
        # Instead, use DictionaryLoader for on-demand lookup via _dict_loader.lookup()
        self._dict_loader = DictionaryLoader(config_dir, load_cbdb_to_memory=False)
        total_loaded = self._dict_loader.load_all()
        
        if total_loaded > 0:
            logger.info("Classical dictionaries loaded: %d entries (YAML only)", total_loaded)

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
        if cs is None:
            # Span cannot be resolved reliably (bad token indices and
            # ambiguous text occurrence) — drop the entity rather than
            # emit a wrong span.
            return None

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

        # ── Layer 1: NER models (with conflict detection — Step 8) ──
        # Phase A: Collect candidates from all NER sources
        all_ner_candidates: list[Entity] = []
        for ner_key in self._label_mapper.ner_sources:
            for raw_ent in raw.get(ner_key, []):
                mapped = self.map(raw_ent, ner_key, tokens, text)
                if mapped:
                    all_ner_candidates.append(mapped)

        # Phase B: Resolve conflicts (IoU > 0.8, different labels)
        _resolve_ner_conflicts(all_ner_candidates)

        # Phase C: Deduplicate exact-text duplicates and add to entities
        for candidate in all_ner_candidates:
            if not EntityDeduplicator.is_duplicate(candidate, entities):
                entities.append(candidate)

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
        entities = self._merger.merge_same_category(entities, text)
        entities.sort(key=lambda e: e.span[0])
        entities = self._merger.merge_cross_category(entities, text)

        if not has_ner and not has_srl:
            entities = self._merger.merge_det_entities(entities, tokens, raw, self._id_gen)

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
    ) -> Optional[tuple[int, int]]:
        """Resolve an entity's character span.

        Returns None when the span cannot be resolved reliably:
        - Token indices out of range AND the entity text occurs more than
          once in the text (``text.find`` would silently map the mention to
          its first occurrence, producing a wrong span for later mentions).
        """
        if 0 <= tok_s < len(tokens) and 0 < tok_e <= len(tokens):
            return tokens[tok_s].span[0], tokens[tok_e - 1].span[1]
        if text and ent_text:
            first = text.find(ent_text)
            if first >= 0:
                # Only safe when the text occurs exactly once
                if text.find(ent_text, first + 1) < 0:
                    return first, first + len(ent_text)
        return None

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

        # NOTE: single-char suffixes (会/社/山/河/市/区/院/所/部/局...)
        # are far too aggressive — they misclassify common words like
        # 社会→ORGANIZATION, 黄山→LOCATION. Only multi-char,
        # high-precision suffixes are kept, and suffix matching is
        # additionally gated by a minimum text length below.
        ORG_SUFFIXES = {"公司", "集团", "有限", "股份", "中心",
                       "大学", "学院", "学校", "医院", "银行",
                       "委员会", "协会", "学会"}
        LOC_SUFFIXES = {"省", "市", "区", "县", "公路", "大街",
                       "省城", "市区"}
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
        # Suffix matching is only meaningful for multi-char names;
        # 2-char words like 社会/黄山 must not be classified by suffix.
        if len(text) < 3:
            return "UNKNOWN"
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
            # Skip function words (e.g. 之/其 mis-tagged as PROPN)
            if t.text in _CLASSICAL_FUNCTION_WORDS:
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

        # (Function-word filtering uses the module-level
        # _CLASSICAL_FUNCTION_WORDS constant, applied in both loops above.)

        for i, t in enumerate(tokens):
            span_key = (t.span[0], t.span[1])
            if span_key in entity_spans:
                continue
            # Skip function words
            if t.text in _CLASSICAL_FUNCTION_WORDS:
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
        entities = self._merger.merge_same_category(entities, text)

        return entities

    @staticmethod
    def _parse_xpos_category(xpos: str) -> Optional[str]:
        """Parse xpos tag to entity category.
        
        Supports both modern and classical Chinese xpos tags.
        Classical Chinese extensions:
        - 官职名 → TITLE
        - 时代名 → ERA
        - 典章制度 → INSTITUTION
        - 天文历法 → ASTRONOMY
        - 爵位名 → TITLE
        - 朝代名 → ERA
        
        xpos format examples:
        - "名詞,地名" → LOCATION
        - "名詞,人名" → PERSON
        - "名詞,官職名" → TITLE
        """
        if not xpos:
            return None
        
        # Handle both comma-separated and simple formats.
        # CTB xpos format is "大类,小类" (e.g. "名詞,地名"):
        # parts[0] is the major class, parts[1:] are the minor classes.
        if "," in xpos:
            parts = xpos.split(",")
            major_class = parts[0].strip()
            # Category checks run against the minor classes
            # (e.g. "名詞,一般名詞,動物" → "一般名詞,動物")
            minor_classes = ",".join(p.strip() for p in parts[1:])
        else:
            # Simple format like "名詞" or "地名"
            major_class = xpos
            minor_classes = xpos

        # Check if it's a noun class (major class is the first part)
        if "名詞" not in major_class:
            # Try direct matching for simple formats
            if "官職" in xpos or "官名" in xpos or "爵位" in xpos:
                return "TITLE"
            if "朝代" in xpos or "时代" in xpos or "年代" in xpos:
                return "ERA"
            if "典章" in xpos or "制度" in xpos or "礼制" in xpos:
                return "INSTITUTION"
            if "天文" in xpos or "历法" in xpos or "星宿" in xpos:
                return "ASTRONOMY"
            if "地名" in xpos or "地形" in xpos:
                return "LOCATION"
            if "人名" in xpos:
                return "PERSON"
            if "組織" in xpos:
                return "ORGANIZATION"
            if "作品" in xpos:
                return "PRODUCT"
            return None
        
        # Classical Chinese categories
        if "官职" in minor_classes or "官名" in minor_classes or "爵位" in minor_classes or "官職" in minor_classes:
            return "TITLE"
        if "时代" in minor_classes or "朝代" in minor_classes or "年代" in minor_classes:
            return "ERA"
        if "典章" in minor_classes or "制度" in minor_classes or "礼制" in minor_classes:
            return "INSTITUTION"
        if "天文" in minor_classes or "历法" in minor_classes or "星宿" in minor_classes:
            return "ASTRONOMY"
        
        # Standard categories
        if "地名" in minor_classes or "地形" in minor_classes:
            return "LOCATION"
        if "人" in minor_classes and "名" in minor_classes:
            return "PERSON"
        if "組織" in minor_classes:
            return "ORGANIZATION"
        if "作品" in minor_classes:
            return "PRODUCT"
        if "固有名詞" in minor_classes:
            return "LOCATION"
        if "固定物" in minor_classes:
            return "LOCATION"
        if "主体" in minor_classes or "動物" in minor_classes or "植物" in minor_classes:
            return "UNKNOWN"
        return None
