"""
Entity Mapping Rules — NER Unification Layer.

HanLP MTL NER output: list of (text, label, tok_start, tok_end) tuples.
tok_start/tok_end are token-level indices into tok/fine.
We convert them to character offsets using the token list.
"""

from __future__ import annotations
from typing import Optional
from .schema import Entity, EntityCategory, Token

_PKU_MAP = {"nr":"PERSON","ns":"LOCATION","nt":"ORGANIZATION","nz":"PRODUCT"}
_MSRA_MAP = {"PERSON":"PERSON","LOCATION":"LOCATION","ORGANIZATION":"ORGANIZATION","DATE":"DATE"}
_ONTONOTES_MAP = {"PERSON":"PERSON","NORP":"ORGANIZATION","FAC":"FACILITY","ORG":"ORGANIZATION",
    "GPE":"LOCATION","LOC":"LOCATION","PRODUCT":"PRODUCT","DATE":"DATE","TIME":"DATE",
    "PERCENT":"NUMBER","MONEY":"NUMBER","QUANTITY":"NUMBER","CARDINAL":"NUMBER",
    "ORDINAL":"NUMBER","LAW":"STANDARD","EVENT":"UNKNOWN","WORK_OF_ART":"UNKNOWN","LANGUAGE":"UNKNOWN"}

_MATERIAL = frozenset({
    # Metals & alloys
    "钢","铁","铜","铝","钛","锌","镍","铬","锰","锡","铅","镁","钨","钴",
    "合金","不锈钢","碳钢","铸铁","铝合金","钛合金","镁合金","铜合金",
    # Non-metals
    "塑料","橡胶","陶瓷","玻璃","纤维","复合材料","碳纤维",
    # Elements (single-char, for compound detection)
    "碳","硅","硼","硫","磷",
})
_STANDARD = frozenset({"GB","GB/T","ISO","ASTM","DIN","JIS","EN","标准","规范"})
_PARAMETER = frozenset({"强度","硬度","韧性","密度","熔点","沸点","抗拉强度","屈服强度","延伸率"})


class EntityMappingRules:
    SOURCE_MAPS = {"ner/pku":_PKU_MAP,"ner/msra":_MSRA_MAP,"ner/ontonotes":_ONTONOTES_MAP}

    def __init__(self):
        self._counter = 0

    def map(self, raw_tuple: tuple, source: str, tokens: list[Token], text: str = "") -> Optional[Entity]:
        if len(raw_tuple) < 4:
            return None
        ent_text = str(raw_tuple[0]).strip()
        label = str(raw_tuple[1]).strip()
        tok_s, tok_e = int(raw_tuple[2]), int(raw_tuple[3])
        confidence = float(raw_tuple[4]) if len(raw_tuple) >= 5 else 1.0
        if not ent_text or not label:
            return None
        category = self.SOURCE_MAPS.get(source, {}).get(label) or self._keyword(ent_text)
        if category is None:
            return None
        if 0 <= tok_s < len(tokens) and 0 < tok_e <= len(tokens):
            cs, ce = tokens[tok_s].span[0], tokens[tok_e - 1].span[1]
        elif text and ent_text:
            idx = text.find(ent_text)
            cs, ce = (idx, idx + len(ent_text)) if idx >= 0 else (0, len(ent_text))
        else:
            cs, ce = 0, len(ent_text)
        self._counter += 1
        return Entity(id=f"ent_{self._counter:03d}", text=ent_text, category=category,
                      span=(cs, ce), normalized=ent_text, source=source, confidence=confidence)

    def map_all(self, text: str, raw: dict, tokens: list[Token],
                entity_dict: dict[str, str] | None = None) -> list[Entity]:
        entities = []
        for ner_key in ["ner/pku","ner/msra","ner/ontonotes"]:
            for raw_ent in raw.get(ner_key, []):
                mapped = self.map(raw_ent, ner_key, tokens, text)
                if mapped and not self._dup(mapped, entities):
                    entities.append(mapped)

        # Post-processing
        entities.sort(key=lambda e: e.span[0])
        entities = self._merge_adjacent(entities)
        entities = self._discover_keyword_entities(entities, tokens, text)

        # Classical Chinese fallback: if no NER entities found and no SRL,
        # use model-native signals: PROPN upos + xpos semantic parsing.
        has_ner = any(
            raw.get(k) for k in ["ner/pku", "ner/msra", "ner/ontonotes"]
        )
        has_srl = bool(raw.get("srl"))
        if not has_ner and not has_srl:
            classical_entities = self.map_classical_entities(
                tokens, text, raw, entity_dict,
            )
            for ce in classical_entities:
                if not self._dup(ce, entities):
                    entities.append(ce)

        entities.sort(key=lambda e: e.span[0])
        return entities

    @staticmethod
    def _keyword(text: str) -> Optional[str]:
        # Modern domain keywords only (no classical dictionaries)
        if text in _MATERIAL: return "MATERIAL"
        if text in _STANDARD: return "STANDARD"
        if text in _PARAMETER: return "PARAMETER"
        if len(text) >= 3:
            for kw in _PARAMETER:
                if text.endswith(kw) and len(kw) >= 2:
                    return "PARAMETER"
        return None

    # ── xpos semantic tag parser ──

    @staticmethod
    def _parse_xpos_category(xpos: str) -> Optional[str]:
        """Parse LZH xpos semantic categories into NSP entity categories.

        LZH xpos format (UniDic-based): ``<pos>,<品詞>,<サブカテゴリ>,<詳細>``.

        Entity-relevant patterns (only for ``名詞``, skip ``代名詞``/pronouns):
          - n,名詞,*,地名        → LOCATION   (place name)
          - n,名詞,人,名         → PERSON     (person name)
          - n,名詞,*,組織        → ORGANIZATION
          - n,名詞,*,作品        → PRODUCT
          - n,名詞,固有名詞,...  → UNKNOWN    (proper noun, too general)
          - n,名詞,主体,...      → UNKNOWN    (concrete entity: 魚, 鳥, …)

        Pronouns (代名詞) and other non-nominal tags are skipped to
        avoid false positives like ``其`` (n,代名詞,人称,起格).
        """
        if not xpos or "," not in xpos:
            return None
        parts = xpos.split(",")
        if len(parts) < 2:
            return None
        pos_class = parts[1].strip()
        if pos_class != "名詞":
            return None
        rest = ",".join(parts[2:]) if len(parts) > 2 else ""
        if "地名" in rest:
            return "LOCATION"
        if "地形" in rest:
            return "LOCATION"
        if "人" in rest and "名" in rest:
            return "PERSON"
        if "組織" in rest:
            return "ORGANIZATION"
        if "作品" in rest:
            return "PRODUCT"
        if "固有名詞" in rest:
            return "LOCATION"  # proper noun, default to LOCATION
        # Concrete entities: 固定物 (mountains, rivers, buildings),
        # 主体 (animals: 魚, 鳥), 動物, 植物
        if "固定物" in rest:
            return "LOCATION"  # mountains, buildings → LOCATION
        if "主体" in rest or "動物" in rest or "植物" in rest:
            return "UNKNOWN"
        return None

    def map_classical_entities(
        self, tokens: list[Token], text: str, raw: dict | None = None,
        entity_dict: dict[str, str] | None = None,
    ) -> list[Entity]:
        """Extract entities from classical Chinese using model-native signals.

        Zero hardcoded entity dictionaries. Three strategies:

        1. (Optional) User-provided ``entity_dict`` full-text span matching.
        2. PROPN upos tokens — proper nouns are entity candidates;
           xpos semantic tags (地名→LOCATION, 人,名→PERSON) provide category.
        3. xpos-only — non-PROPN nouns whose xpos carries a semantic hint.

        Args:
            tokens: Token list with upos tags.
            text: Original text for span computation.
            raw: Full HanLP output dict (needs ``pos/xpos``).
            entity_dict: Optional ``{term: NSP_category}`` mapping supplied
                         by the caller (API / frontend).
        """
        entities: list[Entity] = []
        entity_spans: set[tuple[int, int]] = set()
        xpos_tags: list[str] = raw.get("pos/xpos", []) if raw else []

        # ── Strategy 1 (optional): user-provided entity dictionary ──
        if entity_dict:
            # Sort longest-first to prefer longer matches
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
                        self._counter += 1
                        entities.append(Entity(
                            id=f"ent_{self._counter:03d}",
                            text=term, category=cat,
                            span=(idx, idx + len(term)),
                            normalized=term, source="entity_dict",
                            confidence=0.95,
                        ))
                        entity_spans.add(span_key)
                    start = idx + 1

        # ── Strategy 2: PROPN upos (proper nouns) ──
        for i, t in enumerate(tokens):
            span_key = (t.span[0], t.span[1])
            if span_key in entity_spans:
                continue
            if t.pos in ("PROPN", "NR"):
                # Try xpos semantic tag first
                xpos = xpos_tags[i] if i < len(xpos_tags) else ""
                xcat = self._parse_xpos_category(xpos) if xpos else None
                cat = xcat or ("LOCATION" if len(t.text) >= 2 else None)
                if cat is None:
                    continue
                self._counter += 1
                entities.append(Entity(
                    id=f"ent_{self._counter:03d}",
                    text=t.text, category=cat, span=t.span,
                    normalized=t.text,
                    source=f"classical_{'xpos' if xcat else 'propn'}",
                    confidence=0.80,
                ))
                entity_spans.add(span_key)

        # ── Strategy 3: xpos-only (semantic tags on non-PROPN nouns) ──
        for i, t in enumerate(tokens):
            span_key = (t.span[0], t.span[1])
            if span_key in entity_spans:
                continue
            xpos = xpos_tags[i] if i < len(xpos_tags) else ""
            if not xpos:
                continue
            xcat = self._parse_xpos_category(xpos)
            # Exclude classifiers / measure words (助数詞, 度量衡: 里, 个, 匹, …)
            if "助数詞" in xpos or "度量衡" in xpos:
                continue
            # Fallback: NOUN tokens without clear xpos category still
            # get UNKNOWN (e.g. 鹏 mis-tagged as verb by LZH).
            if xcat is None and t.pos == "NOUN":
                xcat = "UNKNOWN"
            if xcat is None:
                continue
            self._counter += 1
            entities.append(Entity(
                id=f"ent_{self._counter:03d}",
                text=t.text, category=xcat, span=t.span,
                normalized=t.text, source="classical_xpos",
                confidence=0.65,
            ))
            entity_spans.add(span_key)

        # Adjacent merge
        entities.sort(key=lambda e: e.span[0])
        entities = self._merge_adjacent(entities)

        # ── Det-compound merge (其 + 名 → 其名) ──
        entities = self._merge_det_entities(entities, tokens, raw)

        return entities

    def _merge_det_entities(
        self, entities: list[Entity], tokens: list[Token], raw: dict | None,
    ) -> list[Entity]:
        """Merge det children into their entity head (其+名→其名)."""
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

        merged: list[Entity] = []
        merged_indices: set[int] = set()

        for i, e in enumerate(entities):
            # Find which token index this entity corresponds to
            e_idx = None
            for idx, ent in ent_by_idx.items():
                if ent is e:
                    e_idx = idx
                    break
            if e_idx is None or e_idx in merged_indices:
                continue

            # Collect det children that point to this entity's token
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
            merged_text = "".join(tokens[i].text for i in all_idx)
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
            ))
            merged_indices.add(e_idx)
            merged_indices.update(det_children)

        return merged

    @staticmethod
    def _dup(candidate: Entity, existing: list[Entity]) -> bool:
        for e in existing:
            if e.text == candidate.text and candidate.span[0] < e.span[1] and candidate.span[1] > e.span[0]:
                return True
        return False

    # ---- Post-processing ----

    @staticmethod
    def _merge_adjacent(entities: list[Entity]) -> list[Entity]:
        """Merge adjacent same-category entities into compound entities.

        '北京'(LOC) + '立方庭'(LOC) → '北京立方庭'(LOC)
        '碳'(MATERIAL) + '钢'(MATERIAL) → '碳钢'(MATERIAL)
        """
        if len(entities) < 2:
            return entities
        merged = []
        i = 0
        while i < len(entities):
            cur = entities[i]
            j = i + 1
            while j < len(entities):
                nxt = entities[j]
                if (cur.category == nxt.category
                        and cur.span[1] == nxt.span[0]):
                    cur = Entity(
                        id=cur.id,
                        text=cur.text + nxt.text,
                        category=cur.category,
                        span=(cur.span[0], nxt.span[1]),
                        normalized=cur.normalized + nxt.normalized,
                        source=cur.source,
                        confidence=min(cur.confidence, nxt.confidence),
                    )
                    j += 1
                else:
                    break
            merged.append(cur)
            i = j
        return merged

    def _discover_keyword_entities(
        self, entities: list[Entity], tokens: list[Token], text: str
    ) -> list[Entity]:
        """Scan tokens for domain keywords not caught by NER.

        Multi-char tokens matching _MATERIAL/_STANDARD/_PARAMETER → entities.
        Single-char tokens only matched against _MATERIAL (e.g., 钢, 铁, 铜).
        """
        result = list(entities)
        entity_spans = {(e.span[0], e.span[1]) for e in entities}

        for t in tokens:
            span_key = (t.span[0], t.span[1])
            if span_key in entity_spans:
                continue
            # Single-char: only match materials (钢, 铁, 铜, 铝...)
            if len(t.text) == 1 and t.text in _MATERIAL:
                cat = "MATERIAL"
            elif len(t.text) >= 2:
                cat = self._keyword(t.text)
            else:
                continue
            if cat and cat != "UNKNOWN":
                self._counter += 1
                result.append(Entity(
                    id=f"ent_{self._counter:03d}",
                    text=t.text,
                    category=cat,
                    span=t.span,
                    normalized=t.text,
                    source="keyword",
                    confidence=1.0,
                ))
                entity_spans.add(span_key)

        result.sort(key=lambda e: e.span[0])
        return result
