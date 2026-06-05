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

    def map_all(self, text: str, raw: dict, tokens: list[Token]) -> list[Entity]:
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
        return entities

    @staticmethod
    def _keyword(text: str) -> Optional[str]:
        if text in _MATERIAL: return "MATERIAL"
        if text in _STANDARD: return "STANDARD"
        if text in _PARAMETER: return "PARAMETER"
        return "UNKNOWN"

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
