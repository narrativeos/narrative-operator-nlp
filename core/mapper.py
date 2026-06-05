"""
HanlpSchemaMapper — HanLP Raw Output → NSP Standard Format.

Maps HanLP MTL model output to NSP NarrativeDocument.
Handles the real HanLP output structures:

    tok/fine   : list[str]                          — individual characters
    pos/ctb    : list[str]                          — POS tags aligned with tok
    ner/*      : list[tuple[str,str,int,int]]       — (text, label, tok_start, tok_end)
    dep        : list[tuple[int,str]]               — (head_1based, deprel)
    srl        : list[list[tuple[str,str,int,int]]]  — frames
"""

from __future__ import annotations

from datetime import datetime, timezone

from .entity_mapper import EntityMappingRules
from .relation_mapper import RelationExtractionRules
from .schema import (
    NarrativeContent,
    NarrativeDocument,
    NarrativeMeta,
    Token,
)


class HanlpSchemaMapper:
    """Maps HanLP raw MTL output to NSP NarrativeDocument."""

    def __init__(self) -> None:
        self.entity_rules = EntityMappingRules()
        self.relation_rules = RelationExtractionRules()

    def map(self, text: str, raw: dict, source: str = "hanlp_v2") -> NarrativeDocument:
        tokens = self._map_tokens(text, raw)
        entities = self._map_entities(text, raw, tokens)
        self._assign_attributes(text, raw, tokens, entities)
        # PARAMETERs are properties, not standalone entities
        entities = [e for e in entities if e.category != "PARAMETER"]
        relations = self._map_relations(text, raw, tokens, entities)
        patterns = self._build_patterns(text, raw, entities, relations)
        return NarrativeDocument(
            meta=NarrativeMeta(
                source=source,
                version="1.0",
                timestamp=datetime.now(timezone.utc).isoformat(),
                text_length=len(text),
            ),
            content=NarrativeContent(
                tokens=tokens,
                entities=entities,
                relations=relations,
                patterns=patterns,
                structural=raw,
            ),
        )

    def _map_tokens(self, text: str, raw: dict) -> list[Token]:
        tok_fine = raw.get("tok/fine", [])
        pos = raw.get("pos/ctb", [])
        tok_conf = raw.get("tok/fine_conf") or raw.get("tok/coarse_conf")
        # Flatten if nested (batch-level list)
        if tok_conf and isinstance(tok_conf[0], list):
            tok_conf = tok_conf[0]
        tokens: list[Token] = []
        cursor = 0
        for i in range(len(tok_fine)):
            token_text = tok_fine[i]
            token_pos = pos[i] if i < len(pos) else "X"
            idx = text.find(token_text, cursor)
            if idx >= 0:
                start, end = idx, idx + len(token_text)
                cursor = end
            else:
                start, end = cursor, cursor + len(token_text)
                cursor = end
            conf = float(tok_conf[i]) if tok_conf and i < len(tok_conf) else 1.0
            tokens.append(Token(id=i, text=token_text, pos=token_pos, span=(start, end), confidence=conf))
        return tokens

    def _map_entities(self, text: str, raw: dict, tokens: list[Token]) -> list:
        return self.entity_rules.map_all(text, raw, tokens)

    def _map_relations(self, text: str, raw: dict, tokens: list[Token],
                       entities: list) -> list:
        return self.relation_rules.extract_all(text, raw, tokens, entities)

    def _assign_attributes(self, text: str, raw: dict, tokens: list,
                           entities: list) -> None:
        """Assign entity attributes from SRL property frames.

        When SRL gives ARG0=entity, ARG1=compound of PARAMETER entities
        (e.g., '高强度和高韧性'), split into individual attributes
        and attach to the ARG0 entity.
        """
        from .schema import EntityAttribute

        # Build entity lookup by text
        entity_by_text: dict[str, object] = {e.text: e for e in entities}
        param_entities = {e.text for e in entities if e.category == "PARAMETER"}

        for f in raw.get("srl", []):
            if not isinstance(f, list):
                continue
            a0_text, a1_text = "", ""
            pred_text = ""
            for item in f:
                if len(item) < 4:
                    continue
                role = str(item[1]).upper()
                if "ARG0" in role:
                    a0_text = str(item[0])
                elif "ARG1" in role:
                    a1_text = str(item[0])
                elif role == "PRED":
                    pred_text = str(item[0])

            if not a0_text or not a1_text or not pred_text:
                continue

            # ARG0 must be an entity
            a0_entity = entity_by_text.get(a0_text)
            if a0_entity is None:
                continue

            # Find all PARAMETER entities that are substrings of ARG1
            matched = [p for p in param_entities if p in a1_text]
            if not matched:
                continue

            # Assign each parameter as an attribute
            for param_text in matched:
                param_entity = entity_by_text.get(param_text)
                if param_entity is None:
                    continue

                # Split "高强度" → key="强度", value="高"
                # If param_text IS a known parameter keyword → no value prefix
                from .entity_mapper import _PARAMETER as KNOWN_PARAMS
                if param_text in KNOWN_PARAMS:
                    key, value = param_text, ""
                else:
                    # Try suffix match: "高强度" ends with "强度" → key="强度", value="高"
                    matched_kw = None
                    for kw in sorted(KNOWN_PARAMS, key=len, reverse=True):
                        if param_text.endswith(kw) and len(kw) >= 2:
                            matched_kw = kw
                            break
                    if matched_kw:
                        key = matched_kw
                        value = param_text[:-len(matched_kw)]
                    else:
                        key = param_text
                        value = ""

                attr = EntityAttribute(
                    key=key,
                    value=value,
                    predicate_verb=pred_text,
                    confidence=0.85,
                )
                a0_entity.attributes.append(attr)

    def _build_patterns(self, text: str, raw: dict, entities: list,
                        relations: list) -> list:
        """Build sentence-level structural patterns for statistical aggregation."""
        from .schema import SentencePattern

        patterns: list = []
        for f in raw.get("srl", []):
            if not isinstance(f, list):
                continue

            preds: list[str] = []
            entity_cats: list[str] = []
            for item in f:
                if len(item) < 4:
                    continue
                role = str(item[1]).upper()
                itxt = str(item[0])
                if role == "PRED":
                    preds.append(itxt)
                if role in ("ARG0", "ARG1"):
                    # ARG text already tells us the entity span — just match by text
                    matched_cat = None
                    for e in entities:
                        if e.text == itxt or e.text in itxt:
                            matched_cat = e.category
                            break
                    entity_cats.append(matched_cat or "?")

            if not preds:
                continue

            template = " ".join(
                f"{entity_cats[i // 2] if i % 2 == 0 else preds[i // 2]}"
                if i < len(entity_cats) * 2 - 1 else ""
                for i in range(len(entity_cats) * 2 - 1)
            )
            # Simpler: interleave
            parts: list[str] = []
            for i in range(max(len(entity_cats), len(preds))):
                if i < len(entity_cats):
                    parts.append(entity_cats[i])
                if i < len(preds):
                    parts.append(preds[i])
            template = " ".join(parts)

            # Relation summary
            rel_summaries: list[str] = []
            for r in relations:
                if r.predicate_verb in preds:
                    rel_summaries.append(f"{r.subject}→{r.predicate_verb}→{r.object}")

            attr_count = sum(len(e.attributes) for e in entities)

            # ── Auto-detect syntactic features ──
            sentence_type = _detect_sentence_type(text)
            polarity = _detect_polarity(text)
            voice = _detect_voice(text)
            sub_types = _detect_sub_types(text)
            word_count = len(raw.get("tok/fine", []))
            clause_count = _count_clauses(text)
            punct = _sentence_punct(text)

            patterns.append(SentencePattern(
                sentence=text,
                sentence_type=sentence_type,
                polarity=polarity,
                voice=voice,
                sub_types=sub_types,
                template=template,
                entity_sequence=entity_cats,
                predicates=preds,
                relation_summary=rel_summaries,
                attribute_count=attr_count,
                word_count=word_count,
                clause_count=clause_count,
                punctuation_mark=punct,
            ))

        return patterns


# ── Syntactic feature detectors ──

def _detect_sentence_type(text: str) -> str:
    last = text.strip()[-1] if text.strip() else ""
    if last == "？":
        return "interrogative"
    if last == "！":
        return "exclamatory"
    return "declarative"


def _detect_polarity(text: str) -> str:
    for neg in ("不", "没", "无", "非", "未", "别", "莫", "勿"):
        if neg in text:
            return "negative"
    return "affirmative"


def _detect_voice(text: str) -> str:
    if "被" in text:
        return "passive"
    return "active"


def _detect_sub_types(text: str) -> list[str]:
    types: list[str] = []
    if "把" in text:
        types.append("ba_construction")
    if "被" in text:
        types.append("bei_construction")
    return types


def _count_clauses(text: str) -> int:
    return max(1, text.count("，") + text.count("；") + 1)


def _sentence_punct(text: str) -> str:
    last = text.strip()[-1] if text.strip() else ""
    return last if last in "。？！" else ""
