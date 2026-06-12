"""
Classical Chinese Pattern Extractor — Rule-based relation extraction for 文言文.

古汉语的句式高度结构化，此模块用正则匹配提取核心关系，
弥补 HanLP KYOTO-EVAHAN 模型在 NER/SRL 方面的不足。

支持的句式模式：
1. 判断句: "X者，Y人也" → X LOCATED_AT/IS_A Y
2. 表字: "X，字Y" / "X字Y" → X HAS_STYLE_NAME Y
3. 被动句: "为X所V" / "见V于X" → X CAUSES patient
4. 官职任命: "拜/封/授/除/迁为X" → person HAS_TITLE X
5. 地理位置: "X军/驻/屯Y" → X LOCATED_AT Y
6. 对话引述: "X曰/云/谓Y" → X SAYS Y
7. 名/字同一指代: 基于表字关系建立 EQUIVALENT_TO
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .schema import Entity, Relation, CoreferenceChain, Mention

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern Definitions
# ---------------------------------------------------------------------------

class _Pattern:
    """A single regex-based pattern for classical Chinese relation extraction."""
    __slots__ = ('name', 'regex', 'predicate', 'confidence', 'source')

    def __init__(self, name: str, regex: str, predicate: str,
                 confidence: float = 0.85, source: str = "classical_pattern"):
        self.name = name
        self.regex = re.compile(regex)
        self.predicate = predicate
        self.confidence = confidence
        self.source = source


# All patterns (order matters: more specific first)
_PATTERNS: list[_Pattern] = [

    # 1. 判断句: "X者，Y人也" / "X者Y人也" / "X，Y也"
    _Pattern("judgment_者也", r"(.+?)者[，,]?(.+?)人也", "LOCATED_AT", 0.90),
    _Pattern("judgment_也", r"([^。；,]+)也$", "IS_A", 0.80),

    # 1b. 命名: "其名为X" / "名曰X"
    # "其" is a pronoun - capture the pronoun marker to resolve later via _find_preceding_person
    _Pattern("name_名为", r"[，,。；]其名为([\u4e00-\u9fff]{1,3})[。；,]?", "HAS_PROPERTY", 0.88),
    _Pattern("name_名曰", r"[，,。；]名曰([\u4e00-\u9fff]{1,3})[。；,]?", "HAS_PROPERTY", 0.88),
    _Pattern("name_名", r"([\u4e00-\u9fff]{1,4})名([\u4e00-\u9fff]{2,4})[。；,]?", "HAS_PROPERTY", 0.82),

    # 1c. 存在: "X有Y" (X中有Y) - only concrete entities
    # Skip abstract nouns: 色/功/業/術/勢/理/義/道/德/力/志/氣/情/意/心/念/願/望
    _Pattern("existence_有", r"([\u4e00-\u9fff]{2,4})有([\u4e00-\u9fff]{1,3})[。；,]?", "HAS_PROPERTY", 0.75),

    # 1d. 变化: "化而为X" / "化为X" / "变为X"
    # No subject in regex - resolve via _find_preceding_person
    _Pattern("transform_化为", r"[，,。；]化为([\u4e00-\u9fff]{1,3})[。；,]?", "TRANSFERS_TO", 0.88),
    _Pattern("transform_化而为", r"[，,。；]化而为([\u4e00-\u9fff]{1,3})[。；,]?", "TRANSFERS_TO", 0.90),
    _Pattern("transform_变为", r"[，,。；]变为([\u4e00-\u9fff]{1,3})[。；,]?", "TRANSFERS_TO", 0.88),

    # 2. 表字: "X，字Y" / "X字Y" / "X，字为Y"
    # Pattern A: person name directly before "字" (2-3 chars)
    _Pattern("style_name_字", r"[，,]?([\u4e00-\u9fff]{2,3})字(为)?([\u4e00-\u9fff]{1,2})[\u3002。]", "HAS_STYLE_NAME", 0.92),
    _Pattern("style_name_字_end", r"[，,]?([\u4e00-\u9fff]{2,3})字(为)?([\u4e00-\u9fff]{1,2})$", "HAS_STYLE_NAME", 0.92),
    # Pattern B: "也字涉" - style name directly after "也" (no comma), surname+style as one token
    _Pattern("style_name_也字", r"也([\u4e00-\u9fff]{1,2})字(为)?([\u4e00-\u9fff]{1,2})[\u3002。]", "HAS_STYLE_NAME", 0.90),
    _Pattern("style_name_也字_end", r"也([\u4e00-\u9fff]{1,2})字(为)?([\u4e00-\u9fff]{1,2})$", "HAS_STYLE_NAME", 0.90),
    # Pattern C: "人也，字X" - judgment sentence followed by style name
    # NOTE: This pattern captures "X人也，字Y" where X is the LOCATION in the text
    # (e.g., "阳城人也，字涉"). The subject resolution is handled in _match_pattern
    # by looking up the preceding PERSON entity from the judgment pattern.
    # Mark with _standalone suffix so the handler knows to use _find_preceding_person
    _Pattern("style_name_人也字_standalone", r"人也[，,]字(为)?([\u4e00-\u9fff]{1,2})[\u3002。]", "HAS_STYLE_NAME", 0.93),
    _Pattern("style_name_人也字_standalone_end", r"人也[，,]字(为)?([\u4e00-\u9fff]{1,2})$", "HAS_STYLE_NAME", 0.93),
    # Pattern D: "字涉" at sentence boundary - standalone "字X" after comma
    _Pattern("style_name_字_standalone", r"[，,]字(为)?([\u4e00-\u9fff]{1,2})[\u3002。]", "HAS_STYLE_NAME", 0.88),
    _Pattern("style_name_字_standalone_end", r"[，,]字(为)?([\u4e00-\u9fff]{1,2})$", "HAS_STYLE_NAME", 0.88),

    # 3. 表号: "号X" / "，号X"
    _Pattern("style_hao", r"[，,]?([\u4e00-\u9fff]{2,3})号([\u4e00-\u9fff]{1,2})[\u3002。]", "HAS_STYLE_NAME", 0.90),
    _Pattern("style_hao_end", r"[，,]?([\u4e00-\u9fff]{2,3})号([\u4e00-\u9fff]{1,2})$", "HAS_STYLE_NAME", 0.90),

    # 4. 被动句: "为X所V"
    _Pattern("passive_为所", r"(.+?)为(.+?)所(.+)", "CAUSES", 0.80),

    # 5. 被动句: "见V于X"
    _Pattern("passive_见于", r"(.+?)见(.+?)于(.+)", "CAUSES", 0.80),

    # 6. 官职任命: "拜/封/授/除/迁/擢/进/降为X"
    # Use Chinese char bounds + optional trailing punctuation (works at sentence end)
    _Pattern("title_拜为", r"([\u4e00-\u9fff]{2,4})拜为([\u4e00-\u9fff]{2,5})[。.;;，,]?", "HAS_TITLE", 0.88),
    _Pattern("title_封为", r"([\u4e00-\u9fff]{2,4})封为([\u4e00-\u9fff]{2,5})[。.;;，,]?", "HAS_TITLE", 0.88),
    _Pattern("title_授为", r"([\u4e00-\u9fff]{2,4})授为([\u4e00-\u9fff]{2,5})[。.;;，,]?", "HAS_TITLE", 0.88),
    _Pattern("title_除为", r"([\u4e00-\u9fff]{2,4})除为([\u4e00-\u9fff]{2,5})[。.;;，,]?", "HAS_TITLE", 0.88),
    _Pattern("title_迁为", r"([\u4e00-\u9fff]{2,4})迁为([\u4e00-\u9fff]{2,5})[。.;;，,]?", "HAS_TITLE", 0.88),
    _Pattern("title_擢为", r"([\u4e00-\u9fff]{2,4})擢为([\u4e00-\u9fff]{2,5})[。.;;，,]?", "HAS_TITLE", 0.88),
    _Pattern("title_进为", r"([\u4e00-\u9fff]{2,4})进为([\u4e00-\u9fff]{2,5})[。.;;，,]?", "HAS_TITLE", 0.88),
    _Pattern("title_降为", r"([\u4e00-\u9fff]{2,4})降为([\u4e00-\u9fff]{2,5})[。.;;，,]?", "HAS_TITLE", 0.88),

    # 7. 官职 (无"为"): "拜X" / "封X"
    _Pattern("title_拜", r"([\u4e00-\u9fff]{2,4})拜([\u4e00-\u9fff]{2,5})[。.;;，,]?", "HAS_TITLE", 0.75),
    _Pattern("title_封", r"([\u4e00-\u9fff]{2,4})封([\u4e00-\u9fff]{2,5})[。.;;，,]?", "HAS_TITLE", 0.75),

    # 8. 地理位置: "X军/驻/屯/据/守/攻/取Y"
    _Pattern("location_军", r"(.+?)军(.+?)[。.;;，,]", "LOCATED_AT", 0.82),
    _Pattern("location_驻", r"(.+?)驻(.+?)[。.;;，,]", "LOCATED_AT", 0.85),
    _Pattern("location_屯", r"(.+?)屯(.+?)[。.;;，,]", "LOCATED_AT", 0.85),
    _Pattern("location_据", r"(.+?)据(.+?)[。.;;，,]", "LOCATED_AT", 0.82),
    _Pattern("location_守", r"(.+?)守(.+?)[。.;;，,]", "LOCATED_AT", 0.80),
    _Pattern("location_攻", r"(.+?)攻(.+?)[。.;;，,]", "INTERACTS_WITH", 0.75),
    _Pattern("location_取", r"(.+?)取(.+?)[。.;;，,]", "TRANSFERS_TO", 0.75),

    # 9. 对话引述: "X曰/云/谓/对曰Y"
    # Subject must be a clean name (2-4 Chinese chars), no punctuation or verbs
    _Pattern("dialogue_曰", r"([\u4e00-\u9fff]{2,4})曰[：:\"]?(.+?)(?:[\"。.;;!？]+|$)", "SAYS", 0.85),
    _Pattern("dialogue_云", r"([\u4e00-\u9fff]{2,4})云[：:\"]?(.+?)(?:[\"。.;;!？]+|$)", "SAYS", 0.82),
    _Pattern("dialogue_对曰", r"([\u4e00-\u9fff]{2,4})对曰[：:\"]?(.+?)(?:[\"。.;;!？]+|$)", "SAYS", 0.85),

    # 10. 移动: "X至/往/赴/归/还/入/出/过Y"
    # Exclude "以至于" (idiom meaning "eventually to", not physical movement)
    _Pattern("movement_至", r"(.+?)至(.+?)[。.;;，,]", "MOVED_TO", 0.85),
    _Pattern("movement_往", r"(.+?)往(.+?)[。.;;，,]", "MOVED_TO", 0.85),
    _Pattern("movement_赴", r"(.+?)赴(.+?)[。.;;，,]", "MOVED_TO", 0.85),
    _Pattern("movement_归", r"(.+?)归(.+?)[。.;;，,]", "MOVED_TO", 0.82),
    _Pattern("movement_还", r"(.+?)还(.+?)[。.;;，,]", "MOVED_TO", 0.82),
    _Pattern("movement_入", r"(.+?)入(.+?)[。.;;，,]", "MOVED_TO", 0.80),
    _Pattern("movement_出", r"(.+?)出(.+?)[。.;;，,]", "DEPARTED_FROM", 0.80),
    _Pattern("movement_过", r"(.+?)过(.+?)[。.;;，,]", "MOVED_TO", 0.75),

    # 11. 死亡: "X卒/薨/崩/死/殁于Y"
    # 崩/薨/卒/死/殁 without 于 = just "died" (no location), only match with 于
    _Pattern("death_卒于", r"(.+?)卒于(.+?)[。.;;，,]", "DIED_AT", 0.85),
    _Pattern("death_薨于", r"(.+?)薨于(.+?)[。.;;，,]", "DIED_AT", 0.85),
    _Pattern("death_崩于", r"(.+?)崩于(.+?)[。.;;，,]", "DIED_AT", 0.85),
    _Pattern("death_死于", r"(.+?)死于(.+?)[。.;;，,]", "DIED_AT", 0.82),
    _Pattern("death_殁于", r"(.+?)殁于(.+?)[。.;;，,]", "DIED_AT", 0.82),

    # 12. 出生/籍贯: "X，Y人也" (only the "也" ending, not the noisy comma variant)
    # Note: "X者，Y人也" is already handled by judgment_者也 above
    # This pattern catches "X，Y人也" without "者"
    _Pattern("origin_人也", r"([\u4e00-\u9fff]{2,4})，([\u4e00-\u9fff]{2,4})人也", "LOCATED_AT", 0.78),
]


# Words that should NEVER be relation endpoints (function words, common nouns)
_NON_ENDPOINTS = frozenset({
    # 虚词
    "者", "也", "矣", "焉", "乎", "哉", "耶", "欤", "兮", "哉",
    "所", "以", "于", "以", "而", "则", "虽", "然", "之", "其",
    # 常见非实体
    "人", "时", "少", "多", "大", "小", "上", "下",
    "父", "母", "子", "女", "兄", "弟", "妻", "夫",
    "君", "臣", "官", "民", "兵", "将", "军",
    # 单字虚词
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "百", "千", "万", "里", "丈", "尺", "寸",
})


def _is_valid_endpoint(text: str) -> bool:
    """Check if text is a valid relation endpoint."""
    text = text.strip()
    if not text:
        return False
    if text in _NON_ENDPOINTS:
        return False
    # Single character that's a common function word
    if len(text) == 1 and text in _NON_ENDPOINTS:
        return False
    return True


def _find_entity_by_text(entities: list[Entity], text: str) -> Optional[Entity]:
    """Find an entity whose text matches (exact or contains)."""
    # Exact match first
    for e in entities:
        if e.text == text:
            return e
    # Contains match
    best = None
    best_len = 0
    for e in entities:
        if text in e.text and len(e.text) > best_len:
            best = e
            best_len = len(e.text)
        elif e.text in text and len(e.text) > best_len:
            best = e
            best_len = len(e.text)
    return best


class ClassicalPatternExtractor:
    """Rule-based pattern extractor for classical Chinese texts.

    Uses regex patterns to extract relations that the HanLP model misses.
    Designed to work as a post-processing step after the main NLP pipeline.
    """

    def __init__(self):
        self._counter = 0

    def reset(self):
        self._counter = 0

    def extract(
        self,
        text: str,
        entities: list[Entity],
    ) -> tuple[list[Relation], list[CoreferenceChain]]:
        """Extract relations and coreferences from classical Chinese text.

        Args:
            text: The original classical Chinese text.
            entities: Entities already extracted by the NLP pipeline.

        Returns:
            (relations, coreference_chains) tuple.
        """
        self.reset()
        relations: list[Relation] = []
        coreferences: list[CoreferenceChain] = []
        entity_texts = {e.text for e in entities}

        # Track style names for coreference resolution
        style_name_map: dict[str, str] = {}  # style_name -> person_name

        # For standalone style patterns (no subject in regex), find the preceding
        # person entity from the judgment pattern context
        def _find_preceding_person(match_start: int) -> Optional[str]:
            """Find the last person name before the match position.

            Looks for:
            1. Subjects of LOCATED_AT from judgment pattern (HIGHEST priority - most reliable)
            2. PERSON entities (from NER or seed dictionary) - excluding known style names
            3. Any entity before the match position (fallback)
            """
            best = None
            best_end = -1
            # Build set of known style names to exclude (they are not real person names)
            known_style_names = set(style_name_map.keys())
            # First try: subjects of LOCATED_AT from judgment pattern
            # This is the MOST reliable source: "X者，Y人也" -> X is the person
            for r in relations:
                if r.predicate == "LOCATED_AT" and "judgment" in r.source:
                    idx = text.find(r.subject)
                    if idx >= 0 and idx + len(r.subject) <= match_start:
                        end = idx + len(r.subject)
                        if end > best_end:
                            best_end = end
                            best = r.subject
            # Second try: PERSON entities (excluding known style names)
            if best is None:
                for e in entities:
                    if e.category == "PERSON" and e.span[1] <= match_start:
                        # Skip known style names (e.g., "涉" is a style name, not a person)
                        if e.text in known_style_names:
                            continue
                        if e.span[1] > best_end:
                            best_end = e.span[1]
                            best = e.text
            # Third try: any entity before the match position (excluding style names)
            if best is None:
                for e in entities:
                    if e.span[1] <= match_start and e.span[0] >= 0:
                        if e.text in known_style_names:
                            continue
                        if len(e.text) >= 2 and e.span[1] > best_end:
                            best_end = e.span[1]
                            best = e.text
            return best

        for pattern in _PATTERNS:
            for m in pattern.regex.finditer(text):
                groups = m.groups()
                # Handle pronoun/standalone patterns (no subject in regex, resolve via _find_preceding_person)
                if pattern.name.startswith("name_名为") or pattern.name.startswith("name_名曰") \
                   or pattern.name.startswith("transform_"):
                    # Only 1 group: object
                    if len(groups) >= 1:
                        obj = (groups[0] or "").strip()
                        subject = _find_preceding_person(m.start()) or ""
                    else:
                        continue
                elif pattern.name.startswith("style_name") or pattern.name.startswith("style_hao"):
                    # Handle standalone patterns (subject not in regex)
                    if pattern.name.endswith("_standalone") or pattern.name.endswith("_standalone_end"):
                        # Groups: optional "为", style_name
                        if len(groups) >= 2:
                            _ = (groups[0] or "").strip()  # "为" or empty
                            obj = (groups[1] or "").strip()
                            subject = _find_preceding_person(m.start()) or ""
                        else:
                            continue
                    elif pattern.name.endswith("_也字") or pattern.name.endswith("_也字_end"):
                        # Groups: surname_prefix, optional "为", style_name
                        if len(groups) >= 3:
                            subject = (groups[0] or "").strip()
                            _ = (groups[1] or "").strip()
                            obj = (groups[2] or "").strip()
                        elif len(groups) >= 2:
                            subject = (groups[0] or "").strip()
                            obj = (groups[1] or "").strip()
                        else:
                            continue
                    elif len(groups) >= 3:
                        subject = (groups[0] or "").strip()
                        _ = (groups[1] or "").strip()  # "为" or empty
                        obj = (groups[2] or "").strip()
                    elif len(groups) >= 2:
                        subject = (groups[0] or "").strip()
                        obj = (groups[1] or "").strip()
                    else:
                        continue
                else:
                    # 2 groups: subject, object
                    if len(groups) < 2:
                        continue
                    subject = (groups[0] or "").strip()
                    obj = (groups[1] or "").strip()

                # Clean up punctuation
                subject = subject.rstrip("，,。.;:!？")
                obj = obj.strip("，,。.;:!？\"")

                if not _is_valid_endpoint(subject) or not _is_valid_endpoint(obj):
                    continue

                # Filter: existence_有 should not match abstract nouns
                if pattern.name.startswith("existence_"):
                    ABSTRACT_NOUNS = frozenset({
                        "色", "功", "業", "術", "勢", "理", "義", "道", "德",
                        "力", "志", "氣", "情", "意", "心", "念", "願", "望",
                        "驕", "謙", "畏", "莊", "禮", "賢", "能", "才", "智",
                        "勇", "仁", "信", "廉", "恥", "恩", "怨", "禍", "福",
                    })
                    # Check if obj is or contains an abstract noun
                    if obj in ABSTRACT_NOUNS:
                        continue
                    if any(a in obj for a in ABSTRACT_NOUNS):
                        continue

                # Filter: movement_至 should not match "以至于" (idiom)
                if pattern.name.startswith("movement_至"):
                    if "以至" in text[m.start():m.end()]:
                        continue

                # Try to resolve to known entities
                subj_entity = _find_entity_by_text(entities, subject)
                obj_entity = _find_entity_by_text(entities, obj)

                # Skip if both endpoints are not entities and not in entity_texts
                # (at least one should be resolvable)
                # EXCEPTION: HAS_STYLE_NAME from standalone patterns - the subject
                # is resolved from the judgment pattern context (not an entity) and
                # the style name is a single character that won't be an entity
                if pattern.predicate != "HAS_STYLE_NAME":
                    subj_resolved = subj_entity is not None or subject in entity_texts
                    obj_resolved = obj_entity is not None or obj in entity_texts
                    if not subj_resolved and not obj_resolved:
                        continue

                # Use entity text if available for cleaner output
                # BUT for style_name patterns, always use raw text to avoid
                # resolving "字涉" to the person entity (which causes self-loops)
                if pattern.predicate == "HAS_STYLE_NAME":
                    subj_final = subject  # Keep raw, don't resolve to entity
                    obj_final = obj       # Keep raw style name
                else:
                    subj_final = subj_entity.text if subj_entity else subject
                    obj_final = obj_entity.text if obj_entity else obj

                # Skip self-loops
                if subj_final == obj_final:
                    continue

                # Deduplicate: skip if same subject+predicate already exists
                key = (subj_final, pattern.predicate, obj_final)
                if key in {(r.subject, r.predicate, r.object) for r in relations}:
                    continue

                self._counter += 1
                rel = Relation(
                    id=f"rel_{self._counter:03d}",
                    subject=subj_final,
                    subject_raw=subject,
                    predicate=pattern.predicate,
                    predicate_verb=pattern.name.split("_")[0] if "_" in pattern.name else pattern.name,
                    object=obj_final,
                    object_raw=obj,
                    object_ent_id=obj_entity.id if obj_entity else None,
                    subject_ent_id=subj_entity.id if subj_entity else None,
                    evidence=text[m.start():m.end()],
                    evidence_span=(m.start(), m.end()),
                    confidence=pattern.confidence,
                    source=f"classical_pattern/{pattern.name}",
                )
                relations.append(rel)

                # Track style names for coreference
                if pattern.predicate == "HAS_STYLE_NAME":
                    style_name_map[obj_final] = subj_final

        # Build coreference chains for name/style_name equivalence
        for style_name, person_name in style_name_map.items():
            # Find the person entity
            person_ent = _find_entity_by_text(entities, person_name)
            # Check if there's an entity that combines surname + style_name (e.g., 陈涉)
            # The style_name alone (涉) might not be an entity, but 陈涉 would be
            for e in entities:
                if e.category == "PERSON" and style_name in e.text and e.text != person_name:
                    # e.text is like "陈涉", person_name is "陈胜"
                    # Create EQUIVALENT_TO relation
                    self._counter += 1
                    relations.append(Relation(
                        id=f"rel_{self._counter:03d}",
                        subject=person_name,
                        subject_raw=person_name,
                        predicate="EQUIVALENT_TO",
                        predicate_verb="coref_name_style",
                        object=e.text,
                        object_raw=e.text,
                        subject_ent_id=person_ent.id if person_ent else None,
                        object_ent_id=e.id,
                        evidence=f"{person_name}...{e.text}",
                        evidence_span=(
                            text.find(person_name) if text.find(person_name) >= 0 else 0,
                            text.find(e.text) + len(e.text) if text.find(e.text) >= 0 else len(text),
                        ),
                        confidence=0.90,
                        source="classical_pattern/name_style_coref",
                    ))
                    # Create coreference chain
                    coreferences.append(CoreferenceChain(
                        chain_id=f"coref_{len(coreferences)+1:03d}",
                        mentions=[
                            Mention(text=person_name, span=person_ent.span if person_ent else (0, len(person_name)),
                                    mention_type="entity", entity_id=person_ent.id if person_ent else None,
                                    is_principal=True),
                            Mention(text=e.text, span=e.span, mention_type="entity", entity_id=e.id,
                                    is_principal=False),
                        ],
                        representative=person_name,
                        confidence=0.90,
                        language="classical",
                        quality_flag="high",
                        model_version="classical_pattern_v1",
                    ))
                    break  # Only one equivalence per person

        return relations, coreferences


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def extract_classical_patterns(
    text: str,
    entities: list[Entity],
) -> tuple[list[Relation], list[CoreferenceChain]]:
    """Extract classical Chinese patterns. Convenience wrapper."""
    extractor = ClassicalPatternExtractor()
    return extractor.extract(text, entities)