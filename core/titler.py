"""
Title Generator - Entity + Category Description + TextRank Truncation.

Strategy: Now (human-readable titles)
- generate_title(doc): Uses entity category + meaningful relations to compose readable titles.
- generate_title_text(text): TextRank top-1 sentence truncated to target length.

That formats:
- "{entity}——{description}" e.g., "阿里巴巴——中国电商与云计算巨头"
- "{entity}：{description}" e.g., "马云：阿里巴巴创始人"
- TextRank sentence truncated: "阿里巴巴集团成立于1999年，由马云..."

Modes: "chars" (default 20) or "ratio" (default 0.05).
"""
from __future__ import annotations
import re
from typing import Optional

from .schema import (
    Entity, Event, NarrativeContent, NarrativeDocument, Relation,
    SentenceLanguage, Title,
)

_DEFAULT_CHARS = 20
_DEFAULT_RATIO = 0.05
_EMPTY_RE = re.compile(r"^[^\w\s]*$")

# ---------------------------------------------------------------------------
# Predicate -> Chinese verb mapping for readable titles
# ---------------------------------------------------------------------------
_PREDICATE_CN: dict[str, str] = {
    "IS_A": "是",
    "PART_OF": "属于",
    "HAS_PROPERTY": "具有",
    "PROPERTY_OF": "属性",
    "LOCATED_AT": "位于",
    "TEMPORAL_AT": "时间",
    "CAUSES": "导致",
    "AFFECTS": "影响",
    "PRODUCES": "推出",
    "COMPOSED_OF": "由...组成",
    "TRANSFERS_TO": "转移至",
    "DEPENDS_ON": "依赖",
    "MOVED_TO": "迁至",
    "DEPARTED_FROM": "离开",
    "INTERACTS_WITH": "与...互动",
    "CONTROLS": "控制",
    "BENEFITS": "惠及",
    "EQUIVALENT_TO": "等同于",
    "REFERENCE_OF": "参考",
    "CONSTRAINT_OF": "约束",
    "HAS_STYLE_NAME": "字",
    "HAS_TITLE": "任",
    "SAYS": "曰",
    "DIED_AT": "卒于",
    "EMPLOYS": "雇佣",
    "WORKS_FOR": "任职于",
    "FOUNDED_BY": "由...创立",
    "RELATES_TO": "相关",
}

# ---------------------------------------------------------------------------
# Entity category -> Chinese noun mapping
# ---------------------------------------------------------------------------
_CATEGORY_CN: dict[str, str] = {
    "PERSON": "人物",
    "ORGANIZATION": "企业",
    "LOCATION": "地点",
    "FACILITY": "设施",
    "PRODUCT": "产品",
    "DATE": "日期",
    "NUMBER": "数字",
    "MATERIAL": "材料",
    "STANDARD": "标准",
    "PARAMETER": "参数",
    "UNKNOWN": "",
    "TITLE": "官职",
    "ERA": "朝代",
    "INSTITUTION": "制度",
    "ASTRONOMY": "天文",
}

# Predicates that produce good descriptive objects for titles
_TITLE_PREDICATES = [
    "IS_A", "HAS_PROPERTY", "PRODUCES", "LOCATED_AT",
    "HAS_TITLE", "CONTROLS", "PART_OF", "CAUSES",
]

# Entity category weights for centrality scoring
_CATEGORY_WEIGHT: dict[str, float] = {
    "PERSON": 3.0,
    "ORGANIZATION": 2.5,
    "LOCATION": 2.0,
    "TITLE": 2.0,
    "ERA": 1.5,
    "INSTITUTION": 1.5,
    "PRODUCT": 1.5,
    "FACILITY": 1.0,
}

# Predicate information scores per entity category.
# Higher score = more informative for title generation.
# LOCATED_AT is low-value for PERSON/ORGANIZATION (noisy), high for LOCATION.
_PREDICATE_INFO_SCORE: dict[str, dict[str, float]] = {
    "PERSON": {
        "HAS_TITLE": 3.0,
        "IS_A": 2.5,
        "WORKS_FOR": 2.0,
        "PRODUCES": 2.0,
        "HAS_STYLE_NAME": 1.5,
        "CONTROLS": 1.5,
        "FOUNDED_BY": 1.5,
        "LOCATED_AT": 0.5,
    },
    "ORGANIZATION": {
        "IS_A": 3.0,
        "PRODUCES": 2.5,
        "CONTROLS": 2.0,
        "FOUNDED_BY": 2.0,
        "HAS_PROPERTY": 1.8,
        "LOCATED_AT": 0.8,
    },
    "LOCATION": {
        "LOCATED_AT": 2.5,
        "IS_A": 2.0,
        "PRODUCES": 1.5,
    },
    "PRODUCT": {
        "IS_A": 3.0,
        "PRODUCES": 2.0,
        "HAS_PROPERTY": 1.8,
        "LOCATED_AT": 0.5,
    },
}

# Predicates whose objects serve as primary description (not modifiers)
_PRIMARY_PREDICATES = frozenset({
    "IS_A", "HAS_TITLE", "PRODUCES", "CONTROLS", "HAS_PROPERTY",
    "FOUNDED_BY", "WORKS_FOR", "HAS_STYLE_NAME",
})


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, adding ellipsis if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 1] + "\u2026"


def _compute_textrank(
    sentences: list[SentenceLanguage],
    max_iter: int = 30, damping: float = 0.85, threshold: float = 1e-6,
) -> dict[int, float]:
    n = len(sentences)
    if n == 0:
        return {}
    if n == 1:
        return {0: 1.0}
    token_sets = [set(re.findall(r"\w+", s.text)) for s in sentences]
    sim: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            uni = len(token_sets[i] | token_sets[j])
            if uni > 0:
                v = len(token_sets[i] & token_sets[j]) / uni
                sim[i][j] = v
                sim[j][i] = v
    scores = [1.0 / n] * n
    for _ in range(max_iter):
        new_scores = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i != j and sim[j][i] > 0:
                    out = sum(sim[j])
                    if out > 0:
                        s += sim[j][i] / out * scores[j]
            new_scores[i] = (1 - damping) + damping * s
        diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores
        if diff < threshold:
            break
    mx = max(scores) if scores else 1.0
    if mx > 0:
        scores = [s / mx for s in scores]
    return {i: scores[i] for i in range(n)}


def _compose_title_from_entities(
    content: NarrativeContent,
    target_chars: int,
) -> tuple[str, list[str], list[str], list[str]]:
    """
    Compose a human-readable title from entities and relations.

    Multi-relation fusion strategy:
    1. Find the most central entity (by relation count × category weight).
    2. Score each relation by informativeness (category-aware predicate scoring).
    3. Separate LOCATED_AT as location modifier, others as primary description.
    4. Combine: "{location}{primary_desc}" e.g., "中国科技公司".
    5. Format: "{entity}——{desc}" or "{entity}：{title}".

    This avoids "阿里巴巴——位于中国" by demoting LOCATED_AT for non-LOCATION entities.
    """
    entities = content.entities
    relations = content.relations
    events = content.events

    if not entities:
        return ("", [], [], ["no_entities"])

    # --- Step 1: count relations per entity ---
    entity_rel_count: dict[str, int] = {}
    entity_relations: dict[str, list[Relation]] = {}
    for r in relations:
        for eid in (r.subject_ent_id, r.object_ent_id):
            if eid:
                entity_rel_count[eid] = entity_rel_count.get(eid, 0) + 1
                entity_relations.setdefault(eid, []).append(r)

    for evt in events:
        for arg in evt.arguments:
            if arg.entity_id:
                entity_rel_count[arg.entity_id] = entity_rel_count.get(arg.entity_id, 0) + 1

    if not entity_rel_count:
        sorted_ents = sorted(entities, key=lambda e: e.confidence, reverse=True)
        main_entity = sorted_ents[0]
        title = _truncate(main_entity.text, target_chars)
        return (title, [main_entity.text], [], ["single_entity"])

    # Score entities by relation count weighted by category importance
    def _entity_score(eid: str) -> float:
        ent = next((e for e in entities if e.id == eid), None)
        cat = ent.category if ent else "UNKNOWN"
        weight = _CATEGORY_WEIGHT.get(cat, 1.0)
        return entity_rel_count.get(eid, 0) * weight

    main_eid = max(entity_rel_count, key=_entity_score)
    main_entity = next((e for e in entities if e.id == main_eid), None)
    if not main_entity:
        main_entity = entities[0]
        main_eid = main_entity.id

    rels_for_entity = entity_relations.get(main_eid, [])
    if not rels_for_entity:
        title = _truncate(main_entity.text, target_chars)
        return (title, [main_entity.text], [], ["entity_no_relations"])

    # --- Step 2: score each relation by informativeness ---
    entity_map = {e.id: e for e in entities}
    cat_scores = _PREDICATE_INFO_SCORE.get(main_entity.category, {})
    default_scores = _PREDICATE_INFO_SCORE.get("PRODUCT", {})

    # (score, predicate, object_text, is_forward)
    scored_rels: list[tuple[float, str, str, bool]] = []

    for r in rels_for_entity:
        if r.subject_ent_id == main_eid:
            other_ent = entity_map.get(r.object_ent_id)
            other_text = other_ent.text if other_ent else r.object
            score = cat_scores.get(r.predicate, default_scores.get(r.predicate, 1.0))
            scored_rels.append((score, r.predicate, other_text, True))
        else:
            other_ent = entity_map.get(r.subject_ent_id)
            other_text = other_ent.text if other_ent else r.subject
            score = cat_scores.get(r.predicate, default_scores.get(r.predicate, 1.0)) * 0.7
            scored_rels.append((score, r.predicate, other_text, False))

    # Sort by score descending
    scored_rels.sort(key=lambda x: x[0], reverse=True)

    # --- Step 3: separate location modifier from primary description ---
    location: Optional[str] = None
    primary_parts: list[str] = []
    seen_objects: set[str] = set()
    used_predicates: list[str] = []

    for score, pred, obj_text, is_forward in scored_rels:
        if obj_text in seen_objects or not obj_text:
            continue
        seen_objects.add(obj_text)

        if pred == "LOCATED_AT":
            if location is None:
                location = obj_text
        elif pred in _PRIMARY_PREDICATES:
            primary_parts.append(obj_text)
            used_predicates.append(pred)
            if len(primary_parts) >= 2:
                break

    # --- Step 4: build description ---
    if not primary_parts:
        # No primary description — use category or raw object as fallback
        cat_cn = _CATEGORY_CN.get(main_entity.category, "")
        if main_entity.category == "LOCATION" and location:
            # For LOCATION with only LOCATED_AT, just use the location object directly
            desc = location
        elif location and cat_cn:
            desc = f"{location}{cat_cn}"
        elif location:
            desc = location
        elif cat_cn:
            desc = cat_cn
        else:
            desc = scored_rels[0][2]
    else:
        # Decide whether location modifier makes sense:
        # Good: "中国科技公司" (country + generic description)
        # Bad:  "深圳微信" (city + proper product name)
        # Rule: only add location when primary description looks like a category
        # (contains category keywords) or location is clearly country-level.
        add_location = False
        if location:
            if main_entity.category == "LOCATION":
                add_location = True
            elif main_entity.category in ("ORGANIZATION", "PERSON", "FACILITY"):
                # Only add location if primary description contains category-like words
                # or location is country-level (2-3 chars, common countries)
                _country_like = len(location) <= 3 and location not in ("深圳", "上海", "北京", "广州", "杭州")
                _is_category_like = any(kw in primary_parts[0] for kw in
                    ("公司", "企业", "人物", "组织", "机构", "人物", "作家", "画家",
                     "科学家", "企业家", "政治家", "学者"))
                if _country_like or _is_category_like:
                    add_location = True

        if add_location:
            desc = f"{location}{primary_parts[0]}"
            if len(primary_parts) > 1:
                desc = f"{desc}与{primary_parts[1]}"
        else:
            desc = "、".join(primary_parts[:2])

    # --- Step 5: choose format based on best predicate ---
    best_pred = used_predicates[0] if used_predicates else scored_rels[0][1]

    if best_pred == "HAS_TITLE":
        title = f"{main_entity.text}：{desc}"
    else:
        title = f"{main_entity.text}——{desc}"

    used_predicates = list(dict.fromkeys(used_predicates))
    entities_used = [main_entity.text] + [o for o in seen_objects if o != main_entity.text]

    title = _truncate(title, target_chars)
    reasons = [
        f"central_entity({main_entity.text}, {entity_rel_count[main_eid]}rels)",
        f"category({main_entity.category})",
    ]
    if used_predicates:
        reasons.append(f"predicates({','.join(used_predicates)})")
    if location:
        reasons.append(f"location({location})")

    return (title, entities_used, used_predicates if used_predicates else [best_pred], reasons)


def generate_title(
    doc: NarrativeDocument,
    mode: str = "chars",
    target_chars: int = _DEFAULT_CHARS,
    target_ratio: float = _DEFAULT_RATIO,
) -> Title:
    """
    Generate a title from a fully analyzed NarrativeDocument.

    Uses entity composition: finds the most central entity and its
    most important predicate to form a concise title.
    Falls back to TextRank top sentence if no entities available.
    """
    content = doc.content
    sentences = content.sentences

    # In ratio mode, compute effective target_chars from total text length
    effective_chars = target_chars
    if mode == "ratio":
        total_len = sum(len(s.text) for s in sentences)
        effective_chars = max(1, int(total_len * target_ratio))

    # Try entity composition first
    title_text, entities_used, predicates_used, reasons = _compose_title_from_entities(
        content, effective_chars
    )

    if title_text and entities_used:
        return Title(
            title_text=title_text,
            method="entity_composition",
            mode=mode,
            target_chars=target_chars,
            target_ratio=target_ratio,
            source_sentence_index=-1,
            source_text="",
            score=0.0,
            entities_used=entities_used,
            predicates_used=predicates_used,
            reasons=reasons,
        )

    # Fallback: use TextRank top sentence
    tr_scores = _compute_textrank(sentences)
    if not tr_scores:
        return Title(
            title_text="", method="entity_composition", mode=mode,
            target_chars=target_chars, target_ratio=target_ratio,
            reasons=["no_sentences"],
        )

    n_total = len(sentences)

    # Apply position bonus BEFORE selecting top sentence
    adjusted_scores = dict(tr_scores)
    for i in range(n_total):
        if i == 0:
            adjusted_scores[i] += 0.15
        elif i == n_total - 1 and n_total > 1:
            adjusted_scores[i] += 0.10

    top_idx = max(adjusted_scores, key=adjusted_scores.get)
    pos_bonus = adjusted_scores[top_idx] - tr_scores[top_idx]
    fused_score = min(adjusted_scores[top_idx], 1.0)

    top_sent = sentences[top_idx]
    title_text = _truncate(top_sent.text, target_chars)

    return Title(
        title_text=title_text,
        method="textrank_fallback",
        mode=mode,
        target_chars=target_chars,
        target_ratio=target_ratio,
        source_sentence_index=top_idx,
        source_text=top_sent.text,
        score=round(fused_score, 4),
        entities_used=[],
        predicates_used=[],
        reasons=[f"textrank({tr_scores[top_idx]:.2f})", f"position_prior({pos_bonus:.2f})"],
    )


def generate_title_text(
    text: str,
    mode: str = "chars",
    target_chars: int = _DEFAULT_CHARS,
    target_ratio: float = _DEFAULT_RATIO,
    language: str = "auto",
) -> Title:
    """
    Generate a title from raw text without full NLP analysis.

    Uses TextRank to find the most important sentence, then truncates
    to target_chars. No entities or predicates available.
    """
    from .analyzer import _split_sentences
    from .language_detector import detect_language

    raw_sents = _split_sentences(text)
    if not raw_sents:
        return Title(
            title_text="", method="textrank_truncate", mode=mode,
            target_chars=target_chars, target_ratio=target_ratio,
            reasons=["empty_text"],
        )

    sentences: list[SentenceLanguage] = []
    for st, off in raw_sents:
        lbl = language if language != "auto" else detect_language(st)[0]
        sentences.append(SentenceLanguage(
            text=st, span=(off, off + len(st)), label=lbl, confidence=0.5,
        ))

    tr_scores = _compute_textrank(sentences)
    if not tr_scores:
        return Title(
            title_text="", method="textrank_truncate", mode=mode,
            target_chars=target_chars, target_ratio=target_ratio,
            reasons=["no_sentences"],
        )

    n_total = len(sentences)

    if mode == "ratio":
        total_len = sum(len(s.text) for s in sentences)
        target_chars = max(1, int(total_len * target_ratio))

    # Apply position bonus BEFORE selecting top sentence
    adjusted_scores = dict(tr_scores)
    for i in range(n_total):
        if i == 0:
            adjusted_scores[i] += 0.15
        elif i == n_total - 1 and n_total > 1:
            adjusted_scores[i] += 0.10

    top_idx = max(adjusted_scores, key=adjusted_scores.get)
    pos_bonus = adjusted_scores[top_idx] - tr_scores[top_idx]
    fused_score = min(adjusted_scores[top_idx], 1.0)

    top_sent = sentences[top_idx]
    title_text = _truncate(top_sent.text, target_chars)

    reasons = [f"textrank({tr_scores[top_idx]:.2f})"]
    if pos_bonus > 0:
        reasons.append(f"position_prior({pos_bonus:.2f})")

    return Title(
        title_text=title_text,
        method="textrank_truncate",
        mode=mode,
        target_chars=target_chars,
        target_ratio=target_ratio,
        source_sentence_index=top_idx,
        source_text=top_sent.text,
        score=round(fused_score, 4),
        entities_used=[],
        predicates_used=[],
        reasons=reasons,
    )
