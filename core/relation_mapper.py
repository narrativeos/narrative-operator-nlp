"""
Relation Extraction — SRL-first with DEP supplementary.

Strategy (redesigned):
1. SRL (semantic role labeling) is the primary source — it gives clean
   (ARG0, PRED, ARG1) triples that directly map to NSP predicates.
2. DEP (dependency parsing) is used only for supplementary relations
   that SRL doesn't capture: adjective→noun properties, etc.
3. Entity-aware: relations must involve at least one entity.
4. Function words (是, 的, 和, 一, ...) are excluded as endpoints.
5. Compound-internal dependencies (nn, assmod, assm) are skipped —
   they're entity merging hints, not cross-entity relations.

Optimizations (v2):
- Extended SRL pairs: ARG0→ARG2, ARG1→ARGM-TMP, etc.
- Predicate refinement: verb-based mapping (位于→LOCATED_AT, etc.)
- Relaxed entity filter: at least ONE endpoint must be an entity
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import Token

from .schema import Relation, EntityAttribute

# ── Compound-internal dep rels (entity merging, not cross-entity) ──
_COMPOUND_INTERNAL = {"nn", "assmod", "assm", "nummod", "clf", "det", "punct", "cc", "conj", "root", "top", "attr", "pass", "etmp", "prep", "pobj", "appos", "lobj", "plmod", "tmod", "advcl", "rcmod", "nsubjpass"}

# POS that are NOT valid relation endpoints (function words / quantities)
_NON_ENDPOINT_POS: frozenset[str] = frozenset({
    "NUM", "ADV", "PART", "PUNCT", "PRON", "DET", "AUX", "SCONJ", "CCONJ",
    "INTJ", "SYM", "X",
})

# Classical Chinese function words that should NOT be relation endpoints
# These are common in classical texts but are not meaningful entities
_CLASSICAL_NON_ENDPOINTS: frozenset[str] = frozenset({
    # 代词
    "其", "之", "彼", "此", "是", "斯", "厥", "我", "汝", "尔", "乃", "若", "而",
    "吾", "余", "予", "吾人", "余人",
    # 指示词
    "此", "彼", "是", "斯", "夫", "盖",
    # 虚词/助词
    "者", "也", "矣", "焉", "乎", "哉", "耶", "欤", "兮", "哉",
    "所", "以", "所", "之", "于", "以", "而", "则", "虽", "然",
    # 量词/数量词
    "名", "千", "万", "百", "十", "里", "丈", "尺", "寸", "斤", "两", "钱", "贯",
    "几", "数", "多", "少", "大", "小", "长", "短", "高", "低", "深", "浅",
    # 常见虚词组合
    "其名", "其大", "其小", "其长", "其高", "其深", "其广", "其翼", "其尾",
    "不知", "不亦", "未尝", "未始",
    # 动词作虚词（不应作为关系端点）
    "去", "徙", "击", "抟", "息", "化", "为", "曰", "云", "谓",
    # 数量组合
    "三千", "九万", "三千里", "九万里", "六月",
})

# ── Verb-based predicate mapping (Modern Chinese) ──
# Maps predicate_verb → more specific NSP predicate
_VERB_PREDICATE_MAP: dict[str, str] = {
    # Location
    "位于": "LOCATED_AT",
    "在": "LOCATED_AT",
    "坐落于": "LOCATED_AT",
    "地处": "LOCATED_AT",
    # Production
    "生产": "PRODUCES",
    "制造": "PRODUCES",
    "制造出": "PRODUCES",
    # Composition
    "组成": "COMPOSED_OF",
    "构成": "COMPOSED_OF",
    "由...组成": "COMPOSED_OF",
    # Causal
    "导致": "CAUSES",
    "引起": "CAUSES",
    "造成": "CAUSES",
    # Control
    "控制": "CONTROLS",
    "管理": "CONTROLS",
    "领导": "CONTROLS",
    # Transfer
    "转移到": "TRANSFERS_TO",
    "运往": "TRANSFERS_TO",
    # Movement
    "移动到": "MOVED_TO",
    "迁往": "MOVED_TO",
    "出发": "DEPARTED_FROM",
    # Interaction
    "与...合作": "INTERACTS_WITH",
    "和...合作": "INTERACTS_WITH",
}

# ── Semantic class mapping (Step 2: dual-track system) ──
# Maps predicate_verb → semantic_class for RELATES_TO fallback.
# This is an OPEN SET auxiliary annotation, NOT part of RelationPredicate enum.
# Usage guide:
# - predicate != "RELATES_TO" → use predicate directly (exact mapping)
# - predicate == "RELATES_TO" and semantic_class is set → use semantic_class as edge type
# - predicate == "RELATES_TO" and semantic_class is None → generic edge, keep predicate_verb
_VERB_SEMANTIC_CLASS_MAP: dict[str, str] = {
    # 使役 (Causation)
    "让": "CAUSATION", "使": "CAUSATION", "令": "CAUSATION", "叫": "CAUSATION",
    "导致": "CAUSATION", "引起": "CAUSATION", "造成": "CAUSATION",
    # 感知 (Perception)
    "看": "PERCEPTION", "见": "PERCEPTION", "听": "PERCEPTION", "闻": "PERCEPTION",
    "触": "PERCEPTION", "观": "PERCEPTION",
    # 言语 (Communication)
    "说": "COMMUNICATION", "讲": "COMMUNICATION", "祝": "COMMUNICATION",
    "问": "COMMUNICATION", "答": "COMMUNICATION", "曰": "COMMUNICATION",
    "云": "COMMUNICATION", "谓": "COMMUNICATION",
    # 移动 (Movement)
    "去": "MOVEMENT", "来": "MOVEMENT", "进": "MOVEMENT", "出": "MOVEMENT",
    "过": "MOVEMENT", "到": "MOVEMENT", "奔赴": "MOVEMENT", "至": "MOVEMENT",
    "往": "MOVEMENT", "适": "MOVEMENT", "迁": "MOVEMENT", "赴": "MOVEMENT",
    # 存在 (Existence)
    "有": "EXISTENCE", "在": "EXISTENCE", "存": "EXISTENCE", "留": "EXISTENCE",
    "坐落": "EXISTENCE",
    # 创建 (Creation)
    "建": "CREATION", "造": "CREATION", "修": "CREATION", "筑": "CREATION",
    "写": "CREATION", "修建": "CREATION",
}


# ── Classical Chinese verb-based predicate mapping (minimal seed) ──
# 古汉语动词谓词映射（最小种子，用于辅助推断）
# 注意：这个映射只是辅助，主要依赖基于实体类型的动态推断
_CLASSICAL_VERB_PREDICATE_MAP: dict[str, str] = {
    # 判断/系词 (Copula/Judgment)
    "为": "IS_A",
    "乃": "IS_A",
    "即": "IS_A",
    "系": "IS_A",
    # 引述 (Quotation/Statement)
    "曰": "RELATES_TO",
    "云": "RELATES_TO",
    "谓": "RELATES_TO",
    # 表字 (Courtesy name)
    "字": "HAS_PROPERTY",
    # 移动 (Movement)
    "至": "MOVED_TO",
    "往": "MOVED_TO",
    "来": "DEPARTED_FROM",
    "适": "MOVED_TO",
    # 使令 (Causative)
    "使": "CAUSES",
    "令": "CAUSES",
    "命": "CAUSES",
    # 给予 (Giving)
    "赐": "TRANSFERS_TO",
    "予": "TRANSFERS_TO",
    "赠": "TRANSFERS_TO",
}


# ── Entity Type-Based Predicate Inference ──
# 基于实体类型组合的动态谓词推断
#
# 注意：这个映射覆盖**所有**实体类型组合，不仅限于古汉语。
# 古汉语实际常见的实体类型：PERSON, LOCATION, TITLE, ERA, INSTITUTION, PRODUCT
# 现代汉语额外支持：MATERIAL, STANDARD, PARAMETER, FACILITY, ORGANIZATION, EVENT
#
# Key: (subject_category, object_category) → predicate

_TYPE_BASED_PREDICATE_MAP: dict[tuple[str, str], str] = {
    # ── PERSON (古今通用) ──
    ("PERSON", "LOCATION"): "LOCATED_AT",
    ("PERSON", "ERA"): "TEMPORAL_AT",
    ("PERSON", "TITLE"): "HAS_PROPERTY",
    ("PERSON", "INSTITUTION"): "PART_OF",
    ("PERSON", "ORGANIZATION"): "PART_OF",
    ("PERSON", "PERSON"): "INTERACTS_WITH",
    ("PERSON", "PRODUCT"): "PRODUCES",
    ("PERSON", "EVENT"): "INTERACTS_WITH",
    ("PERSON", "MATERIAL"): "PRODUCES",
    ("PERSON", "STANDARD"): "HAS_PROPERTY",
    ("PERSON", "ASTRONOMY"): "HAS_PROPERTY",
    ("PERSON", "FACILITY"): "LOCATED_AT",
    ("PERSON", "DATE"): "TEMPORAL_AT",
    ("PERSON", "NUMBER"): "HAS_PROPERTY",
    ("PERSON", "PARAMETER"): "HAS_PROPERTY",
    ("PERSON", "UNKNOWN"): "RELATES_TO",
    
    # ── LOCATION (古今通用) ──
    ("LOCATION", "LOCATION"): "LOCATED_AT",
    ("LOCATION", "ERA"): "TEMPORAL_AT",
    ("LOCATION", "INSTITUTION"): "HAS_PROPERTY",
    ("LOCATION", "ORGANIZATION"): "HAS_PROPERTY",
    ("LOCATION", "PERSON"): "HAS_PROPERTY",
    ("LOCATION", "MATERIAL"): "PRODUCES",
    ("LOCATION", "PRODUCT"): "PRODUCES",
    ("LOCATION", "EVENT"): "LOCATED_AT",
    ("LOCATION", "ASTRONOMY"): "HAS_PROPERTY",
    ("LOCATION", "FACILITY"): "HAS_PROPERTY",
    ("LOCATION", "DATE"): "TEMPORAL_AT",
    ("LOCATION", "STANDARD"): "HAS_PROPERTY",
    ("LOCATION", "UNKNOWN"): "RELATES_TO",
    
    # ── ERA (古今通用) ──
    ("ERA", "ERA"): "TEMPORAL_AT",
    ("ERA", "PERSON"): "HAS_PROPERTY",
    ("ERA", "LOCATION"): "HAS_PROPERTY",
    ("ERA", "EVENT"): "TEMPORAL_AT",
    ("ERA", "INSTITUTION"): "HAS_PROPERTY",
    ("ERA", "PRODUCT"): "HAS_PROPERTY",
    ("ERA", "ASTRONOMY"): "HAS_PROPERTY",
    ("ERA", "ORGANIZATION"): "HAS_PROPERTY",
    ("ERA", "FACILITY"): "HAS_PROPERTY",
    ("ERA", "TITLE"): "HAS_PROPERTY",
    ("ERA", "UNKNOWN"): "RELATES_TO",
    
    # ── TITLE (古汉语特有) ──
    ("TITLE", "PERSON"): "HAS_PROPERTY",
    ("TITLE", "INSTITUTION"): "PART_OF",
    ("TITLE", "LOCATION"): "LOCATED_AT",
    ("TITLE", "TITLE"): "PART_OF",
    ("TITLE", "ERA"): "TEMPORAL_AT",
    ("TITLE", "UNKNOWN"): "RELATES_TO",
    
    # ── INSTITUTION (古今通用) ──
    ("INSTITUTION", "INSTITUTION"): "PART_OF",
    ("INSTITUTION", "PERSON"): "HAS_PROPERTY",
    ("INSTITUTION", "LOCATION"): "LOCATED_AT",
    ("INSTITUTION", "ERA"): "TEMPORAL_AT",
    ("INSTITUTION", "ORGANIZATION"): "PART_OF",
    ("INSTITUTION", "PRODUCT"): "PRODUCES",
    ("INSTITUTION", "STANDARD"): "HAS_PROPERTY",
    ("INSTITUTION", "FACILITY"): "LOCATED_AT",
    ("INSTITUTION", "TITLE"): "HAS_PROPERTY",
    ("INSTITUTION", "UNKNOWN"): "RELATES_TO",
    
    # ── ORGANIZATION (现代为主) ──
    ("ORGANIZATION", "ORGANIZATION"): "PART_OF",
    ("ORGANIZATION", "PERSON"): "HAS_PROPERTY",
    ("ORGANIZATION", "LOCATION"): "LOCATED_AT",
    ("ORGANIZATION", "ERA"): "TEMPORAL_AT",
    ("ORGANIZATION", "INSTITUTION"): "PART_OF",
    ("ORGANIZATION", "PRODUCT"): "PRODUCES",
    ("ORGANIZATION", "FACILITY"): "LOCATED_AT",
    ("ORGANIZATION", "STANDARD"): "HAS_PROPERTY",
    ("ORGANIZATION", "UNKNOWN"): "RELATES_TO",
    
    # ── PRODUCT (古今通用) ──
    ("PRODUCT", "PRODUCT"): "RELATES_TO",
    ("PRODUCT", "PERSON"): "HAS_PROPERTY",
    ("PRODUCT", "MATERIAL"): "COMPOSED_OF",
    ("PRODUCT", "STANDARD"): "DEPENDS_ON",
    ("PRODUCT", "LOCATION"): "LOCATED_AT",
    ("PRODUCT", "ERA"): "TEMPORAL_AT",
    ("PRODUCT", "FACILITY"): "LOCATED_AT",
    ("PRODUCT", "PARAMETER"): "HAS_PROPERTY",
    ("PRODUCT", "UNKNOWN"): "RELATES_TO",
    
    # ── MATERIAL (现代为主) ──
    ("MATERIAL", "MATERIAL"): "COMPOSED_OF",
    ("MATERIAL", "LOCATION"): "LOCATED_AT",
    ("MATERIAL", "STANDARD"): "DEPENDS_ON",
    ("MATERIAL", "PRODUCT"): "COMPOSED_OF",
    ("MATERIAL", "PARAMETER"): "HAS_PROPERTY",
    ("MATERIAL", "PERSON"): "HAS_PROPERTY",
    ("MATERIAL", "FACILITY"): "LOCATED_AT",
    ("MATERIAL", "UNKNOWN"): "RELATES_TO",
    
    # ── STANDARD (现代为主) ──
    ("STANDARD", "STANDARD"): "DEPENDS_ON",
    ("STANDARD", "PARAMETER"): "HAS_PROPERTY",
    ("STANDARD", "MATERIAL"): "CONSTRAINT_OF",
    ("STANDARD", "PRODUCT"): "CONSTRAINT_OF",
    ("STANDARD", "PERSON"): "HAS_PROPERTY",
    ("STANDARD", "ORGANIZATION"): "HAS_PROPERTY",
    ("STANDARD", "UNKNOWN"): "RELATES_TO",
    
    # ── PARAMETER (现代为主) ──
    ("PARAMETER", "PARAMETER"): "RELATES_TO",
    ("PARAMETER", "MATERIAL"): "HAS_PROPERTY",
    ("PARAMETER", "PRODUCT"): "HAS_PROPERTY",
    ("PARAMETER", "STANDARD"): "DEPENDS_ON",
    ("PARAMETER", "PERSON"): "HAS_PROPERTY",
    ("PARAMETER", "UNKNOWN"): "RELATES_TO",
    
    # ── FACILITY (现代为主) ──
    ("FACILITY", "LOCATION"): "LOCATED_AT",
    ("FACILITY", "FACILITY"): "LOCATED_AT",
    ("FACILITY", "ORGANIZATION"): "PART_OF",
    ("FACILITY", "PERSON"): "HAS_PROPERTY",
    ("FACILITY", "ERA"): "TEMPORAL_AT",
    ("FACILITY", "UNKNOWN"): "RELATES_TO",
    
    # ── EVENT (古今通用) ──
    ("EVENT", "EVENT"): "RELATES_TO",
    ("EVENT", "PERSON"): "HAS_PROPERTY",
    ("EVENT", "LOCATION"): "LOCATED_AT",
    ("EVENT", "ERA"): "TEMPORAL_AT",
    ("EVENT", "INSTITUTION"): "HAS_PROPERTY",
    ("EVENT", "ORGANIZATION"): "HAS_PROPERTY",
    ("EVENT", "DATE"): "TEMPORAL_AT",
    ("EVENT", "UNKNOWN"): "RELATES_TO",
    
    # ── ASTRONOMY (古汉语特有) ──
    ("ASTRONOMY", "ASTRONOMY"): "RELATES_TO",
    ("ASTRONOMY", "LOCATION"): "LOCATED_AT",
    ("ASTRONOMY", "ERA"): "TEMPORAL_AT",
    ("ASTRONOMY", "PERSON"): "HAS_PROPERTY",
    ("ASTRONOMY", "UNKNOWN"): "RELATES_TO",
    
    # ── DATE ──
    ("DATE", "DATE"): "TEMPORAL_AT",
    ("DATE", "PERSON"): "HAS_PROPERTY",
    ("DATE", "EVENT"): "TEMPORAL_AT",
    ("DATE", "LOCATION"): "HAS_PROPERTY",
    ("DATE", "ERA"): "TEMPORAL_AT",
    ("DATE", "UNKNOWN"): "RELATES_TO",
    
    # ── NUMBER ──
    ("NUMBER", "NUMBER"): "RELATES_TO",
    ("NUMBER", "PERSON"): "HAS_PROPERTY",
    ("NUMBER", "PRODUCT"): "HAS_PROPERTY",
    ("NUMBER", "MATERIAL"): "HAS_PROPERTY",
    ("NUMBER", "UNKNOWN"): "RELATES_TO",
    
    # ── UNKNOWN (兜底) ──
    ("UNKNOWN", "PERSON"): "RELATES_TO",
    ("UNKNOWN", "LOCATION"): "RELATES_TO",
    ("UNKNOWN", "UNKNOWN"): "RELATES_TO",
}


def _infer_predicate_from_types(
    subject_category: str,
    object_category: str,
    predicate_verb: str = "",
) -> str:
    """Infer predicate from entity types and verb context.
    
    Uses a combination of:
    1. Entity type pair (subject_type, object_type)
    2. Predicate verb hints
    
    Args:
        subject_category: Subject entity category (e.g., "PERSON", "LOCATION")
        object_category: Object entity category
        predicate_verb: Original verb for additional context
        
    Returns:
        NSP predicate string.
    """
    # Try direct type-based lookup
    type_pair = (subject_category, object_category)
    if type_pair in _TYPE_BASED_PREDICATE_MAP:
        return _TYPE_BASED_PREDICATE_MAP[type_pair]
    
    # Try verb-based lookup (as supplement)
    if predicate_verb in _CLASSICAL_VERB_PREDICATE_MAP:
        return _CLASSICAL_VERB_PREDICATE_MAP[predicate_verb]
    
    # Try verb-based lookup for modern Chinese
    if predicate_verb in _VERB_PREDICATE_MAP:
        return _VERB_PREDICATE_MAP[predicate_verb]
    
    # Fallback: use semantic heuristics
    # Movement verbs + LOCATION → MOVED_TO
    movement_verbs = {"至", "往", "来", "适", "迁", "赴", "归", "还", "入", "出"}
    if predicate_verb in movement_verbs and object_category == "LOCATION":
        return "MOVED_TO"
    if predicate_verb in movement_verbs and subject_category == "LOCATION":
        return "DEPARTED_FROM"
    
    # Judgment/copula verbs → IS_A
    judgment_verbs = {"为", "乃", "即", "系", "是"}
    if predicate_verb in judgment_verbs:
        return "IS_A"
    
    # Default fallback
    return "RELATES_TO"


def _is_relation_endpoint(token, text: str = "") -> bool:
    """A token is a valid relation endpoint unless it is a function word.
    
    Also filters out classical Chinese function words that are not meaningful entities.
    """
    if token is None:
        return False
    if token.pos in _NON_ENDPOINT_POS:
        return False
    # Filter classical Chinese function words
    if text in _CLASSICAL_NON_ENDPOINTS:
        return False
    return True


def _find_entity(entities: list, text: str):
    """Find an entity object by its text."""
    for e in entities:
        if e.text == text:
            return e
    return None


def _attach_attribute(
    entities: list, owner_text: str, prop_text: str,
    confidence: float = 0.70, source: str = "",
) -> None:
    """Attach a property to an entity as EntityAttribute (deduplicated)."""
    owner = _find_entity(entities, owner_text)
    if owner is None:
        return
    if any(a.key == prop_text for a in owner.attributes):
        return
    attr = EntityAttribute(
        key=prop_text,
        value="",
        predicate_verb="",
        confidence=confidence,
    )
    owner.attributes.append(attr)


# Classical Chinese deprel types that indicate semantic relations
_CLASSICAL_DEP_RELS: dict[str, str] = {
    "nsubj": "RELATES_TO",
    "obj": "RELATES_TO",
    "dobj": "RELATES_TO",
    "nmod": "HAS_PROPERTY",
    "amod": "HAS_PROPERTY",
    "iobj": "RELATES_TO",
}


def _span_between(sp1: tuple, sp2: tuple, text: str) -> str:
    """Extract text covering two spans."""
    start = min(sp1[0], sp2[0])
    end = max(sp1[1], sp2[1])
    if start < 0 or end > len(text):
        return ""
    return text[start:end]


def _resolve_predicate(pred_text: str, arg_role: str = "") -> tuple[str, Optional[str]]:
    """Resolve predicate_verb to (predicate, semantic_class).

    Dual-track system:
    - predicate: NSP standard predicate (always from RelationPredicate enum)
    - semantic_class: Open-set auxiliary annotation for RELATES_TO fallback

    Decision chain:
    1. _VERB_PREDICATE_MAP hit → (mapped_predicate, None)
    2. _CLASSICAL_VERB_PREDICATE_MAP hit → (mapped_predicate, None)
    3. Role-based mapping hit → (mapped_predicate, None)
    4. Fallback → ("RELATES_TO", semantic_class or None)

    Args:
        pred_text: The verb/predicate text (e.g., "位于", "生产")
        arg_role: The argument role for context (e.g., "argm-loc")

    Returns:
        (predicate, semantic_class) tuple
    """
    # Phase 1: Direct verb mapping (exact predicate, no semantic_class)
    if pred_text in _VERB_PREDICATE_MAP:
        return (_VERB_PREDICATE_MAP[pred_text], None)

    # Phase 2: Classical verb mapping
    if pred_text in _CLASSICAL_VERB_PREDICATE_MAP:
        return (_CLASSICAL_VERB_PREDICATE_MAP[pred_text], None)

    # Phase 3: Role-based mapping
    role_map = {
        "argm-loc": "LOCATED_AT",
        "argm-tmp": "TEMPORAL_AT",
        "argm-mnr": "HAS_PROPERTY",
        "argm-cau": "CAUSES",
        "argm-prd": "PRODUCES",
        "argm-ben": "BENEFITS",
        "argm-rec": "TRANSFERS_TO",
        "argm-src": "DEPARTED_FROM",
    }
    if arg_role in role_map:
        return (role_map[arg_role], None)

    # Phase 4: Fallback to RELATES_TO + semantic_class
    semantic_class = _VERB_SEMANTIC_CLASS_MAP.get(pred_text)
    return ("RELATES_TO", semantic_class)


class RelationExtractionRules:
    """SRL-first, DEP-supplementary relation extractor."""

    def __init__(self):
        self._counter = 0

    def reset(self):
        """Reset per-request state. Must be called before each new analysis."""
        self._counter = 0

    # ── SRL Extraction (primary) ──

    def extract_from_srl(self, frame: list, tokens: list,
                         text: str = "") -> list[Relation]:
        """Extract relations from SRL frame.

        Extended pairs:
        - ARG0→ARG1 (primary)
        - ARG0→ARG2 (secondary object)
        - ARG0→ARGM-LOC (location)
        - ARG0→ARGM-TMP (time)
        - ARG1→ARGM-LOC (object location)
        - ARG1→ARG2 (object→secondary)
        """
        args: dict[str, tuple[str, tuple]] = {}
        pred_text = ""

        for item in frame:
            if len(item) < 4:
                continue
            role = str(item[1]).upper()
            itxt = str(item[0])
            ts, te = int(item[2]), int(item[3])
            if 0 <= ts < len(tokens) and 0 < te <= len(tokens):
                cs = tokens[ts].span[0]
                ce = tokens[te - 1].span[1]
            else:
                cs, ce = 0, 0
            if role == "PRED":
                pred_text = itxt
            elif role.startswith("ARG"):
                args.setdefault(role.lower(), (itxt, (cs, ce)))

        a0 = args.get("arg0")
        a1 = args.get("arg1")
        a2 = args.get("arg2")
        loc = args.get("argm-loc")
        tmp = args.get("argm-tmp")
        mnr = args.get("argm-mnr")  # Manner
        cau = args.get("argm-cau")  # Cause
        prd = args.get("argm-prd")  # Product/Result
        ben = args.get("argm-ben")  # Beneficiary
        rec = args.get("argm-rec")  # Recipient
        src = args.get("argm-src")  # Source

        # ── Generate all applicable relation pairs ──
        pairs: list[tuple[tuple, tuple, str]] = []

        # Primary: ARG0→ARG1
        if a0 and a1:
            pairs.append((a0, a1, "arg1"))

        # ARG0→ARG2
        if a0 and a2:
            pairs.append((a0, a2, "arg2"))

        # ARG0→ARGM-LOC (location)
        if a0 and loc:
            pairs.append((a0, loc, "argm-loc"))

        # ARG0→ARGM-TMP (time)
        if a0 and tmp:
            pairs.append((a0, tmp, "argm-tmp"))

        # ARG0→ARGM-MNR (manner)
        if a0 and mnr:
            pairs.append((a0, mnr, "argm-mnr"))

        # ARG0→ARGM-CAU (cause)
        if a0 and cau:
            pairs.append((a0, cau, "argm-cau"))

        # ARG0→ARGM-PRD (product/result)
        if a0 and prd:
            pairs.append((a0, prd, "argm-prd"))

        # ARG0→ARGM-BEN (beneficiary)
        if a0 and ben:
            pairs.append((a0, ben, "argm-ben"))

        # ARG0→ARGM-REC (recipient)
        if a0 and rec:
            pairs.append((a0, rec, "argm-rec"))

        # ARG0→ARGM-SRC (source)
        if a0 and src:
            pairs.append((a0, src, "argm-src"))

        # ARG1→ARGM-LOC
        if a1 and loc:
            pairs.append((a1, loc, "argm-loc"))

        # ARG1→ARG2
        if a1 and a2:
            pairs.append((a1, a2, "arg2"))

        # ARG1→ARGM-PRD
        if a1 and prd:
            pairs.append((a1, prd, "argm-prd"))

        relations = []
        for subj, obj, role in pairs:
            predicate, semantic_class = _resolve_predicate(pred_text, role)
            evidence = _span_between(subj[1], obj[1], text)
            if not evidence:
                evidence = f"{subj[0]} {pred_text} {obj[0]}"

            self._counter += 1
            relations.append(Relation(
                id=f"rel_{self._counter:03d}",
                subject=subj[0].strip(),
                predicate=predicate,
                predicate_verb=pred_text,
                semantic_class=semantic_class,
                object=obj[0].strip(),
                evidence=evidence,
                evidence_span=(min(subj[1][0], obj[1][0]), max(subj[1][1], obj[1][1])),
                confidence=0.60,
                source=f"srl/{pred_text}",
            ))
        return relations

    # ── DEP Extraction (supplementary, amod only) ──

    def extract_from_dep(self, child_idx: int, deprel: str, head_idx: int,
                         tokens: list, text: str = "") -> Optional[Relation]:
        """Extract adjective-property relations from DEP (amod only)."""
        deprel = str(deprel).strip().lower()

        if deprel in _COMPOUND_INTERNAL:
            return None

        if deprel != "amod":
            return None

        head_0 = head_idx - 1
        if not (0 <= child_idx < len(tokens) and 0 <= head_0 < len(tokens)):
            return None

        ct, ht = tokens[child_idx], tokens[head_0]

        if ct.text == ht.text:
            return None

        evidence = _span_between(ct.span, ht.span, text)
        if not evidence:
            evidence = f"{ht.text}{ct.text}"

        self._counter += 1
        return Relation(
            id=f"rel_{self._counter:03d}",
            subject=ht.text,
            predicate="HAS_PROPERTY",
            object=ct.text,
            evidence=evidence,
            evidence_span=(min(ct.span[0], ht.span[0]), max(ct.span[1], ht.span[1])),
            confidence=0.80,
            source="dep/amod",
        )

    # ── Classical DEP Extraction (nsubj, dobj, nmod) ──

    def extract_from_dep_classical(
        self, child_idx: int, deprel: str, head_idx: int,
        tokens: list, text: str = "",
    ) -> Optional[Relation]:
        """Extract semantic relations from classical Chinese dep output."""
        deprel = str(deprel).strip().lower()

        if deprel in _COMPOUND_INTERNAL:
            return None

        mapped_pred = _CLASSICAL_DEP_RELS.get(deprel)
        if mapped_pred is None:
            return None

        head_0 = head_idx - 1
        if not (0 <= child_idx < len(tokens) and 0 <= head_0 < len(tokens)):
            return None

        ct, ht = tokens[child_idx], tokens[head_0]

        if ct.text == ht.text:
            return None

        if deprel in ("nsubj", "nmod", "amod"):
            subj, obj = ct, ht
        elif deprel in ("dobj", "obj", "iobj"):
            subj, obj = ht, ct
        else:
            return None

        evidence = _span_between(subj.span, obj.span, text)
        if not evidence:
            evidence = f"{subj.text}{obj.text}"

        self._counter += 1
        return Relation(
            id=f"rel_{self._counter:03d}",
            subject=subj.text,
            predicate=mapped_pred,
            object=obj.text,
            evidence=evidence,
            evidence_span=(min(subj.span[0], obj.span[0]), max(subj.span[1], obj.span[1])),
            confidence=0.55,
            source=f"dep/{deprel}",
        )

    # ── Token helpers ──

    @staticmethod
    def _merge_det_compound(
        idx: int, dep: list, tokens: list,
    ) -> tuple[str, tuple[int, int]]:
        """Merge a token with its det children into a compound."""
        merged_indices = {idx}
        for ci, d in enumerate(dep):
            if not isinstance(d, (list, tuple)) or len(d) < 2:
                continue
            if int(d[0]) - 1 == idx and str(d[1]).strip().lower() == "det":
                merged_indices.add(ci)
        sorted_idx = sorted(merged_indices)
        text = "".join(tokens[i].text for i in sorted_idx)
        span = (
            tokens[sorted_idx[0]].span[0],
            tokens[sorted_idx[-1]].span[1],
        )
        return text, span

    # ── Aggregate ──

    def extract_all(self, text: str, raw: dict, tokens: list,
                    entities: list | None = None) -> list:
        """Extract relations: SRL first, then supplementary DEP (amod).

        Relaxed entity filter: at least ONE endpoint must be an entity.
        """
        entity_texts: set[str] = set()
        if entities:
            for e in entities:
                entity_texts.add(e.text)

        relations: list = []
        has_srl = bool(raw.get("srl"))
        if has_srl:
            for f in raw.get("srl", []):
                if isinstance(f, list):
                    rels = self.extract_from_srl(f, tokens, text)
                    for rel in rels:
                        rel = self._entity_normalize(rel, entity_texts, strict=False, entities=entities)
                        if rel:
                            relations.append(rel)

        if not has_srl:
            bridges = self._bridge_subj_obj(
                raw.get("dep", []), tokens, text, entity_texts,
            )
            cop_bridges = self._bridge_cop(
                raw.get("dep", []), tokens, text, entity_texts,
            )
            relations.extend(bridges)
            relations.extend(cop_bridges)

            bridged_verbs: set[str] = set()
            for r in bridges + cop_bridges:
                if r.predicate_verb:
                    bridged_verbs.add(r.predicate_verb)

            for i, d in enumerate(raw.get("dep", [])):
                if not isinstance(d, (list, tuple)) or len(d) < 2:
                    continue
                deprel = str(d[1]).strip().lower()
                head_1based = int(d[0])
                head_0 = head_1based - 1

                if deprel in ("nsubj", "obj", "dobj", "iobj"):
                    head_tok = tokens[head_0] if 0 <= head_0 < len(tokens) else None
                    head_text = head_tok.text if head_tok else ""
                    if head_text in bridged_verbs:
                        continue
                    child_tok = tokens[i] if 0 <= i < len(tokens) else None
                    if not _is_relation_endpoint(head_tok) or not _is_relation_endpoint(child_tok):
                        continue
                    if head_tok and child_tok:
                        if (head_tok.pos == "VERB" and child_tok.pos == "NOUN" and len(child_tok.text) == 1):
                            continue
                        if (child_tok.pos == "VERB" and head_tok.pos == "NOUN" and len(head_tok.text) == 1):
                            continue
                    rel = self.extract_from_dep_classical(
                        i, deprel, head_1based, tokens, text,
                    )
                    if rel:
                        rel = self._entity_normalize(rel, entity_texts, strict=False, entities=entities)
                        if rel:
                            relations.append(rel)

                elif deprel in ("nmod", "amod"):
                    if not (0 <= i < len(tokens) and 0 <= head_0 < len(tokens)):
                        continue
                    child_tok, head_tok = tokens[i], tokens[head_0]
                    if head_tok.text in entity_texts:
                        owner_text, prop_text = head_tok.text, child_tok.text
                    elif child_tok.text in entity_texts:
                        owner_text, prop_text = child_tok.text, head_tok.text
                    else:
                        continue
                    if entities:
                        _attach_attribute(entities, owner_text, prop_text,
                                          confidence=0.70, source=f"dep/{deprel}")

            # 之-interpolation + NUM/measure heuristic
            for e_text in entity_texts:
                occurrences = [t for t in tokens if t.text == e_text]
                for e_tok in occurrences:
                    e_start, e_end = e_tok.span
                    found_attr = False
                    for i, t in enumerate(tokens):
                        if t.span[0] < e_end:
                            continue
                        if t.text == "之" and t.pos in ("SCONJ", "PART"):
                            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
                            if nxt and nxt.pos in ("VERB", "ADJ", "ADV"):
                                if entities:
                                    _attach_attribute(
                                        entities, e_text, nxt.text,
                                        confidence=0.50, source="heuristic/之",
                                    )
                                found_attr = True
                                j = i + 2
                                while j < len(tokens):
                                    tt = tokens[j]
                                    if tt.pos in ("PUNCT",) and tt.text in ("。", "！", "？", "；"):
                                        break
                                    if tt.pos == "NUM":
                                        parts = [tt.text]
                                        k = j + 1
                                        while k < len(tokens) and tokens[k].pos in ("NOUN", "NUM"):
                                            xp = (raw.get("pos/xpos", [])[k]
                                                  if raw and k < len(raw.get("pos/xpos", []))
                                                  else "")
                                            if "助数詞" in xp or "度量衡" in xp or "数詞" in xp:
                                                parts.append(tokens[k].text)
                                                k += 1
                                            elif tokens[k].pos == "NUM":
                                                parts.append(tokens[k].text)
                                                k += 1
                                            else:
                                                if tokens[k].pos == "NOUN":
                                                    parts.append(tokens[k].text)
                                                    k += 1
                                                break
                                        compound = "".join(parts)
                                        if entities:
                                            _attach_attribute(
                                                entities, e_text, compound,
                                                confidence=0.45, source="heuristic/num_measure",
                                            )
                                        j = k
                                        continue
                                    if tt.pos == "ADJ":
                                        if entities:
                                            _attach_attribute(
                                                entities, e_text, tt.text,
                                                confidence=0.40, source="heuristic/cont",
                                            )
                                    j += 1
                            break
                        break
                        if found_attr:
                            break

        # ── Split conjunction-connected objects ──
        relations = self._split_conjunction_relations(relations)

        return relations

    # ── Conjunction Splitting ──

    def _split_conjunction_relations(
        self, relations: list[Relation]
    ) -> list[Relation]:
        """Split relations whose subject or object contains conjunctions.

        E.g., '碳钢 → RELATES_TO → 高强度和高韧性' becomes:
          - '碳钢 → RELATES_TO → 高强度'
          - '碳钢 → RELATES_TO → 高韧性'

        Supports Chinese conjunctions: 和, 与, 及, 或, 以及, 、
        Supports English conjunctions: and, or
        """
        # Conjunction patterns (sorted by length to match longer first)
        # Modern Chinese
        CN_CONJS = ["以及", "和", "与", "及", "或", "、"]
        # Classical Chinese conjunctions (古汉语连词)
        CLASSICAL_CONJS = ["暨", "幷", "並", "共", "俱", "皆", "仍", "复", "复又"]
        # English
        EN_CONJS = [" and ", " or "]

        def split_conjunctions(text: str) -> list[str]:
            """Split text by conjunctions, returning clean parts.
            
            Supports modern Chinese, classical Chinese, and English conjunctions.
            """
            # Modern Chinese
            for conj in CN_CONJS:
                text = text.replace(conj, "||SPLIT||")
            # Classical Chinese (古汉语连词)
            for conj in CLASSICAL_CONJS:
                text = text.replace(conj, "||SPLIT||")
            # English
            for conj in EN_CONJS:
                text = text.replace(conj, "||SPLIT||")
            parts = [p.strip() for p in text.split("||SPLIT||") if p.strip()]
            return parts if len(parts) > 1 else [text]

        result: list[Relation] = []
        for rel in relations:
            # Check if subject or object contains conjunctions
            subj_parts = split_conjunctions(rel.subject)
            obj_parts = split_conjunctions(rel.object)

            if len(subj_parts) == 1 and len(obj_parts) == 1:
                # No conjunctions — keep as-is
                result.append(rel)
            else:
                # Split: create one relation per combination
                for subj in subj_parts:
                    for obj in obj_parts:
                        self._counter += 1
                        new_rel = Relation(
                            id=f"rel_{self._counter:03d}",
                            subject=subj,
                            predicate=rel.predicate,
                            predicate_verb=rel.predicate_verb,
                            object=obj,
                            evidence=rel.evidence,
                            evidence_span=rel.evidence_span,
                            confidence=rel.confidence,
                            source=rel.source,
                        )
                        # Preserve raw fields
                        if subj == rel.subject:
                            new_rel.subject_raw = rel.subject_raw
                        else:
                            new_rel.subject_raw = subj
                        if obj == rel.object:
                            new_rel.object_raw = rel.object_raw
                        else:
                            new_rel.object_raw = obj
                        result.append(new_rel)

        return result

    # ── Classical nsubj+obj bridging ──

    def _bridge_subj_obj(
        self, dep: list, tokens: list, text: str, entity_texts: set[str],
    ) -> list[Relation]:
        """Bridge nsubj→verb←obj into subject→object relations.
        
        Filters out classical Chinese function words as endpoints.
        """
        head_to_children: dict[int, list[tuple[int, str]]] = {}
        for child_idx, d in enumerate(dep):
            if not isinstance(d, (list, tuple)) or len(d) < 2:
                continue
            head_1based = int(d[0])
            deprel = str(d[1]).strip().lower()
            if deprel in ("nsubj", "obj", "dobj", "iobj"):
                head_0based = head_1based - 1
                head_to_children.setdefault(head_0based, []).append(
                    (child_idx, deprel),
                )

        bridged: list[Relation] = []
        for head_idx, children in head_to_children.items():
            nsubjs = [(ci, dr) for ci, dr in children if dr == "nsubj"]
            objs = [(ci, dr) for ci, dr in children if dr in ("obj", "dobj", "iobj")]
            if not nsubjs or not objs:
                continue

            head_token = tokens[head_idx] if 0 <= head_idx < len(tokens) else None
            verb_text = head_token.text if head_token else ""

            for subj_idx, _ in nsubjs:
                for obj_idx, _ in objs:
                    if not (0 <= subj_idx < len(tokens) and 0 <= obj_idx < len(tokens)):
                        continue

                    subj_text, subj_span = self._merge_det_compound(
                        subj_idx, dep, tokens,
                    )
                    obj_text, obj_span = self._merge_det_compound(
                        obj_idx, dep, tokens,
                    )
                    
                    # Filter classical Chinese function words
                    if subj_text in _CLASSICAL_NON_ENDPOINTS or obj_text in _CLASSICAL_NON_ENDPOINTS:
                        continue

                    subj_is_ent = subj_text in entity_texts
                    obj_is_ent = obj_text in entity_texts
                    if not subj_is_ent and not obj_is_ent:
                        continue

                    evidence = _span_between(subj_span, obj_span, text)
                    if not evidence:
                        evidence = f"{subj_text}{verb_text}{obj_text}"

                    self._counter += 1
                    bridged.append(Relation(
                        id=f"rel_{self._counter:03d}",
                        subject=subj_text,
                        predicate="RELATES_TO",
                        predicate_verb=verb_text,
                        object=obj_text,
                        evidence=evidence,
                        evidence_span=(
                            min(subj_span[0], obj_span[0]),
                            max(subj_span[1], obj_span[1]),
                        ),
                        confidence=0.60,
                        source=f"bridge/{verb_text}" if verb_text else "bridge/nsubj_obj",
                    ))

        return bridged

    def _bridge_cop(
        self, dep: list, tokens: list, text: str, entity_texts: set[str],
    ) -> list[Relation]:
        """Bridge copula relations: 为(cop)→鲲 + 名(obj)→知 → 名→鲲(为).
        
        Filters out classical Chinese function words as endpoints.
        """
        head_children: dict[int, list[tuple[int, str]]] = {}
        child_head: dict[int, int] = {}
        for ci, d in enumerate(dep):
            if not isinstance(d, (list, tuple)) or len(d) < 2:
                continue
            head = int(d[0]) - 1
            deprel = str(d[1]).strip().lower()
            head_children.setdefault(head, []).append((ci, deprel))
            child_head[ci] = head

        bridged: list[Relation] = []
        for ci, d in enumerate(dep):
            if not isinstance(d, (list, tuple)) or len(d) < 2:
                continue
            deprel = str(d[1]).strip().lower()
            if deprel != "cop":
                continue
            cop_verb = tokens[ci]
            pred_idx = int(d[0]) - 1

            if not (0 <= pred_idx < len(tokens)):
                continue
            pred_tok = tokens[pred_idx]

            pred_head = child_head.get(pred_idx)
            if pred_head is None:
                continue

            for sib_idx, sib_rel in head_children.get(pred_head, []):
                if sib_idx == pred_idx or sib_idx == ci:
                    continue
                if sib_rel not in ("nsubj", "obj", "dobj"):
                    continue
                sib_tok = tokens[sib_idx]

                sib_text, sib_span = self._merge_det_compound(
                    sib_idx, dep, tokens,
                )

                # Filter function words
                if sib_text in _CLASSICAL_NON_ENDPOINTS:
                    continue
                if pred_tok.text in _CLASSICAL_NON_ENDPOINTS:
                    continue

                if (sib_text not in entity_texts
                        and pred_tok.text not in entity_texts):
                    continue

                evidence = _span_between(sib_span, pred_tok.span, text)
                if not evidence:
                    evidence = f"{sib_text}{cop_verb.text}{pred_tok.text}"

                self._counter += 1
                bridged.append(Relation(
                    id=f"rel_{self._counter:03d}",
                    subject=sib_text,
                    predicate="RELATES_TO",
                    predicate_verb=cop_verb.text,
                    object=pred_tok.text,
                    evidence=evidence,
                    evidence_span=(
                        min(sib_tok.span[0], pred_tok.span[0]),
                        max(sib_tok.span[1], pred_tok.span[1]),
                    ),
                    confidence=0.55,
                    source=f"bridge/{cop_verb.text}",
                ))

        return bridged

    def _entity_normalize(self, rel: Relation,
                          entity_texts: set[str],
                          strict: bool = True,
                          entities: list | None = None,
                          ) -> Relation | None:
        """Normalize endpoints: exact match → sub-entity match → reject.

        Relaxed: at least ONE endpoint must match an entity.
        
        Also refines the predicate using entity type-based inference when
        both endpoints are resolved to entities.
        
        Filters out classical Chinese function words as endpoints — if either
        subject or object is a function word, the entire relation is discarded.
        """
        if not entity_texts:
            return rel

        # ── Filter function words: discard relation if endpoint is a function word ──
        if rel.subject in _CLASSICAL_NON_ENDPOINTS:
            return None
        if rel.object in _CLASSICAL_NON_ENDPOINTS:
            return None

        hit_count = 0

        for attr in ("subject", "object"):
            text = getattr(rel, attr)
            
            if text in entity_texts:
                hit_count += 1
                continue

            matches = sorted([e for e in entity_texts if e in text], key=len, reverse=True)
            if matches:
                # Use the longest matching entity, but skip if it's a function word
                matched = matches[0]
                if matched in _CLASSICAL_NON_ENDPOINTS:
                    # Function word matched — discard this relation
                    return None
                setattr(rel, attr, matched)
                hit_count += 1
                continue

        # Relaxed: at least one endpoint must match
        if hit_count == 0:
            if not entity_texts:
                return rel
            return None

        # ── Final check: ensure endpoints are not function words after normalization ──
        if rel.subject in _CLASSICAL_NON_ENDPOINTS:
            return None
        if rel.object in _CLASSICAL_NON_ENDPOINTS:
            return None

        # ── Refine predicate using entity type inference ──
        if entities:
            subj_entity = _find_entity(entities, rel.subject)
            obj_entity = _find_entity(entities, rel.object)
            
            if subj_entity and obj_entity:
                inferred = _infer_predicate_from_types(
                    subj_entity.category,
                    obj_entity.category,
                    rel.predicate_verb or "",
                )
                if inferred != rel.predicate:
                    rel.predicate = inferred

        return rel
