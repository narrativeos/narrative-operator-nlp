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
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import Token

from .schema import Relation

# ── Function words to never use as relation endpoints ──
_FUNCTION_WORDS = {
    "是", "的", "了", "和", "与", "或", "不", "也", "都", "就", "一", "二", "三",
    "四", "五", "六", "七", "八", "九", "十", "个", "种", "次", "回", "这", "那",
    "哪", "每", "着", "过", "得", "地", "之", "把", "被", "让", "给", "对", "从",
    "到", "向", "由", "以", "为", "所", "而", "且", "但", "却", "只", "还", "又",
    "再", "才", "将", "能", "会", "可", "要", "用", "做", "来", "去", "出", "进",
    "开", "关", "有", "没", "说", "想", "看", "听", "吃", "喝", "走", "跑",
    "吗", "呢", "吧", "啊", "嘛", "呀", "第", "如", "等",
}

# ── Compound-internal dep rels (entity merging, not cross-entity) ──
_COMPOUND_INTERNAL = {"nn", "assmod", "assm", "nummod", "clf", "det", "punct", "cc", "conj", "root", "top", "attr", "pass", "etmp", "prep", "pobj", "appos", "nmod", "lobj", "plmod", "tmod", "advcl", "rcmod", "nsubjpass"}

# ── SRL predicate → NSP predicate mapping ──
# High-precision explicit map for common predicates.
_SRL_PRED_MAP: dict[str, tuple[str, float]] = {
    # Copula / classification
    "是": ("IS_A", 0.95), "为": ("IS_A", 0.95), "属于": ("IS_A", 0.92),
    "即": ("IS_A", 0.93), "系": ("IS_A", 0.90),
    # Possession / property
    "具有": ("HAS_PROPERTY", 0.90), "拥有": ("HAS_PROPERTY", 0.90),
    "具备": ("HAS_PROPERTY", 0.90), "含有": ("HAS_PROPERTY", 0.85),
    "包含": ("HAS_PROPERTY", 0.85), "包括": ("HAS_PROPERTY", 0.85),
    # Location
    "位于": ("LOCATED_AT", 0.95), "坐落": ("LOCATED_AT", 0.90),
    "座落": ("LOCATED_AT", 0.90), "地处": ("LOCATED_AT", 0.90),
    "产于": ("LOCATED_AT", 0.85), "住在": ("LOCATED_AT", 0.85),
    # Temporal
    "成立于": ("TEMPORAL_AT", 0.85), "建于": ("TEMPORAL_AT", 0.80),
    "始于": ("TEMPORAL_AT", 0.80), "发生于": ("TEMPORAL_AT", 0.85),
    # Production / creation
    "生产": ("PRODUCES", 0.88), "制造": ("PRODUCES", 0.88),
    "创建": ("PRODUCES", 0.88), "发明": ("PRODUCES", 0.85),
    "开发": ("PRODUCES", 0.85), "生成": ("PRODUCES", 0.83),
    "组成": ("COMPOSED_OF", 0.88), "构成": ("COMPOSED_OF", 0.88),
    # Causation
    "导致": ("CAUSES", 0.85), "引起": ("CAUSES", 0.85),
    "造成": ("CAUSES", 0.85), "影响": ("AFFECTS", 0.82),
    "改变": ("AFFECTS", 0.80), "降低": ("AFFECTS", 0.80),
    "提高": ("AFFECTS", 0.80), "增加": ("AFFECTS", 0.78),
    "减少": ("AFFECTS", 0.78),
    # Transfer / exchange
    "出口": ("TRANSFERS_TO", 0.85), "进口": ("TRANSFERS_TO", 0.85),
    "销售": ("TRANSFERS_TO", 0.85), "提供": ("TRANSFERS_TO", 0.83),
    "供应": ("TRANSFERS_TO", 0.83),
    # Movement
    "来到": ("MOVED_TO", 0.85), "到达": ("MOVED_TO", 0.85),
    "前往": ("MOVED_TO", 0.85), "进入": ("MOVED_TO", 0.83),
    "离开": ("DEPARTED_FROM", 0.85), "出发": ("DEPARTED_FROM", 0.83),
    # Interaction
    "参观": ("INTERACTS_WITH", 0.82), "访问": ("INTERACTS_WITH", 0.82),
    "会见": ("INTERACTS_WITH", 0.85), "合作": ("INTERACTS_WITH", 0.83),
    # Control / ownership
    "拥有": ("HAS_PROPERTY", 0.90), "持有": ("HAS_PROPERTY", 0.85),
    "控制": ("CONTROLS", 0.82), "管理": ("CONTROLS", 0.80),
}

# ── Verb-class fallback: keyword → NSP predicate ──
# When SRL predicate is not in the explicit map, search for these
# substrings to infer the relation type.
_VERB_CLASS_FALLBACK: list[tuple[str, str, float]] = [
    # (keyword substring, NSP predicate, confidence)
    ("产", "PRODUCES", 0.75),
    ("造", "PRODUCES", 0.72),
    ("建", "PRODUCES", 0.72),
    ("组", "COMPOSED_OF", 0.75),
    ("构", "COMPOSED_OF", 0.72),
    ("致", "CAUSES", 0.75),
    ("引起", "CAUSES", 0.78),
    ("影响", "AFFECTS", 0.78),
    ("改变", "AFFECTS", 0.75),
    ("变化", "AFFECTS", 0.72),
    ("升高", "AFFECTS", 0.75),
    ("降低", "AFFECTS", 0.75),
    ("提高", "AFFECTS", 0.75),
    ("增加", "AFFECTS", 0.72),
    ("减少", "AFFECTS", 0.72),
    ("增长", "AFFECTS", 0.72),
    ("出口", "TRANSFERS_TO", 0.80),
    ("进口", "TRANSFERS_TO", 0.80),
    ("售", "TRANSFERS_TO", 0.75),
    ("供", "TRANSFERS_TO", 0.72),
    ("来到", "MOVED_TO", 0.80),
    ("到达", "MOVED_TO", 0.80),
    ("前往", "MOVED_TO", 0.78),
    ("进入", "MOVED_TO", 0.75),
    ("离开", "DEPARTED_FROM", 0.78),
    ("出发", "DEPARTED_FROM", 0.75),
    ("参观", "INTERACTS_WITH", 0.78),
    ("访问", "INTERACTS_WITH", 0.78),
    ("会见", "INTERACTS_WITH", 0.80),
    ("合作", "INTERACTS_WITH", 0.78),
    ("控制", "CONTROLS", 0.78),
    ("管理", "CONTROLS", 0.75),
    ("位于", "LOCATED_AT", 0.85),
    ("地处", "LOCATED_AT", 0.85),
    ("住在", "LOCATED_AT", 0.82),
    ("具有", "HAS_PROPERTY", 0.85),
    ("含有", "HAS_PROPERTY", 0.82),
    ("包含", "HAS_PROPERTY", 0.82),
    ("拥有", "HAS_PROPERTY", 0.85),
]

# ── DEP predicate → NSP predicate (only meaningful cross-entity edges) ──
_DEP_PRED_MAP = {
    "amod": ("HAS_PROPERTY", 0.80),
}


def _resolve_predicate(pred_text: str, a0_text: str = "", a1_text: str = "",
                       tokens: list | None = None) -> tuple[str, float]:
    """Resolve SRL predicate to NSP predicate with fallback chain.

    1. Exact match in _SRL_PRED_MAP → high confidence
    2. Substring match in _VERB_CLASS_FALLBACK → medium confidence
    3. Entity-type heuristic → low confidence
    4. Generic fallback → PROPERTY_OF (minimal confidence)
    """
    # 1. Exact match
    if pred_text in _SRL_PRED_MAP:
        return _SRL_PRED_MAP[pred_text]

    # 2. Verb-class substring fallback (longest match first)
    best_match: tuple[str, float] | None = None
    best_len = 0
    for keyword, pred, conf in _VERB_CLASS_FALLBACK:
        if keyword in pred_text and len(keyword) > best_len:
            best_match = (pred, conf)
            best_len = len(keyword)
    if best_match:
        return best_match

    # 3. Entity-type heuristic: if subject is LOCATION → LOCATED_AT
    #    This would need entity lookup — kept as future extension

    # 4. Generic fallback
    return ("PROPERTY_OF", 0.65)
    """Check if a token is meaningful content (not a function word)."""
    text = token.text if hasattr(token, 'text') else str(token)
    return text not in _FUNCTION_WORDS and len(text) >= 1


def _span_between(sp1: tuple, sp2: tuple, text: str) -> str:
    """Extract text covering two spans."""
    start = min(sp1[0], sp2[0])
    end = max(sp1[1], sp2[1])
    if start < 0 or end > len(text):
        return ""
    return text[start:end]


class RelationExtractionRules:
    """SRL-first, DEP-supplementary relation extractor."""

    def __init__(self):
        self._counter = 0

    # ── SRL Extraction (primary) ──

    def extract_from_srl(self, frame: list, tokens: list,
                         text: str = "") -> Optional[Relation]:
        """Extract relation from SRL frame (ARG0, PRED, ARG1)."""
        a0_text, a1_text = "", ""
        a0_span, a1_span = (0, 0), (0, 0)
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
            if "ARG0" in role:
                a0_text, a0_span = itxt, (cs, ce)
            elif "ARG1" in role:
                a1_text, a1_span = itxt, (cs, ce)
            elif role == "PRED":
                pred_text = itxt

        if not a0_text or not a1_text:
            return None

        pred, conf = _resolve_predicate(pred_text, a0_text, a1_text)

        evidence = _span_between(a0_span, a1_span, text)
        if not evidence:
            evidence = f"{a0_text} {pred_text} {a1_text}"

        self._counter += 1
        return Relation(
            id=f"rel_{self._counter:03d}",
            subject=a0_text.strip(),
            predicate=pred,
            object=a1_text.strip(),
            evidence=evidence,
            evidence_span=(a0_span[0], a1_span[1]),
            confidence=conf,
            source=f"srl/{pred_text}",
        )

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

        if not _is_content(ct) or not _is_content(ht):
            return None
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

    # ── Aggregate ──

    def extract_all(self, text: str, raw: dict, tokens: list) -> list:
        """Extract relations: SRL first, then supplementary DEP (amod)."""
        relations: list = []

        for f in raw.get("srl", []):
            if isinstance(f, list):
                rel = self.extract_from_srl(f, tokens, text)
                if rel:
                    relations.append(rel)

        for i, d in enumerate(raw.get("dep", [])):
            if isinstance(d, (list, tuple)) and len(d) >= 2:
                rel = self.extract_from_dep(i, str(d[1]), int(d[0]), tokens, text)
                if rel:
                    relations.append(rel)

        return relations
