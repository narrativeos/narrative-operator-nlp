"""
Modifier Extractor — Extract relation modifiers from evidence text.

Uses built-in dictionaries (with user customization support) to identify
modifier words in the evidence text surrounding a relation.

Design principles:
- Built-in dictionaries for common modifiers (transparent, auditable)
- User can extend/override with custom dictionaries
- Each modifier includes matched_dict source for transparency
"""

from __future__ import annotations

from typing import Optional

from .schema import Relation, RelationModifier


# ── Built-in modifier dictionaries ──
# These are general-purpose modifiers that work across domains.
# Users can extend or override via the modifier_dict parameter.

_BUILTIN_MODIFIER_DICT: dict[str, frozenset[str]] = {
    "degree": frozenset({
        # Chinese degree modifiers
        "非常", "极其", "略微", "相当", "十分", "特别", "极为", "颇",
        "高度", "深度", "极度", "超", "最", "更", "较", "较为", "更为",
        # Classical Chinese degree modifiers (古汉语程度副词)
        "甚", "极", "至", "颇", "稍", "略", "微", "绝", "殊", "尤", "至为", "最为",
        # English degree modifiers
        "very", "extremely", "slightly", "quite", "rather", "highly",
        "deeply", "utterly", "totally", "completely", "absolutely",
        "particularly", "especially", "especially", "most", "more",
    }),
    "scope": frozenset({
        # Chinese scope modifiers
        "主要", "部分", "全部", "几乎", "大部分", "少数", "所有", "一切",
        "整体", "局部", "广泛", "普遍",
        # Classical Chinese scope modifiers (古汉语范围副词)
        "悉", "皆", "俱", "咸", "尽", "毕", "总", "俱", "咸", "并", "俱",
        # English scope modifiers
        "mainly", "partially", "mostly", "entirely", "wholly",
        "generally", "widely", "universally", "largely",
    }),
    "negation": frozenset({
        # Chinese negation
        "不", "未", "非", "无", "没", "没有", "并非", "不曾", "勿", "别",
        # Classical Chinese negation (古汉语否定词)
        "弗", "毋", "莫", "罔", "微", "未尝", "未始", "未能", "不曾",
        # English negation
        "not", "never", "no", "none", "neither", "nor", "n't",
    }),
    "quantity": frozenset({
        # Chinese quantity approximations
        "约", "大约", "超过", "不足", "近", "将近", "左右", "上下",
        "多", "余", "以上", "以下", "约莫", "大概",
        # Classical Chinese quantity approximations (古汉语数量近似)
        "盖", "大抵", "略", "稍", "约", "几", "将", "且",
        # English quantity approximations
        "approximately", "about", "around", "over", "under", "below",
        "above", "roughly", "nearly", "almost", "exactly", "precisely",
    }),
    "temporal": frozenset({
        # Chinese temporal modifiers
        "曾经", "目前", "将来", "一直", "仍然", "已经", "正在", "曾",
        "尚", "还", "仍", "已", "将", "即将", "曾经", "过去", "现在",
        "当前", "今后", "未来", "以前", "之前", "之后",
        # Classical Chinese temporal modifiers (古汉语时态副词)
        "尝", "曾", "已", "既", "方", "正", "将", "欲", "昔", "今", "后",
        "向", "曩", "遽", "忽", "俄", "旋", "寻", "既而", "须臾",
        # English temporal modifiers
        "currently", "previously", "still", "already", "already",
        "formerly", "now", "presently", "recently", "soon", "later",
        "before", "after", "once", "until", "since",
    }),
    "comparison": frozenset({
        # Chinese comparison
        "比", "较", "较为", "更为", "相对", "相较于", "相比",
        # Classical Chinese comparison (古汉语比较)
        "逾", "过", "胜", "若", "如", "似", "堪比",
        # English comparison
        "compared", "versus", "vs", "relative", "relatively",
    }),
    "emphasis": frozenset({
        # Chinese emphasis
        "确实", "真的", "的确", "实在", "的确", "确", "确是",
        # Classical Chinese emphasis (古汉语强调)
        "诚", "信", "固", "本", "原", "实", "乃", "即", "即",
        # English emphasis
        "indeed", "actually", "really", "truly", "certainly",
        "definitely", "undoubtedly",
    }),
    # Classical Chinese interrogative (古汉语疑问词)
    "interrogative": frozenset({
        "何", "胡", "奚", "曷", "安", "焉", "孰", "盍", "讵", "岂",
        "何故", "何以", "何如", "何若", "若何", "奈何",
    }),
}


class ModifierExtractor:
    """Extract relation modifiers from evidence text using dictionary matching.

    Supports both built-in and user-provided modifier dictionaries.
    Each extracted modifier includes a matched_dict field indicating
    whether it came from the built-in or custom dictionary.
    """

    def __init__(
        self,
        modifier_dict: dict[str, set[str]] | None = None,
    ):
        """Initialize the modifier extractor.

        Args:
            modifier_dict: Optional user-provided modifier dictionary.
                Keys are modifier types (degree, scope, negation, etc.).
                Values are sets of modifier words.
                These words are merged with (extend) the built-in dictionary.
        """
        # Start with built-in dictionaries (convert to mutable sets)
        self._builtin: dict[str, set[str]] = {
            k: set(v) for k, v in _BUILTIN_MODIFIER_DICT.items()
        }
        self._custom: dict[str, set[str]] = {}

        # Merge user-provided dictionary
        if modifier_dict:
            for mtype, words in modifier_dict.items():
                if mtype in self._builtin:
                    # Extend built-in with custom words
                    self._builtin[mtype].update(words)
                else:
                    # New modifier type
                    self._custom[mtype] = set(words)

        # Build reverse lookup: word -> (type, source)
        self._word_to_info: dict[str, tuple[str, str]] = {}
        for mtype, words in self._builtin.items():
            for word in words:
                if word not in self._word_to_info:
                    self._word_to_info[word] = (mtype, "builtin")
        for mtype, words in self._custom.items():
            for word in words:
                if word not in self._word_to_info:
                    self._word_to_info[word] = (mtype, "custom")

    def extract(
        self,
        text: str,
        rel: Relation,
    ) -> list[RelationModifier]:
        """Extract modifiers from the relation's evidence context.

        Searches the original text around the relation's evidence span
        for modifier words.

        Args:
            text: The full original text.
            rel: The relation to extract modifiers for.

        Returns:
            List of RelationModifier objects found in the evidence context.
        """
        modifiers: list[RelationModifier] = []

        # Search in a window around the evidence span
        # Expand by 10 characters on each side to catch nearby modifiers
        start = max(0, rel.evidence_span[0] - 10)
        end = min(len(text), rel.evidence_span[1] + 10)
        context = text[start:end]

        # Track matched positions to avoid duplicates
        matched_spans: list[tuple[int, int]] = []

        for word, (mtype, source) in self._word_to_info.items():
            pos = 0
            while pos < len(context):
                idx = context.find(word, pos)
                if idx == -1:
                    break

                abs_start = start + idx
                abs_end = abs_start + len(word)

                # Check for overlap with already matched spans
                overlap = False
                for ms, me in matched_spans:
                    if abs_start < me and abs_end > ms:
                        overlap = True
                        break
                if overlap:
                    pos = abs_start + 1
                    continue

                modifiers.append(RelationModifier(
                    text=word,
                    type=mtype,
                    span=(abs_start, abs_end),
                    matched_dict=source,
                ))
                matched_spans.append((abs_start, abs_end))
                pos = abs_end

        # Sort by position in text
        modifiers.sort(key=lambda m: m.span[0])
        return modifiers

    def extract_batch(
        self,
        text: str,
        relations: list[Relation],
    ) -> None:
        """Extract modifiers for a batch of relations in-place.

        Args:
            text: The full original text.
            relations: List of relations to process (modified in-place).
        """
        for rel in relations:
            rel.modifiers = self.extract(text, rel)

    @staticmethod
    def get_builtin_dict() -> dict[str, frozenset[str]]:
        """Return a copy of the built-in modifier dictionary for inspection.

        Users can use this to understand what's included and decide
        what to extend or override.
        """
        return dict(_BUILTIN_MODIFIER_DICT)