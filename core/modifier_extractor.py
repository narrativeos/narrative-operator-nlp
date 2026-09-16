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

import bisect
from typing import Optional

from .schema import Relation, RelationModifier, Token


# ── Built-in modifier dictionaries ──
# These are general-purpose modifiers that work across domains.
# Users can extend or override via the modifier_dict parameter.

_BUILTIN_MODIFIER_DICT: dict[str, frozenset[str]] = {
    "degree": frozenset({
        # Chinese degree modifiers
        "非常", "极其", "略微", "相当", "十分", "特别", "极为", "颇",
        "高度", "深度", "极度", "超", "最", "更", "较", "较为", "更为",
        # Classical Chinese degree modifiers (古汉语程度副词)
        # NOTE: "至" (preposition 介词) and "微" (negation word 否定词) are
        # function words (虚词), not degree adverbs — removed. The adverbial
        # use of "至" is covered by "至为".
        "甚", "极", "稍", "略", "绝", "殊", "尤", "至为", "最为",
        # English degree modifiers
        "very", "extremely", "slightly", "quite", "rather", "highly",
        "deeply", "utterly", "totally", "completely", "absolutely",
        "particularly", "especially", "most", "more",
    }),
    "scope": frozenset({
        # Chinese scope modifiers
        "主要", "部分", "全部", "几乎", "大部分", "少数", "所有", "一切",
        "整体", "局部", "广泛", "普遍",
        # Classical Chinese scope modifiers (古汉语范围副词)
        # NOTE: "并" is a conjunction (连词), not a scope adverb — removed.
        "悉", "皆", "俱", "咸", "尽", "毕", "总",
        # English scope modifiers
        "mainly", "partially", "mostly", "entirely", "wholly",
        "generally", "widely", "universally", "largely",
    }),
    "negation": frozenset({
        # Chinese negation
        "不", "未", "非", "无", "没", "没有", "并非", "不曾", "勿", "别",
        # Classical Chinese negation (古汉语否定词)
        "弗", "毋", "莫", "罔", "微", "未尝", "未始", "未能",
        # English negation
        "not", "never", "no", "none", "neither", "nor", "n't",
    }),
    "quantity": frozenset({
        # Chinese quantity approximations
        # NOTE: "余" (first-person pronoun 第一人称代词) is a function word — removed.
        "约", "大约", "超过", "不足", "近", "将近", "左右", "上下",
        "多", "以上", "以下", "约莫", "大概",
        # Classical Chinese quantity approximations (古汉语数量近似)
        # NOTE: "盖" (modal particle 语气词), "几" (interrogative pronoun
        # 疑问代词) and "且" (conjunction 连词) are function words — removed.
        # "将" is kept but POS-gated (see _POS_GATED_WORDS).
        "大抵", "略", "稍", "约", "将",
        # English quantity approximations
        "approximately", "about", "around", "over", "under", "below",
        "above", "roughly", "nearly", "almost", "exactly", "precisely",
    }),
    "temporal": frozenset({
        # Chinese temporal modifiers
        "曾经", "目前", "将来", "一直", "仍然", "已经", "正在", "曾",
        "尚", "还", "仍", "已", "将", "即将", "过去", "现在",
        "当前", "今后", "未来", "以前", "之前", "之后",
        # Classical Chinese temporal modifiers (古汉语时态副词)
        # NOTE: "向" (preposition 介词) is a function word — removed.
        # "将"/"既" are kept but POS-gated (see _POS_GATED_WORDS).
        "尝", "既", "方", "正", "欲", "昔", "今", "后",
        "曩", "遽", "忽", "俄", "旋", "寻", "既而", "须臾",
        # English temporal modifiers
        "currently", "previously", "still", "already",
        "formerly", "now", "presently", "recently", "soon", "later",
        "before", "after", "once", "until", "since",
    }),
    "comparison": frozenset({
        # Chinese comparison
        # NOTE: "比" is a preposition (介词), not a comparison adverb — removed.
        "较", "较为", "更为", "相对", "相较于", "相比",
        # Classical Chinese comparison (古汉语比较)
        # NOTE: "若"/"如"/"似" are prepositions/conjunctions (介词/连词) — removed.
        "逾", "过", "胜", "堪比",
        # English comparison
        "compared", "versus", "vs", "relative", "relatively",
    }),
    "emphasis": frozenset({
        # Chinese emphasis
        "确实", "真的", "的确", "实在", "确", "确是",
        # Classical Chinese emphasis (古汉语强调)
        # NOTE: "乃"/"即" also serve as pronoun/conjunction (虚词) in other
        # contexts — kept but POS-gated (see _POS_GATED_WORDS).
        "诚", "信", "固", "本", "原", "实", "乃", "即",
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

# ── POS gating for polysemous words ──
# These words have BOTH adverbial and function-word (虚词) uses:
#   将 — adverb/auxiliary "将要" vs preposition "把" (如：将计就计)
#   既 — adverb "已经" vs conjunction "既…又" (如：既来之则安之)
#   乃 — adverb "于是/就是" vs pronoun "你" (如：乃不知有汉)
#   即 — adverb "立即/就是" vs conjunction "即使"
# When token POS info is available, they only match if the covering token
# is tagged with an adverb-compatible POS. Without POS info they match as
# before (backward compatible).
_POS_GATED_WORDS: frozenset[str] = frozenset({"将", "既", "乃", "即"})

# Adverb-compatible POS tags (Universal POS + CTB).
# CTB tags auxiliaries (将/即 as 助动词) as VV, hence VV is included.
_ADVERBIAL_POS: frozenset[str] = frozenset({"ADV", "AUX", "AD", "VV"})


def _build_pos_index(
    tokens: Optional[list[Token]],
) -> Optional[tuple[list[int], list[Token]]]:
    """Build (starts, tokens) sorted by span start for O(log n) POS lookup."""
    if not tokens:
        return None
    sorted_tokens = sorted(tokens, key=lambda t: t.span[0])
    return [t.span[0] for t in sorted_tokens], sorted_tokens


def _pos_allows_adverbial(
    pos_index: Optional[tuple[list[int], list[Token]]],
    abs_start: int,
    abs_end: int,
) -> bool:
    """Check whether the token covering [abs_start, abs_end) is adverbial.

    Returns True when POS info is unavailable (no index, no covering token,
    or the match does not align with a single token) to preserve recall.
    Returns False only when a covering token is tagged with a non-adverbial
    POS (preposition/conjunction/pronoun/...), i.e. the word is used as a
    function word (虚词) at this position.
    """
    if pos_index is None:
        return True
    starts, sorted_tokens = pos_index
    i = bisect.bisect_right(starts, abs_start) - 1
    if i < 0:
        return True
    t = sorted_tokens[i]
    if t.span[0] > abs_start or abs_end > t.span[1]:
        return True  # not aligned with a single token; cannot validate
    return t.pos in _ADVERBIAL_POS


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
        tokens: Optional[list[Token]] = None,
    ) -> list[RelationModifier]:
        """Extract modifiers from the relation's evidence context.

        Searches the original text around the relation's evidence span
        for modifier words.

        Args:
            text: The full original text.
            rel: The relation to extract modifiers for.
            tokens: Optional token list with POS tags. When provided,
                polysemous words (see _POS_GATED_WORDS) only match if the
                covering token is tagged with an adverb-compatible POS,
                preventing function-word (虚词) uses from being extracted
                as adverbial modifiers.

        Returns:
            List of RelationModifier objects found in the evidence context.
        """
        return self._extract(text, rel, _build_pos_index(tokens))

    def _extract(
        self,
        text: str,
        rel: Relation,
        pos_index: Optional[tuple[list[int], list[Token]]],
    ) -> list[RelationModifier]:
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

                # POS gate: skip polysemous words used as function words
                if word in _POS_GATED_WORDS and not _pos_allows_adverbial(
                    pos_index, abs_start, abs_end
                ):
                    pos = abs_start + 1
                    continue

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
        tokens: Optional[list[Token]] = None,
    ) -> None:
        """Extract modifiers for a batch of relations in-place.

        Args:
            text: The full original text.
            relations: List of relations to process (modified in-place).
            tokens: Optional token list with POS tags (see extract()).
        """
        pos_index = _build_pos_index(tokens)
        for rel in relations:
            rel.modifiers = self._extract(text, rel, pos_index)

    @staticmethod
    def get_builtin_dict() -> dict[str, frozenset[str]]:
        """Return a copy of the built-in modifier dictionary for inspection.

        Users can use this to understand what's included and decide
        what to extend or override.
        """
        return dict(_BUILTIN_MODIFIER_DICT)