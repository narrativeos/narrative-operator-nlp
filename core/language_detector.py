"""
Language Detector — Multi-Layer Confidence Scoring.

Classifies Chinese text as modern or classical using a three-layer
multi-feature scoring approach:

  Layer 1 — Character-Level: 虚词/代词/语气词密度、句末词、否定/疑问
  Layer 2 — Lexical/Grammatical: 古汉语搭配模式、词级特征、现代词汇惩罚
  Layer 3 — Syntactic Sequence: 句型模板、韵律模式、标点使用特征

Returns confidence scores used by the two-stage pipeline in analyzer.py
to decide which NLP model to apply first (and whether to fall back).
"""

from __future__ import annotations

import re
from typing import Literal

# ===========================================================================
# Layer 1 — Character-Level Feature Sets
# ===========================================================================

# Classical function words (虚词) — comprehensive set
_CLASSICAL_FUNCTION: frozenset[str] = frozenset(
    "之乎者也焉矣其乃若夫盖兮耳哉欤耶于以而则且与所为所虽"
)

# Classical sentence-final particles (语气词)
_CLASSICAL_FINAL: frozenset[str] = frozenset(
    "也矣焉耳乎哉欤耶兮夫而已云尔盖诸"
)

# Classical personal pronouns (人称代词)
_CLASSICAL_PRONOUNS: frozenset[str] = frozenset(
    "吾余予汝尔卿寡人朕孤臣妾仆某"
)

# Classical demonstratives / determiners (指示词)
_CLASSICAL_DEMONSTRATIVES: frozenset[str] = frozenset(
    "此是斯彼兹其厥"
)

# Classical copula / existential verbs (系词/存在词)
_CLASSICAL_COPULA_CHARS: frozenset[str] = frozenset(
    "为乃即系惟"
)

# Classical existential / possessive pattern markers
_CLASSICAL_EXISTENTIAL: frozenset[str] = frozenset("有无存")

# Classical negation characters (否定词)
_CLASSICAL_NEGATION_CHARS: frozenset[str] = frozenset(
    "不未弗毋勿非微莫罔无"
)

# Classical interrogative characters (疑问词)
_CLASSICAL_INTERROG_CHARS: frozenset[str] = frozenset(
    "何胡奚曷安焉恶孰谁盍讵岂那"
)

# Classical measure words (古典量词)
_CLASSICAL_MEASURE: frozenset[str] = frozenset("里尺寸丈寻仞斗升石钧顷亩")

# Classical honorific / self-deprecating
_CLASSICAL_HONORIFIC: frozenset[str] = frozenset("陛下殿下阁下足下寡人臣妾")

# Classical quoting verbs
_CLASSICAL_QUOTE: frozenset[str] = frozenset("曰云谓")

# Modern Chinese particles (现代助词) — strong negative signal
_MODERN_PARTICLES: frozenset[str] = frozenset("的了着们过吗吧呢啊嘛")

# Modern Chinese pronouns (现代人称代词)
_MODERN_PRONOUNS: frozenset[str] = frozenset("我你他她它")

# Modern Chinese copula
_MODERN_COPULA: frozenset[str] = frozenset("是")

# Modern Chinese measure words
_MODERN_MEASURE: frozenset[str] = frozenset("个只条张把块本次")


# ===========================================================================
# Layer 2 — Lexical/Grammatical Patterns
# ===========================================================================

# Classical bigram patterns (strong classical indicators)
_CLASSICAL_BIGRAMS: tuple[str, ...] = (
    "其名", "名为", "谓之", "所谓", "是以", "何以", "然则",
    "若夫", "至若", "且夫", "盖夫", "夫唯",
    "故曰", "或曰", "子曰", "诗云", "书曰",
    "者乎", "者欤", "者耶", "者邪", "者哉",
    "之谓", "之为", "之至", "之大", "之多", "之远",
    "不下", "不止", "不多", "不胜", "不啻",
    "未有", "未尝", "未始", "未能",
    "有以", "无以", "可以", "足以", "难以",
    "之所以", "之所以然",
    "向北", "以南", "之内", "之外",
    "何如", "何若", "若何", "奈何",
    "于是", "至于", "及至", "至于",
    "此之", "彼之", "其之", "斯之",
    "君子", "小人", "圣人", "贤人", "仁者",
    "天下", "四海", "九州", "万民", "百姓",
    "德行", "仁义", "礼乐", "忠信", "孝悌",
    "天地", "阴阳", "万物", "大道",
    "必先", "然后", "而后", "是以故",
    "何谓", "安得", "可得", "岂不",
    "不若", "莫若",
)

# Classical trigram patterns
_CLASSICAL_TRIGRAMS: tuple[str, ...] = (
    "不亦乐", "不亦说", "何以故", "何故也", "是以故",
    "之谓也", "之为言", "之所以", "之所有",
    "未有以", "无以异", "有以异", "不足以",
    "未之有", "莫之能", "莫之敢",
    "岂非以", "岂能以", "岂可不",
    "之谓乎", "之谓欤", "之谓也",
    "何难之", "何忧之", "何患之",
    "何其大", "何其远", "何其盛",
    "自古以", "由是以", "是故",
    "不得已", "不可不", "不得不",
    "如之何", "若之何", "奈之何",
)

# Modern Chinese word patterns (strong negative signal)
_MODERN_WORDS: tuple[str, ...] = (
    "我们", "你们", "他们", "她们", "它们",
    "这个", "那个", "哪个", "这些", "那些",
    "因为", "所以", "而且", "但是", "虽然", "如果", "可以",
    "应该", "已经", "正在", "什么", "怎么", "为什么",
    "非常", "比较", "特别", "尤其", "更加",
    "通过", "根据", "按照", "对于", "关于",
    "一个", "一种", "一样", "一起", "一些",
    "进行", "实现", "发展", "建设", "提高",
    "记载", "认为", "研究", "表明", "分析",
    "所谓", "其中", "例如", "包括",
)

# Modern sentence-final particles (strong modern signal)
_MODERN_FINAL: frozenset[str] = frozenset("吧吗呢嘛啊啦呀噢喽咯")


# ===========================================================================
# Layer 3 — Syntactic Sequence Patterns
# ===========================================================================

# Classical sentence template patterns (regex)
_CLASSICAL_TEMPLATES: list[re.Pattern] = [
    # 者...也 copula pattern
    re.compile(r"者[^。！？\n]{1,30}也"),
    # Subject + 之 + Noun (possessive)
    re.compile(r"[\u4e00-\u9fff]之[\u4e00-\u9fff]"),
    # Verb + 于 + Noun (locative/dative) — exclude modern "位于"/"处于"
    re.compile(r"(?:[^位处][\u4e00-\u9fff]|[\u4e00-\u9fff]{2})于[\u4e00-\u9fff]"),
    # 以...为... pattern
    re.compile(r"以[\u4e00-\u9fff]{1,4}为"),
    # 何...之有 pattern
    re.compile(r"何[\u4e00-\u9fff]{1,6}之有"),
    # 不亦...乎 pattern
    re.compile(r"不亦[\u4e00-\u9fff]{1,6}乎"),
    # 岂...哉 pattern
    re.compile(r"岂[\u4e00-\u9fff]{1,8}哉"),
    # 非...而何 pattern
    re.compile(r"非[\u4e00-\u9fff]{1,8}而何"),
    # 莫...于 pattern
    re.compile(r"莫[\u4e00-\u9fff]{1,6}于"),
    # 未有...者
    re.compile(r"未有[\u4e00-\u9fff]{1,12}者"),
    # ...者，...也 (comma-separated copula)
    re.compile(r"者[，,][^。！？\n]{1,30}也"),
    # ...以为... (classical "take...as")
    re.compile(r"[\u4e00-\u9fff]{1,6}以为"),
    # 故...也 (therefore...)
    re.compile(r"故[\u4e00-\u9fff]{1,20}也"),
    # Subject+有+Noun (classical existential)
    re.compile(r"[\u4e00-\u9fff]{1,4}有[\u4e00-\u9fff]{1,4}[，。；、]"),
    # 可X可X pattern (classical antithesis)
    re.compile(r"可[\u4e00-\u9fff]可[\u4e00-\u9fff]"),
    # 非X非X pattern (classical antithesis)  
    re.compile(r"非[\u4e00-\u9fff]非[\u4e00-\u9fff]"),
    # Subject+以+V pattern (classical instrumental)
    re.compile(r"[\u4e00-\u9fff]{1,4}以[\u4e00-\u9fff]{1,4}"),
]

# Modern text template patterns (negative signal)
_MODERN_TEMPLATES: list[re.Pattern] = [
    # 的+noun (modern possessive)
    re.compile(r"的[\u4e00-\u9fff]"),
    # 了+punctuation (modern perfective)
    re.compile(r"了[，。！？\n]"),
    # subject+们 (modern plural)
    re.compile(r"[\u4e00-\u9fff]们"),
    # 正在+verb (modern progressive)
    re.compile(r"正在[\u4e00-\u9fff]"),
    # numbers with units (modern measurement)
    re.compile(r"\d+[个只条张块次]"),
    # 是...的 (modern emphatic construction)
    re.compile(r"是[\u4e00-\u9fff]{1,20}的"),
    # 位于/处于 (modern locative verbs — not classical 于)
    re.compile(r"[位处]于"),
]

# Classical rhythm patterns: 4-character blocks are highly characteristic
# of classical Chinese prose
_CLASSICAL_RHYTHM = re.compile(r"[\u4e00-\u9fff]{4}[，。；、]")

# Presence of Arabic digits / Latin letters (strong modern signal)
_MODERN_ALPHANUM = re.compile(r"[a-zA-Z0-9]")

# ===========================================================================
# English Detection Feature Sets
# ===========================================================================

# English common words (to confirm English text)
_ENGLISH_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "shall",
    "this", "that", "these", "those", "it", "its",
    "i", "me", "my", "we", "us", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their",
    "not", "no", "nor", "but", "and", "or", "for", "so", "yet",
    "if", "then", "else", "when", "where", "what", "which", "who",
    "how", "why", "all", "each", "every", "some", "any", "many",
    "much", "few", "more", "most", "other", "such", "only", "own",
    "in", "on", "at", "to", "for", "with", "by", "from", "of",
    "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "over", "without",
    "one", "two", "three", "first", "last", "next",
    "here", "there", "now", "then", "always", "never", "often",
    "very", "too", "also", "just", "still", "already", "almost",
})

# English structural patterns
_ENGLISH_CONJUNCTIONS: frozenset[str] = frozenset({
    "because", "although", "while", "since", "unless",
    "whereas", "moreover", "furthermore", "nevertheless",
    "consequently", "therefore", "accordingly", "besides",
    "likewise", "meanwhile", "otherwise", "nonetheless",
})

# ASCII letter ratio threshold — if >= this ratio, text is probably English
_ENGLISH_ASCII_RATIO = 0.6


# ===========================================================================
# Multi-Layer Scoring Engine
# ===========================================================================

def classical_confidence(text: str) -> float:
    """
    Compute a confidence score [0.0, 1.0] that ``text`` is classical Chinese,
    using a three-layer multi-feature analysis.

    0.0 = definitively modern
    0.5 = ambiguous
    1.0 = definitively classical

    Layer 1 (character):  虚词密度 / 句末词 / 人称 / 系词 / 否定/疑问  → max +0.52
    Layer 2 (lexical):    古汉语搭配 / 词级模式 / 现代词汇惩罚          → max ±0.42
    Layer 3 (syntactic):  句型模板 / 韵律 / 标点 / 敬语                → max ±0.32
    """
    text = text.strip()
    n = len(text)
    if n < 3:
        return 0.0

    # Strip punctuation for character-level analysis
    text_no_punct = re.sub(r"[，。！？；、：\s]", "", text)
    len_no_punct = len(text_no_punct)
    if len_no_punct == 0:
        return 0.0

    score = 0.0

    # ================================================================
    # Layer 1 — Character-Level Signals (max +0.52, -0.25)
    # ================================================================

    # 1a. Classical function word density (0.0–0.24)
    func_count = sum(1 for c in text_no_punct if c in _CLASSICAL_FUNCTION)
    func_density = func_count / max(len_no_punct, 1)
    score += min(func_density * 5.0, 0.24)

    # 1b. Classical sentence-final particles (0.0–0.10)
    final_count = sum(1 for c in text_no_punct if c in _CLASSICAL_FINAL)
    last_is_final = 1.0 if text_no_punct and text_no_punct[-1] in _CLASSICAL_FINAL else 0.0
    score += min(final_count * 0.04 + last_is_final * 0.04, 0.10)

    # 1c. Classical pronouns & demonstratives (0.0–0.10)
    pronoun_count = sum(1 for c in text_no_punct if c in _CLASSICAL_PRONOUNS)
    demonstr_count = sum(1 for c in text_no_punct if c in _CLASSICAL_DEMONSTRATIVES)
    score += min(pronoun_count * 0.05 + demonstr_count * 0.04, 0.10)

    # 1d. Classical copula & existential verbs (0.0–0.06)
    copula_count = sum(1 for c in text_no_punct
                       if c in _CLASSICAL_COPULA_CHARS
                       and not (c == "是" and ("于是" in text or "是故" in text)))
    existential_count = sum(1 for c in text_no_punct if c in _CLASSICAL_EXISTENTIAL)
    score += min(copula_count * 0.03 + existential_count * 0.03, 0.06)

    # 1e. Classical quoting verbs (0.0–0.04)
    quote_count = sum(1 for c in text_no_punct if c in _CLASSICAL_QUOTE)
    score += min(quote_count * 0.04, 0.04)

    # --- Negative signals (modern character-level) ---

    # 1f. Modern particle penalty (0.0–0.12)
    modern_part_count = sum(1 for c in text_no_punct if c in _MODERN_PARTICLES)
    score -= min(modern_part_count * 0.05, 0.12)

    # 1g. Modern pronoun penalty (0.0–0.08)
    modern_pronoun_count = sum(1 for c in text_no_punct if c in _MODERN_PRONOUNS)
    score -= min(modern_pronoun_count * 0.04, 0.08)

    # 1h. Modern copula penalty (0.0–0.05)
    modern_copula_count = sum(1 for c in text_no_punct if c in _MODERN_COPULA)
    score -= min(modern_copula_count * 0.05, 0.05)

    # 1i. Arabic digits / Latin letters (strong modern, -0.08)
    if _MODERN_ALPHANUM.search(text):
        score -= 0.08

    # 1j. Modern punctuation markers: 《》 (book titles, strong modern signal)
    if "《" in text or "》" in text:
        score -= 0.06

    # ================================================================
    # Layer 2 — Lexical/Grammatical Patterns (max +0.42, -0.22)
    # ================================================================

    # 2a. Classical bigram matches (0.0–0.15)
    bigram_hits = sum(1 for bg in _CLASSICAL_BIGRAMS if bg in text)
    score += min(bigram_hits * 0.03, 0.15)

    # 2b. Classical trigram matches (0.0–0.10)
    trigram_hits = sum(1 for tg in _CLASSICAL_TRIGRAMS if tg in text)
    score += min(trigram_hits * 0.05, 0.10)

    # 2c. Classical negation + final particle pairing (0.0–0.07)
    negation_count = sum(1 for c in text_no_punct if c in _CLASSICAL_NEGATION_CHARS)
    if negation_count > 0:
        neg_bonus = negation_count * 0.02
        if final_count > 0:
            neg_bonus += 0.03
        score += min(neg_bonus, 0.07)

    # 2d. Classical interrogative (0.0–0.05)
    interrog_count = sum(1 for c in text_no_punct if c in _CLASSICAL_INTERROG_CHARS)
    score += min(interrog_count * 0.05, 0.05)

    # 2e. Classical measure words (0.0–0.03)
    classical_measure_count = sum(1 for c in text_no_punct if c in _CLASSICAL_MEASURE)
    score += min(classical_measure_count * 0.03, 0.03)

    # 2f. Short sentence classical density bonus (0.0–0.05)
    if len_no_punct <= 15:
        classical_char_count = (
            func_count + final_count + pronoun_count + demonstr_count +
            copula_count + existential_count + interrog_count
        )
        classical_density = classical_char_count / max(len_no_punct, 1)
        if classical_density >= 0.10:
            score += min(0.02 + classical_density * 0.15, 0.05)

    # --- Negative signals (modern lexical) ---

    # 2g. Modern word penalty (0.0–0.12)
    modern_word_count = sum(1 for w in _MODERN_WORDS if w in text)
    score -= min(modern_word_count * 0.04, 0.12)

    # 2h. Modern sentence-final particle penalty (0.0–0.05)
    modern_final_count = sum(1 for c in text_no_punct if c in _MODERN_FINAL)
    score -= min(modern_final_count * 0.04, 0.05)

    # 2i. Modern measure word penalty (0.0–0.03)
    modern_measure_count = sum(1 for c in text_no_punct if c in _MODERN_MEASURE)
    score -= min(modern_measure_count * 0.02, 0.03)

    # ================================================================
    # Layer 3 — Syntactic Sequence Analysis (max +0.32, -0.12)
    # ================================================================

    # 3a. Classical template matches (0.0–0.20)
    template_hits = sum(1 for tmpl in _CLASSICAL_TEMPLATES if tmpl.search(text))
    score += min(template_hits * 0.07, 0.20)

    # 3b. Four-character rhythm blocks (0.0–0.10)
    rhythm_blocks = len(_CLASSICAL_RHYTHM.findall(text))
    rhythm_score = min(rhythm_blocks * 0.04, 0.06)
    if len_no_punct <= 20 and rhythm_blocks >= 1:
        rhythm_score += 0.04
    score += min(rhythm_score, 0.10)

    # 3c. Punctuation usage profile (0.0–0.04)
    comma_count = text.count("，") + text.count(",")
    period_count = text.count("。") + text.count(".")
    if period_count > 0:
        comma_to_period = comma_count / period_count
        if comma_to_period < 1.5:
            score += 0.02
        if comma_to_period < 0.8:
            score += 0.02

    # 3d. Classical honorific pattern (0.0–0.02)
    if any(h in text for h in _CLASSICAL_HONORIFIC):
        score += 0.02

    # --- Negative signals (modern syntactic) ---

    # 3e. Modern template penalty (0.0–0.08)
    modern_tmpl_hits = sum(1 for mt in _MODERN_TEMPLATES if mt.search(text))
    score -= min(modern_tmpl_hits * 0.04, 0.08)

    # 3f. Modern measure + copula co-occurrence (0.0–0.04)
    if modern_measure_count > 0 and modern_copula_count > 0:
        score -= 0.02
    if modern_measure_count > 1:
        score -= 0.02

    return max(0.0, min(1.0, score))


# ===========================================================================
# Mixed-Signal Penalty
# ===========================================================================

def _apply_mixed_penalty(score: float, text: str, text_no_punct: str) -> float:
    """
    If a sentence has both classical and modern signals, it is likely
    a modern sentence that happens to contain classical-style words
    (e.g., 北京立方庭位于海淀区 where 位于 looks classical but isn't).

    This penalty ensures that mixed sentences default to modern unless
    classical signals are overwhelmingly dominant.
    """
    # Count classical signal types present
    classical_present = 0
    if any(c in _CLASSICAL_FUNCTION for c in text_no_punct):
        classical_present += 1
    if any(c in _CLASSICAL_FINAL for c in text_no_punct):
        classical_present += 1
    if any(c in _CLASSICAL_PRONOUNS for c in text_no_punct):
        classical_present += 1
    if any(c in _CLASSICAL_DEMONSTRATIVES for c in text_no_punct):
        classical_present += 1
    if any(c in _CLASSICAL_COPULA_CHARS for c in text_no_punct):
        classical_present += 1

    # Count modern signal types present (with classical exception for 于是/是故)
    modern_present = 0
    if any(c in _MODERN_PARTICLES for c in text_no_punct):
        modern_present += 1
    if any(c in _MODERN_PRONOUNS for c in text_no_punct):
        modern_present += 1
    # "是" only counts as modern if NOT in classical compound "于是"/"是故"
    if any(c in _MODERN_COPULA for c in text_no_punct):
        if "于是" not in text_no_punct and "是故" not in text_no_punct:
            modern_present += 1
    # Modern book-title brackets (《》, strong modern signal)
    if "《" in text or "》" in text:
        modern_present += 1
    # Modern words found in the text
    if any(w in text for w in _MODERN_WORDS):
        modern_present += 1

    if modern_present >= 1 and classical_present >= 1:
        # Mixed signals — penalize proportionally to modern signal strength
        # More modern signals = stronger penalty
        if modern_present >= 2:
            return score * 0.5  # Strong modern presence: halve the score
        return score * 0.65  # Single modern signal: reduce significantly

    return score


# ===========================================================================
# Three-Way Language Detection
# ===========================================================================

LanguageClass = Literal["modern", "classical", "english"]


def detect_language(text: str) -> tuple[LanguageClass, float]:
    """
    Three-way language detection.

    Returns (language, confidence):
      ("english",   [0,1])   — English text
      ("classical", [0,1])   — Classical Chinese
      ("modern",    [0,1])   — Modern Chinese (default)

    Detection priority: english → classical → modern.
    """
    text = text.strip()
    if len(text) < 3:
        return ("modern", 0.0)

    # ---------- English detection (high priority) ----------
    # Count ASCII letters vs total content characters
    letters = sum(1 for c in text if c.isascii() and c.isalpha())
    total_content = sum(1 for c in text if c.isalpha())
    if total_content > 0:
        ascii_ratio = letters / total_content
        if ascii_ratio >= _ENGLISH_ASCII_RATIO:
            # Confirm with English stop words
            words = text.lower().split()
            stop_hits = sum(1 for w in words if w.strip(".,!?;:'\"()[]") in _ENGLISH_STOP_WORDS)
            # Strong English signal: high ASCII ratio + stop words
            if stop_hits >= 1:
                eng_conf = min(0.5 + stop_hits * 0.05, 0.95)
                return ("english", eng_conf)
            # Pure ASCII with very high ratio but no stop words → likely English
            if ascii_ratio >= 0.85:
                return ("english", 0.60)

    # ---------- Classical Chinese detection ----------
    # Exclude English texts entirely
    if total_content > 0 and letters / total_content > 0.3:
        return ("modern", 0.0)

    conf = classical_confidence(text)
    text_no_punct = re.sub(r"[，。！？；、：\s]", "", text)
    conf = _apply_mixed_penalty(conf, text, text_no_punct)

    if conf >= 0.48:
        return ("classical", conf)
    return ("modern", conf)
