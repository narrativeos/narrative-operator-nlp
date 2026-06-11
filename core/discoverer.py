"""
New Word Discovery — Hybrid PMI + MTL Comparison.

Three strategies combined:
1. PMI + LLR + Left/Right Entropy (statistical, needs enough text).
2. MTL diff: compare MTL tokenizer output against known patterns —
   sequences of low-confidence / unknown-character tokens are candidates.
3. Internal cohesion test: check if characters within a candidate are
   tightly bound (low internal entropy = high cohesion).

Works on both single sentences (MTL diff) and longer texts (PMI).

References
----------
- Sun Maosong et al., "Chinese New Word Identification: A Statistical Approach"
- HanLP 1.x ``com.hankcs.hanlp.mining.word.NewWordDiscover``
- Duan et al., "New Word Discovery with Log-Likelihood Ratio Test"
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .schema import Token


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A discovered word candidate."""

    word: str
    score: float
    pmi: float
    llr: float
    left_entropy: float
    right_entropy: float
    cohesion: float
    frequency: int

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "score": round(self.score, 4),
            "pmi": round(self.pmi, 4),
            "llr": round(self.llr, 4),
            "left_entropy": round(self.left_entropy, 4),
            "right_entropy": round(self.right_entropy, 4),
            "cohesion": round(self.cohesion, 4),
            "frequency": self.frequency,
        }


@dataclass
class DiscoverResult:
    """Result of new word discovery."""

    candidates: list[Candidate] = field(default_factory=list)
    text_length: int = 0
    total_ngrams: int = 0

    def to_dict(self) -> dict:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "text_length": self.text_length,
            "total_ngrams": self.total_ngrams,
            "top_words": [c.word for c in self.candidates[:20]],
        }


# ---------------------------------------------------------------------------
# Core Algorithm
# ---------------------------------------------------------------------------

# Characters that typically mark sentence boundaries (split ngrams here)
_SENT_SPLIT_RE = re.compile(r"[。！？；\n]")
# Punctuation to remove from cleaned text
_PUNCT_RE = re.compile(r"[，、：""''（）《》【】\s]")


def discover(
    text: str,
    min_len: int = 2,
    max_len: int = 6,
    min_freq: int = 2,
    min_score: float = 0.5,
    max_candidates: int = 50,
    mtl_tokens: Optional[list] = None,
) -> DiscoverResult:
    """
    Discover new words using hybrid PMI + LLR + MTL comparison.

    - Short texts (< 100 cleaned chars): MTL hybrid is primary, PMI supplementary.
    - Longer texts: both PMI + LLR and MTL contribute.

    Adaptive parameters:
    - min_freq: auto-adjusted based on text length
    - entropy threshold: auto-adjusted based on text length
    - n-gram range: 2-6 characters (extended from 2-4)

    Parameters
    ----------
    text : str
        Input Chinese text.
    min_len : int
        Minimum n-gram length.
    max_len : int
        Maximum n-gram length (default 6, extended from 4).
    min_freq : int
        Minimum occurrence count for PMI candidates (auto-adjusted).
    min_score : float
        Minimum combined score threshold.
    max_candidates : int
        Max candidates to return.
    mtl_tokens : list, optional
        MTL tokenizer output for hybrid comparison.

    Returns
    -------
    DiscoverResult
    """
    if not text or not text.strip():
        return DiscoverResult(text_length=0, total_ngrams=0)

    sentences = _split_sentences(text)
    cleaned = [_clean_text(s) for s in sentences]
    cleaned = [s for s in cleaned if len(s) >= min_len]

    if not cleaned:
        return DiscoverResult(text_length=len(text), total_ngrams=0)

    total_chars = sum(len(s) for s in cleaned)
    candidates: list[Candidate] = []

    # ---- Adaptive parameters based on text length ----
    adaptive_min_freq = _adaptive_min_freq(total_chars, min_freq)
    adaptive_min_score = _adaptive_min_score(total_chars, min_score)

    # ---- MTL Hybrid (primary for short texts) ----
    if mtl_tokens:
        candidates.extend(_discover_from_mtl(mtl_tokens, cleaned))

    # ---- PMI + LLR (supplementary, only when enough statistics) ----
    if total_chars >= 30:  # lowered from 50 for better short-text support
        ngram_counts: dict[int, Counter] = {}
        for n in range(min_len, max_len + 1):
            ngram_counts[n] = Counter()
            for sent in cleaned:
                ngram_counts[n].update(_count_ngrams(sent, n))

        # Build char-level counts for LLR
        char_counts = Counter()
        for sent in cleaned:
            char_counts.update(sent)

        existing = {c.word for c in candidates}

        for n in range(min_len, max_len + 1):
            for ngram, freq in ngram_counts[n].items():
                if freq < adaptive_min_freq or ngram in existing:
                    continue

                # PMI
                pmi_val = _compute_pmi(ngram, freq, ngram_counts, total_chars)

                # LLR (Log-Likelihood Ratio)
                llr_val = _compute_llr(ngram, freq, char_counts, total_chars)

                # Entropy
                le = _compute_entropy_multisentence(ngram, cleaned, "left")
                re_val = _compute_entropy_multisentence(ngram, cleaned, "right")

                # Cohesion (internal binding strength)
                cohesion = _compute_cohesion(ngram, ngram_counts, total_chars)

                # Combined score: weighted sum
                score = (
                    pmi_val * 0.3 +
                    llr_val * 0.3 +
                    min(le, re_val) * 0.2 +
                    cohesion * 0.2
                )

                if score >= adaptive_min_score:
                    candidates.append(Candidate(
                        word=ngram,
                        score=score,
                        pmi=pmi_val,
                        llr=llr_val,
                        left_entropy=le,
                        right_entropy=re_val,
                        cohesion=cohesion,
                        frequency=freq,
                    ))

    # ---- Sort & Filter ----
    candidates.sort(key=lambda c: (-c.score, -len(c.word)))
    filtered = _subsume_filter(candidates)

    return DiscoverResult(
        candidates=filtered[:max_candidates],
        text_length=total_chars,
        total_ngrams=0,
    )


# ---------------------------------------------------------------------------
# Adaptive Parameters
# ---------------------------------------------------------------------------

def _adaptive_min_freq(total_chars: int, default_min_freq: int) -> int:
    """Adapt min_freq based on text length.

    Short texts (< 50 chars): min_freq = 1
    Medium texts (50-200 chars): min_freq = max(1, default/2)
    Long texts (> 200 chars): min_freq = default
    """
    if total_chars < 50:
        return 1
    if total_chars < 200:
        return max(1, default_min_freq // 2)
    return default_min_freq


def _adaptive_min_score(total_chars: int, default_min_score: float) -> float:
    """Adapt min_score based on text length.

    Short texts: lower threshold (0.3) to allow more candidates
    Long texts: use default threshold
    """
    if total_chars < 50:
        return min(default_min_score, 0.3)
    if total_chars < 100:
        return default_min_score * 0.6
    return default_min_score


# ---------------------------------------------------------------------------
# LLR (Log-Likelihood Ratio) Test
# ---------------------------------------------------------------------------

def _compute_llr(
    ngram: str,
    freq: int,
    char_counts: Counter,
    total_chars: int,
) -> float:
    """Compute log-likelihood ratio for an n-gram.

    LLR tests whether the characters in the n-gram co-occur more often
    than expected by chance. Higher LLR = stronger evidence of a word.

    Formula: LLR = 2 * sum(observed * log(observed/expected))
    """
    n = len(ngram)
    if n == 1 or total_chars < n:
        return 0.0

    # Count individual characters in the n-gram
    ngram_char_counts = Counter(ngram)

    # Expected frequency: product of individual char probabilities * total
    # For each character, compute its standalone probability
    expected_freq = 1.0
    for ch, ch_freq in ngram_char_counts.items():
        ch_prob = char_counts.get(ch, 0) / max(total_chars, 1)
        expected_freq *= ch_prob ** ch_freq

    expected_count = expected_freq * max(total_chars - n + 1, 1)

    if expected_count < 0.001:
        return 0.0

    # LLR = 2 * (observed * log(obs/expected) + (total-obs) * log(...))
    total_ngrams = max(total_chars - n + 1, 1)
    obs = freq
    exp = expected_count

    if obs == 0 or exp == 0:
        return 0.0

    # Simplified LLR (one-sided)
    llr = obs * math.log2(obs / exp)
    return llr


# ---------------------------------------------------------------------------
# Cohesion (Internal Binding Strength)
# ---------------------------------------------------------------------------

def _compute_cohesion(
    ngram: str,
    ngram_counts: dict[int, Counter],
    total_chars: int,
) -> float:
    """Compute internal cohesion of an n-gram.

    Cohesion measures how tightly the characters within a candidate word
    are bound together. A true word has high internal cohesion — its
    sub-parts don't appear independently very often.

    Formula: cohesion = PMI of internal splits / number of splits
    Normalized to [0, 1] range via sigmoid-like function.
    """
    n = len(ngram)
    if n <= 1:
        return 1.0

    pmis = []
    for split in range(1, n):
        left = ngram[:split]
        right = ngram[split:]

        lf = ngram_counts.get(len(left), Counter()).get(left, 0)
        rf = ngram_counts.get(len(right), Counter()).get(right, 0)
        joint = ngram_counts.get(n, Counter()).get(ngram, 0)

        if lf == 0 or rf == 0 or joint == 0:
            pmis.append(0)
            continue

        p_joint = joint / max(total_chars - n + 1, 1)
        p_left = lf / max(total_chars - len(left) + 1, 1)
        p_right = rf / max(total_chars - len(right) + 1, 1)

        if p_left * p_right > 0:
            pmi = math.log2(p_joint / (p_left * p_right))
        else:
            pmi = 0

        pmis.append(pmi)

    avg_pmi = sum(pmis) / len(pmis) if pmis else 0

    # Normalize to [0, 1] via sigmoid: 1 / (1 + exp(-avg_pmi))
    # This maps PMI (-inf to +inf) to (0, 1)
    try:
        cohesion = 1.0 / (1.0 + math.exp(-avg_pmi))
    except OverflowError:
        cohesion = 1.0 if avg_pmi > 0 else 0.0

    return cohesion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split at sentence-boundary punctuation."""
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _clean_text(text: str) -> str:
    """Keep CJK chars + digits; remove punctuation."""
    text = _PUNCT_RE.sub("", text)
    return "".join(
        ch for ch in text
        if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf" or ch.isdigit()
    )


def _count_ngrams(text: str, n: int) -> Counter:
    """Count all n-grams in text."""
    counts: Counter = Counter()
    for i in range(len(text) - n + 1):
        ngram = text[i : i + n]
        counts[ngram] += 1
    return counts


def _compute_pmi(
    ngram: str,
    freq: int,
    ngram_counts: dict[int, Counter],
    total_chars: int,
) -> float:
    """Compute average PMI for an n-gram."""
    n = len(ngram)
    if n == 2:
        c1 = ngram_counts[2].get(ngram, 0)
        if total_chars < 2:
            return 0.0
        p_ab = c1 / (total_chars - 1)
        c_a = ngram_counts.get(1, Counter()).get(ngram[0], 0) or _count_char_fallback(ngram[0], ngram_counts)
        c_b = ngram_counts.get(1, Counter()).get(ngram[1], 0) or _count_char_fallback(ngram[1], ngram_counts)
        if c_a == 0 or c_b == 0:
            return 0.0
        p_a = c_a / total_chars
        p_b = c_b / total_chars
        return math.log2(p_ab / (p_a * p_b)) if p_a * p_b > 0 else 0.0
    else:
        pmis = []
        for split in range(1, n):
            left = ngram[:split]
            right = ngram[split:]
            lf = ngram_counts.get(len(left), Counter()).get(left, 1)
            rf = ngram_counts.get(len(right), Counter()).get(right, 1)
            if lf == 0 or rf == 0:
                continue
            p_joint = freq / max(total_chars - n + 1, 1)
            p_left = lf / max(total_chars - len(left) + 1, 1)
            p_right = rf / max(total_chars - len(right) + 1, 1)
            if p_joint > 0 and p_left * p_right > 0:
                pmis.append(math.log2(p_joint / (p_left * p_right)))
        return sum(pmis) / len(pmis) if pmis else 0.0


def _count_char_fallback(ch: str, ngram_counts: dict[int, Counter]) -> int:
    """Count single character occurrences across all n-gram levels."""
    count = 0
    for n in [2, 3, 4]:
        if n in ngram_counts:
            for ng, c in ngram_counts[n].items():
                count += ng.count(ch) * c
    return max(count, 1)


def _compute_entropy_multisentence(ngram: str, sentences: list[str], side: str) -> float:
    """Compute branching entropy across multiple sentences."""
    neighbors: Counter = Counter()
    n = len(ngram)
    for text in sentences:
        idx = 0
        while True:
            idx = text.find(ngram, idx)
            if idx == -1:
                break
            if side == "left" and idx > 0:
                neighbors[text[idx - 1]] += 1
            elif side == "right" and idx + n < len(text):
                neighbors[text[idx + n]] += 1
            idx += 1
    total = sum(neighbors.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in neighbors.values())


# ---------------------------------------------------------------------------
# MTL-Based Discovery
# ---------------------------------------------------------------------------

_COMPOUND_POS: set[str] = {"NN", "NR", "JJ", "NT"}

# Only merge these POS types (nouns + adjectives)
_MERGE_POS: set[str] = {"NN", "NR", "JJ", "NT"}

# Don't merge if one side is a common single char
_DONT_MERGE_AFTER: set[str] = {"的", "了", "和", "与", "或", "是", "有"}

# Common single-char tokens unlikely to form new words
_COMMON_SINGLE: set[str] = {
    "的", "了", "是", "在", "和", "与", "或", "不", "也", "都", "就",
    "把", "被", "让", "给", "对", "从", "到", "向", "由", "以", "为",
    "因", "所", "而", "且", "但", "却", "只", "还", "又", "再", "才",
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "个", "些", "种", "次", "回", "遍", "趟", "这", "那", "哪", "每",
    "我", "你", "他", "她", "它", "们", "其",
    "着", "过", "得", "地", "之", "已", "将", "能", "会", "可", "要",
    "上", "下", "中", "里", "外", "前", "后", "左", "右",
    "大", "小", "多", "少", "新", "旧", "好", "坏", "高", "低",
    "用", "做", "来", "去", "出", "进", "开", "关", "有", "没",
    "年", "月", "日", "时", "分", "秒", "今", "明", "昨",
    "说", "想", "看", "听", "吃", "喝", "走", "跑", "写", "读",
    "人", "手", "口", "心", "水", "火", "土", "木", "金",
    "第", "如", "等", "吗", "呢", "吧", "啊", "嘛", "呀",
}


def _discover_from_mtl(tokens: list, sentences: list[str]) -> list[Candidate]:
    """Find candidate compound words from MTL token output.

    Strategy:
    - Multi-char tokens → direct candidates
    - Consecutive single-char NN/NR/JJ tokens → merge candidate
    """
    candidates: list[Candidate] = []
    if not tokens:
        return candidates

    all_text = "".join(sentences)

    i = 0
    while i < len(tokens):
        t = tokens[i]

        # Multi-char token: add as candidate (only for content-word POS)
        if len(t.text) >= 2:
            if t.text not in _COMMON_SINGLE and getattr(t, "pos", "") in _COMPOUND_POS:
                freq = all_text.count(t.text)
                candidates.append(Candidate(
                    word=t.text, score=5.0 + (0.5 if freq >= 2 else 0),
                    pmi=0, llr=0, left_entropy=0, right_entropy=0,
                    cohesion=1.0, frequency=freq,
                ))
            i += 1
            continue

        # Single char: try merging consecutive MERGE_POS tokens
        pos_tag = getattr(t, "pos", "")
        if (len(t.text) == 1
                and t.text not in _COMMON_SINGLE
                and pos_tag in _MERGE_POS):
            run_text = t.text
            j = i + 1
            while j < len(tokens) and j - i < 4:  # extended from 3 to 4
                nt = tokens[j]
                nt_pos = getattr(nt, "pos", "")
                if (len(nt.text) == 1
                        and nt.text not in _COMMON_SINGLE
                        and nt.text not in _DONT_MERGE_AFTER
                        and nt_pos in _MERGE_POS):
                    run_text += nt.text
                    j += 1
                else:
                    break
            if len(run_text) >= 2:
                freq = all_text.count(run_text)
                le = _compute_entropy_multisentence(run_text, sentences, "left")
                re_val = _compute_entropy_multisentence(run_text, sentences, "right")
                # Score: merge length * 1.5 + entropy bonus
                score = (j - i) * 1.5 + min(le, re_val) + (0.5 if freq >= 2 else 0)
                candidates.append(Candidate(
                    word=run_text, score=score, pmi=j - i, llr=0,
                    left_entropy=le, right_entropy=re_val,
                    cohesion=0.8, frequency=freq,
                ))
            i = j
        else:
            i += 1

    return candidates


def _subsume_filter(candidates: list[Candidate]) -> list[Candidate]:
    """Remove shorter candidates that are fully contained in longer higher-scored ones.

    If '碳钢' has higher score than both '碳' and '钢', keep only '碳钢'.
    """
    kept: list[Candidate] = []
    for c in candidates:
        subsumed = False
        for k in kept:
            # If this candidate is a substring of a kept one and has lower score
            if c.word != k.word and c.word in k.word and c.score <= k.score:
                subsumed = True
                break
        if not subsumed:
            kept.append(c)
    return kept


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def discover_words(text: str, **kwargs) -> list[str]:
    """Return just the discovered word strings (top 20)."""
    result = discover(text, **kwargs)
    return [c.word for c in result.candidates[:20]]


# ---------------------------------------------------------------------------
# Coarse tokenizer (alternative comparison baseline, no TF needed)
# ---------------------------------------------------------------------------

_coarse_tok: object | None = None


def _get_coarse_tok():
    """Lazy-load coarse ELECTRA tokenizer for comparison. Pure PyTorch."""
    global _coarse_tok
    if _coarse_tok is not None:
        return _coarse_tok
    try:
        import hanlp
        _coarse_tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH, verbose=False)
        return _coarse_tok
    except Exception as exc:
        logger.warning("Coarse tokenizer not available: %s", exc)
        return None


def discover_convseg(text: str) -> dict | None:
    """Run coarse tokenizer as comparison baseline.

    Returns dict with 'tokens' and 'candidates' (multi-char words),
    or None if the model is unavailable.
    """
    model = _get_coarse_tok()
    if model is None:
        return None
    tokens: list[str] = model(text)
    candidates = [t for t in tokens if len(t) >= 2]
    return {
        "engine": "coarse_electra_small",
        "tokens": tokens,
        "candidates": candidates,
    }