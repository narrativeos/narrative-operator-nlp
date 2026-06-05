"""
New Word Discovery — Hybrid PMI + MTL Comparison.

Two strategies combined:
1. PMI + Left/Right Entropy (statistical, needs enough text).
2. MTL diff: compare MTL tokenizer output against known patterns —
   sequences of low-confidence / unknown-character tokens are candidates.

Works on both single sentences (MTL diff) and longer texts (PMI).

Reference
---------
- Sun Maosong et al., "Chinese New Word Identification: A Statistical Approach"
- HanLP 1.x ``com.hankcs.hanlp.mining.word.NewWordDiscover``
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

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
    left_entropy: float
    right_entropy: float
    frequency: int

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "score": round(self.score, 4),
            "pmi": round(self.pmi, 4),
            "left_entropy": round(self.left_entropy, 4),
            "right_entropy": round(self.right_entropy, 4),
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
    max_len: int = 4,
    min_freq: int = 2,
    min_score: float = 0.5,
    max_candidates: int = 50,
    mtl_tokens: Optional[list] = None,
) -> DiscoverResult:
    """
    Discover new words using hybrid PMI + MTL comparison.

    - Short texts (< 100 cleaned chars): MTL hybrid is primary, PMI supplementary.
    - Longer texts: both PMI and MTL contribute.

    Parameters
    ----------
    text : str
        Input Chinese text.
    min_len : int
        Minimum n-gram length.
    max_len : int
        Maximum n-gram length.
    min_freq : int
        Minimum occurrence count for PMI candidates.
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

    # ---- MTL Hybrid (primary for short texts) ----
    if mtl_tokens:
        candidates.extend(_discover_from_mtl(mtl_tokens, cleaned))

    # ---- PMI (supplementary, only when enough statistics) ----
    if total_chars >= 50:
        ngram_counts: dict[int, Counter] = {}
        for n in range(min_len, max_len + 1):
            ngram_counts[n] = Counter()
            for sent in cleaned:
                ngram_counts[n].update(_count_ngrams(sent, n))

        effective_min_freq = max(2, min(min_freq, len(cleaned)))
        existing = {c.word for c in candidates}

        for n in range(min_len, max_len + 1):
            for ngram, freq in ngram_counts[n].items():
                if freq < effective_min_freq or ngram in existing:
                    continue
                pmi_val = _compute_pmi(ngram, freq, ngram_counts, total_chars)
                le = _compute_entropy_multisentence(ngram, cleaned, "left")
                re = _compute_entropy_multisentence(ngram, cleaned, "right")
                score = pmi_val + min(le, re)
                if score >= min_score:
                    candidates.append(Candidate(
                        word=ngram, score=score, pmi=pmi_val,
                        left_entropy=le, right_entropy=re, frequency=freq,
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
                    pmi=0, left_entropy=0, right_entropy=0, frequency=freq,
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
            while j < len(tokens) and j - i < 3:  # max 3-char merge
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
                re = _compute_entropy_multisentence(run_text, sentences, "right")
                # Score: merge length * 1.5 + entropy bonus
                score = (j - i) * 1.5 + min(le, re) + (0.5 if freq >= 2 else 0)
                candidates.append(Candidate(
                    word=run_text, score=score, pmi=j - i,
                    left_entropy=le, right_entropy=re, frequency=freq,
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
# ConvSeg (optional, requires TensorFlow in a clean environment)
# ---------------------------------------------------------------------------

_CONVSEG_VENV = "/tmp/convseg-venv"


def _run_convseg_subprocess(text: str) -> list[str] | None:
    """Run ConvSeg tokenizer in a subprocess using a clean venv.

    Returns list of tokens or None if unavailable.
    """
    import subprocess

    code = f'''
import hanlp
from hanlp.pretrained.tok import PKU_NAME_MERGED_SIX_MONTHS_CONVSEG
convseg = hanlp.load(PKU_NAME_MERGED_SIX_MONTHS_CONVSEG, verbose=False)
import json
result = convseg({text!r})
print(json.dumps(result))
'''
    try:
        proc = subprocess.run(
            [_CONVSEG_VENV + "/bin/python", "-c", code],
            capture_output=True, text=True, timeout=120,
            env={**__import__("os").environ, "PYTHONPATH": ""},
        )
        if proc.returncode != 0:
            return None
        import json as _json
        return _json.loads(proc.stdout.strip().split("\n")[-1])
    except Exception:
        return None


def discover_convseg(text: str) -> dict | None:
    """Run ConvSeg and return comparison data.

    Returns dict with 'tokens' and 'candidates' (words not in MTL output),
    or None if ConvSeg is unavailable.
    """
    tokens = _run_convseg_subprocess(text)
    if tokens is None:
        return None

    # Filter: keep only multi-char tokens as candidates
    candidates = [t for t in tokens if len(t) >= 2]

    return {
        "engine": "convseg_pku_merged",
        "tokens": tokens,
        "candidates": candidates,
    }
