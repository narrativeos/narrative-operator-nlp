"""
New Word Discovery — Hybrid PMI + MTL Comparison.

Three strategies combined:
1. PMI + LLR + Left/Right Entropy (statistical, needs enough text).
2. MTL diff: consecutive single-char content tokens in the MTL output
   are merge candidates — sequences MTL split that should be one word.
3. Internal cohesion test: characters within a candidate are tightly
   bound (high internal PMI = high cohesion).

Candidate filters (applied after scoring):
- Function-char filter: candidates containing pure function characters
  (的/了/和/是/...) are rejected.
- MTL boundary alignment: when MTL tokens are provided, a candidate must
  be a union of whole MTL tokens. Cutting across a token means the
  candidate is a fragment of a known word.
- Locked-fragment filter: a candidate that is almost always extended on
  BOTH sides (a sliding window of a longer repeated unit) is rejected.
  Boundary-marking function chars (和/与/是/…) keep a side open; attaching
  ones (的/在/了/…) do not.

Scores are normalized to [0, 1] before weighting, so every component
contributes on the same scale:
    score = 0.3 * pmi_norm + 0.3 * llr_norm + 0.2 * entropy_norm
            + 0.2 * cohesion

CNG and BM25 are computed and reported per candidate but deliberately DO
NOT enter the score. They were evaluated on a 1051-char domain corpus
(`scripts/bench_discoverer_indicators.py`) and every integration form lost
quality, because both indicators are dominated by raw frequency, which at
document scale correlates with commonness rather than with termhood:
- as a headroom boost they lifted all 131 candidates over the threshold
  (43 new, including junk like 以保证 / 常采用 / 基复);
- as a convex weight they dropped real terms (增材制造, 成型工艺, 温度)
  and added none;
- as a CNG floor of 0.2 they rejected 58 of 88 passing candidates, almost
  all legitimate (高温合金, 蠕变, 层间, 检测).
These indicators assume a corpus, not a document — they belong to Phase 2
(labeled/large-scale data), so they stay observable diagnostics only.

Left/right entropy semantics: a true word combines with diverse contexts
on both sides (high entropy); a fragment of a longer unit has at least
one fixed side (low entropy). When one side has no neighbor at all
(sentence edge), the other side is used instead of zeroing the signal.

Works on both single sentences (MTL merge) and longer texts (PMI).

References
----------
- Sun Maosong et al., "Chinese New Word Identification: A Statistical Approach"
- HanLP 1.x ``com.hankcs.hanlp.mining.word.NewWordDiscover``
- Duan et al., "New Word Discovery with Log-Likelihood Ratio Test"
- Feng Songguo, CNG combination principle (组合度)
- Zhang & Zhang (2017), BCBM25 — the BM25 half of the unsupervised
  keyphrase score, as packaged by ``termweight`` (calmaloner/termweight)
"""

from __future__ import annotations

import logging
import math
import re
import threading
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
    # Diagnostics only — never part of `score` (see module docstring).
    cng: float = 0.0
    bm25: float = 0.0

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "score": round(self.score, 4),
            "pmi": round(self.pmi, 4),
            "llr": round(self.llr, 4),
            "left_entropy": round(self.left_entropy, 4),
            "right_entropy": round(self.right_entropy, 4),
            "cohesion": round(self.cohesion, 4),
            "cng": round(self.cng, 4),
            "bm25": round(self.bm25, 4),
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

# Sentence/clause boundaries (ngrams never span these). Commas and the
# enumeration mark are included so ngrams do not cross clause boundaries.
_SENT_SPLIT_RE = re.compile(r"[。！？；，、\n]")
# Punctuation to remove from cleaned text (a single raw string; _clean_text
# additionally keeps only CJK chars + digits)
_PUNCT_RE = re.compile(r"[，、：；！？“”‘’（）《》【】\s]")

# Pure function characters — a candidate containing any of these is not a
# word. Deliberately conservative: only characters that are (almost)
# exclusively function words. Content morphemes (高/大/新/...) are NOT
# included so compounds like 高铁/新型/高强度 remain discoverable.
_FUNCTION_CHARS: frozenset[str] = frozenset(
    "的了着在过吗呢吧啊嘛呀哦嗯是和与或而且但却就都也又再才只还把被"
)

# Boundary-marking function characters (coordinators, copula, discourse
# markers). A candidate pinned to one of these sits at a NORMAL phrase
# boundary, so that side is NOT locked: 高强度 in "具有高强度和高韧性"
# keeps its right side open because 和 marks a coordination boundary.
#
# The remaining function characters (的/在/了/着/过/吗/...) ATTACH tightly
# to neighboring content — they are part of a longer unit (正在/了的/的X) —
# so a candidate pinned to them IS locked on that side: 成本正 in
# "的成本正在下降" is locked on both sides and rejected as a fragment.
_BOUNDARY_FUNC: frozenset[str] = frozenset("和与或而但就都也又再才只还把被是")

# Do not extend a merge run across these (right-side guard for MTL merges)
_DONT_MERGE_AFTER: frozenset[str] = frozenset("的了和与或是有")

# POS tags that may form compound words. Covers both HanLP CTB tags
# (NN/NR/JJ/NT/VV) and Universal POS tags (NOUN/PROPN/ADJ/VERB) so the
# merge path works regardless of which tagset the pipeline exposes.
_MERGE_POS: frozenset[str] = frozenset(
    {"NN", "NR", "JJ", "NT", "VV", "NOUN", "PROPN", "ADJ", "VERB"}
)

# Score normalization caps (all components mapped to [0, 1] before weighting)
_PMI_NORM_CAP = 10.0               # bits; PMI >= 10 is maximally strong
_LLR_NORM_CAP = math.log(1001.0)   # LLR = 1000 saturates
_ENT_NORM_CAP = 2.0                # bits; 4+ distinct neighbors per side

# Diagnostics only (see the module docstring for why they are not scored).
_FREQ_SAT_K1 = 1.5     # Okapi k1: tf = 1.5 saturates at 0.5, 9 at 0.86


def _norm_pmi(pmi: float) -> float:
    return min(max(pmi, 0.0), _PMI_NORM_CAP) / _PMI_NORM_CAP


def _norm_llr(llr: float) -> float:
    return min(math.log1p(max(llr, 0.0)), _LLR_NORM_CAP) / _LLR_NORM_CAP


def _norm_entropy(ent: float) -> float:
    return min(max(ent, 0.0), _ENT_NORM_CAP) / _ENT_NORM_CAP


def _norm_cng(cng: float, total_chars: int) -> float:
    """Map CNG to [0, 1] on a corpus-size-relative logarithmic scale.

    CNG = freq * min(boundary char freq) is bounded by total_chars**2
    (neither factor can exceed the number of characters), so
    log1p(cng) / (2 * log1p(total_chars)) is guaranteed to land in [0, 1]
    and stays comparable across short and long inputs.
    """
    if cng <= 0.0 or total_chars <= 1:
        return 0.0
    return min(math.log1p(cng) / (2.0 * math.log1p(total_chars)), 1.0)


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

    Pipeline:
    1. Build per-sentence n-gram / char statistics (punctuation removed,
       ngrams never span sentence or clause boundaries).
    2. MTL merge candidates: consecutive single-char content tokens that
       MTL split but should be one word (primary signal for short texts).
    3. PMI + LLR candidates: n-grams with enough frequency, scored with
       PMI + LLR + left/right entropy + internal cohesion (CNG and BM25 are
       attached as diagnostics only).
    4. Filters: function-char, MTL boundary alignment (when tokens are
       given), locked-fragment, subsumption.

    All scores are normalized to [0, 1]:
        score = 0.3*pmi_norm + 0.3*llr_norm + 0.2*entropy_norm + 0.2*cohesion

    Adaptive parameters:
    - min_freq: auto-adjusted based on text length
    - min_score: auto-adjusted based on text length (already on [0, 1])

    Parameters
    ----------
    text : str
        Input Chinese text.
    min_len : int
        Minimum n-gram length.
    max_len : int
        Maximum n-gram length (default 6).
    min_freq : int
        Minimum occurrence count for PMI candidates (auto-adjusted).
    min_score : float
        Minimum combined score threshold in [0, 1] (auto-adjusted).
    max_candidates : int
        Max candidates to return.
    mtl_tokens : list, optional
        MTL tokenizer output (objects with .text and .pos) for the merge
        path and the boundary-alignment filter.

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
    adaptive_min_freq = _adaptive_min_freq(total_chars, min_freq)
    adaptive_min_score = _adaptive_min_score(total_chars, min_score)

    # ---- Statistics (per sentence, so ngrams never cross boundaries) ----
    ngram_counts: dict[int, Counter] = {}
    total_ngrams = 0
    for n in range(min_len, max_len + 1):
        counts: Counter = Counter()
        for sent in cleaned:
            counts.update(_count_ngrams(sent, n))
        ngram_counts[n] = counts
        total_ngrams += len(counts)

    char_counts: Counter = Counter()
    for sent in cleaned:
        char_counts.update(sent)

    # Number of n-gram positions per length (per-sentence sum)
    pos_totals = {
        n: sum(max(len(s) - n + 1, 0) for s in cleaned)
        for n in range(min_len, max_len + 1)
    }

    # ---- MTL boundary index (for the alignment filter) ----
    mtl_full: Optional[str] = None
    mtl_boundaries: Optional[set[int]] = None
    if mtl_tokens:
        mtl_full, mtl_boundaries = _build_mtl_index([t.text for t in mtl_tokens])

    candidates: list[Candidate] = []
    mtl_words: set[str] = set()

    # ---- MTL merge candidates (primary for short texts) ----
    if mtl_tokens:
        for word in _discover_from_mtl(mtl_tokens):
            if not (min_len <= len(word) <= max_len):
                continue
            if _contains_function_char(word):
                continue
            cand = _score_ngram(
                word, ngram_counts, char_counts, cleaned, total_chars, pos_totals
            )
            if cand.frequency == 0:
                continue
            candidates.append(cand)
            mtl_words.add(word)

    # ---- PMI + LLR candidates (supplementary, needs some statistics) ----
    if total_chars >= 30:
        for n in range(min_len, max_len + 1):
            for ngram, freq in ngram_counts[n].items():
                if freq < adaptive_min_freq or ngram in mtl_words:
                    continue
                if _contains_function_char(ngram):
                    continue
                if not _is_mtl_aligned(ngram, mtl_full, mtl_boundaries):
                    continue
                if _is_locked_fragment(ngram, freq, ngram_counts, cleaned, max_len):
                    continue
                cand = _score_ngram(
                    ngram, ngram_counts, char_counts, cleaned, total_chars, pos_totals
                )
                if cand.score >= adaptive_min_score:
                    candidates.append(cand)

    # ---- Sort & Filter ----
    candidates.sort(key=lambda c: (-c.score, -len(c.word)))
    filtered = _subsume_filter(candidates)

    return DiscoverResult(
        candidates=filtered[:max_candidates],
        text_length=total_chars,
        total_ngrams=total_ngrams,
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
    """Adapt min_score (in [0, 1]) based on text length.

    Short texts: lower threshold to allow more candidates.
    """
    if total_chars < 50:
        return min(default_min_score, 0.3)
    if total_chars < 100:
        return min(default_min_score, 0.4)
    return default_min_score


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_ngram(
    word: str,
    ngram_counts: dict[int, Counter],
    char_counts: Counter,
    sentences: list[str],
    total_chars: int,
    pos_totals: dict[int, int],
) -> Candidate:
    """Compute all statistics and the normalized combined score for a word."""
    n = len(word)
    freq = ngram_counts.get(n, Counter()).get(word, 0)
    pmi_val = _compute_pmi(word, freq, ngram_counts, char_counts, total_chars)
    llr_val = _compute_llr(
        word, freq, char_counts, total_chars, pos_totals.get(n, 0)
    )
    le, le_n = _compute_entropy_multisentence(word, sentences, "left")
    re_val, re_n = _compute_entropy_multisentence(word, sentences, "right")
    cohesion = _compute_cohesion(word, ngram_counts, char_counts, total_chars)
    cng_val = _compute_cng(word, freq, char_counts)
    bm25_val = _compute_bm25(freq)
    ent = _effective_entropy(le, le_n, re_val, re_n)
    score = (
        _norm_pmi(pmi_val) * 0.3
        + _norm_llr(llr_val) * 0.3
        + _norm_entropy(ent) * 0.2
        + cohesion * 0.2
    )
    return Candidate(
        word=word,
        score=score,
        pmi=pmi_val,
        llr=llr_val,
        left_entropy=le,
        right_entropy=re_val,
        cohesion=cohesion,
        frequency=freq,
        cng=cng_val,
        bm25=bm25_val,
    )


def _compute_pmi(
    ngram: str,
    freq: int,
    ngram_counts: dict[int, Counter],
    char_counts: Counter,
    total_chars: int,
) -> float:
    """Compute average PMI over all internal splits of an n-gram.

    Single-char sub-parts use the true character counts (char_counts);
    longer sub-parts use the n-gram counts. Splits whose sub-part never
    occurs are skipped (no smoothing bias), so 2-gram and longer n-grams
    are scored on the same footing.
    """
    n = len(ngram)
    if n == 1 or freq == 0 or total_chars < n:
        return 0.0
    pmis = []
    for split in range(1, n):
        left = ngram[:split]
        right = ngram[split:]
        lf = (char_counts.get(left, 0) if len(left) == 1
              else ngram_counts.get(len(left), Counter()).get(left, 0))
        rf = (char_counts.get(right, 0) if len(right) == 1
              else ngram_counts.get(len(right), Counter()).get(right, 0))
        if lf == 0 or rf == 0:
            continue
        p_joint = freq / max(total_chars - n + 1, 1)
        p_left = lf / max(total_chars - len(left) + 1, 1)
        p_right = rf / max(total_chars - len(right) + 1, 1)
        if p_joint > 0 and p_left * p_right > 0:
            pmis.append(math.log2(p_joint / (p_left * p_right)))
    return sum(pmis) / len(pmis) if pmis else 0.0


def _compute_llr(
    ngram: str,
    freq: int,
    char_counts: Counter,
    total_chars: int,
    ngram_positions: int,
) -> float:
    """Compute log-likelihood ratio (G-test) for an n-gram.

    Tests whether the characters in the n-gram co-occur more often than
    expected under character independence. Higher LLR = stronger evidence
    of a word.

    Full four-cell G-test:
        LLR = 2 * sum over cells of O * ln(O / E)
    """
    n = len(ngram)
    if n == 1 or freq == 0 or ngram_positions <= 0 or total_chars < n:
        return 0.0

    # Expected frequency under the independence model
    expected_freq = 1.0
    for ch, ch_freq in Counter(ngram).items():
        ch_prob = char_counts.get(ch, 0) / max(total_chars, 1)
        expected_freq *= ch_prob ** ch_freq

    expected_count = expected_freq * ngram_positions
    if expected_count < 0.001:
        return 0.0

    remaining = ngram_positions - freq
    expected_remaining = ngram_positions - expected_count
    if remaining <= 0 or expected_remaining <= 0:
        return 0.0

    return 2.0 * (
        freq * math.log(freq / expected_count)
        + expected_count * math.log(expected_count / freq)
        + remaining * math.log(remaining / expected_remaining)
        + expected_remaining * math.log(expected_remaining / remaining)
    )


# ---------------------------------------------------------------------------
# Cohesion (Internal Binding Strength)
# ---------------------------------------------------------------------------

def _compute_cohesion(
    ngram: str,
    ngram_counts: dict[int, Counter],
    char_counts: Counter,
    total_chars: int,
) -> float:
    """Compute internal cohesion of an n-gram.

    Cohesion measures how tightly the characters within a candidate word
    are bound together: the average PMI over all internal splits, mapped
    to [0, 1] via a sigmoid (scaled by 2 so typical PMI values of a few
    bits do not saturate the curve).
    """
    n = len(ngram)
    if n <= 1:
        return 1.0

    pmis = []
    for split in range(1, n):
        left = ngram[:split]
        right = ngram[split:]
        lf = (char_counts.get(left, 0) if len(left) == 1
              else ngram_counts.get(len(left), Counter()).get(left, 0))
        rf = (char_counts.get(right, 0) if len(right) == 1
              else ngram_counts.get(len(right), Counter()).get(right, 0))
        joint = ngram_counts.get(n, Counter()).get(ngram, 0)

        if lf == 0 or rf == 0 or joint == 0:
            pmis.append(0.0)
            continue

        p_joint = joint / max(total_chars - n + 1, 1)
        p_left = lf / max(total_chars - len(left) + 1, 1)
        p_right = rf / max(total_chars - len(right) + 1, 1)

        if p_left * p_right > 0:
            pmis.append(math.log2(p_joint / (p_left * p_right)))
        else:
            pmis.append(0.0)

    avg_pmi = sum(pmis) / len(pmis) if pmis else 0.0

    try:
        return 1.0 / (1.0 + math.exp(-avg_pmi / 2.0))
    except OverflowError:
        return 1.0 if avg_pmi > 0 else 0.0


# ---------------------------------------------------------------------------
# Frequency Diagnostics: CNG + BM25 (reported, not scored)
# ---------------------------------------------------------------------------

def _compute_cng(
    ngram: str,
    freq: int,
    char_counts: Counter,
) -> float:
    """Compute CNG (combination principle) for an n-gram — diagnostic only.

    CNG = freq(ngram) * min(freq(first char), freq(last char))

    The boundary characters are what hold a candidate up as a word: a pair
    whose weaker edge character is rare in the corpus is often a slice of a
    longer unit. It is exposed in the response so the demo page and future
    tuning can see it, but it stays out of the score: on a single document
    the min-boundary-char factor tracks character commonness, not termhood,
    and folding it in was measured to reject real terms (see the module
    docstring).
    """
    if len(ngram) < 2 or freq == 0:
        return 0.0
    left_freq = char_counts.get(ngram[0], 0)
    right_freq = char_counts.get(ngram[-1], 0)
    return float(freq) * float(min(left_freq, right_freq))


def _compute_bm25(freq: int) -> float:
    """Saturating frequency diagnostic in [0, 1] — reported, not scored.

        bm25 = tf / (k1 + tf)

    This is the corpus-independent tf-saturation half of the BM25 in the
    BCBM25 keyphrase score (Zhang & Zhang 2017, as packaged by
    ``termweight``). Two deliberate deviations from that recipe:

    - The idf half is DROPPED. BCBM25 counts documents in a corpus;
      ``discover()`` only ever sees one document, so sentence-level df
      marks a genuine term that recurs in every sentence as
      non-discriminative (idf -> 0) — the opposite of the truth.
    - The BC (boundary entropy) half is already covered by the existing
      left/right entropy component, so it would be double-counted.

    What is left is a saturating recurrence curve (a term seen 200 times is
    not 10x better than one seen 20 times). It is useful as a displayed
    diagnostic; as a score component it only promoted frequent common words
    (see the module docstring).
    """
    if freq <= 0:
        return 0.0
    return freq / (_FREQ_SAT_K1 + freq)


# ---------------------------------------------------------------------------
# Left/Right Entropy
# ---------------------------------------------------------------------------

def _compute_entropy_multisentence(
    ngram: str, sentences: list[str], side: str
) -> tuple[float, int]:
    """Compute branching entropy of the neighbor on one side of an n-gram.

    Returns (entropy, neighbor_count). neighbor_count == 0 means the
    n-gram never has a neighbor on that side (e.g. always at sentence
    start) — callers must not treat that as a "stable boundary".
    """
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
        return 0.0, 0
    return -sum((c / total) * math.log2(c / total) for c in neighbors.values()), total


def _effective_entropy(le: float, le_n: int, re_val: float, re_n: int) -> float:
    """Combine left/right entropy into one boundary-diversity signal.

    A true word combines with diverse contexts on BOTH sides; a fragment
    of a longer unit has at least one fixed side. When one side has no
    neighbor at all (sentence edge), fall back to the other side instead
    of letting min() zero out the signal.
    """
    if le_n == 0 and re_n == 0:
        return 0.0
    if le_n == 0:
        return re_val
    if re_n == 0:
        return le
    return min(le, re_val)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _contains_function_char(word: str) -> bool:
    """True if the word contains a pure function character (的/了/和/...)."""
    return any(ch in _FUNCTION_CHARS for ch in word)


def _build_mtl_index(token_texts: list[str]) -> tuple[str, set[int]]:
    """Build the concatenated token string and the set of token boundaries."""
    parts: list[str] = []
    boundaries: set[int] = {0}
    pos = 0
    for t in token_texts:
        parts.append(t)
        pos += len(t)
        boundaries.add(pos)
    return "".join(parts), boundaries


def _is_mtl_aligned(
    word: str, mtl_full: Optional[str], mtl_boundaries: Optional[set[int]]
) -> bool:
    """Check whether a candidate is a union of whole MTL tokens.

    A true new word is something MTL splits into pieces but should be one
    word — it must start and end on token boundaries. A candidate that
    cuts across a token is a fragment of a known word.

    If the token stream does not contain the word at all (e.g. the
    tokenizer normalized the text differently), the check is skipped
    (returns True) rather than filtering on bad data.
    """
    if mtl_full is None or mtl_boundaries is None:
        return True
    n = len(word)
    start = 0
    found_any = False
    while True:
        idx = mtl_full.find(word, start)
        if idx < 0:
            return not found_any
        found_any = True
        if idx in mtl_boundaries and idx + n in mtl_boundaries:
            return True
        start = idx + 1


def _is_locked_fragment(
    word: str,
    freq: int,
    ngram_counts: dict[int, Counter],
    sentences: list[str],
    max_len: int,
) -> bool:
    """Detect fragments locked inside a longer repeated unit.

    A side is "locked" when the candidate is almost always (>= 80% of
    occurrences, at least 2 times) extended on that side by a single fixed
    character, or pinned to a sentence boundary. The one exception is a
    boundary-marking function character (和/与/是/...): it marks a normal
    phrase boundary, so that side stays open (高强度 in "…高强度和高韧性").
    Attaching function characters (的/在/了/…) are NOT an exception —
    they belong to a longer unit, so 成本正 in "的成本正在下降" is locked
    on both sides and rejected.
    If BOTH sides are locked, the candidate is a sliding-window fragment
    of a longer string rather than a word.
    """
    n = len(word)
    if n + 1 > max_len or freq < 2:
        return False
    ext = ngram_counts.get(n + 1, Counter())
    threshold = 0.8 * freq

    left_exts = [(ng, c) for ng, c in ext.items() if ng.endswith(word)]
    right_exts = [(ng, c) for ng, c in ext.items() if ng.startswith(word)]

    left_boundary = _count_at_boundary(word, sentences, "left")
    right_boundary = _count_at_boundary(word, sentences, "right")

    return (_side_locked(left_exts, left_boundary, "left", threshold)
            and _side_locked(right_exts, right_boundary, "right", threshold))


def _side_locked(
    exts: list[tuple[str, int]], boundary_count: int, side: str, threshold: float
) -> bool:
    """True if one side of a candidate is locked (see _is_locked_fragment).

    A dominant single-character extension locks the side unless the
    neighbor is a boundary-marking function character (和/与/是/...),
    which denotes a normal phrase edge rather than a longer word unit.
    """
    for ng, c in exts:
        if c >= 2 and c >= threshold:
            neighbor = ng[0] if side == "left" else ng[-1]
            if neighbor not in _BOUNDARY_FUNC:
                return True
    return boundary_count >= 2 and boundary_count >= threshold


def _count_at_boundary(word: str, sentences: list[str], side: str) -> int:
    """Count occurrences of word at the start/end of a sentence."""
    n = len(word)
    count = 0
    for s in sentences:
        idx = 0
        while True:
            idx = s.find(word, idx)
            if idx < 0:
                break
            if (side == "left" and idx == 0) or (side == "right" and idx + n == len(s)):
                count += 1
            idx += 1
    return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split at sentence/clause-boundary punctuation."""
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
    """Count all (overlapping) n-grams in text."""
    counts: Counter = Counter()
    for i in range(len(text) - n + 1):
        counts[text[i : i + n]] += 1
    return counts


# ---------------------------------------------------------------------------
# MTL-Based Discovery
# ---------------------------------------------------------------------------

def _discover_from_mtl(tokens: list) -> list[str]:
    """Find merge candidates from MTL token output.

    Consecutive single-char content tokens (nouns/verbs/adjectives, both
    CTB and Universal POS tagsets) are merged into candidates — these are
    sequences MTL split that should probably be one word.
    """
    candidates: list[str] = []
    seen: set[str] = set()
    i = 0
    while i < len(tokens):
        t = tokens[i]
        pos_tag = getattr(t, "pos", "")
        if (len(t.text) == 1
                and t.text not in _FUNCTION_CHARS
                and t.text not in _DONT_MERGE_AFTER
                and pos_tag in _MERGE_POS):
            run_text = t.text
            j = i + 1
            while j < len(tokens) and j - i < 4:  # merge runs up to 4 chars
                nt = tokens[j]
                nt_pos = getattr(nt, "pos", "")
                if (len(nt.text) == 1
                        and nt.text not in _FUNCTION_CHARS
                        and nt.text not in _DONT_MERGE_AFTER
                        and nt_pos in _MERGE_POS):
                    run_text += nt.text
                    j += 1
                else:
                    break
            if len(run_text) >= 2 and run_text not in seen:
                seen.add(run_text)
                candidates.append(run_text)
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
_coarse_tok_lock = threading.Lock()


def _get_coarse_tok():
    """Lazy-load coarse ELECTRA tokenizer for comparison. Pure PyTorch."""
    global _coarse_tok
    if _coarse_tok is not None:
        return _coarse_tok
    with _coarse_tok_lock:
        if _coarse_tok is not None:  # double-checked after acquiring lock
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