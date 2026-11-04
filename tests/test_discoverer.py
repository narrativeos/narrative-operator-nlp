"""
Tests: New Word Discovery (core/discoverer.py).

Covers the audit fixes:
- PMI computed from true character counts (no smoothing bias).
- Entropy direction: diverse contexts = valid word; edge fallback.
- POS mapping for both CTB and Universal tagsets.
- Fragment noise: function-char filter, MTL boundary alignment,
  locked-fragment filter (boundary vs attaching function chars).
- Score normalization to [0, 1].
Plus the CNG/BM25 diagnostics: they are reported per candidate but must not
move the calibrated score (see scripts/bench_discoverer_indicators.py).
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from core.discoverer import (
    Candidate,
    _build_mtl_index,
    _clean_text,
    _compute_bm25,
    _compute_cng,
    _compute_cohesion,
    _compute_entropy_multisentence,
    _compute_llr,
    _compute_pmi,
    _contains_function_char,
    _count_ngrams,
    _discover_from_mtl,
    _effective_entropy,
    _is_locked_fragment,
    _is_mtl_aligned,
    _norm_cng,
    _norm_entropy,
    _norm_llr,
    _norm_pmi,
    _score_ngram,
    _side_locked,
    _split_sentences,
    _subsume_filter,
    discover,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tok(text: str, pos: str = "NN"):
    class T:
        pass
    t = T()
    t.text = text
    t.pos = pos
    return t


def _ngram_stats(sentences: list[str], ns: tuple[int, ...] = (2, 3, 4)):
    """Build ngram_counts the same way discover() does."""
    nc: dict[int, Counter] = {}
    for n in ns:
        counts: Counter = Counter()
        for s in sentences:
            counts.update(_count_ngrams(s, n))
        nc[n] = counts
    return nc


def _candidate(word: str, score: float) -> Candidate:
    return Candidate(
        word=word, score=score, pmi=0.0, llr=0.0,
        left_entropy=0.0, right_entropy=0.0, cohesion=0.0, frequency=1,
    )


# ---------------------------------------------------------------------------
# PMI (P0-1: true character counts, no fallback bias)
# ---------------------------------------------------------------------------

class TestPMI:
    def setup_method(self):
        # 我爱学习 / 我爱学习 / 他爱生活
        self.sentences = ["我爱学习", "我爱学习", "他爱生活"]
        self.char_counts: Counter = Counter()
        for s in self.sentences:
            self.char_counts.update(s)
        self.nc = _ngram_stats(self.sentences)
        self.total = sum(len(s) for s in self.sentences)  # 12

    def test_strong_association_high_pmi(self):
        # 学习: both chars only co-occur inside 学习 -> strong PMI
        pmi = _compute_pmi("学习", 2, self.nc, self.char_counts, self.total)
        assert pmi > 2.0

    def test_ordering_strong_beats_weak(self):
        pmi_xuexi = _compute_pmi("学习", 2, self.nc, self.char_counts, self.total)
        pmi_woai = _compute_pmi("我爱", 2, self.nc, self.char_counts, self.total)
        assert pmi_xuexi > pmi_woai

    def test_trigram_averages_splits(self):
        pmi = _compute_pmi("爱学习", 2, self.nc, self.char_counts, self.total)
        assert pmi > 0.0

    def test_zero_freq_returns_zero(self):
        assert _compute_pmi("他习", 0, self.nc, self.char_counts, self.total) == 0.0

    def test_missing_subparts_skipped_no_bias(self):
        # Defensive path: sub-part counts absent -> split skipped, no crash.
        nc = {2: Counter(), 3: Counter({"碳纤维": 5})}
        chars = Counter({"碳": 10})  # 纤 / 维 never counted
        assert _compute_pmi("碳纤维", 5, nc, chars, 100) == 0.0

    def test_single_char_returns_zero(self):
        assert _compute_pmi("学", 5, self.nc, self.char_counts, self.total) == 0.0


# ---------------------------------------------------------------------------
# Entropy (P0-2: high entropy = diverse contexts = valid word)
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_diverse_neighbors_high_entropy(self):
        sents = ["我学习", "他学习", "她学习"]
        ent, n = _compute_entropy_multisentence("学习", sents, "left")
        assert n == 3
        assert ent == pytest.approx(math.log2(3), abs=1e-9)

    def test_fixed_neighbor_zero_entropy(self):
        sents = ["我爱学习", "我爱学习"]
        ent, n = _compute_entropy_multisentence("学习", sents, "left")
        assert n == 2
        assert ent == 0.0

    def test_sentence_edge_reports_no_neighbor(self):
        sents = ["学习", "学习"]
        assert _compute_entropy_multisentence("学习", sents, "left") == (0.0, 0)
        assert _compute_entropy_multisentence("学习", sents, "right") == (0.0, 0)

    def test_effective_entropy_falls_back_to_other_side(self):
        # Left side always at sentence start (no neighbor) -> use right side.
        assert _effective_entropy(0.0, 0, 1.5, 3) == 1.5
        assert _effective_entropy(1.5, 3, 0.0, 0) == 1.5

    def test_effective_entropy_min_when_both_sides_present(self):
        assert _effective_entropy(1.0, 2, 2.0, 2) == 1.0

    def test_effective_entropy_zero_when_no_neighbors_at_all(self):
        assert _effective_entropy(0.0, 0, 0.0, 0) == 0.0


# ---------------------------------------------------------------------------
# Punctuation / sentence boundaries
# ---------------------------------------------------------------------------

class TestPunctuation:
    def test_split_sentences(self):
        assert _split_sentences("你好。世界！再见？好；行、可以\n结束") == [
            "你好", "世界", "再见", "好", "行", "可以", "结束",
        ]

    def test_clean_text_removes_punctuation_keeps_cjk_digits(self):
        # Latin letters are dropped by design (CJK + digits only).
        assert _clean_text("碳纤维，复合材料。强度200MPa") == "碳纤维复合材料强度200"

    def test_ngrams_never_span_boundaries(self):
        sents = _split_sentences("我爱学习。他爱生活")
        counts: Counter = Counter()
        for s in sents:
            counts.update(_count_ngrams(s, 3))
        assert "习他" not in counts  # would require crossing 。
        assert all("。" not in ng for ng in counts)

    def test_discover_output_has_no_punctuation(self):
        text = ("碳纤维复合材料具有高强度。碳纤维复合材料广泛应用。"
                "碳纤维复合材料成本下降。") * 3
        res = discover(text)
        for c in res.candidates:
            assert _clean_text(c.word) == c.word


# ---------------------------------------------------------------------------
# Function-char filter (P0-4)
# ---------------------------------------------------------------------------

class TestFunctionCharFilter:
    def test_pure_function_chars_rejected(self):
        for ch in "的了着在和是过":
            assert _contains_function_char(ch)

    def test_content_words_pass(self):
        assert not _contains_function_char("碳纤维")
        assert not _contains_function_char("高强度")

    def test_discover_never_returns_function_char_words(self):
        # Only repeated unit contains 的 -> must be filtered out.
        text = "碳纤维的复合材料。碳纤维的复合材料。"
        res = discover(text)
        for c in res.candidates:
            assert not _contains_function_char(c.word)


# ---------------------------------------------------------------------------
# MTL boundary alignment
# ---------------------------------------------------------------------------

class TestMtlAlignment:
    def setup_method(self):
        self.full, self.bounds = _build_mtl_index(
            ["碳", "纤", "维", "复合材料"]
        )
        assert self.full == "碳纤维复合材料"
        assert self.bounds == {0, 1, 2, 3, 7}

    def test_union_of_whole_tokens_aligned(self):
        assert _is_mtl_aligned("碳纤维", self.full, self.bounds)
        assert _is_mtl_aligned("碳纤维复合材料", self.full, self.bounds)
        assert _is_mtl_aligned("复合材料", self.full, self.bounds)

    def test_crossing_token_boundary_rejected(self):
        assert not _is_mtl_aligned("维复合", self.full, self.bounds)
        assert not _is_mtl_aligned("纤维复合", self.full, self.bounds)

    def test_word_absent_from_stream_skips_check(self):
        assert _is_mtl_aligned("不存在的词", self.full, self.bounds)

    def test_no_tokens_skips_check(self):
        assert _is_mtl_aligned("碳纤维", None, None)

    def test_discover_filters_misaligned_candidates(self):
        text = "碳纤维复合材料。碳纤维复合材料。"
        tokens = [_tok("碳"), _tok("纤"), _tok("维"), _tok("复合材料"),
                  _tok("。", "PU"),
                  _tok("碳"), _tok("纤"), _tok("维"), _tok("复合材料"),
                  _tok("。", "PU")]
        res = discover(text, mtl_tokens=tokens)
        words = {c.word for c in res.candidates}
        assert "维复合" not in words      # cuts across 维|复合材料
        assert "碳纤维" in words          # merge path, aligned


# ---------------------------------------------------------------------------
# Locked-fragment filter (P0-4: boundary vs attaching function chars)
# ---------------------------------------------------------------------------

class TestLockedFragment:
    def test_fragment_locked_by_attaching_function_chars(self):
        # 成本正 in "的成本正在下降": left 的 (attaching), right 在 (attaching)
        # -> both sides locked -> fragment.
        sents = ["的成本正在下降"] * 3
        nc = _ngram_stats(sents)
        assert _is_locked_fragment("成本正", 3, nc, sents, max_len=6)

    def test_word_open_at_boundary_function_char_kept(self):
        # 高强度 in "具有高强度和高韧性": right neighbor 和 marks a normal
        # phrase boundary -> right side NOT locked -> not a fragment.
        sents = ["具有高强度和高韧性"] * 3
        nc = _ngram_stats(sents)
        assert not _is_locked_fragment("高强度", 3, nc, sents, max_len=6)

    def test_word_open_at_left_boundary_function_char_kept(self):
        # 高韧性: left neighbor 和 (boundary func) open; right side is a
        # sentence boundary (locked) -> only one side locked -> kept.
        sents = ["具有高强度和高韧性"] * 3
        nc = _ngram_stats(sents)
        assert not _is_locked_fragment("高韧性", 3, nc, sents, max_len=6)

    def test_diverse_neighbors_not_locked(self):
        sents = ["购买碳纤维", "测试碳纤维", "使用碳纤维"]
        nc = _ngram_stats(sents)
        assert not _is_locked_fragment("碳纤维", 3, nc, sents, max_len=6)

    def test_low_freq_not_locked(self):
        sents = ["的成本正在下降"]
        nc = _ngram_stats(sents)
        assert not _is_locked_fragment("成本正", 1, nc, sents, max_len=6)

    def test_side_locked_unit_cases(self):
        # Attaching function char neighbor -> locked.
        assert _side_locked([("的成本正", 3)], 0, "left", 2.4)
        # Boundary function char neighbor -> NOT locked.
        assert not _side_locked([("高强度和", 3)], 0, "right", 2.4)
        # Content char neighbor -> locked.
        assert _side_locked([("有高强度", 3)], 0, "left", 2.4)
        # Sentence boundary majority -> locked.
        assert _side_locked([], 3, "right", 2.4)
        # Below the minimum count -> not locked.
        assert not _side_locked([("的成本正", 1)], 0, "left", 2.4)
        assert not _side_locked([], 1, "right", 2.4)


# ---------------------------------------------------------------------------
# MTL merge path (P0-3: CTB + Universal POS)
# ---------------------------------------------------------------------------

class TestMtlMerge:
    def test_csb_tags_merge(self):
        tokens = [_tok("碳"), _tok("钢"), _tok("是", "VC"), _tok("钢"),
                  _tok("的", "DEG")]
        assert _discover_from_mtl(tokens) == ["碳钢"]

    def test_universal_tags_merge(self):
        tokens = [_tok("量", "NOUN"), _tok("子", "NOUN"), _tok("的", "DET")]
        assert _discover_from_mtl(tokens) == ["量子"]

    def test_function_char_stops_run(self):
        tokens = [_tok("碳"), _tok("的", "DEG"), _tok("钢")]
        assert _discover_from_mtl(tokens) == []

    def test_non_content_pos_stops_run(self):
        tokens = [_tok("碳"), _tok("在", "P"), _tok("钢")]
        assert _discover_from_mtl(tokens) == []

    def test_run_capped_at_four_chars(self):
        tokens = [_tok(c) for c in "碳纤维复合材料"]
        cands = _discover_from_mtl(tokens)
        assert all(len(c) <= 4 for c in cands)

    def test_discover_short_text_mtl_merge(self):
        text = "碳钢是钢的一种，具有高强度。"
        tokens = [_tok("碳"), _tok("钢"), _tok("是", "VC"), _tok("钢"),
                  _tok("的", "DEG"), _tok("一", "CD"), _tok("种"),
                  _tok("，", "PU"), _tok("具有"), _tok("高", "JJ"),
                  _tok("强度"), _tok("。", "PU")]
        res = discover(text, mtl_tokens=tokens)
        words = {c.word for c in res.candidates}
        assert "碳钢" in words


# ---------------------------------------------------------------------------
# Score normalization & result shape
# ---------------------------------------------------------------------------

class TestScoresAndResult:
    def test_scores_within_unit_interval(self):
        text = ("碳纤维复合材料具有高强度和高韧性。碳纤维复合材料广泛应用于"
                "航空航天领域。碳纤维复合材料的成本正在下降。") * 3
        res = discover(text)
        assert res.candidates
        for c in res.candidates:
            assert 0.0 <= c.score <= 1.0

    def test_candidates_sorted_by_score_desc(self):
        text = ("碳纤维复合材料具有高强度。碳纤维复合材料广泛应用。"
                "碳纤维复合材料成本下降。") * 3
        res = discover(text)
        scores = [c.score for c in res.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_empty_text(self):
        res = discover("")
        assert res.candidates == []
        assert res.text_length == 0
        res2 = discover("   ")
        assert res2.candidates == []

    def test_to_dict_shape(self):
        res = discover(TestEndToEnd.TEXT)
        d = res.to_dict()
        assert set(d) == {"candidates", "text_length", "total_ngrams", "top_words"}
        assert d["candidates"], "expected candidates to exercise the payload shape"
        assert set(d["candidates"][0]) == {
            "word", "score", "pmi", "llr", "left_entropy", "right_entropy",
            "cohesion", "cng", "bm25", "frequency",
        }

    def test_text_length_is_cleaned_length(self):
        res = discover("碳纤维，复合材料。")
        assert res.text_length == len("碳纤维复合材料")


# ---------------------------------------------------------------------------
# Subsumption
# ---------------------------------------------------------------------------

class TestSubsume:
    def test_shorter_lower_scored_substring_removed(self):
        kept = _subsume_filter([_candidate("钢材", 0.9), _candidate("钢", 0.5)])
        assert [c.word for c in kept] == ["钢材"]

    def test_shorter_higher_scored_kept(self):
        kept = _subsume_filter([_candidate("钢材", 0.5), _candidate("钢", 0.9)])
        assert {c.word for c in kept} == {"钢材", "钢"}

    def test_disjoint_words_all_kept(self):
        kept = _subsume_filter([_candidate("碳纤维", 0.8), _candidate("复合材料", 0.7)])
        assert len(kept) == 2


# ---------------------------------------------------------------------------
# End-to-end: the audit's fragment-noise scenario
# ---------------------------------------------------------------------------

class TestEndToEnd:
    TEXT = ("碳纤维复合材料具有高强度和高韧性。"
            "碳纤维复合材料广泛应用于航空航天领域。"
            "碳纤维复合材料的成本正在下降。") * 3

    def test_fragment_not_discovered_without_mtl(self):
        res = discover(self.TEXT)
        words = {c.word for c in res.candidates}
        assert "成本正" not in words   # sliding window of 成本正在下降
        assert "复合材料" in words
        assert "高强度" in words
        assert "高韧性" in words

    def test_fragment_not_discovered_with_mtl(self):
        lex = {"碳纤维": "NN", "复合材料": "NN", "具有": "VV", "高强度": "NN",
               "高韧性": "NN", "广泛": "AD", "应用于": "VV", "航空航天": "NN",
               "领域": "NN", "成本": "NN", "正在": "AD", "下降": "VV",
               "和": "CC", "的": "DEG", "。": "PU"}
        tokens, i = [], 0
        while i < len(self.TEXT):
            for w, pos in lex.items():
                if self.TEXT.startswith(w, i):
                    tokens.append(_tok(w, pos))
                    i += len(w)
                    break
            else:
                tokens.append(_tok(self.TEXT[i]))
                i += 1
        res = discover(self.TEXT, mtl_tokens=tokens)
        words = {c.word for c in res.candidates}
        assert "成本正" not in words
        assert "复合材料" in words


# ---------------------------------------------------------------------------
# Frequency diagnostics: CNG + BM25 (reported, never scored)
# ---------------------------------------------------------------------------

class TestFrequencyDiagnostics:
    def test_cng_multiplies_freq_by_weaker_boundary_char(self):
        cc = Counter({"碳": 8, "钢": 4, "复": 2})
        assert _compute_cng("碳钢", 3, cc) == 3 * 4
        assert _compute_cng("复钢", 3, cc) == 3 * 2

    def test_cng_zero_for_single_char_or_missing_freq(self):
        cc = Counter({"碳": 5})
        assert _compute_cng("碳", 5, cc) == 0.0
        assert _compute_cng("碳钢", 0, cc) == 0.0

    def test_cng_saturates_at_the_rarer_of_both_edges(self):
        cc = Counter({"钢": 9, "材": 1})
        assert _compute_cng("钢材", 2, cc) == 2 * 1

    def test_bm25_is_monotone_and_saturating(self):
        assert _compute_bm25(0) == 0.0
        vals = [_compute_bm25(f) for f in (1, 2, 5, 20, 200)]
        assert vals == sorted(vals)
        assert all(0.0 <= v < 1.0 for v in vals)
        # 10x the frequency must not give 10x the score
        assert vals[-1] < 10 * vals[-2]

    def test_norm_cng_is_bounded_by_one(self):
        # CNG cannot exceed total_chars**2, so the log scale caps at 1.
        assert _norm_cng(10 ** 9, 120) == 1.0
        assert _norm_cng(0.0, 120) == 0.0
        assert _norm_cng(5.0, 1) == 0.0
        assert 0.0 < _norm_cng(81.0, 144) < 1.0

    def test_diagnostics_do_not_move_the_score(self):
        """The score stays the calibrated 4-way sum; cng/bm25 are extras."""
        sentences = ["碳纤维复合材料具有高强度", "碳纤维复合材料广泛应用"]
        ns = (2, 3, 4, 5, 6, 7)
        nc = _ngram_stats(sentences, ns)
        cc: Counter = Counter()
        for s in sentences:
            cc.update(s)
        total = sum(len(s) for s in sentences)
        pos = {n: sum(max(len(s) - n + 1, 0) for s in sentences) for n in ns}

        word = "复合材料"
        cand = _score_ngram(word, nc, cc, sentences, total, pos)
        freq = nc[len(word)].get(word, 0)
        pmi = _compute_pmi(word, freq, nc, cc, total)
        llr = _compute_llr(word, freq, cc, total, pos[len(word)])
        le, le_n = _compute_entropy_multisentence(word, sentences, "left")
        re, re_n = _compute_entropy_multisentence(word, sentences, "right")
        ent = _effective_entropy(le, le_n, re, re_n)
        coh = _compute_cohesion(word, nc, cc, total)
        expected = (_norm_pmi(pmi) * 0.3 + _norm_llr(llr) * 0.3
                    + _norm_entropy(ent) * 0.2 + coh * 0.2)
        assert cand.score == pytest.approx(expected)
        assert cand.cng > 0.0 and cand.bm25 > 0.0

    def test_discover_reports_nonzero_diagnostics(self):
        res = discover(TestEndToEnd.TEXT)
        assert res.candidates
        assert any(c.cng > 0.0 for c in res.candidates)
        assert any(0.0 < c.bm25 < 1.0 for c in res.candidates)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
