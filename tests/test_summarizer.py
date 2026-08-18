"""
Tests: Extractive Summarizer.

Validates NSP+TextRank hybrid summarizer, dual modes,
language weights, fusion_weights traceability, and edge cases.
"""
from __future__ import annotations
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from core.schema import (
    Entity, Event, KeySentence, NarrativeContent, NarrativeDocument,
    NarrativeMeta, Relation, SentenceLanguage, Summary,
)
from core.summarizer import (
    summarize, summarize_text,
    _compute_nsp_score, _compute_textrank, _detect_dominant_language,
    _select_sentences, _DEFAULT_WEIGHTS,
)


def _make_sentence(text: str, span: tuple[int, int] | None = None, label: str = "modern") -> SentenceLanguage:
    if span is None:
        span = (0, len(text))
    return SentenceLanguage(text=text, span=span, label=label, confidence=0.8)


def _make_entity(text: str, span: tuple[int, int]) -> Entity:
    return Entity(id="ent_0", text=text, category="PERSON", span=span)


def _make_relation(span: tuple[int, int]) -> Relation:
    return Relation(id="rel_0", subject="s", predicate="RELATES_TO", object="o", evidence="e", evidence_span=span)


def _make_event(span: tuple[int, int]) -> Event:
    return Event(id="evt_0", event_type="T", trigger="v", trigger_span=span)


def _make_content(sentences, entities=None, relations=None, events=None) -> NarrativeContent:
    return NarrativeContent(
        sentences=sentences, entities=entities or [],
        relations=relations or [], events=events or [],
    )


def _make_doc(content: NarrativeContent) -> NarrativeDocument:
    return NarrativeDocument(
        meta=NarrativeMeta(source="test", text_length=100, language_mode="auto"),
        content=content,
    )


class TestNSPScore:
    def test_empty_sentence(self):
        sent = _make_sentence("...")
        content = _make_content([sent])
        score, reasons = _compute_nsp_score(sent, content)
        assert score == 0.0
        assert "empty_filtered" in reasons

    def test_entity_density(self):
        sent = _make_sentence("张三来了", (0, 4))
        ent = _make_entity("张三", (0, 2))
        content = _make_content([sent], entities=[ent])
        score, reasons = _compute_nsp_score(sent, content)
        assert score > 0
        assert any("entity_density" in r for r in reasons)

    def test_event_density(self):
        sent = _make_sentence("张三来了", (0, 4))
        evt = _make_event((2, 4))
        content = _make_content([sent], events=[evt])
        score, reasons = _compute_nsp_score(sent, content)
        assert score > 0
        assert any("event_density" in r for r in reasons)

    def test_relation_density(self):
        sent = _make_sentence("张三来了", (0, 4))
        rel = _make_relation((0, 4))
        content = _make_content([sent], relations=[rel])
        score, reasons = _compute_nsp_score(sent, content)
        assert score > 0
        assert any("relation_density" in r for r in reasons)

    def test_low_value_penalty(self):
        sent = _make_sentence("来源：xxx")
        content = _make_content([sent])
        score, reasons = _compute_nsp_score(sent, content)
        assert "low_value_pattern" in reasons


class TestTextRank:
    def test_empty(self):
        assert _compute_textrank([]) == {}

    def test_single(self):
        sents = [_make_sentence("hello", (0, 5))]
        scores = _compute_textrank(sents)
        assert scores == {0: 1.0}

    def test_convergence(self):
        sents = [
            _make_sentence("张三来了", (0, 4)),
            _make_sentence("张三走了", (4, 8)),
            _make_sentence("李四来了", (8, 12)),
        ]
        scores = _compute_textrank(sents)
        assert len(scores) == 3
        for s in scores.values():
            assert 0.0 <= s <= 1.0

    def test_overlapping_sentences(self):
        sents = [
            _make_sentence("张三来了张三走了", (0, 10)),
            _make_sentence("张三来了", (0, 4)),
        ]
        scores = _compute_textrank(sents)
        assert all(0.0 <= s <= 1.0 for s in scores.values())


class TestDominantLanguage:
    def test_modern_default(self):
        assert _detect_dominant_language([]) == "modern"

    def test_classical_dominant(self):
        sents = [
            _make_sentence("学而时习之", (0, 5), "classical"),
            _make_sentence("有朋自远方来", (5, 11), "classical"),
            _make_sentence("hello", (11, 16), "english"),
        ]
        assert _detect_dominant_language(sents) == "classical"

    def test_mixed(self):
        sents = [_make_sentence("hello", (0, 5), "english"), _make_sentence("world", (5, 10), "english")]
        assert _detect_dominant_language(sents) == "english"


class TestDefaultWeights:
    def test_weights_defined(self):
        for lang in ("classical", "modern", "english"):
            assert lang in _DEFAULT_WEIGHTS
            nsp, tr = _DEFAULT_WEIGHTS[lang]
            assert 0.0 <= nsp <= 1.0 and 0.0 <= tr <= 1.0


class TestSelectSentences:
    def test_empty(self):
        assert _select_sentences([], "chars", 30, 0.2, 100) == []

    def test_short_text_returns_all(self):
        sents = [_make_sentence("short", (0, 5))]
        scored = [(0, 0.5, ["reason"], sents[0])]
        result = _select_sentences(scored, "chars", 100, 0.2, 5)
        assert len(result) == 1

    def test_greedy_selection(self):
        sents = [
            _make_sentence("a" * 20, (0, 20)),
            _make_sentence("b" * 20, (20, 40)),
            _make_sentence("c" * 20, (40, 60)),
        ]
        scored = [(1, 0.9, ["high"], sents[1]), (0, 0.8, ["med"], sents[0]), (2, 0.5, ["low"], sents[2])]
        result = _select_sentences(scored, "chars", 30, 0.2, 60)
        assert len(result) >= 1
        indices = [r[0] for r in result]
        assert indices == sorted(indices)

    def test_ratio_mode(self):
        sents = [_make_sentence("a" * 30, (0, 30)), _make_sentence("b" * 30, (30, 60))]
        scored = [(0, 0.8, [], sents[0]), (1, 0.6, [], sents[1])]
        result = _select_sentences(scored, "ratio", 30, 0.5, 60)
        assert len(result) >= 1


class TestSummarize:
    def test_empty_sentences(self):
        doc = _make_doc(NarrativeContent())
        summary = summarize(doc)
        assert summary.key_sentences == []
        assert summary.summary_text == ""
        assert summary.total_sentences == 0

    def test_basic_chars_mode(self):
        sents = [_make_sentence("张三来了", (0, 4)), _make_sentence("李四走了", (4, 8))]
        ent = _make_entity("张三", (0, 2))
        content = _make_content(sents, entities=[ent])
        doc = _make_doc(content)
        summary = summarize(doc, mode="chars", target_chars=20)
        assert summary.mode == "chars"
        assert summary.target_chars == 20
        assert summary.total_sentences == 2
        assert len(summary.key_sentences) >= 1
        assert summary.fusion_weights.get("nsp_weight") is not None

    def test_ratio_mode(self):
        sents = [_make_sentence("a" * 20, (0, 20)), _make_sentence("b" * 20, (20, 40))]
        content = _make_content(sents)
        doc = _make_doc(content)
        summary = summarize(doc, mode="ratio", target_ratio=0.5)
        assert summary.mode == "ratio"
        assert summary.target_ratio == 0.5

    def test_custom_weights(self):
        sents = [_make_sentence("test", (0, 4))]
        content = _make_content(sents)
        doc = _make_doc(content)
        summary = summarize(doc, nsp_weight=0.9, textrank_weight=0.1)
        fw = summary.fusion_weights
        assert abs(fw["nsp_weight"] - 0.9) < 0.01
        assert abs(fw["textrank_weight"] - 0.1) < 0.01

    def test_fusion_weights_traceability(self):
        sents = [_make_sentence("test", (0, 4), "classical")]
        content = _make_content(sents)
        doc = _make_doc(content)
        summary = summarize(doc)
        fw = summary.fusion_weights
        assert "default_classical_nsp" in fw
        assert "default_modern_nsp" in fw
        assert "default_english_nsp" in fw

    def test_classical_weight_applied(self):
        sents = [_make_sentence("学而时习之", (0, 5), "classical")]
        content = _make_content(sents)
        doc = _make_doc(content)
        summary = summarize(doc)
        fw = summary.fusion_weights
        assert fw["default_nsp"] == 0.85
        assert fw["default_textrank"] == 0.15

    def test_span_from_original(self):
        sents = [_make_sentence("第一句", (0, 3)), _make_sentence("第二句", (3, 6))]
        content = _make_content(sents)
        doc = _make_doc(content)
        summary = summarize(doc, mode="chars", target_chars=100)
        for ks in summary.key_sentences:
            assert isinstance(ks.span, tuple)
            assert ks.span[0] >= 0

    def test_serialization(self):
        sents = [_make_sentence("test", (0, 4))]
        content = _make_content(sents)
        doc = _make_doc(content)
        summary = summarize(doc)
        d = summary.model_dump()
        assert "key_sentences" in d
        assert "fusion_weights" in d



class TestSummarizeText:
    def test_empty(self):
        summary = summarize_text("")
        assert summary.key_sentences == []
        assert summary.summary_text == ""
        assert summary.method == "textrank_only"

    def test_basic(self):
        text = "张三来了。李四走了。王五到了。"
        summary = summarize_text(text, mode="chars", target_chars=10)
        assert summary.method == "textrank_only"
        assert summary.total_sentences >= 1
        assert len(summary.key_sentences) >= 1

    def test_ratio_mode(self):
        text = "a" * 100 + ". " + "b" * 100 + "."
        summary = summarize_text(text, mode="ratio", target_ratio=0.5)
        assert summary.mode == "ratio"
        assert len(summary.key_sentences) >= 1

    def test_language_override(self):
        text = "学而时习之。有朋自远方来。"
        summary = summarize_text(text, language="classical")
        assert summary.total_sentences >= 1

    def test_short_text_returns_all(self):
        text = "短文本"
        summary = summarize_text(text, mode="chars", target_chars=100)
        assert len(summary.summary_text) > 0


class TestKeySentence:
    def test_creation(self):
        ks = KeySentence(sentence_index=0, text="test", span=(0, 4), score=0.8, reasons=["high"], source="summarizer")
        assert ks.sentence_index == 0
        assert ks.text == "test"

    def test_invalid_span(self):
        with pytest.raises(Exception):
            KeySentence(sentence_index=0, text="test", span=(-1, 4), score=0.8, reasons=[], source="s")


class TestSummary:
    def test_default_values(self):
        s = Summary()
        assert s.key_sentences == []
        assert s.summary_text == ""
        assert s.method == "nsp_textrank"
        assert s.mode == "chars"
        assert s.target_chars == 30
        assert s.target_ratio == 0.2

    def test_serialization_roundtrip(self):
        ks = KeySentence(sentence_index=0, text="t", span=(0, 1), score=0.5, reasons=["r"], source="s")
        s = Summary(
            key_sentences=[ks], summary_text="t", method="nsp_textrank", mode="chars",
            target_chars=30, target_ratio=0.2,
            fusion_weights={"nsp_weight": 0.6, "textrank_weight": 0.4}, total_sentences=1,
        )
        d = s.model_dump()
        assert d["key_sentences"][0]["text"] == "t"
        assert d["fusion_weights"]["nsp_weight"] == 0.6

