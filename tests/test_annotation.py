"""Tests for the annotation contract (core/annotation.py) and its CLI.

No model weights and no LLM are needed: pre-labeling is exercised through the
dictionary engine, and the GLiNER/MTL paths are represented by hand-built
Prelabel objects.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from core.annotation import (
    AgreementTier,
    BookRecord,
    CorpusManifest,
    GoldSpan,
    Prelabel,
    SectionRecord,
    content_sha256,
    merge_proposals,
    review_workload,
    retriage,
    split_book,
    suggest_spans,
    to_eval_jsonl,
    triage,
    validate_section,
)


def _section(text: str, **kw) -> SectionRecord:
    return SectionRecord(section_id=kw.pop("sid", "b#0"), book_id=kw.pop("book", "b"),
                         text=text, offset=kw.pop("offset", (0, len(text))), **kw)


def _pre(span: tuple[int, int], text: str, category: str = "MATERIAL",
         **sources: bool) -> Prelabel:
    return Prelabel(text=text, category=category, span=span,
                    sources=sources or {"dictionary": True})


# ── splitting ──

def test_split_covers_the_book_without_overlap():
    text = "\n".join(["第一段。" * 30, "第二段。" * 30, "第三段。" * 30])
    spans = split_book(text, max_chars=120)
    assert spans, "expected at least one section"
    assert spans[0][0] == 0
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert e1 <= s2, "sections must not overlap"
    assert "".join(text[s:e] for s, e in spans) == text.replace("\n", ""), \
        "every non-blank character must land in exactly one section"


def test_split_respects_max_chars_and_paragraphs():
    text = "甲" * 300 + "\n" + "乙" * 50
    for s, e in split_book(text, max_chars=200):
        assert e - s <= 201, "section exceeded max_chars"
    assert len(split_book(text, max_chars=1000)) == 1, \
        "two paragraphs within budget stay one section"


def test_split_hard_cuts_a_single_line_book_without_boundaries():
    """A one-line file with no sentence-end marks must still honor max_chars."""
    text = "甲" * 1000
    spans = split_book(text, max_chars=300)
    assert all(e - s <= 300 for s, e in spans), "no section may exceed the budget"
    assert "".join(text[s:e] for s, e in spans) == text, "hard cut must not drop text"


def test_split_empty_input():
    assert split_book("") == []
    assert split_book("\n\n  \n") == []


def test_content_sha256_is_stable_and_distinct():
    assert content_sha256("甲") == content_sha256("甲")
    assert content_sha256("甲") != content_sha256("乙")


# ── triage ──

def test_single_engine_cannot_produce_agreement():
    """Regression: one engine used to read as 'all engines agree'."""
    assert triage(_pre((0, 2), "碳钢", dictionary=True)) == AgreementTier.PARTIAL


def test_two_engines_same_span_are_agreement():
    assert triage(_pre((0, 2), "碳钢", dictionary=True, gliner=True)) == AgreementTier.AGREE


def test_two_engines_disagreeing_on_category_is_conflict():
    p = _pre((0, 2), "碳钢", dictionary=True, gliner=True)
    p.disputed = True
    assert triage(p) == AgreementTier.CONFLICT


def test_quality_demotion_is_conflict_regardless_of_agreement():
    p = _pre((0, 2), "碳钢", dictionary=True, gliner=True)
    p.demoted = True
    assert triage(p) == AgreementTier.CONFLICT


def test_retriage_assigns_tiers_in_place():
    rec = _section("碳钢是钢的一种。",
                   prelabels=[_pre((0, 2), "碳钢", dictionary=True, gliner=True),
                              _pre((3, 4), "钢", dictionary=True)])
    retriage(rec)
    assert [p.tier for p in rec.prelabels] == [AgreementTier.AGREE, AgreementTier.PARTIAL]


# ── models ──

def test_gold_span_rejects_unknown_category():
    with pytest.raises(Exception):
        GoldSpan(text="碳钢", category="NOT_A_CATEGORY", span=(0, 2))


def test_section_status_is_validated():
    with pytest.raises(Exception):
        _section("文本", status="done")


# ── validation ──

def test_validate_catches_offset_text_mismatch():
    rec = _section("碳纤维复合材料",
                   labels=[GoldSpan(text="高强度", category="MATERIAL", span=(0, 3))])
    assert any("mismatch" in e for e in validate_section(rec))


def test_validate_catches_out_of_bounds_and_duplicates():
    rec = _section("碳钢是钢", labels=[
        GoldSpan(text="碳钢", category="MATERIAL", span=(0, 2)),
        GoldSpan(text="碳钢", category="MATERIAL", span=(0, 2)),
        GoldSpan(text="钢", category="MATERIAL", span=(3, 9))])
    errors = validate_section(rec)
    assert any("out of section bounds" in e for e in errors)
    assert any("duplicate span" in e for e in errors)


def test_validate_rejects_labeled_without_gold():
    assert validate_section(_section("碳钢", status="labeled"))


def test_validate_accepts_a_consistent_section():
    rec = _section("碳纤维复合材料", status="labeled",
                   labels=[GoldSpan(text="碳纤维", category="MATERIAL", span=(0, 3))])
    assert validate_section(rec) == []


# ── export + workload ──

def test_eval_export_only_includes_labeled_sections():
    labeled = _section("碳钢是钢。", status="labeled",
                       labels=[GoldSpan(text="碳钢", category="MATERIAL", span=(0, 2))])
    pending = _section("钛合金。", sid="b#1", status="pending")
    assert to_eval_jsonl([labeled, pending]) == [
        {"text": "碳钢是钢。", "entities": [{"text": "碳钢", "category": "MATERIAL"}]}]


def test_workload_counts_engines_and_review_items():
    rec = _section("碳钢是钢。", prelabels=[
        _pre((0, 2), "碳钢", dictionary=True, gliner=True),
        _pre((3, 4), "钢", dictionary=True)])
    stats = review_workload([retriage(rec)])
    assert stats["spans_agree"] == 1 and stats["spans_partial"] == 1
    assert stats["engines_seen"] == 2
    assert stats["human_review_items"] == 2  # 1 partial + 0 conflict + ceil(1*0.05)


def test_workload_reports_stale_tiers_as_partial_not_agree():
    """Tiers are stored; an un-retriaged corpus must not look auto-accepted."""
    rec = _section("碳钢是钢。", prelabels=[_pre((0, 2), "碳钢", dictionary=True, gliner=True)])
    assert review_workload([rec])["spans_agree"] == 0


def test_workload_on_empty_corpus_does_not_divide_by_zero():
    assert review_workload([])["auto_accept_share"] == 0.0


# ── manifest + storage ──

def test_manifest_roundtrip(tmp_path, monkeypatch):
    import core.annotation as ann

    monkeypatch.setattr(ann, "ANNOTATION_DIR", tmp_path)
    monkeypatch.setattr(ann, "MANIFEST_PATH", tmp_path / "manifest.json")
    ann.save_manifest(CorpusManifest(books=[
        BookRecord(book_id="b1", title="书一", license="CC-BY-4.0", split="dev")]))
    loaded = ann.load_manifest()
    assert loaded.book("b1").license == "CC-BY-4.0"
    assert loaded.book("missing") is None


def test_sections_roundtrip_keeps_spans(tmp_path, monkeypatch):
    import core.annotation as ann

    monkeypatch.setattr(ann, "ANNOTATION_DIR", tmp_path)
    rec = _section("碳纤维复合材料", status="labeled",
                   labels=[GoldSpan(text="碳纤维", category="MATERIAL", span=(0, 3))])
    ann.write_sections("b2", [rec])
    back = list(ann.iter_sections("b2"))
    assert back[0].labels[0].span == (0, 3)
    assert validate_section(back[0]) == []


# ── pre-labeling ──

def test_dictionary_spans_return_offsets_that_match_the_text():
    from core.annotation import dictionary_spans

    text = "碳纤维复合材料与钢的疲劳对比。"
    for surface, category, (start, end) in dictionary_spans(text):
        assert text[start:end] == surface
        assert category in ("MATERIAL", "STANDARD", "PARAMETER")


def test_suggest_spans_without_gliner_is_never_agree():
    spans = suggest_spans("碳钢具有高强度，按 GB/T 3001 检测。")
    assert spans
    assert all(s.tier == AgreementTier.PARTIAL for s in spans)


def test_suggest_spans_with_gliner_can_reach_agree():
    class FakeEngine:
        def predict_entities(self, text, labels, threshold=0.5):
            i = text.index("钢", 2)  # the standalone occurrence, not the one in 碳钢
            return [{"text": "钢", "label": "MATERIAL", "start": i, "end": i + 1}]

    spans = suggest_spans("碳钢是钢的一种。", gliner_engine=FakeEngine())
    agree = [s for s in spans if s.tier == AgreementTier.AGREE]
    assert any(s.text == "钢" for s in agree), "dictionary+gliner overlap must read as agreement"


def test_unknown_label_is_not_a_category_conflict():
    """An engine that cannot classify is not an engine that disagrees."""
    class FakeEngine:
        def predict_entities(self, text, labels, threshold=0.5):
            i = text.index("钢", 2)
            return [{"text": "钢", "label": "怪标签", "start": i, "end": i + 1}]

    spans = suggest_spans("碳钢是钢的一种。", gliner_engine=FakeEngine())
    hit = [s for s in spans if s.text == "钢"]
    assert hit and not hit[0].disputed
    assert hit[0].category == "MATERIAL" and hit[0].tier == AgreementTier.AGREE


def test_engine_boundary_disagreement_is_not_agreement():
    """An engine that cuts 钢 out of 碳钢 disagrees on the boundary, not agrees."""
    class FakeEngine:
        def predict_entities(self, text, labels, threshold=0.5):
            return [{"text": "钢", "label": "MATERIAL", "start": 1, "end": 2}]

    spans = suggest_spans("碳钢是钢的一种。", gliner_engine=FakeEngine())
    assert not [s for s in spans if s.tier == AgreementTier.AGREE and s.text == "钢"]


def test_merge_proposals_flags_category_conflict():
    spans = merge_proposals([
        ("dictionary", [("钛合金", "MATERIAL", (0, 3))]),
        ("gliner", [("钛合金", "PARAMETER", (0, 3))]),
    ], text_len=10)
    assert len(spans) == 1
    assert spans[0].disputed and spans[0].tier == AgreementTier.CONFLICT


def test_merge_proposals_drops_spans_beyond_the_text():
    spans = merge_proposals([("dictionary", [("越界", "MATERIAL", (5, 50))])], text_len=10)
    assert spans == []


def test_dictionary_spans_are_non_overlapping_longest_match():
    """The dictionary holds both GB and GB/T, 钢 and 碳钢 — never nest them.

    Nested hits inflate the count and become phantom CONFLICTs once a second
    engine proposes only the longer span.
    """
    from core.annotation import dictionary_spans

    text = "参照 GB/T 3001，碳钢与钢的强度不同。"
    found = dictionary_spans(text)
    spans = sorted(s for _, _, s in found)
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        assert a2 >= b1, f"nested spans: {(a1, b1)} inside {(a2, b2)}"

    by_start = {a: text[a:b] for _, _, (a, b) in found}
    gb_at = text.index("GB")
    assert by_start[gb_at] == "GB/T", "GB must not be emitted inside a GB/T reference"
    # A separate occurrence of the shorter term is still a legitimate hit.
    assert any(s == "钢" for s in by_start.values())
    assert any(s == "碳钢" for s in by_start.values())
    assert any(s == "强度" for s in by_start.values())


# ── CLI ──

def test_cli_split_prelabel_validate_stats_export(tmp_path):
    store, book = tmp_path / "store", tmp_path / "book.txt"
    book.write_text("碳纤维复合材料需按 GB/T 3001 检测。\n" + "钢的塑性参数见附表。\n" * 4,
                    encoding="utf-8")
    env = {**os.environ, "NARRATIVE_ANNOTATION_DIR": str(store)}
    cli = str(_project_root / "scripts" / "label_corpus.py")

    def run(*args):
        return subprocess.run([sys.executable, cli, *args], env=env,
                              capture_output=True, text=True)

    r = run("split", "--book", str(book), "--book-id", "clibook", "--license",
            "CC-BY-4.0", "--split", "dev", "--max-chars", "60")
    assert r.returncode == 0, r.stderr
    assert (store / "clibook.jsonl").exists()

    r = run("prelabel", "--book-id", "clibook")
    assert r.returncode == 0, r.stderr
    assert "engines_seen=1" in r.stdout, "dictionary-only run must report one engine"
    assert "cannot produce agreement" in r.stdout

    assert run("validate", "--book-id", "clibook").returncode == 0
    assert "license=CC-BY-4.0" in run("stats", "--book-id", "clibook").stdout

    r = run("export", "--book-id", "clibook", "--out", str(tmp_path / "eval.jsonl"))
    assert r.returncode == 1, "nothing is labeled yet, so export must refuse"
    assert "empty" in r.stderr

