"""Tests: Entity Quality Pipeline (F0/F1/F2) + Noun Signals.

Covers:
- F0: HTML tag stripping
- F1: section-ref relabeling + bare-number demotion
- F2: per-source confidence gate, category corroboration, keyword-injection
- Noun signals: Step A (POS gating) + Step B (syntactic role)
- Backward compatibility: policy=None / noun_signals=None are no-ops
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from core.schema import DependencyEdge, Entity, EntityCategory, Token
from core.entity_quality import (
    QualityPolicy,
    apply_entity_quality,
    clean_html,
)
from core.noun_signal import NounSignalConfig, extract_noun_signals


# ---------------------------------------------------------------------------
# F0: HTML cleaning
# ---------------------------------------------------------------------------

class TestF0CleanHtml:
    def test_strip_sub_tag(self):
        assert clean_html("x<sub>6</sub>y") == "x6y"

    def test_strip_sup_tag(self):
        assert clean_html("H<sup>2</sup>O") == "H2O"

    def test_strip_br(self):
        assert clean_html("a<br>b") == "ab"

    def test_strip_tag_with_attrs(self):
        assert clean_html('a <span class="x">b</span> c') == "a b c"

    def test_decode_entities(self):
        assert clean_html("a&mdash;b") == "a—b"
        assert clean_html("x&amp;y") == "x&y"

    def test_empty_and_plain(self):
        assert clean_html("") == ""
        assert clean_html("plain text") == "plain text"


# ---------------------------------------------------------------------------
# F1: Shape / structure
# ---------------------------------------------------------------------------

def _ent(text, category, source="ner/ontonotes", confidence=1.0, eid="ent_001"):
    return Entity(
        id=eid, text=text, category=category, span=(0, len(text)),
        source=source, confidence=confidence,
    )


class TestF1Shape:
    def test_section_ref_relabel(self):
        pol = QualityPolicy()
        e = _ent("4.3.2", EntityCategory.NUMBER)
        apply_entity_quality([e], pol)
        assert e.category == EntityCategory.SECTION_REF
        assert e.keep is True

    def test_bare_number_demoted(self):
        pol = QualityPolicy()
        e = _ent("100", EntityCategory.NUMBER)
        apply_entity_quality([e], pol)
        assert e.keep is False
        assert e.filter == "F1_shape"

    def test_number_with_unit_not_bare(self):
        pol = QualityPolicy()
        # "100MΩ" has a unit → meaningful measured value → kept (even from NER).
        e = _ent("100MΩ", EntityCategory.NUMBER)
        apply_entity_quality([e], pol)
        assert e.keep is True
        assert e.filter is None

    def test_number_with_percent_kept(self):
        pol = QualityPolicy()
        # "70%" has a unit (%) → kept even though from ner/ontonotes.
        e = _ent("70%", EntityCategory.NUMBER, source="ner/ontonotes", confidence=1.0)
        apply_entity_quality([e], pol)
        assert e.keep is True

    def test_section_ref_disabled(self):
        pol = QualityPolicy.from_dict(
            {"f1_shape": {"section_ref": {"enabled": False}}}
        )
        e = _ent("4.3.2", EntityCategory.NUMBER)
        apply_entity_quality([e], pol)
        assert e.category == EntityCategory.NUMBER  # not relabeled


# ---------------------------------------------------------------------------
# F2: Confidence / evidence gates
# ---------------------------------------------------------------------------

class TestF2Confidence:
    def test_number_numeric_percentage_kept(self):
        pol = QualityPolicy()
        e = _ent("50%", EntityCategory.NUMBER, source="numeric/percentage", confidence=0.75)
        apply_entity_quality([e], pol)
        assert e.keep is True

    def test_number_ner_demoted(self):
        pol = QualityPolicy()
        # 7-digit number: not a bare short int (F1 skips), not a valid date,
        # and from ner/ontonotes (not numeric/*) → F2 demotes it.
        e = _ent("1000000", EntityCategory.NUMBER, source="ner/ontonotes", confidence=1.0)
        apply_entity_quality([e], pol)
        assert e.keep is False
        assert e.filter == "F2_confidence"

    def test_number_invalid_date_demoted(self):
        pol = QualityPolicy()
        # 8-digit ID (month 63 invalid) from numeric/date → not a valid date.
        e = _ent("63906433", EntityCategory.NUMBER, source="numeric/date", confidence=0.75)
        apply_entity_quality([e], pol)
        assert e.keep is False

    def test_number_valid_date_kept(self):
        pol = QualityPolicy()
        e = _ent("20240115", EntityCategory.NUMBER, source="numeric/date", confidence=0.75)
        apply_entity_quality([e], pol)
        assert e.keep is True

    def test_standard_code_kept(self):
        pol = QualityPolicy()
        e = _ent("GB50150", EntityCategory.STANDARD, source="ner/ontonotes", confidence=0.9)
        apply_entity_quality([e], pol)
        assert e.keep is True

    def test_standard_generic_demoted(self):
        pol = QualityPolicy()
        e = _ent("标准", EntityCategory.STANDARD, source="ner/ontonotes", confidence=0.9)
        apply_entity_quality([e], pol)
        assert e.keep is False

    def test_product_low_conf_demoted(self):
        pol = QualityPolicy()
        e = _ent("某产品", EntityCategory.PRODUCT, source="ner/ontonotes", confidence=0.6)
        apply_entity_quality([e], pol)
        assert e.keep is False

    def test_product_high_conf_kept(self):
        pol = QualityPolicy()
        e = _ent("某产品", EntityCategory.PRODUCT, source="ner/ontonotes", confidence=0.9)
        apply_entity_quality([e], pol)
        assert e.keep is True

    def test_unknown_low_conf_demoted(self):
        pol = QualityPolicy()
        e = _ent("绝缘电阻", EntityCategory.UNKNOWN, source="pos/nnp", confidence=0.55)
        apply_entity_quality([e], pol)
        assert e.keep is False

    def test_person_never_demoted(self):
        pol = QualityPolicy()
        e = _ent("张三", EntityCategory.PERSON, source="ner/pku", confidence=0.5)
        apply_entity_quality([e], pol)
        assert e.keep is True
        assert e.filter is None

    def test_keyword_require_injected(self):
        pol = QualityPolicy()
        e = _ent("石墨烯", EntityCategory.MATERIAL, source="keyword", confidence=0.8)
        apply_entity_quality([e], pol, injected_keywords=set())
        assert e.keep is False
        assert e.filter == "F2_confidence"

    def test_keyword_injected_kept(self):
        pol = QualityPolicy()
        e = _ent("石墨烯", EntityCategory.MATERIAL, source="keyword", confidence=0.8)
        apply_entity_quality([e], pol, injected_keywords={"石墨烯"})
        assert e.keep is True

    def test_by_source_threshold(self):
        pol = QualityPolicy.from_dict(
            {
                "f2_confidence": {
                    "enabled": True,
                    "min_confidence": {"default": 0.6, "by_source": {"pos/nnp": 0.55}},
                    "demote_categories": ["UNKNOWN"],
                }
            }
        )
        # UNKNOWN at 0.55: passes by_source (0.55 >= 0.55) but fails corroboration (< 0.75).
        e = _ent("x", EntityCategory.UNKNOWN, source="pos/nnp", confidence=0.55)
        apply_entity_quality([e], pol)
        assert e.keep is False

    def test_standard_code_in_unknown_kept(self):
        pol = QualityPolicy()
        # "GB50150" mis-tagged as NNP/UNKNOWN: standard code → kept, and the
        # low confidence (0.55) is exempted because the code is strong evidence.
        e = _ent("GB50150", EntityCategory.UNKNOWN, source="pos/nnp", confidence=0.55)
        apply_entity_quality([e], pol)
        assert e.keep is True
        assert e.filter is None

    def test_unit_value_exempt_from_conf_gate(self):
        pol = QualityPolicy()
        # "2500V" (UNKNOWN, low conf 0.55): unit-bearing value → corroboration
        # passes AND the by-source confidence gate is skipped.
        e = _ent("2500V", EntityCategory.UNKNOWN, source="pos/nnp", confidence=0.55)
        apply_entity_quality([e], pol)
        assert e.keep is True
        assert e.filter is None

    def test_generic_unknown_low_conf_demoted(self):
        pol = QualityPolicy()
        # A generic low-confidence NNP (no unit, not a standard code) → demoted.
        e = _ent("绝缘", EntityCategory.UNKNOWN, source="pos/nnp", confidence=0.5)
        apply_entity_quality([e], pol)
        assert e.keep is False
        assert e.filter == "F2_confidence"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_policy_none_is_noop(self):
        e = _ent("100", EntityCategory.NUMBER, source="ner/ontonotes", confidence=1.0)
        apply_entity_quality([e], None)
        assert e.keep is True
        assert e.filter is None
        assert e.filter_reason is None

    def test_empty_policy_dict_is_noop(self):
        assert QualityPolicy.from_dict(None) is None
        assert QualityPolicy.from_dict({}) is None
        e = _ent("100", EntityCategory.NUMBER)
        apply_entity_quality([e], QualityPolicy.from_dict(None))
        assert e.keep is True

    def test_noun_signals_default_disabled(self):
        tokens = [Token(id=0, text="绝缘电阻", pos="NN", span=(0, 4))]
        assert extract_noun_signals(tokens, [], None) == []
        assert extract_noun_signals(tokens, [], NounSignalConfig.from_dict(None)) == []


# ---------------------------------------------------------------------------
# Noun signals: Step A + Step B
# ---------------------------------------------------------------------------

class TestNounSignals:
    def _tokens_deps(self):
        tokens = [
            Token(id=0, text="测量", pos="VV", span=(0, 2)),
            Token(id=1, text="绝缘", pos="NN", span=(2, 4)),
            Token(id=2, text="电阻", pos="NN", span=(4, 6)),
            Token(id=3, text="张三", pos="NR", span=(7, 9)),
            Token(id=4, text="检查", pos="VV", span=(9, 11)),
        ]
        deps = [
            DependencyEdge(child=1, head=0, rel="dobj"),
            DependencyEdge(child=2, head=1, rel="nn"),
            DependencyEdge(child=3, head=4, rel="nsubj"),
            DependencyEdge(child=0, head=4, rel="conj"),
            DependencyEdge(child=4, head=-1, rel="root"),
        ]
        return tokens, deps

    def test_step_a_pos_gating(self):
        tokens, deps = self._tokens_deps()
        cfg = NounSignalConfig.from_dict({"enabled": True, "pos_whitelist": ["NN"]})
        sigs = extract_noun_signals(tokens, deps, cfg)
        # Only NN tokens (绝缘, 电阻) — NR (张三) excluded.
        assert all(s.pos == "NN" for s in sigs)
        assert {s.text for s in sigs} == {"绝缘", "电阻"}

    def test_step_b_syntactic_role(self):
        tokens, deps = self._tokens_deps()
        cfg = NounSignalConfig.from_dict({"enabled": True, "pos_whitelist": ["NN", "NR"]})
        sigs = {s.text: s for s in extract_noun_signals(tokens, deps, cfg)}
        assert sigs["绝缘"].syntactic_role == "Object"
        assert sigs["绝缘"].evidence["head_rel"] == "dobj"
        assert sigs["绝缘"].evidence["governing_verb"] == "测量"
        assert sigs["张三"].syntactic_role == "Subject"
        assert sigs["张三"].evidence["governing_verb"] == "检查"

    def test_score_ordering_and_cap(self):
        tokens, deps = self._tokens_deps()
        cfg = NounSignalConfig.from_dict({"enabled": True, "max_per_block": 2})
        sigs = extract_noun_signals(tokens, deps, cfg)
        assert len(sigs) <= 2
        # Sorted by score desc.
        assert sigs[0].score >= sigs[1].score

    def test_min_score_filter(self):
        tokens, deps = self._tokens_deps()
        cfg = NounSignalConfig.from_dict({"enabled": True, "min_score": 0.99})
        sigs = extract_noun_signals(tokens, deps, cfg)
        assert sigs == []

    def test_single_char_skipped(self):
        tokens = [Token(id=0, text="钢", pos="NN", span=(0, 1))]
        cfg = NounSignalConfig.from_dict({"enabled": True})
        assert extract_noun_signals(tokens, [], cfg) == []
