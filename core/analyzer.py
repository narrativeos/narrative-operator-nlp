"""
Unified NLP Analysis Entry Point — Two-Stage Hybrid Pipeline.

Splits text → detects language per sentence → routes to appropriate
HanLP model → falls back if output quality is poor → merges results
with global offset remapping.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .language_detector import classify, should_fallback, classical_confidence
from .mapper import HanlpSchemaMapper
from .schema import NarrativeContent, NarrativeDocument, NarrativeMeta, Token

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded HanLP pipelines
# ---------------------------------------------------------------------------

_modern_pipeline: Optional[object] = None
_classical_pipeline: Optional[object] = None
_mapper: Optional[HanlpSchemaMapper] = None


def _get_modern_pipeline():
    global _modern_pipeline
    if _modern_pipeline is None:
        import hanlp
        _modern_pipeline = hanlp.load(
            hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH
        )
        logger.info("Modern Chinese HanLP pipeline loaded (ELECTRA-small).")
    return _modern_pipeline


def _get_classical_pipeline():
    global _classical_pipeline
    if _classical_pipeline is None:
        import hanlp
        _classical_pipeline = hanlp.load(
            hanlp.pretrained.mtl.KYOTO_EVAHAN_TOK_LEM_POS_UDEP_LZH
        )
        logger.info("Classical Chinese HanLP pipeline loaded (KYOTO-EVAHAN).")
    return _classical_pipeline


def _get_mapper() -> HanlpSchemaMapper:
    global _mapper
    if _mapper is None:
        _mapper = HanlpSchemaMapper()
    return _mapper


# ---------------------------------------------------------------------------
# Sentence Splitting
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；\n])\s*")


def _split_sentences(text: str) -> list[tuple[str, int]]:
    """Split text into (sentence_text, start_offset) pairs."""
    sentences: list[tuple[str, int]] = []
    parts = _SENT_SPLIT_RE.split(text)
    cursor = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        idx = text.find(part, cursor)
        if idx < 0:
            idx = cursor
        sentences.append((part, idx))
        cursor = idx + len(part)
    return sentences


# ---------------------------------------------------------------------------
# Quality Assessment
# ---------------------------------------------------------------------------

def _assess_quality(tokens: list[Token], raw: dict) -> bool:
    """Heuristic quality check. Returns False if output looks like garbage."""
    if not tokens:
        return False
    # >60% NR (proper noun) → model doesn't know these words
    nr_count = sum(1 for t in tokens if t.pos == "NR")
    if nr_count > len(tokens) * 0.6:
        return False
    # >70% "root"/"dep" in dependency → model guessing
    dep_data = raw.get("dep", [])
    if dep_data and len(dep_data) > 3:
        bad = sum(1 for d in dep_data
                  if isinstance(d, (list, tuple)) and len(d) >= 2
                  and str(d[1]).lower() in ("root", "dep"))
        if bad / len(dep_data) > 0.7:
            return False
    return True


# ---------------------------------------------------------------------------
# Per-Segment Analysis
# ---------------------------------------------------------------------------

def _analyze_modern(text: str, offset: int, mapper: HanlpSchemaMapper,
                    dict_combine: Optional[set] = None
                    ) -> tuple[list, list, list, list, bool]:
    try:
        pipeline = _get_modern_pipeline()
        if dict_combine:
            try:
                pipeline['tok/fine'].dict_combine = dict_combine
            except (KeyError, AttributeError):
                pass
        raw = pipeline(text)
    except Exception as exc:
        logger.warning("Modern pipeline failed: %s", exc)
        return [], [], [], [], False
    doc = mapper.map(text, raw, source="hanlp_v2")
    _apply_offset(doc, offset)
    return (doc.content.tokens, doc.content.entities, doc.content.relations,
            doc.content.patterns, _assess_quality(doc.content.tokens, raw))


def _analyze_classical(text: str, offset: int, mapper: HanlpSchemaMapper
                       ) -> tuple[list, list, list, bool]:
    try:
        raw = _get_classical_pipeline()(text)
    except (AttributeError, ImportError):
        logger.warning("Classical Chinese model not available. Skipping.")
        return [], [], [], False
    except Exception as exc:
        logger.warning("Classical pipeline failed: %s", exc)
        return [], [], [], False
    normalized = _normalize_lzh_keys(raw)
    doc = mapper.map(text, normalized, source="hanlp_lzh")
    _apply_offset(doc, offset)
    for t in doc.content.tokens:
        t.source = "hanlp_lzh"
    return (doc.content.tokens, doc.content.entities, doc.content.relations,
            doc.content.patterns, _assess_quality(doc.content.tokens, normalized))


def _apply_offset(doc: NarrativeDocument, offset: int):
    for t in doc.content.tokens:
        t.span = (t.span[0] + offset, t.span[1] + offset)
    for e in doc.content.entities:
        e.span = (e.span[0] + offset, e.span[1] + offset)
    for r in doc.content.relations:
        r.evidence_span = (r.evidence_span[0] + offset, r.evidence_span[1] + offset)


# ---------------------------------------------------------------------------
# Classical Key Normalization (lzh_* → standard mapper keys)
# ---------------------------------------------------------------------------

_LZH_KEY_MAP = {
    "lzh_tok_fine": "tok/fine",
    "lzh_tok_coarse": "tok/coarse",
    "lzh_pos_upos": "pos/ctb",
    "lzh_pos_xpos": "pos/pku",
    "lzh_pos_pku": "pos/pku",
    "lzh_dep": "dep",
    "lzh_lem": None,
}


def _normalize_lzh_keys(raw: dict) -> dict:
    normalized = {}
    for key, value in raw.items():
        mapped = _LZH_KEY_MAP.get(key, key)
        if mapped is not None:
            normalized[mapped] = value
    normalized.setdefault("ner/pku", [])
    normalized.setdefault("ner/msra", [])
    normalized.setdefault("ner/ontonotes", [])
    normalized.setdefault("srl", [])
    normalized.setdefault("con", None)
    return normalized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(text: str, dict_combine: Optional[set] = None) -> NarrativeDocument:
    """
    Analyze text with automatic language detection and model routing.

    Modern Chinese → MTL (ELECTRA-small)
    Classical Chinese → LZH (KYOTO-EVAHAN)
    Mixed text → per-sentence routing with fallback

    Returns a single NarrativeDocument with global offsets.
    """
    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")

    mapper = _get_mapper()
    sentences = _split_sentences(text)

    # Group consecutive same-language sentences
    segments: list[tuple[str, int, str, float]] = []
    for sent_text, sent_offset in sentences:
        lang, conf = classify(sent_text)
        segments.append((sent_text, sent_offset, lang, conf))

    merged: list[tuple[str, int, str, float]] = []
    for sent_text, offset, lang, conf in segments:
        if merged and merged[-1][2] == lang:
            prev_text, prev_offset, _, _ = merged[-1]
            merged[-1] = (text[prev_offset:offset + len(sent_text)], prev_offset, lang, conf)
        else:
            merged.append((sent_text, offset, lang, conf))

    # Process each segment
    all_tokens, all_entities, all_relations, all_patterns = [], [], [], []
    for seg_text, seg_offset, lang, conf in merged:
        if lang == "classical":
            tokens, entities, relations, patterns, ok = _analyze_modern(seg_text, seg_offset, mapper, dict_combine)
            if not ok and should_fallback(conf):
                logger.info("Classical→Modern fallback for: %s...", seg_text[:20])
                tokens, entities, relations, patterns, _ = _analyze_modern(seg_text, seg_offset, mapper, dict_combine)
        else:
            tokens, entities, relations, patterns, ok = _analyze_modern(seg_text, seg_offset, mapper, dict_combine)
            if not ok and should_fallback(conf):
                logger.info("Modern→Classical fallback for: %s...", seg_text[:20])
                tokens, entities, relations, patterns, _ = _analyze_classical(seg_text, seg_offset, mapper)
        all_tokens.extend(tokens)
        all_entities.extend(entities)
        all_relations.extend(relations)
        all_patterns.extend(patterns)

    # Re-number token IDs globally, sorted by position
    for i, t in enumerate(sorted(all_tokens, key=lambda t: t.span[0])):
        t.id = i

    sources = sorted(set(t.source for t in all_tokens if t.source))
    meta_source = "+".join(sources) if sources else "hanlp_v2"

    return NarrativeDocument(
        meta=NarrativeMeta(source=meta_source, text_length=len(text)),
        content=NarrativeContent(tokens=all_tokens, entities=all_entities,
                                 relations=all_relations, patterns=all_patterns, structural={}),
    )
