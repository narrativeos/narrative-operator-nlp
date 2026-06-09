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

from .language_detector import detect_language, classical_confidence, LanguageClass
from .mapper import HanlpSchemaMapper
from .schema import NarrativeContent, NarrativeDocument, NarrativeMeta, SentenceLanguage, Token

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded HanLP pipelines
# ---------------------------------------------------------------------------

_modern_pipeline: Optional[object] = None
_classical_pipeline: Optional[object] = None
_english_pipeline: Optional[object] = None
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


def _get_english_pipeline():
    global _english_pipeline
    if _english_pipeline is None:
        import hanlp
        url = "https://file.hankcs.com/hanlp/mtl/en_tok_lem_pos_ner_srl_udep_sdp_con_modernbert_base_prepend_false_20241229_053838.zip"
        try:
            _english_pipeline = hanlp.load(url)
            logger.info("English HanLP pipeline loaded (MODERNBERT-base).")
        except Exception as exc:
            logger.warning(
                "English pipeline not available (need hanlp>=2.1.0): %s. "
                "Pull upstream commits: git pull upstream main",
                exc,
            )
            return None
    return _english_pipeline


_LZH_MODEL_URL = (
    "https://file.hankcs.com/hanlp/mtl/"
    "kyoto_evahan_tok_lem_pos_udep_bert-ancient-chinese_lr_1_aug_dict_20250112_154422.zip"
)


def _get_classical_pipeline():
    global _classical_pipeline
    if _classical_pipeline is None:
        import hanlp
        try:
            # Try the published pretrained constant first (future HanLP versions)
            _classical_pipeline = hanlp.load(
                hanlp.pretrained.mtl.KYOTO_EVAHAN_TOK_LEM_POS_UDEP_LZH
            )
        except AttributeError:
            # Fallback: load by direct URL (current HanLP version)
            try:
                _classical_pipeline = hanlp.load(_LZH_MODEL_URL)
            except Exception as exc:
                logger.warning(
                    "Classical Chinese model (LZH) not available: %s. "
                    "Install with: python scripts/setup_models.py --model LZH",
                    exc,
                )
                return None
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
                       ) -> tuple[list, list, list, list, bool]:
    try:
        pipeline = _get_classical_pipeline()
        if pipeline is None:
            raise RuntimeError("Classical pipeline not loaded")
        raw = pipeline(text)
    except (AttributeError, ImportError, RuntimeError):
        logger.warning("Classical Chinese model not available. Skipping.")
        return [], [], [], [], False
    except Exception as exc:
        logger.warning("Classical pipeline failed: %s", exc)
        return [], [], [], [], False
    normalized = _normalize_lzh_keys(raw)
    doc = mapper.map(text, normalized, source="hanlp_lzh")
    _apply_offset(doc, offset)
    for t in doc.content.tokens:
        t.source = "hanlp_lzh"
    return (doc.content.tokens, doc.content.entities, doc.content.relations,
            doc.content.patterns, _assess_quality(doc.content.tokens, normalized))


def _analyze_english(text: str, offset: int, mapper: HanlpSchemaMapper
                     ) -> tuple[list, list, list, list, bool]:
    try:
        pipeline = _get_english_pipeline()
        if pipeline is None:
            raise RuntimeError("English pipeline not loaded")
        raw = pipeline(text)
    except (AttributeError, ImportError, RuntimeError):
        logger.warning("English model not available. Skipping.")
        return [], [], [], [], False
    except Exception as exc:
        logger.warning("English pipeline failed: %s", exc)
        return [], [], [], [], False
    # English model uses standard UD keys, map directly
    doc = mapper.map(text, raw, source="en_modernbert")
    _apply_offset(doc, offset)
    for t in doc.content.tokens:
        t.source = "en_modernbert"
    return (doc.content.tokens, doc.content.entities, doc.content.relations,
            doc.content.patterns, _assess_quality(doc.content.tokens, raw))


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

def analyze(text: str, dict_combine: Optional[set] = None,
            language: str = "auto") -> NarrativeDocument:
    """
    Analyze text with automatic language detection and model routing.

    Args:
        text: Raw text to analyze (Chinese or English).
        dict_combine: Optional set of words to force-combine during tokenization.
        language: Language mode:
            - "auto" (default): per-sentence detection
            - "modern": force modern Chinese pipeline
            - "classical": force classical Chinese pipeline
            - "english": force English pipeline

    Modern Chinese  → MTL (ELECTRA-small)
    Classical Chinese → LZH (KYOTO-EVAHAN)
    English          → MODERNBERT-base

    Returns a single NarrativeDocument with global offsets.
    """
    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")

    mapper = _get_mapper()
    sentences = _split_sentences(text)
    language_sentences = []

    if language == "auto":
        # Group consecutive same-language sentences
        segments: list[tuple[str, int, str, float]] = []
        for sent_text, sent_offset in sentences:
            lang, conf = detect_language(sent_text)
            segments.append((sent_text, sent_offset, lang, conf))
            language_sentences.append({
                "text": sent_text, "span": (sent_offset, sent_offset + len(sent_text)),
                "label": lang, "confidence": conf,
            })

        merged: list[tuple[str, int, str, float]] = []
        for sent_text, offset, lang, conf in segments:
            if merged and merged[-1][2] == lang:
                prev_text, prev_offset, _, _ = merged[-1]
                merged[-1] = (text[prev_offset:offset + len(sent_text)], prev_offset, lang, conf)
            else:
                merged.append((sent_text, offset, lang, conf))
    elif language == "classical":
        lang: str = "classical"
        merged = [(text, 0, lang, 1.0)]
        for sent_text, sent_offset in sentences:
            language_sentences.append({
                "text": sent_text, "span": (sent_offset, sent_offset + len(sent_text)),
                "label": lang, "confidence": 1.0,
            })
    elif language == "english":
        lang = "english"
        merged = [(text, 0, lang, 1.0)]
        for sent_text, sent_offset in sentences:
            language_sentences.append({
                "text": sent_text, "span": (sent_offset, sent_offset + len(sent_text)),
                "label": lang, "confidence": 1.0,
            })
    else:  # "modern"
        lang = "modern"
        merged = [(text, 0, lang, 0.0)]
        for sent_text, sent_offset in sentences:
            language_sentences.append({
                "text": sent_text, "span": (sent_offset, sent_offset + len(sent_text)),
                "label": lang, "confidence": 0.0,
            })

    # Process each segment
    all_tokens, all_entities, all_relations, all_patterns = [], [], [], []
    for seg_text, seg_offset, lang, conf in merged:
        if lang == "classical":
            tokens, entities, relations, patterns, ok = _analyze_classical(seg_text, seg_offset, mapper)
            if not ok:
                logger.warning(
                    "Classical segment not analyzed (model unavailable): %s...",
                    seg_text[:20],
                )
        elif lang == "english":
            tokens, entities, relations, patterns, ok = _analyze_english(seg_text, seg_offset, mapper)
            if not ok:
                logger.warning(
                    "English segment not analyzed (model unavailable): %s...",
                    seg_text[:20],
                )
        else:
            tokens, entities, relations, patterns, ok = _analyze_modern(seg_text, seg_offset, mapper, dict_combine)
            if not ok:
                logger.warning(
                    "Modern segment not analyzed: %s...",
                    seg_text[:20],
                )
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
        meta=NarrativeMeta(
            source=meta_source,
            text_length=len(text),
            language_mode=language,
            language_sentences=[SentenceLanguage(**s) for s in language_sentences],
        ),
        content=NarrativeContent(tokens=all_tokens, entities=all_entities,
                                 relations=all_relations, patterns=all_patterns, structural={}),
    )
