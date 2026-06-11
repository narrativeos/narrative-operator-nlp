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
        try:
            _english_pipeline = hanlp.load(
                hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_BASE
            )
            logger.info("English HanLP pipeline loaded (MODERNBERT-base).")
        except Exception as exc:
            logger.warning(
                "English pipeline not available: %s. "
                "Run: python scripts/setup_models.py --model LZH",
                exc,
            )
            return None
    return _english_pipeline


def _get_classical_pipeline():
    global _classical_pipeline
    if _classical_pipeline is None:
        import hanlp
        try:
            _classical_pipeline = hanlp.load(
                hanlp.pretrained.mtl.KYOTO_EVAHAN_TOK_LEM_POS_UDEP_LZH
            )
            logger.info("Classical Chinese HanLP pipeline loaded (KYOTO-EVAHAN).")
        except Exception as exc:
            logger.warning(
                "Classical Chinese model (LZH) not available: %s. "
                "Run: python scripts/setup_models.py --model LZH",
                exc,
            )
            return None
    return _classical_pipeline


def _get_mapper() -> HanlpSchemaMapper:
    global _mapper
    if _mapper is None:
        _mapper = HanlpSchemaMapper()
    return _mapper


# ---------------------------------------------------------------------------
# Sentence Splitting
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(
    r"(?<=[。！？；\n])\s*"           # Chinese punctuation
    r"|(?<=[.!?])\s+(?=[A-Z])"       # English: period + space + capital letter
)


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
    # >60% NR/NNP (proper noun) → model doesn't know these words
    nr_count = sum(1 for t in tokens if t.pos in ("NR", "NNP", "NNPS"))
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

def _analyze_modern(
    text: str,
    offset: int,
    mapper: HanlpSchemaMapper,
    dict_combine: Optional[set] = None,
    entity_categories: dict[str, list[str]] | None = None,
    auto_discover_entities: bool = False,
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
    doc = mapper.map(
        text, raw, source="hanlp_v2",
        entity_categories=entity_categories,
        auto_discover_entities=auto_discover_entities,
    )
    _apply_offset(doc, offset)
    return (doc.content.tokens, doc.content.entities, doc.content.relations,
            doc.content.patterns, _assess_quality(doc.content.tokens, raw))


def _analyze_classical(
    text: str,
    offset: int,
    mapper: HanlpSchemaMapper,
    entity_dict: dict[str, str] | None = None,
    entity_categories: dict[str, list[str]] | None = None,
    auto_discover_entities: bool = False,
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
    doc = mapper.map(
        text, normalized, source="hanlp_lzh",
        entity_dict=entity_dict,
        entity_categories=entity_categories,
        auto_discover_entities=auto_discover_entities,
    )
    _apply_offset(doc, offset)
    for t in doc.content.tokens:
        t.source = "hanlp_lzh"
    return (doc.content.tokens, doc.content.entities, doc.content.relations,
            doc.content.patterns, _assess_quality(doc.content.tokens, normalized))


def _analyze_english(
    text: str,
    offset: int,
    mapper: HanlpSchemaMapper,
    entity_categories: dict[str, list[str]] | None = None,
    auto_discover_entities: bool = False,
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
    doc = mapper.map(
        text, raw, source="en_modernbert",
        entity_categories=entity_categories,
        auto_discover_entities=auto_discover_entities,
    )
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

def analyze(
    text: str,
    dict_combine: Optional[set] = None,
    language: str = "auto",
    entity_dict: dict[str, str] | None = None,
    entity_categories: dict[str, list[str]] | None = None,
    auto_discover_entities: bool = False,
) -> NarrativeDocument:
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
        entity_dict: Optional ``{term: NSP_category}`` dictionary for classical
            Chinese entity recognition. No hardcoded dictionaries — the
            caller (API / frontend) supplies this.
            Example: ``{"北冥": "LOCATION", "鲲": "PERSON"}``.
        entity_categories: Optional ``{category: [keyword1, keyword2, ...]}``
            for domain-specific keyword injection. The caller supplies domain
            keywords at runtime — no hardcoded domain knowledge in the tool.
            Example: ``{"DISEASE": ["乳腺癌", "肿瘤"], "ANATOMY": ["乳腺"]}``.
            Keywords with categories not in EntityCategory.ALL are mapped to UNKNOWN.
        auto_discover_entities: If True, run new word discovery (PMI+MTL)
            and promote high-score candidates to UNKNOWN entities. Default False.

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
            tokens, entities, relations, patterns, ok = _analyze_classical(
                seg_text, seg_offset, mapper, entity_dict, entity_categories,
                auto_discover_entities,
            )
            if not ok:
                logger.warning(
                    "Classical segment not analyzed (model unavailable): %s...",
                    seg_text[:20],
                )
        elif lang == "english":
            tokens, entities, relations, patterns, ok = _analyze_english(
                seg_text, seg_offset, mapper, entity_categories,
                auto_discover_entities,
            )
            if not ok:
                logger.warning(
                    "English segment not analyzed (model unavailable): %s...",
                    seg_text[:20],
                )
        else:
            tokens, entities, relations, patterns, ok = _analyze_modern(
                seg_text, seg_offset, mapper, dict_combine, entity_categories,
                auto_discover_entities,
            )
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

    sentence_objects = [SentenceLanguage(**s) for s in language_sentences]

    return NarrativeDocument(
        meta=NarrativeMeta(
            source=meta_source,
            text_length=len(text),
            language_mode=language,
        ),
        content=NarrativeContent(
            tokens=all_tokens, entities=all_entities,
            relations=all_relations, patterns=all_patterns,
            sentences=sentence_objects, structural={},
        ),
    )
