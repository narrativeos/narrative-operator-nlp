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
from .schema import CoreferenceChain, DependencyEdge, Event, NarrativeContent, NarrativeDocument, NarrativeMeta, SentenceLanguage, Token
from .coref_resolver import CorefResolver
from .modifier_extractor import ModifierExtractor
from .relation_classifier import RelationClassifier
from .entity_hierarchy import EntityHierarchyBuilder
from .classical_pattern_extractor import ClassicalPatternExtractor
from .event_extractor import EventExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded HanLP pipelines + cached dictionary loader
# ---------------------------------------------------------------------------

_modern_pipeline: Optional[object] = None
_classical_pipeline: Optional[object] = None
_english_pipeline: Optional[object] = None
_mapper: Optional[HanlpSchemaMapper] = None
_dict_loader = None  # Cached DictionaryLoader with CBDB loaded


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
) -> tuple[list, list, list, list, list, list, bool]:
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
        return [], [], [], [], [], [], False
    doc = mapper.map(
        text, raw, source="hanlp_v2",
        entity_categories=entity_categories,
        auto_discover_entities=auto_discover_entities,
    )
    _apply_offset(doc, offset)
    deps = _extract_deps(raw)
    srl_frames = _validate_srl_frames(raw.get("srl", []))
    return (doc.content.tokens, doc.content.entities, doc.content.relations,
            doc.content.patterns, deps, srl_frames, _assess_quality(doc.content.tokens, raw))


def _analyze_classical(
    text: str,
    offset: int,
    mapper: HanlpSchemaMapper,
    dict_combine: Optional[set] = None,
    entity_dict: dict[str, str] | None = None,
    entity_categories: dict[str, list[str]] | None = None,
    auto_discover_entities: bool = False,
) -> tuple[list, list, list, list, list, bool]:
    try:
        pipeline = _get_classical_pipeline()
        if pipeline is None:
            raise RuntimeError("Classical pipeline not loaded")
        # Apply custom dictionary + seed dictionaries to classical pipeline
        # Only load YAML seed dictionaries (small scale) to avoid memory issues
        all_dict_words = set()
        if dict_combine:
            all_dict_words |= dict_combine

        # Load only YAML seed dictionaries for dict_combine (not full CBDB, too slow for tokenizer)
        # CBDB is used for post-NER classification via the mapper's keyword_extractor
        try:
            from .dictionary_loader import DictionaryLoader
            from pathlib import Path
            config_dir = str(Path(__file__).parent.parent / "config")
            # Only load YAML dictionaries (not CBDB) for tokenizer dict_combine
            loader_yaml = DictionaryLoader(config_dir, load_cbdb_to_memory=False)
            loader_yaml.load_all()
            for category in loader_yaml.FILE_TO_CATEGORY.values():
                keywords = loader_yaml.get_keywords(category)
                if keywords:
                    all_dict_words |= keywords
        except Exception as exc:
            logger.debug("Failed to load YAML dictionaries for tokenization: %s", exc)

        if all_dict_words:
            try:
                pipeline['tok/fine'].dict_combine = all_dict_words
            except (KeyError, AttributeError):
                pass
        raw = pipeline(text)
    except (AttributeError, ImportError, RuntimeError):
        logger.warning("Classical Chinese model not available. Skipping.")
        return [], [], [], [], [], False
    except Exception as exc:
        logger.warning("Classical pipeline failed: %s", exc)
        return [], [], [], [], [], False
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

    # Fill evidence for entities using on-demand CBDB lookup
    # (CBDB is queried via SQLite, not loaded to memory - avoids timeout)
    try:
        from .dictionary_loader import DictionaryLoader
        from pathlib import Path
        config_dir = str(Path(__file__).parent.parent / "config")
        dict_loader = DictionaryLoader(config_dir, load_cbdb_to_memory=False)
        dict_loader.load_all()
        # Build token list for smart lookup
        token_list = [(t.text, t.span[0], t.span[1]) for t in doc.content.tokens]
        discovered = dict_loader.lookup_tokens(token_list, text)
        discovered_map = {d.keyword: d for d in discovered}
        for entity in doc.content.entities:
            entity.evidence.tokenizer = entity.text in all_dict_words
            entry = dict_loader.get_entry(entity.text)
            if entry is not None:
                entity.evidence.seed_dict = True
                entity.evidence.seed_dict_source = entry.source
            cbdb_entry = discovered_map.get(entity.text) or dict_loader.lookup(entity.text)
            if cbdb_entry is not None and cbdb_entry.source == "CBDB":
                entity.evidence.cbdb = True
                entity.evidence.cbdb_category = cbdb_entry.category
    except Exception as exc:
        logger.debug("Failed to fill entity evidence: %s", exc)

    deps = _extract_deps(normalized)
    return (doc.content.tokens, doc.content.entities, doc.content.relations,
            doc.content.patterns, deps, _assess_quality(doc.content.tokens, normalized))


def _analyze_english(
    text: str,
    offset: int,
    mapper: HanlpSchemaMapper,
    entity_categories: dict[str, list[str]] | None = None,
    auto_discover_entities: bool = False,
) -> tuple[list, list, list, list, list, list, bool]:
    """Analyze English text.

    Note: The English pipeline does not currently support SRL (Semantic Role Labeling),
    so the returned srl_frames list is always empty.
    """
    try:
        pipeline = _get_english_pipeline()
        if pipeline is None:
            raise RuntimeError("English pipeline not loaded")
        raw = pipeline(text)
    except (AttributeError, ImportError, RuntimeError):
        logger.warning("English model not available. Skipping.")
        return [], [], [], [], [], [], False
    except Exception as exc:
        logger.warning("English pipeline failed: %s", exc)
        return [], [], [], [], [], [], False
    # English model uses standard UD keys, map directly
    doc = mapper.map(
        text, raw, source="en_modernbert",
        entity_categories=entity_categories,
        auto_discover_entities=auto_discover_entities,
    )
    _apply_offset(doc, offset)
    for t in doc.content.tokens:
        t.source = "en_modernbert"
    deps = _extract_deps(raw)
    # English pipeline: SRL not extracted (empty list)
    return (doc.content.tokens, doc.content.entities, doc.content.relations,
            doc.content.patterns, deps, [], _assess_quality(doc.content.tokens, raw))


def _extract_deps(raw: dict) -> list[DependencyEdge]:
    """Extract dependency edges from HanLP raw output (1-indexed to 0-indexed)."""
    deps: list[DependencyEdge] = []
    dep_data = raw.get("dep", [])
    for i, d in enumerate(dep_data):
        if not isinstance(d, (list, tuple)) or len(d) < 2:
            continue
        head_idx = int(d[0]) - 1  # HanLP is 1-indexed, convert to 0-indexed; 0 becomes -1 (ROOT)
        rel = str(d[1])
        deps.append(DependencyEdge(child=i, head=head_idx, rel=rel))
    return deps


def _validate_srl_frames(srl_data: list) -> list:
    """Validate and sanitize SRL frames from HanLP output.

    Expected format: list of frames, each frame is a list of tuples:
        [(text, role, tok_start, tok_end), ...]

    Filters out frames that:
    - Are not lists
    - Don't have at least 2 items
    - Contain items that aren't tuples/lists of length >= 4

    Returns a clean list of valid frames.
    """
    if not isinstance(srl_data, list):
        logger.debug("SRL data is not a list, skipping SRL enrichment")
        return []

    valid_frames = []
    for frame in srl_data:
        if not isinstance(frame, list) or len(frame) < 2:
            continue
        # Validate each item in the frame
        valid_items = []
        for item in frame:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            try:
                # Validate that positions are numeric
                text = str(item[0])
                role = str(item[1])
                tok_start = int(item[2])
                tok_end = int(item[3])
                valid_items.append((text, role, tok_start, tok_end))
            except (ValueError, TypeError):
                continue
        if valid_items:
            valid_frames.append(valid_items)

    if len(valid_frames) != len(srl_data):
        logger.debug(
            "SRL frames sanitized: %d original → %d valid",
            len(srl_data), len(valid_frames),
        )

    return valid_frames


def _apply_offset(doc: NarrativeDocument, offset: int):
    for t in doc.content.tokens:
        t.span = (t.span[0] + offset, t.span[1] + offset)
    for e in doc.content.entities:
        e.span = (e.span[0] + offset, e.span[1] + offset)
    for r in doc.content.relations:
        r.evidence_span = (r.evidence_span[0] + offset, r.evidence_span[1] + offset)


def _resolve_coreferences(
    text: str,
    tokens: list[Token],
    entities: list,
    language: str,
) -> list[CoreferenceChain]:
    """Run coreference resolution on the analyzed text."""
    resolver = CorefResolver()
    # Determine language for coref
    if language == "auto":
        # Use the dominant language from segments
        lang = "modern"  # default
    else:
        lang = language
    result = resolver.resolve(text, entities, tokens, lang)
    return result.chains


def _normalize_relations_by_coref(
    relations: list,
    coreferences: list[CoreferenceChain],
    entities: list,
) -> None:
    """Use coreference chains to normalize relation endpoints.

    Two-level resolution:
    1. Exact match: relation subject/object matches a coref mention directly
    2. Prefix match: relation subject starts with a coref mention
       (e.g., "该公司总部" starts with "该公司")

    For coref chains whose representative is a demonstrative+noun
    (该公司/此产品), resolve to the actual entity by finding the
    entity whose text is contained in or matches the noun part.
    """
    # Demonstrative prefixes
    DEMO_PREFIXES = {"该", "此", "本", "其", "彼", "是"}

    # Build mention->chain lookup
    mention_to_chain: dict[str, CoreferenceChain] = {}
    for chain in coreferences:
        for mention in chain.mentions:
            mention_to_chain[mention.text] = chain

    # Build entity text set for resolution
    entity_texts = {e.text for e in entities}

    # Helper: resolve chain representative to actual entity
    def resolve_representative(chain: CoreferenceChain) -> str:
        rep = chain.representative
        # If representative starts with a demonstrative prefix, strip it
        for prefix in DEMO_PREFIXES:
            if rep.startswith(prefix):
                noun_part = rep[len(prefix):]
                # Find entity that contains or equals this noun part
                for et in entity_texts:
                    if noun_part in et or et == noun_part:
                        return et
                # Fallback: use noun part itself
                return noun_part
        # Check if representative itself is an entity
        if rep in entity_texts:
            return rep
        return rep

    # Also build entity text set for normalization
    for rel in relations:
        # Skip normalization for HAS_STYLE_NAME and EQUIVALENT_TO from classical patterns
        # (the style name and equivalent names are intentional, not coref mentions)
        if rel.predicate in ("HAS_STYLE_NAME", "EQUIVALENT_TO"):
            if "classical_pattern" in rel.source:
                continue

        # Normalize subject
        matched_chain = None

        # Level 1: exact match
        if rel.subject_raw in mention_to_chain:
            matched_chain = mention_to_chain[rel.subject_raw]
        else:
            # Level 2: prefix match - find longest mention that's a prefix
            best_prefix = ""
            for mention_text, chain in mention_to_chain.items():
                if rel.subject_raw.startswith(mention_text) and len(mention_text) > len(best_prefix):
                    best_prefix = mention_text
                    matched_chain = chain

        if matched_chain:
            resolved = resolve_representative(matched_chain)
            if resolved != rel.subject_raw:
                rel.subject = resolved

        # Level 3: entity text normalization - find entity that contains subject
        # Only apply if the subject is a proper substring (not the full entity name)
        # and it's shorter than the entity (to avoid false matches like "钢"->"碳钢")
        orig_subject = rel.subject
        best_entity_match = None
        best_entity_len = 0
        for et in entity_texts:
            if (rel.subject in et or rel.subject_raw in et) and len(rel.subject) < len(et):
                if len(et) > best_entity_len:
                    best_entity_len = len(et)
                    best_entity_match = et
        if best_entity_match and best_entity_match != rel.subject:
            rel.subject = best_entity_match

        # Normalize object
        matched_chain = None

        if rel.object_raw in mention_to_chain:
            matched_chain = mention_to_chain[rel.object_raw]
        else:
            best_prefix = ""
            for mention_text, chain in mention_to_chain.items():
                if rel.object_raw.startswith(mention_text) and len(mention_text) > len(best_prefix):
                    best_prefix = mention_text
                    matched_chain = chain

        if matched_chain:
            resolved = resolve_representative(matched_chain)
            if resolved != rel.object_raw:
                rel.object = resolved

        # Level 3: entity text normalization - find entity that contains object
        orig_object = rel.object
        best_entity_match = None
        best_entity_len = 0
        for et in entity_texts:
            if (rel.object in et or rel.object_raw in et) and len(rel.object) < len(et):
                if len(et) > best_entity_len:
                    best_entity_len = len(et)
                    best_entity_match = et
        if best_entity_match and best_entity_match != rel.object:
            rel.object = best_entity_match

        # Prevent self-loops: if subject == object after normalization, revert
        if rel.subject == rel.object:
            rel.subject = orig_subject
            rel.object = orig_object


# ---------------------------------------------------------------------------
# Classical Key Normalization (lzh_* -> standard mapper keys)
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
            Chinese entity recognition. No hardcoded dictionaries - the
            caller (API / frontend) supplies this.
            Example: ``{"北冥": "LOCATION", "鲲": "PERSON"}``.
        entity_categories: Optional ``{category: [keyword1, keyword2, ...]}``
            for domain-specific keyword injection. The caller supplies domain
            keywords at runtime - no hardcoded domain knowledge in the tool.
            Example: ``{"DISEASE": ["乳腺癌", "肿瘤"], "ANATOMY": ["乳腺"]}``.
            Keywords with categories not in EntityCategory.ALL are mapped to UNKNOWN.
        auto_discover_entities: If True, run new word discovery (PMI+MTL)
            and promote high-score candidates to UNKNOWN entities. Default False.

    Modern Chinese  -> MTL (ELECTRA-small)
    Classical Chinese -> LZH (KYOTO-EVAHAN)
    English          -> MODERNBERT-base

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
    all_tokens, all_entities, all_relations, all_patterns, all_deps, all_srl = [], [], [], [], [], []
    token_offset = 0  # Global token index offset for dep remapping
    for seg_text, seg_offset, lang, conf in merged:
        if lang == "classical":
            tokens, entities, relations, patterns, deps, ok = _analyze_classical(
                seg_text, seg_offset, mapper, dict_combine, entity_dict, entity_categories,
                auto_discover_entities,
            )
            if not ok:
                logger.warning(
                    "Classical segment not analyzed (model unavailable): %s...",
                    seg_text[:20],
                )
                token_offset += len(tokens)
                continue
            # Classical pipeline has no SRL
            seg_srl: list = []
        elif lang == "english":
            tokens, entities, relations, patterns, deps, seg_srl, ok = _analyze_english(
                seg_text, seg_offset, mapper, entity_categories,
                auto_discover_entities,
            )
            if not ok:
                logger.warning(
                    "English segment not analyzed (model unavailable): %s...",
                    seg_text[:20],
                )
                token_offset += len(tokens)
                continue
            # Note: seg_srl from English pipeline is always empty (SRL not yet supported)
        else:
            tokens, entities, relations, patterns, deps, seg_srl, ok = _analyze_modern(
                seg_text, seg_offset, mapper, dict_combine, entity_categories,
                auto_discover_entities,
            )
            if not ok:
                logger.warning(
                    "Modern segment not analyzed: %s...",
                    seg_text[:20],
                )
                token_offset += len(tokens)
                continue

        # Remap dep indices from segment-local to global
        for dep in deps:
            dep.child += token_offset
            if dep.head >= 0:
                dep.head += token_offset
        all_deps.extend(deps)

        all_tokens.extend(tokens)
        all_entities.extend(entities)
        all_relations.extend(relations)
        all_patterns.extend(patterns)
        all_srl.extend(seg_srl)
        token_offset += len(tokens)

    # Re-number token IDs globally, sorted by position
    for i, t in enumerate(sorted(all_tokens, key=lambda t: t.span[0])):
        t.id = i

    # Event Extraction from Relation clustering (V3)
    # Entity → Relation → Event pipeline:
    # Groups relations by (sentence, predicate_verb) to form events,
    # enriched by SRL frames for richer argument roles.
    event_extractor = EventExtractor()
    sentence_objects = [SentenceLanguage(**s) for s in language_sentences]
    all_events = event_extractor.extract(
        text, all_relations, all_entities,
        sentence_objects, all_patterns,
        srl_frames=all_srl,
        deps=all_deps,
        tokens=all_tokens,
    )

    # Classical Chinese pattern extraction (post-processing)
    # Extract relations using regex patterns for classical Chinese sentence structures
    if language == "classical" or (language == "auto" and any(s["label"] == "classical" for s in language_sentences)):
        pattern_extractor = ClassicalPatternExtractor()
        pattern_rels, pattern_corefs = pattern_extractor.extract(text, all_entities)
        all_relations.extend(pattern_rels)
        # These coreferences will be merged with the standard coref results below

    # Coreference resolution
    all_coreferences = _resolve_coreferences(text, all_tokens, all_entities, language)

    # Merge classical pattern coreferences with standard coreferences
    if language == "classical" or (language == "auto" and any(s["label"] == "classical" for s in language_sentences)):
        pattern_extractor = ClassicalPatternExtractor()
        _, pattern_corefs = pattern_extractor.extract(text, all_entities)
        all_coreferences.extend(pattern_corefs)

    # Coref-Relation Integration
    # Use coreference chains to further normalize relation endpoints
    _normalize_relations_by_coref(all_relations, all_coreferences, all_entities)

    # Modifier Extraction
    # Extract relation modifiers (degree, negation, scope, etc.)
    modifier_extractor = ModifierExtractor()
    modifier_extractor.extract_batch(text, all_relations)

    # Relation Classification
    # Classify relations into entity relations vs entity attributes
    classifier = RelationClassifier()
    classifier.classify(all_relations, all_entities)

    # Entity Hierarchy
    # Detect containment relationships and generate PART_OF relations
    hierarchy_builder = EntityHierarchyBuilder(relation_counter=len(all_relations))
    hierarchy_relations = hierarchy_builder.build(all_entities, text)
    all_relations.extend(hierarchy_relations)

    sources = sorted(set(t.source for t in all_tokens if t.source))
    meta_source = "+".join(sources) if sources else "hanlp_v2"

    return NarrativeDocument(
        meta=NarrativeMeta(
            source=meta_source,
            text_length=len(text),
            language_mode=language,
        ),
        content=NarrativeContent(
            tokens=all_tokens, entities=all_entities,
            relations=all_relations, events=all_events,
            deps=all_deps, patterns=all_patterns,
            coreferences=all_coreferences,
            sentences=sentence_objects, structural={},
        ),
    )
