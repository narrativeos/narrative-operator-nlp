"""
HanlpSchemaMapper — HanLP Raw Output → NSP Standard Format.

Maps HanLP MTL model output to NSP NarrativeDocument.
Handles the real HanLP output structures:

    tok/fine   : list[str]                          — individual characters
    pos/ctb    : list[str]                          — POS tags aligned with tok
    ner/*      : list[tuple[str,str,int,int]]       — (text, label, tok_start, tok_end)
    dep        : list[tuple[int,str]]               — (head_1based, deprel)
    srl        : list[list[tuple[str,str,int,int]]]  — frames
"""

from __future__ import annotations

from datetime import datetime, timezone

from .entity_mapper import EntityMappingRules
from .relation_mapper import RelationExtractionRules
from .schema import (
    NarrativeContent,
    NarrativeDocument,
    NarrativeMeta,
    Token,
)


class HanlpSchemaMapper:
    """Maps HanLP raw MTL output to NSP NarrativeDocument."""

    def __init__(self) -> None:
        self.entity_rules = EntityMappingRules()
        self.relation_rules = RelationExtractionRules()

    def map(self, text: str, raw: dict, source: str = "hanlp_v2") -> NarrativeDocument:
        tokens = self._map_tokens(text, raw)
        entities = self._map_entities(text, raw, tokens)
        self._assign_attributes(text, raw, tokens, entities)
        # PARAMETERs are properties, not standalone entities
        entities = [e for e in entities if e.category != "PARAMETER"]
        relations = self._map_relations(text, raw, tokens, entities)
        patterns = self._build_patterns(text, raw, entities, relations)
        return NarrativeDocument(
            meta=NarrativeMeta(
                source=source,
                version="1.0",
                timestamp=datetime.now(timezone.utc).isoformat(),
                text_length=len(text),
            ),
            content=NarrativeContent(
                tokens=tokens,
                entities=entities,
                relations=relations,
                patterns=patterns,
                structural=raw,
            ),
        )

    def _map_tokens(self, text: str, raw: dict) -> list[Token]:
        tok_fine = raw.get("tok/fine", [])
        pos = raw.get("pos/ctb", [])
        tok_conf = raw.get("tok/fine_conf") or raw.get("tok/coarse_conf")
        # Flatten if nested (batch-level list)
        if tok_conf and isinstance(tok_conf[0], list):
            tok_conf = tok_conf[0]
        tokens: list[Token] = []
        cursor = 0
        for i in range(len(tok_fine)):
            token_text = tok_fine[i]
            token_pos = pos[i] if i < len(pos) else "X"
            idx = text.find(token_text, cursor)
            if idx >= 0:
                start, end = idx, idx + len(token_text)
                cursor = end
            else:
                start, end = cursor, cursor + len(token_text)
                cursor = end
            conf = float(tok_conf[i]) if tok_conf and i < len(tok_conf) else 1.0
            tokens.append(Token(id=i, text=token_text, pos=token_pos, span=(start, end), confidence=conf))
        return tokens

    def _map_entities(self, text: str, raw: dict, tokens: list[Token]) -> list:
        return self.entity_rules.map_all(text, raw, tokens)

    def _map_relations(self, text: str, raw: dict, tokens: list[Token],
                       entities: list) -> list:
        return self.relation_rules.extract_all(text, raw, tokens, entities)

    def _assign_attributes(self, text: str, raw: dict, tokens: list,
                           entities: list) -> None:
        """Assign entity attributes from SRL property frames.

        When SRL gives ARG0=entity, ARG1=compound of PARAMETER entities
        (e.g., '高强度和高韧性'), split into individual attributes
        and attach to the ARG0 entity.
        """
        from .schema import EntityAttribute

        # Build entity lookup by text
        entity_by_text: dict[str, object] = {e.text: e for e in entities}
        param_entities = {e.text for e in entities if e.category == "PARAMETER"}

        for f in raw.get("srl", []):
            if not isinstance(f, list):
                continue
            a0_text, a1_text = "", ""
            pred_text = ""
            for item in f:
                if len(item) < 4:
                    continue
                role = str(item[1]).upper()
                if "ARG0" in role:
                    a0_text = str(item[0])
                elif "ARG1" in role:
                    a1_text = str(item[0])
                elif role == "PRED":
                    pred_text = str(item[0])

            if not a0_text or not a1_text or not pred_text:
                continue

            # ARG0 must be an entity
            a0_entity = entity_by_text.get(a0_text)
            if a0_entity is None:
                continue

            # Find all PARAMETER entities that are substrings of ARG1
            matched = [p for p in param_entities if p in a1_text]
            if not matched:
                continue

            # Assign each parameter as an attribute
            for param_text in matched:
                param_entity = entity_by_text.get(param_text)
                if param_entity is None:
                    continue

                # Split "高强度" → key="强度", value="高"
                # If param_text IS a known parameter keyword → no value prefix
                from .entity_mapper import _PARAMETER as KNOWN_PARAMS
                if param_text in KNOWN_PARAMS:
                    key, value = param_text, ""
                else:
                    # Try suffix match: "高强度" ends with "强度" → key="强度", value="高"
                    matched_kw = None
                    for kw in sorted(KNOWN_PARAMS, key=len, reverse=True):
                        if param_text.endswith(kw) and len(kw) >= 2:
                            matched_kw = kw
                            break
                    if matched_kw:
                        key = matched_kw
                        value = param_text[:-len(matched_kw)]
                    else:
                        key = param_text
                        value = ""

                attr = EntityAttribute(
                    key=key,
                    value=value,
                    predicate_verb=pred_text,
                    confidence=0.85,
                )
                a0_entity.attributes.append(attr)

    def _build_patterns(self, text: str, raw: dict, entities: list,
                        relations: list) -> list:
        """Build sentence-level structural patterns for statistical aggregation.

        Each sentence (split by 。！？) gets one pattern combining all its SRL frames.
        """
        from .schema import SentencePattern
        import re

        srl_frames = [f for f in raw.get("srl", []) if isinstance(f, list)]
        if not srl_frames:
            return []

        # Split into sentences
        sentences = re.split(r"(?<=[。！？])", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Group SRL frames by which sentence their ARG0 appears in
        sent_frames: list[list] = [[] for _ in sentences]
        for f in srl_frames:
            a0 = ""
            for item in f:
                if len(item) >= 4 and str(item[1]).upper() == "ARG0":
                    a0 = str(item[0])
                    break
            if not a0:
                continue
            for si, sent in enumerate(sentences):
                if a0 in sent:
                    sent_frames[si].append(f)
                    break

        patterns: list = []
        for si, frames in enumerate(sent_frames):
            sent = sentences[si]

            if not frames:
                # Sentence with no SRL — still emit a pattern with syntactic features
                est_words = _count_sentence_words(sent, raw)
                patterns.append(SentencePattern(
                    sentence=sent,
                    sentence_type=_detect_sentence_type(sent),
                    structural_type="unknown",
                    polarity=_detect_polarity(sent),
                    voice=_detect_voice(sent),
                    sub_types=_detect_sub_types(sent),
                    rhetorical_form=_detect_rhetorical(sent),
                    sentence_length_tier=_detect_length_tier(sent),
                    template="",
                    entity_sequence=[],
                    predicates=[],
                    relation_summary=[],
                    attribute_count=0,
                    word_count=est_words,
                    clause_count=_count_clauses(sent),
                    punctuation_mark=_sentence_punct(sent),
                    limitations=_collect_limitations(sent, frames),
                ))
                continue
            preds, entity_cats, rel_summaries = [], [], []

            for f in frames:
                for item in f:
                    if len(item) < 4:
                        continue
                    role = str(item[1]).upper()
                    itxt = str(item[0])
                    if role == "PRED":
                        preds.append(itxt)
                    if role in ("ARG0", "ARG1"):
                        mc = None
                        for e in entities:
                            if e.text == itxt or e.text in itxt:
                                mc = e.category
                                break
                        entity_cats.append(mc or "?")

            if not preds:
                continue

            parts: list[str] = []
            for i in range(max(len(entity_cats), len(preds))):
                if i < len(entity_cats):
                    parts.append(entity_cats[i])
                if i < len(preds):
                    parts.append("谓词")
            template = " ".join(parts)

            for r in relations:
                if r.predicate_verb in preds:
                    rel_summaries.append(f"{r.subject}→{r.predicate_verb}→{r.object}")

            attr_count = sum(len(e.attributes) for e in entities)

            patterns.append(SentencePattern(
                sentence=sent,
                sentence_type=_detect_sentence_type(sent),
                structural_type=_detect_structural_type(sent, frames),
                polarity=_detect_polarity(sent),
                voice=_detect_voice(sent),
                sub_types=_detect_sub_types(sent),
                rhetorical_form=_detect_rhetorical(sent),
                sentence_length_tier=_detect_length_tier(sent),
                template=template,
                entity_sequence=entity_cats,
                predicates=preds,
                relation_summary=rel_summaries,
                attribute_count=attr_count,
                word_count=_count_sentence_words(sent, raw),
                clause_count=_count_clauses(sent),
                punctuation_mark=_sentence_punct(sent),
                limitations=_collect_limitations(sent, frames),
            ))

        return patterns


# ── Syntactic feature detectors ──

def _detect_sentence_type(text: str) -> str:
    """Detect sentence mood from ending punctuation only.

    Punctuation-based detection is near-100%:
    - ？→ interrogative
    - ！→ exclamatory
    - Otherwise → declarative (safe catch-all)

    Imperative detection via keywords (请/别/勿/禁止) is NOT near-100%
    — hints go to limitations for downstream resolution.
    """
    last = text.strip()[-1] if text.strip() else ""
    if last == "？":
        return "interrogative"
    if last == "！":
        return "exclamatory"
    return "declarative"


def _detect_structural_type(text: str,
                            frames: list) -> str:
    """Detect subject-predicate vs non-subject-predicate structure."""
    for f in frames:
        has_arg0 = has_pred = False
        for item in f:
            if len(item) >= 4:
                role = str(item[1]).upper()
                if role == "ARG0":
                    has_arg0 = True
                elif role == "PRED":
                    has_pred = True
        if has_arg0 and has_pred:
            return "subject_predicate"
    return "non_subject_predicate"


def _detect_polarity(text: str) -> str:
    """Returns 'affirmative' as safe default.

    Chinese negation detection requires syntactic scope resolution
    beyond current NLP capability. HanLP MTL output has no dedicated
    negation feature. Downstream (LLM/rules) must resolve.
    """
    return "affirmative"


def _detect_voice(text: str) -> str:
    """Returns 'active' as safe default.

    '被' keyword is ~90% reliable for passive, but misses:
    - Lexical passives (遭受, 受到, 得到, 给, 让, 叫 — ambiguous with
      causative/pivotal).
    - Semantic passives without marker (饭吃完了).

    Passive hints go to limitations for downstream.
    """
    return "active"


def _detect_sub_types(text: str) -> list[str]:
    """Returns empty list as safe default.

    Special constructions (ba_construction, bei_construction, serial_verb,
    pivotal, ellipsis) cannot be reliably detected via keyword matching.
    Hints go to limitations for downstream resolution.
    """
    return []


def _detect_rhetorical(text: str) -> str:
    """Returns 'unknown' as safe default.

    Rhetorical structure detection via comma counting is NOT near-100%.
    A sentence with ≥2 commas may be a list, not parallel structure.
    Hints go to limitations for downstream resolution.
    """
    return "unknown"


def _detect_length_tier(text: str) -> str:
    length = len(text.strip())
    if length <= 10:
        return "short"
    if length <= 30:
        return "medium"
    return "long"


def _count_clauses(text: str) -> int:
    return max(1, text.count("，") + text.count("；") + 1)


def _sentence_punct(text: str) -> str:
    last = text.strip()[-1] if text.strip() else ""
    return last if last in "。？！" else ""


def _count_sentence_words(sentence: str, raw: dict) -> int:
    """Count tokens from NLP output that belong to this sentence, by text position."""
    tok_fine = raw.get("tok/fine", [])
    if not tok_fine:
        return len(sentence.strip()) // 2  # fallback estimate

    # Reconstruct token positions using same logic as _map_tokens
    # but for the full text; then count tokens falling within sentence span
    text = "".join(tok_fine)  # HanLP fine-grained tokens reconstruct the text
    sent_start = text.find(sentence)
    if sent_start < 0:
        return len(sentence.strip()) // 2
    sent_end = sent_start + len(sentence)

    count = 0
    cursor = 0
    for t in tok_fine:
        idx = text.find(t, cursor)
        if idx >= 0:
            if sent_start <= idx < sent_end:
                count += 1
            cursor = idx + len(t)
        else:
            cursor += len(t)
    return count


def _collect_limitations(text: str, frames: list) -> list[str]:
    """Collect NLP capability gaps — ONLY from NLP output, NOT raw text.

    Principle: SentencePattern must be computed from NLP pipeline results
    (token/POS/NER/DEP/SRL), never from raw text string matching.

    Features NOT computable from current NLP output (documented for downstream):
    - Negation scope/polarity — no negation feature in HanLP MTL output
    - Passive voice — DEP may have clues but not reliably for Chinese
    - Ba-construction — no dedicated feature in MTL output
    - Imperative mood — no mood feature in MTL output
    - Rhetorical structure — no discourse parsing in MTL output
    - Pivotal construction — requires deep syntactic analysis
    - Ellipsis — no reliable detection method

    Only SRL-derived signals are provided here.
    """
    limits: list[str] = []

    # ── serial_verb: multiple ARG0s suggest serial verb clauses ──
    # Derived from SRL output (NLP), not raw text.
    nsubj_count = 0
    for f in frames:
        for item in f:
            if len(item) >= 4 and str(item[1]).upper() == "ARG0":
                nsubj_count += 1
    if nsubj_count >= 3:
        limits.append("hint:serial_verb")

    return limits
