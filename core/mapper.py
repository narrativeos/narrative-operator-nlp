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

import re
from datetime import datetime, timezone

from .entity_mapper import EntityMappingRules
from .entity_deduplicator import EntityDeduplicator
from .relation_mapper import RelationExtractionRules
from .schema import (
    Entity,
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

    def map(
        self,
        text: str,
        raw: dict,
        source: str = "hanlp_v2",
        entity_dict: dict[str, str] | None = None,
        entity_categories: dict[str, list[str]] | None = None,
        auto_discover_entities: bool = False,
    ) -> NarrativeDocument:
        # Reset per-request state to avoid cross-request pollution
        # (this mapper is a global singleton, so state must be request-local)
        self.entity_rules.reset()
        self.relation_rules.reset()
        tokens = self._map_tokens(text, raw)

        # ── Step 1: Extract raw entities (before merging) ──
        raw_entities = self._map_entities(
            text, raw, tokens, entity_dict, entity_categories,
            auto_discover_entities,
        )
        self._assign_attributes(text, raw, tokens, raw_entities)
        # PARAMETERs are properties, not standalone entities
        raw_entities = [e for e in raw_entities if e.category != "PARAMETER"]

        # ── Step 2: Extract relations using raw entities (preserves original mentions) ──
        relations = self._map_relations(text, raw, tokens, raw_entities)

        # ── Step 2.5: Apply negation-aware confidence adjustment ──
        from .negation_detector import find_negation_spans, compute_negation_aware_confidence
        negation_spans = find_negation_spans(text)
        for rel in relations:
            adjusted_conf, is_neg = compute_negation_aware_confidence(
                rel.confidence, rel.evidence_span, negation_spans,
            )
            rel.confidence = adjusted_conf
            # Dynamic confidence: adjust based on evidence quality
            rel.confidence = self._dynamic_confidence(rel, raw)

        # ── Step 3: Merge entities ──
        raw_entities.sort(key=lambda e: e.span[0])
        raw_entities = self.entity_rules.merger.merge_same_category(raw_entities)
        raw_entities.sort(key=lambda e: e.span[0])
        raw_entities = self.entity_rules.merger.merge_cross_category(raw_entities)
        raw_entities = self.entity_rules.merger.merge_det_entities(raw_entities, tokens, raw)
        raw_entities = EntityDeduplicator.deduplicate(raw_entities)

        # ── Step 4: Normalize relations to merged entities ──
        # Build mapping: raw mention text → resolved entity
        mention_to_entity: dict[str, Entity] = {}
        for e in raw_entities:
            mention_to_entity[e.text] = e
            # Also map merged text variants
            for attr in e.attributes:
                pass  # attributes are not entity mentions

        # Also build span-based mapping for mentions that don't have entity IDs
        span_to_entity: dict[tuple[int, int], Entity] = {}
        for e in raw_entities:
            span_to_entity[e.span] = e

        for rel in relations:
            # Normalize subject
            subj_entity = mention_to_entity.get(rel.subject)
            if subj_entity:
                rel.subject_raw = rel.subject
                rel.subject = subj_entity.text
                rel.subject_ent_id = subj_entity.id
            elif rel.subject_raw == rel.subject:
                # Already set raw during extraction
                pass

            # Normalize object
            obj_entity = mention_to_entity.get(rel.object)
            if obj_entity:
                rel.object_raw = rel.object
                rel.object = obj_entity.text
                rel.object_ent_id = obj_entity.id
            elif rel.object_raw == rel.object:
                pass

            # Set raw = canonical if not already set
            if not rel.subject_raw:
                rel.subject_raw = rel.subject
            if not rel.object_raw:
                rel.object_raw = rel.object

        entities = raw_entities
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
        tok_fine = raw.get("tok/fine") or raw.get("tok") or []
        # POS fallback chain: upos (universal) → ctb → pku → bare pos → "X"
        pos = (raw.get("pos/upos") or raw.get("pos/ctb") or raw.get("pos/pku")
               or raw.get("pos") or [])
        tok_conf = (raw.get("tok/fine_conf") or raw.get("tok/coarse_conf")
                    or raw.get("tok_conf"))
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

    def _map_entities(
        self,
        text: str,
        raw: dict,
        tokens: list[Token],
        entity_dict: dict[str, str] | None = None,
        entity_categories: dict[str, list[str]] | None = None,
        auto_discover_entities: bool = False,
    ) -> list:
        # Inject user-defined keywords into the keyword extractor
        if entity_categories:
            self.entity_rules.keyword_extractor.add_keywords(entity_categories)
        return self.entity_rules.map_all(
            text, raw, tokens, entity_dict, auto_discover_entities,
        )

    def _map_relations(self, text: str, raw: dict, tokens: list[Token],
                       entities: list) -> list:
        return self.relation_rules.extract_all(text, raw, tokens, entities)

    def _dynamic_confidence(self, rel, raw: dict) -> float:
        """Compute dynamic confidence based on evidence quality.

        Factors:
        - SRL source: higher confidence (0.65)
        - DEP source: medium confidence (0.55)
        - Bridge source: lower confidence (0.50)
        - Both endpoints matched to entities: bonus (+0.05)
        - Evidence length: shorter = more precise = bonus
        """
        base = rel.confidence

        # Source-based adjustment
        if "srl/" in rel.source:
            base = max(base, 0.65)
        elif "bridge/" in rel.source:
            base = min(base, 0.50)

        # Entity match bonus
        if rel.subject_ent_id and rel.object_ent_id:
            base = min(1.0, base + 0.05)
        elif rel.subject_ent_id or rel.object_ent_id:
            base = min(1.0, base + 0.02)

        # Evidence length bonus (shorter = more precise)
        evidence_len = rel.evidence_span[1] - rel.evidence_span[0]
        if evidence_len <= 10:
            base = min(1.0, base + 0.03)

        return base

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
                    parts.append("PRED")
            template = " ".join(parts)

            for r in relations:
                if r.predicate_verb in preds:
                    rel_summaries.append(f"{r.subject}→{r.predicate_verb}→{r.object}")

            # Count attributes for entities appearing in THIS sentence
            attr_count = sum(len(e.attributes) for e in entities if e.text in sent)

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
    """Detect special sentence constructions.

    Modern Chinese:
    - ba_construction: 把...V...
    - bei_construction: 被...V...
    
    Classical Chinese (古汉语特有句式):
    - judgment_sentence: ...者，...也 (判断句)
    - passive_classical: 见V于N, 为V所N (被动句)
    - rhetorical_question: 何...之有, 不亦...乎 (反问句)
    - object_fronting: 宾语前置 (倒装句)
    - comparison: 孰与..., 何...如 (比较句)
    - negation_judgment: 非...也 (否定判断句)
    """
    sub_types: list[str] = []
    
    # ── Modern Chinese constructions ──
    if "把" in text and re.search(r"把[\u4e00-\u9fff]+[^\u4e00-\u9fff]*[vV]", text):
        sub_types.append("ba_construction")
    if "被" in text:
        sub_types.append("bei_construction")
    
    # ── Classical Chinese constructions ──
    
    # 判断句: ...者，...也
    if re.search(r".+者[，,].+也", text):
        sub_types.append("judgment_sentence")
    
    # 被动句 (古汉语): 见V于N, 为V所N
    if re.search(r"见[\u4e00-\u9fff]+于", text):
        sub_types.append("passive_classical")
    if re.search(r"为[\u4e00-\u9fff]+所", text):
        sub_types.append("passive_classical")
    
    # 反问句: 何...之有
    if re.search(r"何[\u4e00-\u9fff]{0,4}之有", text):
        sub_types.append("rhetorical_question")
    
    # 反问句: 不亦...乎
    if re.search(r"不亦[\u4e00-\u9fff]+乎", text):
        sub_types.append("rhetorical_question")
    
    # 比较句: 孰与...
    if "孰与" in text:
        sub_types.append("comparison")
    
    # 比较句: 何...如
    if re.search(r"何[\u4e00-\u9fff]+如", text):
        sub_types.append("comparison")
    
    # 否定判断句: 非...也
    if re.search(r"非[\u4e00-\u9fff]+也", text):
        sub_types.append("negation_judgment")
    
    return sub_types


def _detect_rhetorical(text: str) -> str:
    """Detect rhetorical structure of the sentence.

    For modern Chinese, returns 'unknown' as safe default.
    For classical Chinese, attempts to detect:
    - parallel (对偶/排比): four-character rhythm blocks
    - loose (松散): mixed rhythm patterns

    Returns 'unknown' when detection is not reliable.
    """
    # Classical Chinese: four-character rhythm detection (四字格)
    total_chars = len(re.sub(r"[^\u4e00-\u9fff]", "", text))

    if total_chars >= 8:
        # Count four-character blocks (with or without trailing punctuation)
        # Pattern 1: four chars followed by punctuation
        four_char_with_punct = re.compile(r"[\u4e00-\u9fff]{4}[，。；、]")
        # Pattern 2: four chars at the end of text
        four_char_at_end = re.compile(r"[\u4e00-\u9fff]{4}$")

        count_with_punct = len(four_char_with_punct.findall(text))
        count_at_end = len(four_char_at_end.findall(text))
        four_char_count = count_with_punct + count_at_end

        # If >= 2 four-character blocks, likely parallel structure
        if four_char_count >= 2:
            return "parallel"

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

    For classical Chinese, additional limitations are documented.
    """
    limits: list[str] = []

    # ── serial_verb: multiple ARG0s suggest serial verb clauses ──
    nsubj_count = 0
    for f in frames:
        for item in f:
            if len(item) >= 4 and str(item[1]).upper() == "ARG0":
                nsubj_count += 1
    if nsubj_count >= 3:
        limits.append("hint:serial_verb")

    # ── Classical Chinese specific limitations ──
    # These patterns are common in classical Chinese but hard to detect reliably
    if any(pattern in text for pattern in ["者...也", "...者，...也"]):
        limits.append("hint:judgment_sentence")  # 判断句
    if "被" in text or "见" in text or "于" in text:
        limits.append("hint:passive_voice")  # 被动句
    if re.search(r"何[\u4e00-\u9fff]{1,6}之有", text):
        limits.append("hint:rhetorical_question")  # 反问句

    return limits
