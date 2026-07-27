"""
Event Extractor V3 — Entity → Relation → Event Pipeline (Hybrid SRL + Relation).

Approach:
1. Entity: NER identifies "who/what" (already done upstream)
2. Relation: dependency syntax extracts subject-predicate-object triples (already done upstream)
3. Event: relations with the same predicate_verb in the same sentence are clustered
   into a single event instance; SRL frames enrich arguments and validate confidence.

Hybrid strategy:
- Relation clustering provides the event skeleton (grouping + trigger word)
- SRL frames provide richer argument roles (ARGM-TMP→Time, ARGM-LOC→Location, etc.)
  that Relations alone cannot capture
- SRL also validates: if SRL's ARG0/ARG1 match the Relation's subject/object,
  confidence gets a boost

This creates a clear traceability chain:
  Event.source_relation_ids → Relation.id
  Relation.subject_ent_id/object_ent_id → Entity.id

Quality filtering:
- Skip relations without predicate_verb (no action trigger)
- Skip relations from classical_pattern (rule-based, not event-like)
- Skip stopwords as triggers (是/有/在/为)
- Skip SRL frames that are not content predicates (no ARG0+ARG1 → prepositions/light verbs)
- Filter by sentence pattern type (prioritize declarative)
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from .schema import (
    DependencyEdge, Entity, Event, EventArgument, Relation,
    SentenceLanguage, SentencePattern,
)


# ── Stopwords to filter as triggers ──
_TRIGGER_STOPWORDS: set[str] = {
    "是", "有", "在", "为", "作", "成", "当", "像", "够",
    "了", "着", "过", "吗", "吧", "呢", "啊", "呀",
    "把", "被", "给", "对", "对于", "关于",
    "和", "与", "及", "或", "而", "但",
    "做", "搞", "弄",
    "通过", "经过", "按照", "根据",
}


# ── Relation predicate → Object's event argument role mapping ──
# Subject always maps to "Agent"; this table defines the object's role.
_PREDICATE_TO_ROLE: dict[str, str] = {
    # Action verbs: object is the result/patient
    "CAUSES": "Result",
    "PRODUCES": "Product",
    "CONTROLS": "Patient",
    "AFFECTS": "Patient",
    "MOVED_TO": "Destination",
    "DEPARTED_FROM": "Origin",
    "INTERACTS_WITH": "Counterpart",
    "SAYS": "Utterance",
    "TRANSFERS_TO": "Target",
    # Stative / classificatory: object is the category/location/property value
    "IS_A": "Type",
    "PART_OF": "Whole",
    "LOCATED_AT": "Location",
    "TEMPORAL_AT": "Time",
    "DIED_AT": "Location",
    "DEPENDS_ON": "Dependency",
    "COMPOSED_OF": "Component",
    # Attribute relations: object is the attribute value
    "HAS_TITLE": "Title",
    "HAS_STYLE_NAME": "StyleName",
    "HAS_PROPERTY": "Property",
    "PROPERTY_OF": "Owner",
    "BENEFITS": "Beneficiary",
    # Generic
    "EQUIVALENT_TO": "Equivalent",
    "REFERENCE_OF": "Reference",
    "CONSTRAINT_OF": "Constraint",
    "RELATES_TO": "Related",
}


# ── SRL ARGM-* → Event argument role mapping ──
_ARGM_ROLE_MAP: dict[str, str] = {
    "ARGM-TMP": "Time",
    "ARGM-LOC": "Location",
    "ARGM-MNR": "Manner",
    "ARGM-CAU": "Cause",
    "ARGM-PRD": "Product",
    "ARGM-BEN": "Beneficiary",
    "ARGM-REC": "Recipient",
    "ARGM-SRC": "Source",
    "ARGM-ADJ": "Adjunct",
    "ARGM-DIR": "Direction",
    "ARGM-PRP": "Purpose",
    "ARGM-ADV": "Adverbial",
    "ARGM-EXT": "Extent",
    "ARGM-FRQ": "Frequency",
    "ARGM-DGR": "Degree",
}


def _is_content_predicate(frame: list) -> bool:
    """Check if an SRL frame represents a content predicate (not a function word).

    Relaxed criteria (V4.1):
    - Must have ARG0 (agent/subject)
    - Must have ARG1 OR at least one ARGM-* role

    This allows frames like "过去三年 [ARG0] 实现 [PRED]" where the time
    expression is the only argument besides ARG0, which was previously
    filtered out requiring both ARG0+ARG1.

    Frames with only ARGM-* (no ARG0) are still function words
    (prepositions, coverbs, light verbs) and should be skipped.
    """
    roles = {str(item[1]).upper() for item in frame if len(item) >= 2}
    has_arg0 = "ARG0" in roles
    has_arg1 = "ARG1" in roles
    has_argm = any(r.startswith("ARGM-") for r in roles)
    return has_arg0 and (has_arg1 or has_argm)


class EventExtractor:
    """Extract structured events by clustering relations with the same predicate verb,
    enriched by SRL semantic role labeling.

    Pipeline: Entity → Relation → Event (hybrid SRL + Relation clustering)
    - Groups relations by (sentence_index, predicate_verb)
    - Each group becomes one Event skeleton
    - SRL frames enrich arguments with ARGM-* roles (Time, Location, Manner, etc.)
    - SRL validates confidence when ARG0/ARG1 match relation endpoints
    - Traceability via source_relation_ids
    """

    def __init__(self) -> None:
        self._counter = 0

    def reset(self) -> None:
        self._counter = 0

    def extract(
        self,
        text: str,
        relations: list[Relation],
        entities: list[Entity],
        sentences: list[SentenceLanguage],
        patterns: list[SentencePattern],
        srl_frames: Optional[list] = None,
        deps: Optional[list[DependencyEdge]] = None,
        tokens: Optional[list] = None,
    ) -> list[Event]:
        """Extract events by clustering relations, enriched with SRL.

        V4.2 improvements over V4:
        1. Agent-dimension clustering: relations with the same (sentence, verb, agent)
           are grouped into one event, preventing different agents from being merged.
        2. Event deduplication: events with identical (trigger_span, agent, patient)
           are deduplicated, keeping the one with highest confidence.
        3. Sub-event hierarchy: relations with no Agent (e.g., adjunct-only) are linked
           as sub-events to the main event they belong to.
        4. Dependency-driven extraction: when SRL is missing (e.g., HanLP ELECTRA-small
           doesn't output SRL for all verbs), use dependency relations to extract events
           from conj+ccomp patterns.

        Args:
            text: Original text for span resolution.
            relations: Relations already extracted from dependency syntax.
            entities: Entity list for argument linking.
            sentences: Sentence boundaries and language labels.
            patterns: Sentence patterns for filtering.
            srl_frames: Raw HanLP SRL frames (list of list of (text, role, tok_start, tok_end)).
                        If None or empty, SRL enrichment is skipped.
            deps: Dependency edges from HanLP. Used as a fallback when SRL is missing.
            tokens: Token list from HanLP. Used for dependency-driven extraction.

        Returns:
            A list of Event instances.
        """
        if not relations:
            return []

        self.reset()
        srl_frames = srl_frames or []

        # Build lookups
        entity_by_text: dict[str, Entity] = {e.text: e for e in entities}

        # Store tokens as list for methods that need positional access
        tokens_list: list = tokens if tokens else []

        # Build token index by span for dependency-driven extraction
        # Store (text, span, pos) for POS-aware filtering
        token_by_idx: dict[int, tuple[str, tuple[int, int], str]] = {}
        if tokens:
            for tok in tokens:
                tok_id = getattr(tok, 'id', None)
                tok_text = getattr(tok, 'text', '')
                tok_span = getattr(tok, 'span', (0, 0))
                tok_pos = getattr(tok, 'pos', 'X')
                if tok_id is not None:
                    token_by_idx[tok_id] = (tok_text, tok_span, tok_pos)

        # Build sentence span lookup
        sent_spans: list[tuple[int, int]] = [(s.span[0], s.span[1]) for s in sentences]

        # Build pattern lookup by sentence index
        pattern_by_idx: dict[int, SentencePattern] = {}
        for i, p in enumerate(patterns):
            pattern_by_idx[i] = p

        # ── Pre-process SRL frames: build (sentence_idx, predicate_verb) → frame lookup ──
        srl_by_key: dict[tuple[int, str], list[dict]] = defaultdict(list)
        for frame in srl_frames:
            if not isinstance(frame, list) or len(frame) < 2:
                continue
            if not _is_content_predicate(frame):
                continue

            # Extract predicate verb
            pred_text = ""
            for item in frame:
                if len(item) >= 2 and str(item[1]).upper() == "PRED":
                    pred_text = str(item[0])
                    break
            if not pred_text or pred_text in _TRIGGER_STOPWORDS:
                continue

            # Determine sentence index: use SRL token positions as hint for precise matching.
            srl_hint_span: Optional[tuple[int, int]] = None
            if len(frame) >= 1 and len(frame[0]) >= 4:
                try:
                    srl_hint_span = (int(frame[0][2]), int(frame[0][3]))
                except (ValueError, TypeError):
                    pass
            sent_idx = self._find_sentence_by_text(
                sentences,
                str(frame[0][0]) if frame else "",
                hint_span=srl_hint_span,
            )

            srl_by_key[(sent_idx, pred_text)].append({
                "frame": frame,
                "pred_text": pred_text,
            })

        # Determine sentence index for each relation by evidence_span
        def _get_sentence_index(span_start: int) -> int:
            for idx, (ss, se) in enumerate(sent_spans):
                if ss <= span_start < se:
                    return idx
            return -1

        # ── Step 1: Group relations by (sentence_index, predicate_verb, primary_agent) ──
        # Agent-dimension clustering: different agents with the same verb form separate events
        groups: dict[tuple[int, str, str], list[Relation]] = defaultdict(list)

        for rel in relations:
            # Skip relations without predicate_verb (no action trigger)
            if not rel.predicate_verb:
                continue

            # Skip stopwords as triggers
            if rel.predicate_verb in _TRIGGER_STOPWORDS:
                continue

            # Skip classical pattern relations (rule-based, not event-like)
            if "classical_pattern" in rel.source:
                continue

            # Skip self-loops
            if rel.subject == rel.object:
                continue

            sent_idx = _get_sentence_index(rel.evidence_span[0])

            # Check sentence pattern for filtering
            pat = pattern_by_idx.get(sent_idx)
            if pat and pat.sentence_type == "interrogative":
                continue

            # Use the subject (Agent) as part of the clustering key
            # Relations without a clear subject use empty string as agent key
            agent_key = rel.subject or ""
            key = (sent_idx, rel.predicate_verb, agent_key)
            groups[key].append(rel)

        # ── Build entity_by_id lookup for NER conflict detection ──
        entity_by_id: dict[str, Entity] = {e.id: e for e in entities}

        # ── Step 2: Convert groups to events ──
        events: list[Event] = []

        for (sent_idx, trigger_verb, agent_key), rels in sorted(groups.items()):
            if not rels:
                continue

            # Build arguments from relations
            arguments: list[EventArgument] = []
            source_rel_ids: list[str] = []

            for rel in rels:
                source_rel_ids.append(rel.id)

                # Subject → Agent
                if rel.subject:
                    agent_entity_id: Optional[str] = rel.subject_ent_id
                    if not agent_entity_id:
                        agent_entity_id = self._find_entity_id(rel.subject, entity_by_text)

                    arguments.append(EventArgument(
                        role="Agent",
                        text=rel.subject,
                        entity_id=agent_entity_id,
                        span=self._find_char_span(text, rel.subject, rel.evidence_span),
                    ))

                # Object → role based on predicate
                if rel.object:
                    role = _PREDICATE_TO_ROLE.get(rel.predicate, "Patient")

                    obj_entity_id: Optional[str] = rel.object_ent_id
                    if not obj_entity_id:
                        obj_entity_id = self._find_entity_id(rel.object, entity_by_text)

                    arguments.append(EventArgument(
                        role=role,
                        text=rel.object,
                        entity_id=obj_entity_id,
                        span=self._find_char_span(text, rel.object, rel.evidence_span),
                    ))

            # ── SRL enrichment: add ARGM-* arguments and validate confidence ──
            srl_entries = srl_by_key.get((sent_idx, trigger_verb), [])
            srl_boost = False
            for entry in srl_entries:
                frame = entry["frame"]
                for item in frame:
                    if len(item) < 4:
                        continue
                    arg_text, role = str(item[0]), str(item[1]).upper()

                    # ARGM-* roles → additional EventArguments
                    if role in _ARGM_ROLE_MAP:
                        mapped_role = _ARGM_ROLE_MAP[role]
                        # Use SRL token positions as hint for accurate span resolution
                        srl_hint: Optional[tuple[int, int]] = None
                        if len(item) >= 4:
                            try:
                                tok_start = int(item[2])
                                tok_end = int(item[3])
                                srl_hint = (tok_start, tok_end)
                            except (ValueError, TypeError):
                                pass
                        arg_span = self._find_char_span(text, arg_text, hint_span=srl_hint)
                        if arg_span is None or arg_span == (0, 0):
                            continue
                        # Avoid duplicate arguments (same role + same text)
                        if any(a.role == mapped_role and a.text == arg_text for a in arguments):
                            continue
                        arguments.append(EventArgument(
                            role=mapped_role,
                            text=arg_text,
                            entity_id=self._find_entity_id(arg_text, entity_by_text),
                            span=arg_span,
                        ))

                    # SRL validation: ARG0/ARG1 confirm Relation endpoints → confidence boost
                    if role == "ARG0" and any(
                        arg_text in a.text or a.text in arg_text
                        for a in arguments if a.role == "Agent"
                    ):
                        srl_boost = True
                    if role == "ARG1" and any(
                        arg_text in a.text or a.text in arg_text
                        for a in arguments if a.role != "Agent"
                    ):
                        srl_boost = True

            # Skip if no meaningful arguments
            if not arguments:
                continue

            # ── Dynamic confidence (Step 3) ──
            rel_predicate = rels[0].predicate if rels else "RELATES_TO"
            rel_sem_class = rels[0].semantic_class if rels else None
            confidence = self._compute_confidence(
                trigger_verb=trigger_verb,
                arguments=arguments,
                deps=deps or [],
                tokens=tokens_list,
                relation_predicate=rel_predicate,
                semantic_class=rel_sem_class,
                entities_by_id=entity_by_id,
            )
            if srl_boost:
                confidence = min(1.0, confidence + 0.1)

            # Find trigger span from original text, using first relation's evidence_span as hint
            trigger_span = self._find_char_span(
                text, trigger_verb,
                hint_span=rels[0].evidence_span,
                fallback=rels[0].evidence_span,
            )

            # Create event
            self._counter += 1
            evt = Event(
                id=f"evt_{self._counter:03d}",
                event_type=trigger_verb,
                trigger=trigger_verb,
                trigger_span=trigger_span,
                arguments=arguments,
                sentence_index=sent_idx,
                is_main_event=True,
                sub_events=[],
                source_relation_ids=source_rel_ids,
                confidence=confidence,
                source="relation_cluster" if not srl_entries else "srl+relation_cluster",
            )
            events.append(evt)

        # ── Step 3: Event deduplication ──
        # Deduplicate events with identical (trigger_span, agent_text, patient_text)
        events = self._deduplicate_events(events)

        # ── Step 4: Extract events directly from SRL frames ──
        # When Relations are missing (e.g., "引发到处封路管控"), SRL frames may still
        # contain the event structure. Extract events from SRL frames that don't have
        # a corresponding Relation-based event.
        srl_events = self._extract_events_from_srl_only(
            text, srl_by_key, entity_by_text, sentences, sent_spans,
            pattern_by_idx, events, tokens_list, entity_by_id,
        )
        events.extend(srl_events)

        # ── Step 4b: Extract events from dependency relations ──
        # When both Relations and SRL are missing for a verb (e.g., "引发" in
        # "出现在...，引发到处封路管控"), use dependency tree to extract the event.
        # Pattern: conj verb → ccomp/dobj → result
        deps = deps or []
        dep_events = self._extract_events_from_deps(
            text, deps, token_by_idx, entity_by_text, sentences, sent_spans,
            pattern_by_idx, events, tokens_list, entity_by_id,
        )
        events.extend(dep_events)

        # ── Step 5: Enrich with temporal/location entities from NER ──
        self._enrich_with_temporal_entities(text, events, entities, sent_spans)

        # ── Step 5b: Assign syntactic roles (Step 4) ──
        if deps and tokens_list:
            self._assign_syntactic_roles(events, deps, tokens_list)

        # ── Step 6: Build sub-event hierarchy ──
        self._build_sub_event_hierarchy(events)

        # ── Step 7: Fill token_span for all event arguments (Step 5) ──
        if tokens_list:
            self._fill_event_token_spans(events, tokens_list, entity_by_id)

        return events

    def _extract_events_from_deps(
        self,
        text: str,
        deps: list[DependencyEdge],
        token_by_idx: dict[int, tuple[str, tuple[int, int], str]],
        entity_by_text: dict[str, Entity],
        sentences: list[SentenceLanguage],
        sent_spans: list[tuple[int, int]],
        pattern_by_idx: dict[int, SentencePattern],
        existing_events: list[Event],
        tokens_list: list,
        entity_by_id: dict[str, Entity],
    ) -> list[Event]:
        """Extract events from dependency relations when SRL is missing.

        Pattern: Find conj verbs (parallel predicates) that have ccomp/dobj children.
        Example: "出现在...，引发到处封路管控"
        - 14(引发) → conj of 7(出现)  → "引发" is a parallel verb
        - 17(管控) → ccomp of 14(引发) → "管控" is the result of "引发"

        Agent extraction: find nsubj of the parent verb, filter to nouns only,
        merge adjacent nouns into a single phrase.
        """
        if not deps or not token_by_idx:
            return []

        # Noun POS tags that qualify as Agent
        _noun_poses = {"NN", "NR", "NP", "NS", "NT", "NR1", "NR2", "NN1", "NN2"}

        # Build existing keys to avoid duplication
        existing_keys = {(evt.sentence_index, evt.trigger) for evt in existing_events}

        # Build children index: head_idx -> [(child_idx, rel), ...]
        children_by_head: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for dep in deps:
            children_by_head[dep.head].append((dep.child, dep.rel))

        dep_events: list[Event] = []

        for dep in deps:
            if dep.rel != 'conj':
                continue

            conj_child = dep.child

            # Get trigger text from token_by_idx
            tok_info = token_by_idx.get(conj_child)
            if not tok_info:
                continue
            trigger_text, trigger_span, _trigger_pos = tok_info

            if trigger_text in _TRIGGER_STOPWORDS:
                continue

            # Find sentence index by span
            sent_idx = self._find_sentence_by_span(sentences, trigger_span)
            if sent_idx < 0:
                continue

            if (sent_idx, trigger_text) in existing_keys:
                continue

            pat = pattern_by_idx.get(sent_idx)
            if pat and pat.sentence_type == "interrogative":
                continue

            # Find ccomp/dobj children → Result
            arguments: list[EventArgument] = []
            for child_idx, rel in children_by_head.get(conj_child, []):
                if rel in ('ccomp', 'dobj', 'iobj'):
                    child_info = token_by_idx.get(child_idx)
                    if not child_info:
                        continue
                    obj_text, obj_span, _obj_pos = child_info
                    arguments.append(EventArgument(
                        role="Result",
                        text=obj_text,
                        entity_id=self._find_entity_id(obj_text, entity_by_text),
                        span=obj_span,
                    ))

            # Find Agent from parent verb's nsubj (shared subject)
            # Strategy: find nsubj tokens, then walk up their nn/modifier chains
            # to get the full noun phrase (e.g., "泰国" nn-> "总理" nsubj-> "出现")
            conj_head = dep.head
            
            # Collect all nsubj token indices that are nouns
            nsubj_indices: list[int] = []
            for child_idx, rel in children_by_head.get(conj_head, []):
                if rel == 'nsubj':
                    agent_info = token_by_idx.get(child_idx)
                    if not agent_info:
                        continue
                    _agent_text, _agent_span, agent_pos = agent_info
                    if agent_pos in _noun_poses:
                        nsubj_indices.append(child_idx)

            # For each nsubj, walk up the nn chain to collect the full noun phrase
            agent_groups: list[list[int]] = []
            for nsubj_idx in nsubj_indices:
                # Walk up: find all nn modifiers that chain to this nsubj
                group: list[int] = [nsubj_idx]
                current = nsubj_idx
                # Follow nn links upward (nn children of this token are modifiers)
                visited: set[int] = {current}
                for nn_child, nn_rel in children_by_head.get(current, []):
                    if nn_rel == 'nn':
                        nn_info = token_by_idx.get(nn_child)
                        if nn_info and nn_info[2] in _noun_poses:
                            group.append(nn_child)
                            visited.add(nn_child)
                            # Continue walking up from this nn child
                            for deeper_child, deeper_rel in children_by_head.get(nn_child, []):
                                if deeper_rel == 'nn' and deeper_child not in visited:
                                    deeper_info = token_by_idx.get(deeper_child)
                                    if deeper_info and deeper_info[2] in _noun_poses:
                                        group.append(deeper_child)
                                        visited.add(deeper_child)
                # Sort by span to get correct text order
                group.sort(key=lambda idx: token_by_idx[idx][1][0] if idx in token_by_idx else 0)
                agent_groups.append(group)

            # Pick the best group: prefer the one closest to the conj verb (the trigger)
            # This avoids picking temporal expressions ("7月18日") over the actual subject
            # ("泰国总理") when both appear as nsubj of the parent verb.
            # The conj verb shares the subject of the clause it belongs to, which is
            # the nsubj group closest to it in the text.
            if agent_groups:
                best_group = min(
                    agent_groups,
                    key=lambda g: (
                        # Primary: closer to the conj verb is better (smaller distance)
                        min(abs(token_by_idx[idx][1][0] - trigger_span[0])
                            for idx in g if idx in token_by_idx),
                        # Secondary: longer noun phrase is better (negate for min)
                        -len(g),
                    ),
                )
                agent_parts = []
                agent_span_start = None
                agent_span_end = None
                for idx in best_group:
                    info = token_by_idx.get(idx)
                    if info:
                        agent_parts.append(info[0])
                        if agent_span_start is None:
                            agent_span_start = info[1][0]
                        agent_span_end = info[1][1]

                if agent_parts and agent_span_start is not None and agent_span_end is not None:
                    merged_agent = "".join(agent_parts)
                    arguments.append(EventArgument(
                        role="Agent",
                        text=merged_agent,
                        entity_id=self._find_entity_id(merged_agent, entity_by_text),
                        span=(agent_span_start, agent_span_end),
                    ))

            if not arguments:
                continue

            # Dynamic confidence for DEP-only events (Step 3)
            dep_confidence = self._compute_confidence(
                trigger_verb=trigger_text,
                arguments=arguments,
                deps=deps,
                tokens=tokens_list,
                relation_predicate="RELATES_TO",
                semantic_class=None,
                entities_by_id=entity_by_id,
            )
            self._counter += 1
            evt = Event(
                id=f"evt_{self._counter:03d}",
                event_type=trigger_text,
                trigger=trigger_text,
                trigger_span=trigger_span,
                arguments=arguments,
                sentence_index=sent_idx,
                is_main_event=True,
                sub_events=[],
                source_relation_ids=[],
                confidence=dep_confidence,
                source="dep_only",
            )
            dep_events.append(evt)

        return dep_events

    def _find_sentence_by_span(
        self, sentences: list[SentenceLanguage], span: tuple[int, int]
    ) -> int:
        """Find which sentence contains the given character span."""
        for i, sent in enumerate(sentences):
            if sent.span[0] <= span[0] < sent.span[1]:
                return i
        return -1

    def _extract_events_from_srl_only(
        self,
        text: str,
        srl_by_key: dict[tuple[int, str], list[dict]],
        entity_by_text: dict[str, Entity],
        sentences: list[SentenceLanguage],
        sent_spans: list[tuple[int, int]],
        pattern_by_idx: dict[int, SentencePattern],
        existing_events: list[Event],
        tokens_list: list,
        entity_by_id: dict[str, Entity],
    ) -> list[Event]:
        """Extract events directly from SRL frames when no Relation-based event exists.

        This handles cases where the dependency parser fails to extract a relation
        (e.g., "引发到处封路管控" where "引发" is the predicate but no relation
        was extracted), but SRL still has the semantic roles.

        Only creates an event if there is NO existing event with the same
        (sentence_index, trigger) to avoid duplication.
        """
        if not srl_by_key:
            return []

        # Build set of existing (sent_idx, trigger) pairs to avoid duplication
        existing_keys = {(evt.sentence_index, evt.trigger) for evt in existing_events}

        srl_events: list[Event] = []

        for (sent_idx, trigger_verb), entries in sorted(srl_by_key.items()):
            # Skip if already have an event for this (sentence, trigger)
            if (sent_idx, trigger_verb) in existing_keys:
                continue

            # Skip stopwords as triggers (prepositions, light verbs)
            if trigger_verb in _TRIGGER_STOPWORDS:
                continue

            # Check sentence pattern for filtering
            pat = pattern_by_idx.get(sent_idx)
            if pat and pat.sentence_type == "interrogative":
                continue

            for entry in entries:
                frame = entry["frame"]
                arguments: list[EventArgument] = []

                for item in frame:
                    if len(item) < 4:
                        continue
                    arg_text, role = str(item[0]), str(item[1]).upper()

                    if role == "ARG0":
                        # ARG0 → Agent
                        srl_hint: Optional[tuple[int, int]] = None
                        try:
                            srl_hint = (int(item[2]), int(item[3]))
                        except (ValueError, TypeError):
                            pass
                        arg_span = self._find_char_span(text, arg_text, hint_span=srl_hint)
                        if arg_span == (0, 0):
                            continue
                        arguments.append(EventArgument(
                            role="Agent",
                            text=arg_text,
                            entity_id=self._find_entity_id(arg_text, entity_by_text),
                            span=arg_span,
                        ))
                    elif role == "ARG1":
                        # ARG1 → Patient/Result
                        srl_hint = None
                        try:
                            srl_hint = (int(item[2]), int(item[3]))
                        except (ValueError, TypeError):
                            pass
                        arg_span = self._find_char_span(text, arg_text, hint_span=srl_hint)
                        if arg_span == (0, 0):
                            continue
                        arguments.append(EventArgument(
                            role="Patient",
                            text=arg_text,
                            entity_id=self._find_entity_id(arg_text, entity_by_text),
                            span=arg_span,
                        ))
                    elif role in _ARGM_ROLE_MAP:
                        mapped_role = _ARGM_ROLE_MAP[role]
                        srl_hint = None
                        try:
                            srl_hint = (int(item[2]), int(item[3]))
                        except (ValueError, TypeError):
                            pass
                        arg_span = self._find_char_span(text, arg_text, hint_span=srl_hint)
                        if arg_span is None or arg_span == (0, 0):
                            continue
                        if any(a.role == mapped_role and a.text == arg_text for a in arguments):
                            continue
                        arguments.append(EventArgument(
                            role=mapped_role,
                            text=arg_text,
                            entity_id=self._find_entity_id(arg_text, entity_by_text),
                            span=arg_span,
                        ))

                if not arguments:
                    continue

                # Find trigger span
                trigger_span = self._find_char_span(
                    text, trigger_verb,
                    fallback=sent_spans[sent_idx] if sent_idx >= 0 and sent_idx < len(sent_spans) else (0, 0),
                )

                # Dynamic confidence for SRL-only events (Step 3)
                srl_confidence = self._compute_confidence(
                    trigger_verb=trigger_verb,
                    arguments=arguments,
                    deps=[],
                    tokens=tokens_list,
                    relation_predicate="RELATES_TO",
                    semantic_class=None,
                    entities_by_id=entity_by_id,
                )
                self._counter += 1
                evt = Event(
                    id=f"evt_{self._counter:03d}",
                    event_type=trigger_verb,
                    trigger=trigger_verb,
                    trigger_span=trigger_span,
                    arguments=arguments,
                    sentence_index=sent_idx,
                    is_main_event=True,
                    sub_events=[],
                    source_relation_ids=[],
                    confidence=srl_confidence,
                    source="srl_only",
                )
                srl_events.append(evt)

        return srl_events

    def _enrich_with_temporal_entities(
        self,
        text: str,
        events: list[Event],
        entities: list[Entity],
        sent_spans: list[tuple[int, int]],
    ) -> None:
        """Enrich events with Time/Location arguments from NER-detected entities.

        When SRL fails to identify time/location expressions (common with
        relative time expressions like '过去三年', '2024年', etc.), we use
        NER-detected DATE/LOCATION/FACILITY entities to fill the gap.

        Strategy:
        1. For each event, find its sentence span
        2. Look for DATE entities in the same sentence without an existing Time arg
        3. Look for LOCATION/FACILITY entities without an existing Location arg
        4. Only add if the entity is NOT already used as Agent/Patient (avoid duplication)

        Modifies events in-place.
        """
        if not entities:
            return

        # Pre-index entities by category
        date_entities = [e for e in entities if e.category == "DATE"]
        location_entities = [e for e in entities if e.category in ("LOCATION", "FACILITY")]

        # Build set of entity texts already used as Agent/Patient/etc. (to avoid duplication)
        # This is checked per-event below

        for evt in events:
            if evt.sentence_index < 0:
                continue

            sent_start, sent_end = sent_spans[evt.sentence_index]

            # Get existing argument texts to avoid duplication
            existing_texts = {a.text for a in evt.arguments}

            # Check if Time argument is missing
            has_time = any(a.role == "Time" for a in evt.arguments)
            if not has_time and date_entities:
                # Find the DATE entity closest to the trigger in this sentence
                trigger_center = (evt.trigger_span[0] + evt.trigger_span[1]) / 2
                best_entity: Optional[Entity] = None
                best_distance = float('inf')

                # Time unit keywords that indicate a proper date expression
                _time_units = {"日", "月", "年", "时", "分", "秒", "周", "天", "点", "刻"}
                
                for ent in date_entities:
                    # Check if entity is in the same sentence
                    if not (sent_start <= ent.span[0] < sent_end):
                        continue
                    # Skip if already used as another argument
                    if ent.text in existing_texts:
                        continue
                    # Skip pure numeric DATE entities (e.g., "18", "7") - they are fragments
                    # of larger date expressions and should not be used as standalone time args
                    if ent.text.isdigit():
                        continue
                    # Prefer date expressions containing time unit words
                    has_time_unit = any(c in _time_units for c in ent.text)
                    if not has_time_unit:
                        continue
                    # Calculate distance to trigger
                    ent_center = (ent.span[0] + ent.span[1]) / 2
                    distance = abs(ent_center - trigger_center)
                    if distance < best_distance:
                        best_distance = distance
                        best_entity = ent

                if best_entity is not None:
                    evt.arguments.append(EventArgument(
                        role="Time",
                        text=best_entity.text,
                        entity_id=best_entity.id,
                        span=best_entity.span,
                    ))
                    existing_texts.add(best_entity.text)

            # Check if Location argument is missing
            has_location = any(a.role == "Location" for a in evt.arguments)
            if not has_location and location_entities:
                trigger_center = (evt.trigger_span[0] + evt.trigger_span[1]) / 2
                best_entity = None
                best_distance = float('inf')

                for ent in location_entities:
                    if not (sent_start <= ent.span[0] < sent_end):
                        continue
                    if ent.text in existing_texts:
                        continue
                    ent_center = (ent.span[0] + ent.span[1]) / 2
                    distance = abs(ent_center - trigger_center)
                    if distance < best_distance:
                        best_distance = distance
                        best_entity = ent

                if best_entity is not None:
                    evt.arguments.append(EventArgument(
                        role="Location",
                        text=best_entity.text,
                        entity_id=best_entity.id,
                        span=best_entity.span,
                    ))

    def _deduplicate_events(self, events: list[Event]) -> list[Event]:
        """Deduplicate events with identical core arguments.

        Two events are considered duplicates if they share:
        - Same trigger_span
        - Same Agent text (if present)
        - Same Patient/Result text (if present)

        Keeps the event with the highest confidence.
        """
        if len(events) <= 1:
            return events

        seen: dict[tuple, Event] = {}

        for evt in events:
            agent_text = next((a.text for a in evt.arguments if a.role == "Agent"), "")
            patient_text = next(
                (a.text for a in evt.arguments
                 if a.role in ("Patient", "Result", "Product")),
                "",
            )
            key = (evt.trigger_span, agent_text, patient_text)

            if key in seen:
                # Keep the one with higher confidence
                if evt.confidence > seen[key].confidence:
                    seen[key] = evt
            else:
                seen[key] = evt

        return list(seen.values())

    def _build_sub_event_hierarchy(self, events: list[Event]) -> None:
        """Link sub-events to their parent main events.

        An event is considered a sub-event if:
        - It shares the same (sentence_index, trigger) as another event
        - It has no Agent argument (adjunct-only event)
        - The parent event has an Agent argument

        Modifies events in-place to set is_main_event and sub_events fields.
        """
        if not events:
            return

        # Group events by (sentence_index, trigger)
        trigger_groups: dict[tuple[int, str], list[Event]] = defaultdict(list)
        for evt in events:
            trigger_groups[(evt.sentence_index, evt.trigger)].append(evt)

        for group in trigger_groups.values():
            if len(group) == 1:
                continue

            main_events = [e for e in group if any(a.role == "Agent" for a in e.arguments)]
            sub_events = [e for e in group if not any(a.role == "Agent" for a in e.arguments)]

            if not sub_events:
                continue

            # If no clear main event, pick the one with highest confidence as main
            if not main_events:
                group.sort(key=lambda e: e.confidence, reverse=True)
                main_events = [group[0]]
                sub_events = group[1:]

            for main in main_events:
                main.is_main_event = True
                main.sub_events = [sub.id for sub in sub_events]

            for sub in sub_events:
                sub.is_main_event = False

    # ── Confidence scoring (Step 3) ──

    # Tuneable coefficients (P0: extracted from magic numbers for maintainability)
    _CONF_BASE: float = 0.70
    _CONF_ROOT_VERB: float = 0.15
    _CONF_AGENT_COMPLETE: float = 0.10
    _CONF_PATIENT_COMPLETE: float = 0.10
    _CONF_PENALTY_LONG_OBJ: float = 0.15
    _CONF_PENALTY_PUNCT: float = 0.20
    _CONF_PENALTY_NER_DISPUTED: float = 0.10
    _CONF_PENALTY_GENERIC_PRED: float = 0.10
    _CONF_LONG_OBJ_THRESHOLD: int = 15

    # Only sentence-boundary punctuation signals a broken argument.
    # Quotes, brackets, colons legitimately appear in event text
    # (e.g. "曰：'学而时习之'"). 。！？ are the true red flags.
    _CONF_SENTENCE_BOUNDARY_PUNCT_RE = re.compile(r'[。！？]')

    @staticmethod
    def _compute_confidence(
        trigger_verb: str,
        arguments: list[EventArgument],
        deps: list[DependencyEdge],
        tokens: list,
        relation_predicate: str,
        semantic_class: Optional[str],
        entities_by_id: dict[str, Entity],
    ) -> float:
        """Compute dynamic confidence score based on structural completeness.

        Formula (coefficients are class constants for tuneability):
          base _CONF_BASE
          +_CONF_ROOT_VERB : verb is root node (core predicate)
          +_CONF_AGENT_COMPLETE : agent is complete (non-empty)
          +_CONF_PATIENT_COMPLETE : patient is complete (non-empty)
          -_CONF_PENALTY_LONG_OBJ : longest non-Agent arg > _CONF_LONG_OBJ_THRESHOLD chars
          -_CONF_PENALTY_PUNCT : non-Agent arg contains sentence-boundary punctuation
          -_CONF_PENALTY_NER_DISPUTED : NER multi-model conflict on linked entity
          -_CONF_PENALTY_GENERIC_PRED : predicate is RELATES_TO with no semantic_class
          → clamped to [0.0, 1.0]

        Design rationale:
        - Only sentence-boundary punctuation (。！？) is penalised because
          quotes, brackets, colons legitimately appear in event arguments
          (especially classical Chinese dialogue markers like "曰：").
        - Coefficients are class-level constants so they can be overridden
          or tuned without modifying method logic.
        """
        score = EventExtractor._CONF_BASE

        # +root_verb
        if deps and tokens:
            is_root = any(
                dep.head == -1
                and 0 <= dep.child < len(tokens)
                and getattr(tokens[dep.child], 'text', '') == trigger_verb
                for dep in deps
            )
            if is_root:
                score += EventExtractor._CONF_ROOT_VERB

        # +agent_complete
        has_agent = any(a.role == "Agent" and a.text for a in arguments)
        if has_agent:
            score += EventExtractor._CONF_AGENT_COMPLETE

        # +patient_complete
        has_patient = any(a.role != "Agent" and a.text for a in arguments)
        if has_patient:
            score += EventExtractor._CONF_PATIENT_COMPLETE

        # -long_obj / -punct: non-Agent argument checks
        non_agent_args = [a for a in arguments if a.role != "Agent"]
        if non_agent_args:
            max_len = max(len(a.text) for a in non_agent_args)
            if max_len > EventExtractor._CONF_LONG_OBJ_THRESHOLD:
                score -= EventExtractor._CONF_PENALTY_LONG_OBJ
            if any(EventExtractor._CONF_SENTENCE_BOUNDARY_PUNCT_RE.search(a.text)
                   for a in non_agent_args):
                score -= EventExtractor._CONF_PENALTY_PUNCT

        # -ner_disputed
        for arg in arguments:
            if arg.entity_id and arg.entity_id in entities_by_id:
                ent = entities_by_id[arg.entity_id]
                if getattr(ent, 'ner_disputed', False):
                    score -= EventExtractor._CONF_PENALTY_NER_DISPUTED
                    break

        # -generic_pred
        if relation_predicate == "RELATES_TO" and not semantic_class:
            score -= EventExtractor._CONF_PENALTY_GENERIC_PRED

        return max(0.0, min(1.0, score))

    # ── Syntactic role assignment (Step 4) ──

    @staticmethod
    def _assign_syntactic_roles(
        events: list[Event],
        deps: list[DependencyEdge],
        tokens: list,
    ) -> None:
        """Assign syntactic roles to all event arguments from the dependency tree.

        For each EventArgument, finds its corresponding token and queries the
        dependency relation to determine:
        - syntactic_role: Subject (nsubj), Object (dobj), Adverbial (loc/lobj),
          or Attributive (nn/amod).
        - governing_verb: The first verb ancestor in the dependency tree.

        Token matching strategy:
        Uses maximum overlap to find the best-matching token. For multi-word
        arguments (e.g., "张老三" tokenized as "张/NR 老三/NR"), this typically
        selects the core noun ("老三") where dependency relations (nsubj/dobj)
        are usually attached. This is acceptable because HanLP dependency parsers
        typically attach relations to the head word within a noun phrase.

        Modifies events in-place.
        """
        if not deps or not tokens:
            return

        parent_of: dict[int, tuple[int, str]] = {}
        for dep in deps:
            if dep.child >= 0:
                parent_of[dep.child] = (dep.head, dep.rel)

        def _find_token_idx(arg_span: tuple[int, int]) -> Optional[int]:
            best_idx: Optional[int] = None
            best_overlap = 0
            for i, tok in enumerate(tokens):
                tok_span = getattr(tok, 'span', (0, 0))
                overlap_start = max(arg_span[0], tok_span[0])
                overlap_end = min(arg_span[1], tok_span[1])
                if overlap_end > overlap_start:
                    overlap = overlap_end - overlap_start
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_idx = i
            return best_idx

        _DEP_TO_SYNTACTIC: dict[str, str] = {
            "nsubj": "Subject", "nsubjpass": "Subject",
            "dobj": "Object", "iobj": "Object",
            "loc": "Adverbial", "lobj": "Adverbial",
            "nn": "Attributive", "amod": "Attributive",
        }

        for evt in events:
            for arg in evt.arguments:
                token_idx = _find_token_idx(arg.span)
                if token_idx is None:
                    continue

                if token_idx in parent_of:
                    _head_idx, dep_rel = parent_of[token_idx]
                    dep_rel_lower = dep_rel.lower() if dep_rel else ""
                    syn_role = _DEP_TO_SYNTACTIC.get(dep_rel_lower)
                    if syn_role is not None:
                        arg.syntactic_role = syn_role

                # Walk up to first V-pos ancestor
                current = token_idx
                visited: set[int] = set()
                while current in parent_of and current not in visited:
                    visited.add(current)
                    head_idx, _rel = parent_of[current]
                    if head_idx < 0:
                        break
                    if 0 <= head_idx < len(tokens):
                        head_pos = getattr(tokens[head_idx], 'pos', '')
                        if head_pos and head_pos[0] == 'V':
                            arg.governing_verb = getattr(tokens[head_idx], 'text', '')
                            break
                    current = head_idx

    # ── Token span filling (Step 5) ──

    @staticmethod
    def _fill_event_token_spans(
        events: list[Event],
        tokens: list,
        entity_by_id: dict[str, Entity],
    ) -> None:
        """Fill EventArgument.token_span from entity lookup or span→token index.

        For each EventArgument:
        - If entity_id is set, copy token_span from the matching Entity.
        - Otherwise, compute from character span → token index overlap.

        Modifies events in-place.
        """
        if not events or not tokens:
            return

        token_spans: list[tuple[int, int]] = []
        for tok in tokens:
            # P2: defensive check — skip tokens without span attribute
            if not hasattr(tok, 'span'):
                continue
            token_spans.append(tok.span)

        for evt in events:
            for arg in evt.arguments:
                if arg.token_span is not None:
                    continue

                if arg.entity_id and arg.entity_id in entity_by_id:
                    ent = entity_by_id[arg.entity_id]
                    ent_ts = getattr(ent, 'token_span', None)
                    if ent_ts is not None:
                        arg.token_span = ent_ts
                        continue

                first_idx: int | None = None
                last_idx: int | None = None
                for i, (ts, te) in enumerate(token_spans):
                    if ts < arg.span[1] and te > arg.span[0]:
                        if first_idx is None:
                            first_idx = i
                        last_idx = i
                if first_idx is not None and last_idx is not None:
                    arg.token_span = (first_idx, last_idx + 1)

    # ── Helper methods ──

    def _resolve_srl_arg_span(
        self, text: str, arg_text: str, srl_tok_start: int, srl_tok_end: int,
        tokens: list, hint_span: Optional[tuple[int, int]] = None,
    ) -> Optional[tuple[int, int]]:
        """Resolve SRL argument span with token-level fallback.

        Strategy:
        1. Try exact match via _find_char_span with hint
        2. If result text doesn't match SRL arg_text, fallback to token boundary
           text[tokens[ts].span[0] : tokens[te-1].span[1]]
        3. Last resort: original _find_char_span behavior

        Args:
            text: Original text
            arg_text: SRL argument text from HanLP
            srl_tok_start: SRL token start index (0-based, inclusive) — matches HanLP's tok_start
            srl_tok_end: SRL token end index (0-based, exclusive) — matches HanLP's tok_end
            tokens: Token list from HanLP (0-based indexing)
            hint_span: Optional character-level hint for search

        Note: HanLP SRL outputs tok_start/tok_end as 0-based indices where
        tok_start is inclusive and tok_end is exclusive (Python slice style).
        See relation_mapper.py:520-523 for the same convention.
        """
        # Phase 1: Try precise match
        span = self._find_char_span(text, arg_text, hint_span=hint_span)
        if span and span != (0, 0):
            extracted = text[span[0]:span[1]]
            if extracted == arg_text:
                return span

        # Phase 2: Fallback to token boundary (SRL tokens are 0-based, te exclusive)
        ts, te = srl_tok_start, srl_tok_end
        if tokens and 0 <= ts < len(tokens) and 0 < te <= len(tokens):
            return (tokens[ts].span[0], tokens[te - 1].span[1])

        # Phase 3: Last resort - original behavior
        return self._find_char_span(text, arg_text, hint_span=hint_span)

    def _find_char_span(
        self, text: str, target: str,
        hint_span: Optional[tuple[int, int]] = None,
        fallback: Optional[tuple[int, int]] = None,
    ) -> tuple[int, int]:
        """Find the character span of `target` in `text`, near a hint position.

        Uses hint_span (e.g., evidence_span) to constrain the search area,
        avoiding incorrect matches when the same text appears multiple times.

        Strategy:
        1. If hint_span provided, first try exact match within hint range
        2. Then expand outward from hint center using nearest-match strategy
        3. Fall back to global search only if hint is unavailable

        Returns (start, end) or the fallback span if not found.
        """
        if not target or not text:
            return fallback or (0, 0)

        # Find all occurrences of target in text
        occurrences: list[int] = []
        start = 0
        while True:
            idx = text.find(target, start)
            if idx < 0:
                break
            occurrences.append(idx)
            start = idx + 1

        if not occurrences:
            return fallback or (0, 0)

        if hint_span is not None:
            # Prefer an occurrence whose center is closest to the hint center
            hint_center = (hint_span[0] + hint_span[1]) / 2
            best_idx = min(
                occurrences,
                key=lambda idx: abs((idx + len(target) / 2) - hint_center),
            )
            return (best_idx, best_idx + len(target))

        # No hint: return first occurrence
        return (occurrences[0], occurrences[0] + len(target))

    def _find_sentence_by_text(
        self, sentences: list[SentenceLanguage], arg_text: str,
        hint_span: Optional[tuple[int, int]] = None,
    ) -> int:
        """Find which sentence contains the given argument text.

        Uses hint_span (from SRL token positions) for precise matching when available.
        Falls back to text containment check, preferring the first match.

        Returns sentence index or -1 if not found.
        """
        if not arg_text:
            return -1

        # If we have a character-level hint (e.g., from SRL tok positions),
        # use span containment for precise matching
        if hint_span is not None:
            for i, sent in enumerate(sentences):
                if sent.span[0] <= hint_span[0] < sent.span[1]:
                    return i

        # Fallback: text containment check
        for i, sent in enumerate(sentences):
            if arg_text in sent.text:
                return i
        return -1

    def _find_entity_id(self, text: str, entity_by_text: dict[str, Entity]) -> Optional[str]:
        """Find entity ID by text matching.

        Strategy:
        1. Exact match (highest priority)
        2. Substring match with 80% length threshold to avoid spurious associations
           (e.g., "创作者" should not match "创作者联盟" - the entity must be
           at least 80% of the argument text length, or vice versa)
        """
        if text in entity_by_text:
            return entity_by_text[text].id

        # Try substring match with length threshold
        text_len = len(text)
        best: Optional[Entity] = None
        best_overlap = 0

        for ent in entity_by_text.values():
            ent_len = len(ent.text)
            if text == ent.text:
                return ent.id

            # Check containment in either direction
            if text in ent.text:
                # text is a substring of entity text
                overlap = text_len
            elif ent.text in text:
                # entity text is a substring of argument text
                overlap = ent_len
            else:
                continue

            # Require at least 80% overlap to avoid spurious matches
            min_len = min(text_len, ent_len)
            if overlap >= min_len * 0.8 and overlap > best_overlap:
                best = ent
                best_overlap = overlap

        return best.id if best else None
