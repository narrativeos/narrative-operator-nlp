"""
Coreference Resolution — Rule-based + Statistical.

Multi-language coreference resolution without LLM dependency.
Supports modern Chinese, English, and classical Chinese (degraded mode).

Architecture:
  Layer 1: Pronoun resolution (rule-based)
  Layer 2: Same-name entity merging (exact match)
  Layer 3: Heuristic clustering (type + distance + attributes)

References
----------
- Lee, Toutanova, "End-to-End Neural Coreference Resolution" (ACL 2011)
- Clark et al., "Glove-based Coreference Resolution" (EMNLP 2018)
- HanLP 1.x ``com.hankcs.hanlp.mining.coref.CorefResolver``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .schema import CoreferenceChain, Entity, Mention, Token

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chinese pronouns and their gender/number features
# ---------------------------------------------------------------------------

# Person pronouns
_CN_PERSON_PRONUN = {
    "我": {"person": "1st", "number": "sg"},
    "你": {"person": "2nd", "number": "sg"},
    "他": {"person": "3rd", "gender": "M", "number": "sg"},
    "她": {"person": "3rd", "gender": "F", "number": "sg"},
    "它": {"person": "3rd", "gender": "N", "number": "sg"},
    "我们": {"person": "1st", "number": "pl"},
    "你们": {"person": "2nd", "number": "pl"},
    "他们": {"person": "3rd", "gender": "M", "number": "pl"},
    "她们": {"person": "3rd", "gender": "F", "number": "pl"},
    "它们": {"person": "3rd", "gender": "N", "number": "pl"},
}

# Demonstrative + noun patterns (该/此/本 + N)
# Demonstrative prefixes that form nominal mentions (该/此/本/其/彼 + N).
# 是 is deliberately excluded: it is already a classical pronoun entry, and
# as a prefix it mostly yields false positives (是时/是人 = "at that time").
_CN_DEMONSTRATIVE_PREFIXES = {"该", "此", "本", "其", "彼"}

# POS tags that disqualify a token from being a pronoun, regardless of
# dictionary membership. 而 is a conjunction in virtually all classical
# texts (its archaic "you" sense is rare); KYOTO tags it CC.
_PRONOUN_EXCLUDED_POS = {"CC", "CCONJ"}

# Classical Chinese pronouns (古汉语代词)
# Enhanced with case/role information for better resolution
_CN_CLASSICAL_PRONUN = {
    # Third person (第三人称)
    "之": {"person": "3rd", "case": "obj"},      # 宾语位置
    "其": {"person": "3rd", "case": "gen"},      # 所有格/定语
    "彼": {"person": "3rd", "case": "subj"},     # 主语位置
    "此": {"person": "3rd", "case": "demonstrative"},  # 近指
    "是": {"person": "3rd", "case": "demonstrative"},  # 近指
    "斯": {"person": "3rd", "case": "demonstrative"},  # 近指
    "厥": {"person": "3rd", "case": "demonstrative"},  # 远指
    # First person (第一人称)
    "吾": {"person": "1st", "case": "subj"},     # 主语位置
    "余": {"person": "1st", "case": "subj"},     # 主语位置
    "我": {"person": "1st", "case": "universal"}, # 通用
    "予": {"person": "1st", "case": "obj"},      # 宾语位置
    # Second person (第二人称)
    "汝": {"person": "2nd", "case": "obj"},
    "尔": {"person": "2nd", "case": "obj"},
    "君": {"person": "2nd", "case": "subj"},
    # 子 removed as a pronoun: its "Master/you" uses are title-like and are
    # better captured as nominals (PROPN/NR) or NER (子路/子贡); keeping it
    # here made common-noun 子 (child, in 子女/弟子) a pronoun mention.
    "卿": {"person": "2nd", "case": "subj"},
    "若": {"person": "2nd", "case": "obj"},
    # 而 removed: it is a conjunction (CC), not a pronoun, in classical
    # texts — keeping it here made every clause linker a mention.
}

# English pronouns
_EN_PRONUN = {
    "I": {"person": "1st", "number": "sg"},
    "me": {"person": "1st", "number": "sg"},
    "my": {"person": "1st", "number": "sg"},
    "mine": {"person": "1st", "number": "sg"},
    "you": {"person": "2nd"},
    "your": {"person": "2nd"},
    "yours": {"person": "2nd"},
    "he": {"person": "3rd", "gender": "M", "number": "sg"},
    "him": {"person": "3rd", "gender": "M", "number": "sg"},
    "his": {"person": "3rd", "gender": "M", "number": "sg"},
    "she": {"person": "3rd", "gender": "F", "number": "sg"},
    "her": {"person": "3rd", "gender": "F", "number": "sg"},
    "hers": {"person": "3rd", "gender": "F", "number": "sg"},
    "it": {"person": "3rd", "gender": "N", "number": "sg"},
    "its": {"person": "3rd", "gender": "N", "number": "sg"},
    "we": {"person": "1st", "number": "pl"},
    "us": {"person": "1st", "number": "pl"},
    "our": {"person": "1st", "number": "pl"},
    "ours": {"person": "1st", "number": "pl"},
    "they": {"person": "3rd", "number": "pl"},
    "them": {"person": "3rd", "number": "pl"},
    "their": {"person": "3rd", "number": "pl"},
    "theirs": {"person": "3rd", "number": "pl"},
}

# Case-insensitive lookup set for English pronouns (sentence-initial
# "She"/"He"/"They" must match the lowercase dictionary keys).
_EN_PRONUN_LOWER = frozenset(w.lower() for w in _EN_PRONUN)

# ---------------------------------------------------------------------------
# Pronoun resolution strategy (batch 3)
# ---------------------------------------------------------------------------

# Sentence-final punctuation that bounds coreference distance.
# Unpunctuated classical text simply has no boundaries — every anchor
# stays "same sentence", which is the intended degraded behavior.
_SENTENCE_END_CHARS = frozenset("。！？!?…\n")

# Maximum character distance for CROSS-sentence pronoun resolution.
# Same-sentence anchors are never distance-capped (sentences are bounded).
_CROSS_SENTENCE_MAX_DIST = 60

# Penalty added to the distance score of cross-sentence anchors so that
# same-sentence anchors always win at comparable distance.
_CROSS_SENTENCE_PENALTY = 100

# Tier separator: person pronouns (他/她/he/she) strongly prefer PERSON
# anchors over known non-PERSON anchors (e.g. 他 after "…去了北京" should
# not take the LOCATION object). Anchors with unknown category (nominals)
# are never penalized. Deliberately larger than any realistic distance or
# the cross-sentence penalty so it acts as a hard tier.
_NON_PERSON_PENALTY = 1000


# ---------------------------------------------------------------------------
# Core Resolver
# ---------------------------------------------------------------------------

@dataclass
class CorefResult:
    """Result of coreference resolution."""
    chains: list[CoreferenceChain] = field(default_factory=list)
    language: str = "modern"
    model_version: str = "rule_based_v1"


class CorefResolver:
    """Rule-based coreference resolver.

    Supports:
    - Modern Chinese (he/she/it + demonstrative patterns)
    - English (pronoun resolution)
    - Classical Chinese (degraded mode, limited pronoun set)

    Usage:
        resolver = CorefResolver()
        result = resolver.resolve(text, entities, tokens, language="modern")
    """

    def resolve(
        self,
        text: str,
        entities: list[Entity],
        tokens: list[Token],
        language: str = "modern",
    ) -> CorefResult:
        """Resolve coreference for the given text.

        Args:
            text: Original text.
            entities: Recognized entities (may be empty if NER fails).
            tokens: Tokenized tokens.
            language: Language mode: modern|classical|english.

        Returns:
            CorefResult with resolved chains.

        Note: Even when NER produces no entities, this resolver still works
        by extracting noun phrases from POS tags as mention candidates.
        """
        # Step 1: Build mention candidates (entities + pronouns + noun phrases)
        mentions = self._build_mentions(text, entities, tokens, language)
        if not mentions:
            return CorefResult(language=language)

        # Step 2: Group mentions into chains
        chains = self._cluster_mentions(mentions, language, text, tokens)

        # Step 3: Build CoreferenceChain objects
        result_chains = self._build_chains(chains, language)

        return CorefResult(
            chains=result_chains,
            language=language,
        )

    def _build_mentions(
        self,
        text: str,
        entities: list[Entity],
        tokens: list[Token],
        language: str,
    ) -> list[Mention]:
        """Build mention candidates from entities, pronouns, and noun phrases.

        Key insight: even when NER produces no entities, we can still extract
        noun phrases from POS tags as mention candidates. This allows coreference
        resolution to work independently of NER quality.
        """
        mentions: list[Mention] = []
        pronoun_dict = self._get_pronoun_dict(language)

        # ── 1. Entity mentions (from NER) ──
        for ent in entities:
            mentions.append(Mention(
                text=ent.text,
                span=ent.span,
                mention_type="entity",
                entity_id=ent.id,
                category=ent.category,
                is_principal=True,
            ))

        # ── 2. Noun phrase mentions (from POS tags, independent of NER) ──
        # This is the key fix: extract noun phrases even when NER fails
        noun_mention_spans = self._extract_noun_phrases(tokens, language, entities)
        for np_text, np_span in noun_mention_spans:
            # Skip if already covered by an entity
            if self._is_covered_by_entity(np_span, entities):
                continue
            # Skip if it's a pronoun (handled separately)
            if np_text in pronoun_dict:
                continue
            mentions.append(Mention(
                text=np_text,
                span=np_span,
                mention_type="nominal",
                entity_id=None,
                is_principal=False,
            ))

        # ── 3. Pronoun mentions ──
        en_pronoun_lower = _EN_PRONUN_LOWER if language == "english" else None
        for tok in tokens:
            # POS guard: conjunction-tagged tokens (而/和/and) are never
            # pronouns even if a dictionary lists the surface form.
            if tok.pos in _PRONOUN_EXCLUDED_POS:
                continue
            if (tok.text.lower() in en_pronoun_lower if en_pronoun_lower
                    else tok.text in pronoun_dict):
                # Check if this pronoun is already covered by an entity or noun phrase
                if self._is_covered_by_entity(tok.span, entities):
                    continue
                if any(tok.span[0] >= np[1][0] and tok.span[1] <= np[1][1] for np in noun_mention_spans):
                    continue
                mentions.append(Mention(
                    text=tok.text,
                    span=tok.span,
                    mention_type="pronoun",
                    entity_id=None,
                    is_principal=False,
                ))

        # ── 4. Demonstrative + noun mentions (该/此/本 + N) ──
        if language in ("modern", "classical"):
            mentions.extend(self._find_demonstrative_mentions(text, tokens, entities))

        # Sort by span
        mentions.sort(key=lambda m: m.span[0])
        return mentions

    def _extract_noun_phrases(
        self,
        tokens: list[Token],
        language: str,
        entities: list[Entity] | None = None,
    ) -> list[tuple[str, tuple[int, int]]]:
        """Extract noun phrases from POS tags.

        Returns a list of (text, span) tuples for multi-word noun phrases
        and significant single-word nouns.

        Strategy:
        - Chinese: consecutive NR/NN tokens, or NR alone
        - English: consecutive NOUN/PROPN tokens, or PROPN alone
        - Extension stops at the start of a recognized entity, so two
          adjacent entities (北京 上海) are never merged into one NP.
        """
        phrases: list[tuple[str, tuple[int, int]]] = []
        if not tokens:
            return phrases

        # Character offsets where a recognized entity begins — a noun
        # chunk must not extend past one of these boundaries.
        entity_starts = {e.span[0] for e in (entities or [])}

        if language == "english":
            # English: PROPN sequences are entity-like mentions
            noun_poses = {"PROPN", "NOUN", "NNP", "NN"}
            i = 0
            while i < len(tokens):
                if tokens[i].pos in noun_poses:
                    start = i
                    i += 1  # first token always included in the chunk
                    # Extend while consecutive nouns (stop at entity starts)
                    while (i < len(tokens) and tokens[i].pos in noun_poses
                           and tokens[i].span[0] not in entity_starts):
                        i += 1
                    # Only keep if: PROPN (always) or multi-word or significant NN
                    chunk_tokens = tokens[start:i]
                    has_proper = any(t.pos in ("PROPN", "NNP") for t in chunk_tokens)
                    is_multi = len(chunk_tokens) >= 2
                    if has_proper or is_multi:
                        text = " ".join(t.text for t in chunk_tokens)
                        span = (chunk_tokens[0].span[0], chunk_tokens[-1].span[1])
                        phrases.append((text, span))
                else:
                    i += 1
        else:
            # Chinese: NR (proper noun) or NN sequences.
            # Accept both CTB tags (NR/NN/NT) and Universal tags
            # (NOUN/PROPN) — the classical pipeline (KYOTO EVAHAN)
            # emits upos values even though they are stored under
            # the pos/ctb key.
            noun_poses = {"NR", "NN", "NT", "NOUN", "PROPN"}
            i = 0
            while i < len(tokens):
                if tokens[i].pos in noun_poses:
                    start = i
                    i += 1  # first token always included in the chunk
                    # Extend while consecutive nouns (stop at entity starts)
                    while (i < len(tokens) and tokens[i].pos in noun_poses
                           and tokens[i].span[0] not in entity_starts):
                        i += 1
                    chunk_tokens = tokens[start:i]
                    has_proper = any(t.pos in ("NR", "PROPN") for t in chunk_tokens)
                    is_multi = len(chunk_tokens) >= 2
                    # Always extract proper nouns, or multi-word noun phrases
                    if has_proper or is_multi:
                        text = "".join(t.text for t in chunk_tokens)
                        span = (chunk_tokens[0].span[0], chunk_tokens[-1].span[1])
                        phrases.append((text, span))
                else:
                    i += 1

        return phrases

    def _get_pronoun_dict(self, language: str) -> dict[str, dict]:
        """Get pronoun dictionary for the given language."""
        if language == "english":
            return _EN_PRONUN
        if language == "classical":
            return _CN_CLASSICAL_PRONUN
        return _CN_PERSON_PRONUN

    def _is_covered_by_entity(self, span: tuple[int, int], entities: list[Entity]) -> bool:
        """Check if a span is already covered by an entity."""
        for ent in entities:
            if ent.span[0] <= span[0] and ent.span[1] >= span[1]:
                return True
        return False

    def _find_demonstrative_mentions(
        self,
        text: str,
        tokens: list[Token],
        entities: list[Entity],
    ) -> list[Mention]:
        """Find demonstrative + noun mentions (该/此/本 + N)."""
        mentions: list[Mention] = []
        for i, tok in enumerate(tokens):
            if tok.text in _CN_DEMONSTRATIVE_PREFIXES:
                # Look for following noun
                if i + 1 < len(tokens):
                    next_tok = tokens[i + 1]
                    if next_tok.pos in ("NN", "NR", "JJ", "NOUN", "PROPN"):
                        combined_text = tok.text + next_tok.text
                        combined_span = (tok.span[0], next_tok.span[1])
                        if not self._is_covered_by_entity(combined_span, entities):
                            mentions.append(Mention(
                                text=combined_text,
                                span=combined_span,
                                mention_type="nominal",
                                entity_id=None,
                                is_principal=False,
                            ))
        return mentions

    def _cluster_mentions(
        self,
        mentions: list[Mention],
        language: str,
        text: str,
        tokens: list[Token],
    ) -> list[list[Mention]]:
        """Cluster mentions into coreference chains.

        Algorithm:
        1. Group by exact text match (same-name entities AND nominals)
        2. Resolve pronouns to best compatible anchor (entity or nominal)
        3. Resolve nominal mentions to nearest compatible anchor
        """
        clusters: list[list[Mention]] = []
        used = set()

        # Step 1: Group by exact text match (entities AND nominals)
        # This is key: even without NER, repeated noun phrases form clusters.
        # Same-text entities with CONFLICTING categories are different
        # referents (e.g. person 王 vs place 王) and must be split into
        # separate groups; nominals attach to the nearest same-text group.
        text_groups: dict[str, list[Mention]] = {}
        for m in mentions:
            if m.mention_type in ("entity", "nominal"):
                text_groups.setdefault(m.text, []).append(m)

        final_groups: list[list[Mention]] = []
        for group in text_groups.values():
            final_groups.extend(self._split_category_conflicts(group))

        for sub_group in final_groups:
            if len(sub_group) >= 2:
                clusters.append(sub_group)
                for m in sub_group:
                    used.add(id(m))

        # Also add single entities AND nominals as singleton clusters
        # Entities are needed as anchors for nominal/pronoun resolution
        # Nominals can attract pronouns and other nominals
        # HOWEVER: demonstrative nominals (该/此/本 + N) should NOT be added
        # as final singletons here — they need to be resolved in Step 3
        DEMO_PREFIXES = {"该", "此", "本", "其", "彼", "是"}
        for sub_group in final_groups:
            if len(sub_group) == 1:
                m = sub_group[0]
                if m.mention_type in ("entity", "nominal") and id(m) not in used:
                    # Skip demonstrative nominals — they will be resolved in Step 3
                    if m.mention_type == "nominal":
                        is_demonstrative = any(
                            m.text.startswith(p) for p in DEMO_PREFIXES
                        )
                        if is_demonstrative:
                            continue
                    clusters.append([m])
                    used.add(id(m))

        # Step 2: Resolve pronouns to best compatible anchor
        for m in mentions:
            if m.mention_type == "pronoun" and id(m) not in used:
                target = self._resolve_pronoun(
                    m, mentions, clusters, language, text, tokens,
                )
                if target is not None:
                    target.append(m)
                    used.add(id(m))
                else:
                    # Unresolved pronoun: create singleton chain
                    clusters.append([m])
                    used.add(id(m))

            # Step 3: Resolve nominal mentions to nearest compatible anchor
            elif m.mention_type == "nominal" and id(m) not in used:
                target = self._resolve_nominal(m, mentions, clusters)
                if target is not None:
                    target.append(m)
                    used.add(id(m))

        # Sort clusters by first mention's span
        clusters.sort(key=lambda c: c[0].span[0])
        return clusters

    def _split_category_conflicts(
        self,
        group: list[Mention],
    ) -> list[list[Mention]]:
        """Split a same-text group whose entities have conflicting categories.

        Same-text mentions normally form one cluster (same-name merging).
        But if the group contains entities of DIFFERENT categories
        (e.g. person 王 vs place 王), they are different referents and must
        be separated. Nominals (category=None) attach to the nearest
        sub-group by span distance.

        Returns ``[group]`` unchanged when there is no category conflict.
        """
        entities = [m for m in group if m.mention_type == "entity"]
        categories = {m.category for m in entities if m.category}
        if len(categories) <= 1:
            return [group]

        by_category: dict[str, list[Mention]] = {}
        for m in entities:
            by_category.setdefault(m.category or "UNKNOWN", []).append(m)

        for n in (m for m in group if m.mention_type == "nominal"):
            best: Optional[list[Mention]] = None
            best_dist = float("inf")
            for sub in by_category.values():
                for em in sub:
                    dist = abs(n.span[0] - em.span[0])
                    if dist < best_dist:
                        best_dist = dist
                        best = sub
            if best is not None:
                best.append(n)

        return list(by_category.values())

    def _resolve_pronoun(
        self,
        pronoun: Mention,
        all_mentions: list[Mention],
        clusters: list[list[Mention]],
        language: str,
        text: str,
        tokens: list[Token],
    ) -> Optional[list[Mention]]:
        """Resolve a pronoun to the best compatible anchor cluster.

        Strategy (rule-based):
        1. Candidate anchors: entity/nominal mentions before the pronoun.
        2. Hard filters: person compatibility (``_is_compatible``) and
           gender consistency with pronouns already in the cluster.
        3. Scoring: character distance; cross-sentence anchors pay a
           penalty and are rejected beyond ``_CROSS_SENTENCE_MAX_DIST``;
           person pronouns (他/她/he/she) pay a tier penalty for known
           non-PERSON anchors so a nearby location/object loses to a
           farther person.
        4. First-conjunct preference: if the winning anchor is the right
           conjunct of a coordinated pair (a CC token sits between the
           two anchors), the left conjunct's cluster wins instead.
        """
        pronoun_features = self._get_pronoun_features(pronoun.text, language)

        best_cluster: Optional[list[Mention]] = None
        best_anchor: Optional[Mention] = None
        best_score = float("inf")

        for cluster in clusters:
            if not self._cluster_gender_compatible(cluster, pronoun_features, language):
                continue
            for m in cluster:
                if m.mention_type not in ("entity", "nominal"):
                    continue
                if m.span[1] > pronoun.span[0]:
                    continue
                if not self._is_compatible(m, pronoun_features, language):
                    continue
                distance = pronoun.span[0] - m.span[1]
                gap = text[m.span[1]:pronoun.span[0]]
                if any(ch in _SENTENCE_END_CHARS for ch in gap):
                    # Cross-sentence: bounded distance + penalty so that
                    # same-sentence anchors win at comparable distance.
                    if distance > _CROSS_SENTENCE_MAX_DIST:
                        continue
                    score = distance + _CROSS_SENTENCE_PENALTY
                else:
                    score = distance
                if not self._person_tier_ok(m, pronoun_features):
                    score += _NON_PERSON_PENALTY
                if score < best_score:
                    best_score = score
                    best_cluster = cluster
                    best_anchor = m

        if best_anchor is not None:
            left = self._left_conjunct_of(
                pronoun, best_anchor, best_cluster,
                clusters, language, text, tokens,
            )
            if left is not None:
                best_cluster = left

        return best_cluster

    def _cluster_gender_compatible(
        self,
        cluster: list[Mention],
        pronoun_features: dict,
        language: str,
    ) -> bool:
        """A gendered pronoun must not join a cluster that already contains
        a gendered pronoun of a different gender (他 vs 她 / he vs she).

        Entity gender is not annotated anywhere in this pipeline, so only
        pronoun-vs-pronoun agreement can be enforced (documented limit).
        """
        gender = pronoun_features.get("gender")
        if gender not in ("M", "F"):
            return True
        for m in cluster:
            if m.mention_type != "pronoun":
                continue
            other_gender = self._get_pronoun_features(m.text, language).get("gender")
            if other_gender in ("M", "F") and other_gender != gender:
                return False
        return True

    def _person_tier_ok(self, anchor: Mention, pronoun_features: dict) -> bool:
        """Person pronouns (他/她/he/she) prefer PERSON anchors.

        Anchors with unknown category (nominals) always pass — we never
        guess against missing information.
        """
        if pronoun_features.get("gender") not in ("M", "F"):
            return True
        return anchor.category in (None, "PERSON")

    def _left_conjunct_of(
        self,
        pronoun: Mention,
        best_anchor: Mention,
        best_cluster: list[Mention],
        clusters: list[list[Mention]],
        language: str,
        text: str,
        tokens: list[Token],
    ) -> Optional[list[Mention]]:
        """First-conjunct preference for coordinated pairs (张三 和 李四).

        If another compatible anchor sits to the LEFT of the winning anchor
        with a CC (conjunction) token between them, the pair is a
        coordinated subject/object and the left conjunct is the preferred
        antecedent. The check is POS-based (CC in both CTB and Universal
        tagsets), so verbs like 不及 (VV) do not trigger it. The two
        anchors must share a sentence (the pair may sit in the previous
        sentence relative to the pronoun).
        """
        pronoun_features = self._get_pronoun_features(pronoun.text, language)
        for cluster in clusters:
            if cluster is best_cluster:
                continue
            if not self._cluster_gender_compatible(cluster, pronoun_features, language):
                continue
            for m in cluster:
                if m.mention_type not in ("entity", "nominal"):
                    continue
                if not (m.span[1] <= best_anchor.span[0] <= pronoun.span[0]):
                    continue
                if not self._is_compatible(m, pronoun_features, language):
                    continue
                if not self._person_tier_ok(m, pronoun_features):
                    continue
                # The two anchors must share a sentence (a coordinated
                # pair never spans a sentence boundary). The pair may
                # itself sit in the previous sentence.
                gap = text[m.span[1]:best_anchor.span[0]]
                if any(ch in _SENTENCE_END_CHARS for ch in gap):
                    continue
                # A conjunction token strictly between the two anchors
                lo, hi = m.span[1], best_anchor.span[0]
                has_conjunction = any(
                    t.pos == "CC" and t.span[0] >= lo and t.span[1] <= hi
                    for t in tokens
                )
                if has_conjunction:
                    return cluster
        return None

    def _resolve_nominal(
        self,
        nominal: Mention,
        all_mentions: list[Mention],
        clusters: list[list[Mention]],
    ) -> Optional[list[Mention]]:
        """Resolve a nominal mention (该/此 + N) to the nearest compatible cluster.

        Matches by: text containment, suffix matching, or partial overlap.
        """
        # Extract the noun part (after the demonstrative prefix)
        noun_text = nominal.text[1:] if len(nominal.text) > 1 else nominal.text

        best_cluster = None
        best_distance = float("inf")

        for cluster in clusters:
            for m in cluster:
                if m.span[1] <= nominal.span[0] and m.mention_type in ("entity", "nominal"):
                    # Check if the anchor text matches or contains the noun
                    if noun_text in m.text or m.text.endswith(noun_text) or m.text in noun_text:
                        distance = nominal.span[0] - m.span[1]
                        if distance < best_distance:
                            best_distance = distance
                            best_cluster = cluster

        return best_cluster

    def _get_pronoun_features(self, pronoun_text: str, language: str) -> dict:
        """Get features of a pronoun (gender, number, person)."""
        if language == "english":
            return _EN_PRONUN.get(pronoun_text.lower(), {})
        if language == "classical":
            return _CN_CLASSICAL_PRONUN.get(pronoun_text, {})
        return _CN_PERSON_PRONUN.get(pronoun_text, {})

    def _is_compatible(self, entity: Mention, pronoun_features: dict, language: str) -> bool:
        """Check if an entity is compatible with a pronoun's features."""
        if not pronoun_features:
            return True

        # Person compatibility
        if "person" in pronoun_features:
            person = pronoun_features["person"]
            if person == "3rd":
                # 3rd person pronouns can refer to any entity
                pass
            elif person in ("1st", "2nd"):
                # 1st/2nd person: only matches PERSON entities (simplified).
                # Check the entity category — entity IDs (ent_001 /
                # ent_<hash>) never encode the category.
                if entity.category is not None and entity.category != "PERSON":
                    return False

        return True

    def _build_chains(
        self,
        clusters: list[list[Mention]],
        language: str,
    ) -> list[CoreferenceChain]:
        """Build CoreferenceChain objects from clusters."""
        chains: list[CoreferenceChain] = []
        chain_counter = 0

        for cluster in clusters:
            if not cluster:
                continue

            # Sort mentions by span
            cluster.sort(key=lambda m: m.span[0])

            # Find principal mention (first entity mention, or first mention).
            # Reset flags first: entity mentions are created with
            # is_principal=True, so without this a chain with two entities
            # would carry two principals.
            for m in cluster:
                m.is_principal = False
            principal = None
            for m in cluster:
                if m.mention_type == "entity":
                    principal = m
                    break
            if principal is None:
                principal = cluster[0]
            principal.is_principal = True

            # Determine quality flag
            quality = self._determine_quality(cluster, language)

            # Calculate confidence
            confidence = self._calculate_confidence(cluster, language)

            chain_id = f"coref_{chain_counter:04d}"
            chain_counter += 1

            chains.append(CoreferenceChain(
                chain_id=chain_id,
                mentions=cluster,
                representative=principal.text,
                confidence=confidence,
                language=language,
                quality_flag=quality,
                model_version="rule_based_v1",
            ))

        return chains

    def _determine_quality(self, cluster: list[Mention], language: str) -> str:
        """Determine quality flag for a chain."""
        if language == "classical":
            return "degraded"

        # Count entity mentions vs pronoun mentions
        entity_count = sum(1 for m in cluster if m.mention_type == "entity")
        pronoun_count = sum(1 for m in cluster if m.mention_type == "pronoun")

        if entity_count >= 2 and pronoun_count == 0:
            return "high"
        if entity_count >= 1:
            return "medium"
        return "low"

    def _calculate_confidence(self, cluster: list[Mention], language: str) -> float:
        """Calculate confidence score for a chain."""
        if language == "classical":
            return 0.4  # Degraded mode

        entity_count = sum(1 for m in cluster if m.mention_type == "entity")
        total = len(cluster)

        if entity_count >= 2:
            return 0.85
        if entity_count == 1:
            return 0.65
        return 0.45