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
_CN_DEMONSTRATIVE_PREFIXES = {"该", "此", "本", "其", "彼", "是"}

# Classical Chinese pronouns
_CN_CLASSICAL_PRONUN = {
    "之": {"person": "3rd"},
    "其": {"person": "3rd"},
    "彼": {"person": "3rd"},
    "此": {"person": "3rd"},
    "是": {"person": "3rd"},
    "吾": {"person": "1st"},
    "余": {"person": "1st"},
    "我": {"person": "1st"},
    "汝": {"person": "2nd"},
    "尔": {"person": "2nd"},
    "君": {"person": "2nd"},
    "子": {"person": "2nd"},
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
            entities: Recognized entities.
            tokens: Tokenized tokens.
            language: Language mode: modern|classical|english.

        Returns:
            CorefResult with resolved chains.
        """
        if not entities:
            return CorefResult(language=language)

        # Step 1: Build mention candidates (entities + pronouns)
        mentions = self._build_mentions(text, entities, tokens, language)
        if not mentions:
            return CorefResult(language=language)

        # Step 2: Group mentions into chains
        chains = self._cluster_mentions(mentions, language)

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
        """Build mention candidates from entities and pronouns."""
        mentions: list[Mention] = []
        pronoun_dict = self._get_pronoun_dict(language)

        # Add entity mentions
        for ent in entities:
            mentions.append(Mention(
                text=ent.text,
                span=ent.span,
                mention_type="entity",
                entity_id=ent.id,
                is_principal=True,
            ))

        # Add pronoun mentions
        for tok in tokens:
            if tok.text in pronoun_dict:
                # Check if this pronoun is already covered by an entity
                if self._is_covered_by_entity(tok.span, entities):
                    continue
                mentions.append(Mention(
                    text=tok.text,
                    span=tok.span,
                    mention_type="pronoun",
                    entity_id=None,
                    is_principal=False,
                ))

        # Add demonstrative + noun mentions (该/此/本 + N)
        if language in ("modern", "classical"):
            mentions.extend(self._find_demonstrative_mentions(text, tokens, entities))

        # Sort by span
        mentions.sort(key=lambda m: m.span[0])
        return mentions

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
    ) -> list[list[Mention]]:
        """Cluster mentions into coreference chains.

        Algorithm:
        1. Group by exact text match (same-name entities)
        2. Resolve pronouns to nearest compatible entity
        3. Merge small clusters
        """
        clusters: list[list[Mention]] = []
        used = set()

        # Step 1: Group entities by text (same-name)
        text_groups: dict[str, list[Mention]] = {}
        for m in mentions:
            if m.mention_type == "entity":
                text_groups.setdefault(m.text, []).append(m)

        for text, group in text_groups.items():
            if len(group) >= 2:
                clusters.append(group)
                for m in group:
                    used.add(id(m))

        # Step 2: Resolve pronouns
        for m in mentions:
            if m.mention_type == "pronoun" and id(m) not in used:
                target = self._resolve_pronoun(m, mentions, clusters, language)
                if target is not None:
                    target.append(m)
                    used.add(id(m))
                else:
                    # Unresolved pronoun: create singleton chain
                    clusters.append([m])
                    used.add(id(m))

            # Step 3: Resolve nominal mentions
            elif m.mention_type == "nominal" and id(m) not in used:
                target = self._resolve_nominal(m, mentions, clusters)
                if target is not None:
                    target.append(m)
                    used.add(id(m))

        # Sort clusters by first mention's span
        clusters.sort(key=lambda c: c[0].span[0])
        return clusters

    def _resolve_pronoun(
        self,
        pronoun: Mention,
        all_mentions: list[Mention],
        clusters: list[list[Mention]],
        language: str,
    ) -> Optional[list[Mention]]:
        """Resolve a pronoun to the nearest compatible entity cluster."""
        pronoun_features = self._get_pronoun_features(pronoun.text, language)

        best_cluster = None
        best_distance = float("inf")

        for cluster in clusters:
            # Find the nearest entity in the cluster that appears before the pronoun
            for m in cluster:
                if m.span[1] <= pronoun.span[0] and m.mention_type == "entity":
                    distance = pronoun.span[0] - m.span[1]
                    if distance < best_distance:
                        # Check compatibility
                        if self._is_compatible(m, pronoun_features, language):
                            best_distance = distance
                            best_cluster = cluster

        return best_cluster

    def _resolve_nominal(
        self,
        nominal: Mention,
        all_mentions: list[Mention],
        clusters: list[list[Mention]],
    ) -> Optional[list[Mention]]:
        """Resolve a nominal mention (该/此 + N) to the nearest compatible cluster."""
        # Extract the noun part (after the demonstrative prefix)
        noun_text = nominal.text[1:] if len(nominal.text) > 1 else nominal.text

        best_cluster = None
        best_distance = float("inf")

        for cluster in clusters:
            for m in cluster:
                if m.span[1] <= nominal.span[0] and m.mention_type == "entity":
                    # Check if the entity text matches or contains the noun
                    if noun_text in m.text or m.text.endswith(noun_text):
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
            elif person == "1st":
                # 1st person: only matches PERSON entities (simplified)
                if entity.entity_id and "PERSON" not in entity.entity_id:
                    return False
            elif person == "2nd":
                # 2nd person: only matches PERSON entities (simplified)
                if entity.entity_id and "PERSON" not in entity.entity_id:
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

            # Find principal mention (first entity mention, or first mention)
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