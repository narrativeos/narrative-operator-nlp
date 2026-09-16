"""Regression tests for coreference resolver bug fixes.

Covers (audit 2026-09):
- S1: 1st/2nd person pronouns resolve via entity category (not entity_id)
- S2: same-text entities with conflicting categories are NOT merged
- S4: English noun phrases keep spaces between words
- S5: sentence-initial capitalized English pronouns are recognized
- S6: Chinese NP extraction works with Universal POS tags (classical pipeline)
- H4: classical pronoun dict has no duplicate keys (彼)
"""

from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from core.coref_resolver import CorefResolver, _CN_CLASSICAL_PRONUN
from core.schema import Entity, Token


def _ent(eid: str, text: str, category: str, span: tuple[int, int]) -> Entity:
    return Entity(
        id=eid,
        text=text,
        category=category,
        span=span,
        normalized=text,
        source="test",
        confidence=0.9,
    )


def _toks(pairs: list[tuple[str, str, int, int]]) -> list[Token]:
    return [
        Token(id=i, text=t, pos=p, span=(s, e))
        for i, (t, p, s, e) in enumerate(pairs)
    ]


def _chains(resolver: CorefResolver, ents, toks, language: str):
    return resolver.resolve("x", ents, toks, language).chains


def _chain_with(chains, text: str):
    """Return the chain containing a mention with the given text."""
    for c in chains:
        if any(m.text == text for m in c.mentions):
            return c
    return None


@pytest.fixture(scope="module")
def resolver() -> CorefResolver:
    return CorefResolver()


class TestFirstSecondPersonResolution:
    """S1: 我/你 must resolve against entity category, not entity_id."""

    def test_first_person_resolves_to_person(self, resolver):
        ents = [_ent("ent_001", "张三", "PERSON", (0, 2))]
        toks = _toks([("张三", "NR", 0, 2), ("说", "VV", 2, 3), ("我", "PRP", 3, 4)])
        chains = _chains(resolver, ents, toks, "modern")
        chain = _chain_with(chains, "我")
        assert chain is not None
        assert any(m.text == "张三" and m.mention_type == "entity"
                   for m in chain.mentions)

    def test_first_person_rejects_non_person(self, resolver):
        ents = [_ent("ent_001", "北京", "LOCATION", (0, 2))]
        toks = _toks([("北京", "NR", 0, 2), ("我", "PRP", 2, 3)])
        chains = _chains(resolver, ents, toks, "modern")
        chain = _chain_with(chains, "我")
        assert chain is not None
        # 我 must NOT attach to a LOCATION entity
        assert all(m.mention_type == "pronoun" for m in chain.mentions)

    def test_english_first_person_resolves(self, resolver):
        ents = [_ent("ent_001", "Alice", "PERSON", (0, 5))]
        toks = _toks([("Alice", "PROPN", 0, 5), ("said", "VERB", 5, 9),
                      ("I", "PRP", 9, 10)])
        chains = _chains(resolver, ents, toks, "english")
        chain = _chain_with(chains, "I")
        assert chain is not None
        assert any(m.text == "Alice" for m in chain.mentions)


class TestEnglishNounPhrases:
    """S4: multi-word English NPs must keep spaces."""

    def test_proper_noun_phrase_keeps_space(self, resolver):
        toks = _toks([("New", "PROPN", 0, 3), ("York", "PROPN", 4, 8)])
        chains = _chains(resolver, [], toks, "english")
        chain = _chain_with(chains, "New York")
        assert chain is not None, "expected nominal 'New York'"
        assert chain.representative == "New York"

    def test_repeated_english_np_forms_chain(self, resolver):
        toks = _toks([("New", "PROPN", 0, 3), ("York", "PROPN", 4, 8),
                      ("and", "CC", 8, 11),
                      ("New", "PROPN", 11, 14), ("York", "PROPN", 15, 19)])
        chains = _chains(resolver, [], toks, "english")
        chain = _chain_with(chains, "New York")
        assert chain is not None
        assert sum(1 for m in chain.mentions if m.text == "New York") == 2


class TestEnglishPronounCase:
    """S5: sentence-initial capitalized pronouns must be recognized."""

    @pytest.mark.parametrize("pronoun", ["She", "He", "They", "It"])
    def test_capitalized_pronoun_detected(self, resolver, pronoun):
        ents = [_ent("ent_001", "Alice", "PERSON", (0, 5))]
        toks = _toks([("Alice", "PROPN", 0, 5), ("went", "VERB", 5, 9),
                      (pronoun, "PRP", 9, 9 + len(pronoun))])
        chains = _chains(resolver, ents, toks, "english")
        assert any(m.mention_type == "pronoun" and m.text == pronoun
                   for c in chains for m in c.mentions)

    def test_capitalized_pronoun_resolves(self, resolver):
        ents = [_ent("ent_001", "Alice", "PERSON", (0, 5))]
        toks = _toks([("Alice", "PROPN", 0, 5), ("went", "VERB", 5, 9),
                      ("She", "PRP", 9, 12)])
        chains = _chains(resolver, ents, toks, "english")
        chain = _chain_with(chains, "She")
        assert chain is not None
        assert any(m.text == "Alice" for m in chain.mentions)


class TestCategoryConflictSplit:
    """S2: same-text entities with different categories must not merge."""

    def test_same_text_different_categories_split(self, resolver):
        ents = [_ent("ent_001", "王", "PERSON", (0, 1)),
                _ent("ent_002", "王", "LOCATION", (4, 5))]
        toks = _toks([("王", "NR", 0, 1), ("与", "CC", 1, 2), ("王", "NR", 4, 5)])
        chains = _chains(resolver, ents, toks, "modern")
        person_chains = [c for c in chains
                         if any(m.category == "PERSON" for m in c.mentions)]
        loc_chains = [c for c in chains
                      if any(m.category == "LOCATION" for m in c.mentions)]
        assert len(person_chains) == 1
        assert len(loc_chains) == 1
        # The two categories must live in separate chains
        for c in chains:
            cats = {m.category for m in c.mentions if m.category}
            assert len(cats) <= 1, f"mixed categories in one chain: {cats}"

    def test_same_text_same_category_still_merges(self, resolver):
        # Repeated same-name same-category mentions are the documented
        # same-name merging behavior and must keep working.
        ents = [_ent("ent_001", "华为", "ORGANIZATION", (0, 2)),
                _ent("ent_002", "华为", "ORGANIZATION", (5, 7))]
        toks = _toks([("华为", "NR", 0, 2), ("发布", "VV", 2, 4),
                      ("华为", "NR", 5, 7)])
        chains = _chains(resolver, ents, toks, "modern")
        chain = _chain_with(chains, "华为")
        assert chain is not None
        assert sum(1 for m in chain.mentions if m.text == "华为") == 2

    def test_nominal_attaches_to_nearest_subgroup(self, resolver):
        # 王(person) … 王(place) … 王(nominal, nearest to place)
        # The trailing 王 is tagged NR and NOT covered by an entity, so it
        # is extracted as a nominal mention.
        ents = [_ent("ent_001", "王", "PERSON", (0, 1)),
                _ent("ent_002", "王", "LOCATION", (6, 7))]
        toks = _toks([("王", "NR", 0, 1), ("去", "VV", 1, 2),
                      ("王", "NR", 6, 7), ("到", "VV", 7, 8),
                      ("王", "NR", 9, 10)])
        chains = _chains(resolver, ents, toks, "modern")
        # The trailing nominal 王 (span 9,10) is nearest to the LOCATION 王
        loc_chain = next(c for c in chains
                         if any(m.category == "LOCATION" for m in c.mentions))
        assert any(m.mention_type == "nominal" and m.span == (9, 10)
                   for m in loc_chain.mentions)


class TestClassicalUposNounPhrases:
    """S6: Chinese NP extraction must accept Universal POS tags.

    The classical pipeline (KYOTO EVAHAN) emits upos values (NOUN/PROPN)
    stored under the pos/ctb key.
    """

    def test_repeated_upos_np_forms_chain(self, resolver):
        toks = _toks([("研究", "NOUN", 0, 2), ("团队", "NOUN", 2, 4),
                      ("和", "CC", 4, 5),
                      ("研究", "NOUN", 5, 7), ("团队", "NOUN", 7, 9)])
        chains = _chains(resolver, [], toks, "classical")
        assert any(sum(1 for m in c.mentions if m.text == "研究团队") == 2
                   for c in chains)

    def test_upos_propn_extracted(self, resolver):
        toks = _toks([("孔子", "PROPN", 0, 2), ("曰", "VERB", 2, 3)])
        chains = _chains(resolver, [], toks, "classical")
        assert any(m.text == "孔子" for c in chains for m in c.mentions)

    def test_ctb_nps_still_work(self, resolver):
        # Regression: modern Chinese CTB tags must keep working.
        toks = _toks([("研究", "NN", 0, 2), ("团队", "NN", 2, 4),
                      ("和", "CC", 4, 5),
                      ("研究", "NN", 5, 7), ("团队", "NN", 7, 9)])
        chains = _chains(resolver, [], toks, "modern")
        assert any(sum(1 for m in c.mentions if m.text == "研究团队") == 2
                   for c in chains)


class TestClassicalPronounDict:
    """H4: no duplicate keys in the classical pronoun dictionary."""

    def test_pi_is_subject_case(self):
        # The duplicate "彼" entry used to overwrite case="subj"
        # with case="demonstrative".
        assert _CN_CLASSICAL_PRONUN["彼"]["case"] == "subj"

    def test_no_duplicate_keys_in_source(self):
        import re
        src = (Path(_project_root) / "core" / "coref_resolver.py").read_text(
            encoding="utf-8")
        # Extract the _CN_CLASSICAL_PRONUN literal's top-level keys
        start = src.index("_CN_CLASSICAL_PRONUN = {")
        end = src.index("\n}", start)  # closing brace of the outer dict
        block = src[start:end]
        keys = re.findall(r'^\s*"(\S+)":\s*\{', block, flags=re.M)
        assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"


class TestMentionCategoryField:
    """Entity mentions carry their category for downstream disambiguation."""

    def test_entity_mention_has_category(self, resolver):
        ents = [_ent("ent_001", "张三", "PERSON", (0, 2))]
        toks = _toks([("张三", "NR", 0, 2)])
        chains = _chains(resolver, ents, toks, "modern")
        ent_mentions = [m for c in chains for m in c.mentions
                        if m.mention_type == "entity"]
        assert ent_mentions and all(m.category == "PERSON"
                                    for m in ent_mentions)
