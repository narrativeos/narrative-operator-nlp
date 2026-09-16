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
    # Reconstruct surface text by placing each token at its span offset
    # (gaps such as English spaces stay blank) so sentence-boundary checks
    # see real punctuation.
    if toks:
        buf = [""] * toks[-1].span[1]
        for t in toks:
            buf[t.span[0]:t.span[1]] = list(t.text)
        text = "".join(buf)
    else:
        text = ""
    return resolver.resolve(text, ents, toks, language).chains


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


class TestFirstConjunctPreference:
    """H1: coordinated pairs (A 和 B) prefer the left conjunct."""

    def test_coordinated_subject_prefers_first(self, resolver):
        # 张三 和 李四 去 北京 。 他 …  →  他 = 张三
        # (person tier skips the LOCATION object, first conjunct wins)
        ents = [_ent("ent_001", "张三", "PERSON", (0, 2)),
                _ent("ent_002", "李四", "PERSON", (3, 5)),
                _ent("ent_003", "北京", "LOCATION", (6, 8))]
        toks = _toks([("张三", "NR", 0, 2), ("和", "CC", 2, 3),
                      ("李四", "NR", 3, 5), ("去", "VV", 5, 6),
                      ("北京", "NR", 6, 8), ("。", "PU", 8, 9),
                      ("他", "PRP", 9, 10)])
        chains = _chains(resolver, ents, toks, "modern")
        chain = _chain_with(chains, "他")
        assert chain is not None
        assert any(m.text == "张三" for m in chain.mentions)
        assert not any(m.text == "李四" for m in chain.mentions)

    def test_object_reference_kept_without_conjunction(self, resolver):
        # 张三 打 了 李四 。 他 …  →  他 = 李四 (no CC between the anchors)
        ents = [_ent("ent_001", "张三", "PERSON", (0, 2)),
                _ent("ent_002", "李四", "PERSON", (4, 6))]
        toks = _toks([("张三", "NR", 0, 2), ("打", "VV", 2, 3),
                      ("了", "AS", 3, 4), ("李四", "NR", 4, 6),
                      ("。", "PU", 6, 7), ("他", "PRP", 7, 8)])
        chains = _chains(resolver, ents, toks, "modern")
        chain = _chain_with(chains, "他")
        assert chain is not None
        assert any(m.text == "李四" for m in chain.mentions)
        assert not any(m.text == "张三" for m in chain.mentions)

    def test_non_conjunction_verb_does_not_trigger(self, resolver):
        # 不及 is a verb (VV), not a conjunction — must NOT trigger the
        # first-conjunct preference.
        ents = [_ent("ent_001", "张三", "PERSON", (0, 2)),
                _ent("ent_002", "李四", "PERSON", (4, 6))]
        toks = _toks([("张三", "NR", 0, 2), ("不及", "VV", 2, 4),
                      ("李四", "NR", 4, 6), ("。", "PU", 6, 7),
                      ("他", "PRP", 7, 8)])
        chains = _chains(resolver, ents, toks, "modern")
        chain = _chain_with(chains, "他")
        assert chain is not None
        assert any(m.text == "李四" for m in chain.mentions)


class TestSentenceBoundaryDistance:
    """H1: cross-sentence resolution is penalized and distance-capped."""

    def test_cross_sentence_short_distance_resolves(self, resolver):
        ents = [_ent("ent_001", "张三", "PERSON", (0, 2))]
        toks = _toks([("张三", "NR", 0, 2), ("说", "VV", 2, 3),
                      ("了", "AS", 3, 4), ("。", "PU", 4, 5),
                      ("他", "PRP", 5, 6)])
        chains = _chains(resolver, ents, toks, "modern")
        chain = _chain_with(chains, "他")
        assert any(m.text == "张三" for m in chain.mentions)

    def test_cross_sentence_long_distance_unresolved(self, resolver):
        # Anchor > 60 chars away across a sentence boundary → no resolution.
        # Filler alternates NN/VV so no long nominal anchor is extracted.
        filler = [(("字", "NN", 6 + i * 4, 8 + i * 4),
                   ("走", "VV", 8 + i * 4, 10 + i * 4)) for i in range(15)]
        pron_start = 66  # distance from 张三 (ends 2) = 64 > 60
        ents = [_ent("ent_001", "张三", "PERSON", (0, 2))]
        toks = _toks([("张三", "NR", 0, 2), ("说", "VV", 2, 3),
                      ("了", "AS", 3, 4), ("。", "PU", 4, 5),
                      ("。", "PU", 5, 6)] + [t for pair in filler for t in pair]
                 + [("他", "PRP", pron_start, pron_start + 1)])
        chains = _chains(resolver, ents, toks, "modern")
        chain = _chain_with(chains, "他")
        assert chain is not None
        assert all(m.mention_type == "pronoun" for m in chain.mentions)

    def test_same_sentence_long_distance_allowed(self, resolver):
        # No sentence-final punctuation → same sentence → no distance cap.
        # Filler alternates NN/VV so no long nominal anchor is extracted.
        filler = [(("字", "NN", 3 + i * 4, 5 + i * 4),
                   ("走", "VV", 5 + i * 4, 7 + i * 4)) for i in range(15)]
        pron_start = 63
        ents = [_ent("ent_001", "张三", "PERSON", (0, 2))]
        toks = _toks([("张三", "NR", 0, 2), ("的", "DEG", 2, 3)]
                 + [t for pair in filler for t in pair]
                 + [("他", "PRP", pron_start, pron_start + 1)])
        chains = _chains(resolver, ents, toks, "modern")
        chain = _chain_with(chains, "他")
        assert any(m.text == "张三" for m in chain.mentions)

    def test_same_sentence_beats_distant_cross_sentence(self, resolver):
        # A far cross-sentence anchor must lose to a near same-sentence one
        ents = [_ent("ent_001", "张三", "PERSON", (0, 2)),
                _ent("ent_002", "李四", "PERSON", (12, 14))]
        toks = _toks([("张三", "NR", 0, 2), ("说", "VV", 2, 3),
                      ("了", "AS", 3, 4), ("。", "PU", 4, 5),
                      ("填", "VV", 5, 12), ("李四", "NR", 12, 14),
                      ("他", "PRP", 14, 15)])
        chains = _chains(resolver, ents, toks, "modern")
        chain = _chain_with(chains, "他")
        assert any(m.text == "李四" for m in chain.mentions)
        assert not any(m.text == "张三" for m in chain.mentions)


class TestPersonTierPreference:
    """S3/H1: 他/她 prefer PERSON anchors over known non-PERSON ones."""

    def test_person_pronoun_skips_location_object(self, resolver):
        # 李四 参观 了 北京 。 他 …  →  他 = 李四, not the nearer 北京
        ents = [_ent("ent_001", "李四", "PERSON", (0, 2)),
                _ent("ent_002", "北京", "LOCATION", (5, 7))]
        toks = _toks([("李四", "NR", 0, 2), ("参观", "VV", 2, 4),
                      ("了", "AS", 4, 5), ("北京", "NR", 5, 7),
                      ("。", "PU", 7, 8), ("他", "PRP", 8, 9)])
        chains = _chains(resolver, ents, toks, "modern")
        chain = _chain_with(chains, "他")
        assert chain is not None
        assert any(m.text == "李四" for m in chain.mentions)
        assert not any(m.text == "北京" for m in chain.mentions)

    def test_english_he_skips_location_object(self, resolver):
        # Alice visited Paris. He …  →  He = Alice (not Paris)
        ents = [_ent("ent_001", "Alice", "PERSON", (0, 5)),
                _ent("ent_002", "Paris", "LOCATION", (13, 18))]
        toks = _toks([("Alice", "PROPN", 0, 5), ("visited", "VERB", 5, 12),
                      ("Paris", "PROPN", 13, 18), (".", "PUNCT", 18, 19),
                      ("He", "PRP", 19, 21)])
        chains = _chains(resolver, ents, toks, "english")
        chain = _chain_with(chains, "He")
        assert chain is not None
        assert any(m.text == "Alice" for m in chain.mentions)
        assert not any(m.text == "Paris" for m in chain.mentions)


class TestGenderConsistency:
    """S3: 他 and 她 must not share one chain."""

    def test_opposite_gender_pronouns_split(self, resolver):
        # 张三 帮 她 买 了 他  →  她 joins 张三; 他 must NOT
        ents = [_ent("ent_001", "张三", "PERSON", (0, 2))]
        toks = _toks([("张三", "NR", 0, 2), ("帮", "VV", 2, 3),
                      ("她", "PRP", 3, 4), ("买", "VV", 4, 5),
                      ("了", "AS", 5, 6), ("他", "PRP", 6, 7)])
        chains = _chains(resolver, ents, toks, "modern")
        she_chain = _chain_with(chains, "她")
        he_chain = _chain_with(chains, "他")
        assert she_chain is not None and he_chain is not None
        assert she_chain is not he_chain, "他 and 她 share a chain"
        assert any(m.text == "张三" for m in she_chain.mentions)
        assert all(m.mention_type == "pronoun" for m in he_chain.mentions)

    def test_same_gender_pronouns_share(self, resolver):
        # 她 … 她 → same chain
        ents = [_ent("ent_001", "张三", "PERSON", (0, 2))]
        toks = _toks([("张三", "NR", 0, 2), ("帮", "VV", 2, 3),
                      ("她", "PRP", 3, 4), ("买", "VV", 4, 5),
                      ("了", "AS", 5, 6), ("她", "PRP", 6, 7)])
        chains = _chains(resolver, ents, toks, "modern")
        chain = _chain_with(chains, "她")
        assert sum(1 for m in chain.mentions if m.text == "她") == 2


class TestClassicalUnpunctuated:
    """Classical text has no punctuation — long-distance must still work."""

    def test_long_distance_without_punctuation(self, resolver):
        # Filler alternates NOUN/VERB so no long nominal anchor appears.
        filler = [(("字", "NOUN", 3 + i * 4, 5 + i * 4),
                   ("走", "VERB", 5 + i * 4, 7 + i * 4)) for i in range(15)]
        pron_start = 63
        ents = [_ent("ent_001", "项羽", "PERSON", (0, 2))]
        toks = _toks([("项羽", "PROPN", 0, 2), ("举", "VERB", 2, 3)]
                 + [t for pair in filler for t in pair]
                 + [("之", "PRON", pron_start, pron_start + 1)])
        chains = _chains(resolver, ents, toks, "classical")
        chain = _chain_with(chains, "之")
        assert chain is not None
        assert any(m.text == "项羽" for m in chain.mentions)
