"""Tests using language-specific datasets.

Each language (Modern Chinese, Classical Chinese, English) has its own
dataset file in tests/datasets/. This test module loads all datasets
and runs entity/relation extraction tests against them.
"""

from __future__ import annotations
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from core.entity_mapper import EntityMappingRules
from core.relation_mapper import RelationExtractionRules
from core.mapper import HanlpSchemaMapper
from core.schema import EntityCategory, Token
from tests.datasets import load_dataset, list_datasets


# ── Helpers ──

def _make_tokens(text: str, raw: dict) -> list[Token]:
    """Build Token list from raw tok/fine and pos/ctb."""
    tok_fine = raw.get("tok/fine", [])
    pos = raw.get("pos/ctb", [])
    tokens = []
    cursor = 0
    for i, tok in enumerate(tok_fine):
        p = pos[i] if i < len(pos) else "X"
        idx = text.find(tok, cursor)
        if idx >= 0:
            start, end = idx, idx + len(tok)
            cursor = end
        else:
            start, end = cursor, cursor + len(tok)
            cursor = end
        tokens.append(Token(id=i, text=tok, pos=p, span=(start, end)))
    return tokens


def _convert_raw(raw: dict) -> dict:
    """Convert list-based NER/dep/srl to tuple format expected by mapper."""
    result = dict(raw)
    for key in ("ner/pku", "ner/msra", "ner/ontonotes", "ner", "ner/conll2003"):
        if key in result:
            result[key] = [tuple(item) for item in result[key]]
    if "dep" in result:
        result["dep"] = [tuple(item) for item in result["dep"]]
    if "srl" in result:
        result["srl"] = [[tuple(item) for item in frame] for frame in result["srl"]]
    return result


# ── Parametrized tests ──

datasets = list_datasets()


@pytest.mark.parametrize("dataset_name", datasets)
class TestEntityExtraction:
    """Entity extraction tests for each language dataset."""

    def test_entity_count(self, dataset_name: str):
        ds = load_dataset(dataset_name)
        for case in ds["cases"]:
            raw = _convert_raw(case["raw"])
            tokens = _make_tokens(case["text"], case["raw"])
            rules = EntityMappingRules()
            entities = rules.map_all(case["text"], raw, tokens)
            expected = case["expected"].get("min_entities", 0)
            assert len(entities) >= expected, (
                f"[{dataset_name}/{case['name']}] "
                f"Expected >= {expected} entities, got {len(entities)}: "
                f"{[(e.text, e.category) for e in entities]}"
            )

    def test_entity_categories(self, dataset_name: str):
        ds = load_dataset(dataset_name)
        for case in ds["cases"]:
            raw = _convert_raw(case["raw"])
            tokens = _make_tokens(case["text"], case["raw"])
            rules = EntityMappingRules()
            entities = rules.map_all(case["text"], raw, tokens)
            cat_map = case["expected"].get("entity_categories", {})
            for text, expected_cat in cat_map.items():
                matched = [e for e in entities if text in e.text]
                assert matched, (
                    f"[{dataset_name}/{case['name']}] "
                    f"Entity '{text}' not found in {[e.text for e in entities]}"
                )
                assert matched[0].category == expected_cat, (
                    f"[{dataset_name}/{case['name']}] "
                    f"Expected '{text}' as {expected_cat}, got {matched[0].category}"
                )

    def test_unique_entity_ids(self, dataset_name: str):
        ds = load_dataset(dataset_name)
        for case in ds["cases"]:
            raw = _convert_raw(case["raw"])
            tokens = _make_tokens(case["text"], case["raw"])
            rules = EntityMappingRules()
            entities = rules.map_all(case["text"], raw, tokens)
            ids = [e.id for e in entities]
            assert len(ids) == len(set(ids)), (
                f"[{dataset_name}/{case['name']}] Duplicate entity IDs: {ids}"
            )


@pytest.mark.parametrize("dataset_name", datasets)
class TestRelationExtraction:
    """Relation extraction tests for each language dataset."""

    def test_relation_count(self, dataset_name: str):
        ds = load_dataset(dataset_name)
        for case in ds["cases"]:
            raw = _convert_raw(case["raw"])
            tokens = _make_tokens(case["text"], case["raw"])
            rules = RelationExtractionRules()
            rels = rules.extract_all(case["text"], raw, tokens)
            expected = case["expected"].get("min_relations", 0)
            assert len(rels) >= expected, (
                f"[{dataset_name}/{case['name']}] "
                f"Expected >= {expected} relations, got {len(rels)}"
            )

    def test_relation_evidence(self, dataset_name: str):
        ds = load_dataset(dataset_name)
        for case in ds["cases"]:
            raw = _convert_raw(case["raw"])
            tokens = _make_tokens(case["text"], case["raw"])
            rules = RelationExtractionRules()
            rels = rules.extract_all(case["text"], raw, tokens)
            for r in rels:
                assert len(r.evidence) > 0, (
                    f"[{dataset_name}/{case['name']}] "
                    f"Relation '{r.subject}→{r.predicate_verb}→{r.object}' has no evidence"
                )

    def test_unique_relation_ids(self, dataset_name: str):
        ds = load_dataset(dataset_name)
        for case in ds["cases"]:
            raw = _convert_raw(case["raw"])
            tokens = _make_tokens(case["text"], case["raw"])
            rules = RelationExtractionRules()
            rels = rules.extract_all(case["text"], raw, tokens)
            ids = [r.id for r in rels]
            assert len(ids) == len(set(ids)), (
                f"[{dataset_name}/{case['name']}] Duplicate relation IDs: {ids}"
            )


@pytest.mark.parametrize("dataset_name", datasets)
class TestMapperIntegration:
    """Full mapper integration tests for each language dataset."""

    def test_map_produces_document(self, dataset_name: str):
        ds = load_dataset(dataset_name)
        for case in ds["cases"]:
            raw = _convert_raw(case["raw"])
            mapper = HanlpSchemaMapper()
            doc = mapper.map(case["text"], raw)
            assert doc.meta.text_length == len(case["text"])
            assert len(doc.content.tokens) > 0
            assert isinstance(doc.content.entities, list)
            assert isinstance(doc.content.relations, list)

    def test_serialization(self, dataset_name: str):
        ds = load_dataset(dataset_name)
        for case in ds["cases"]:
            raw = _convert_raw(case["raw"])
            mapper = HanlpSchemaMapper()
            doc = mapper.map(case["text"], raw)
            data = doc.model_dump()
            assert "meta" in data
            assert "content" in data
            assert len(data["content"]["tokens"]) > 0