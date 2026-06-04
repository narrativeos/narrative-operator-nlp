"""
Tests: NSP Schema Data Models.

Validates the Pydantic data models defined in core/schema.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from pydantic import ValidationError

from core.schema import (
    Entity,
    EntityCategory,
    NarrativeContent,
    NarrativeDocument,
    NarrativeMeta,
    Relation,
    RelationPredicate,
    Token,
)


# ---------------------------------------------------------------------------
# Token Tests
# ---------------------------------------------------------------------------

class TestToken:
    def test_valid_token(self):
        t = Token(id=0, text="碳钢", pos="NN", span=(0, 2))
        assert t.id == 0
        assert t.text == "碳钢"
        assert t.pos == "NN"
        assert t.span == (0, 2)

    def test_invalid_span_reversed(self):
        with pytest.raises(ValidationError):
            Token(id=0, text="碳钢", pos="NN", span=(5, 2))

    def test_invalid_span_negative(self):
        with pytest.raises(ValidationError):
            Token(id=0, text="碳钢", pos="NN", span=(-1, 2))

    def test_empty_text(self):
        with pytest.raises(ValidationError):
            Token(id=0, text="", pos="NN", span=(0, 2))


# ---------------------------------------------------------------------------
# Entity Tests
# ---------------------------------------------------------------------------

class TestEntity:
    def test_valid_entity(self):
        e = Entity(
            id="ent_001",
            text="碳钢",
            category=EntityCategory.MATERIAL,
            span=(0, 2),
        )
        assert e.category == "MATERIAL"
        assert e.normalized == ""
        assert e.confidence == 1.0

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            Entity(
                id="ent_001",
                text="碳钢",
                category="INVALID_CATEGORY",
                span=(0, 2),
            )

    def test_invalid_id_pattern(self):
        with pytest.raises(ValidationError):
            Entity(
                id="bad_id",
                text="碳钢",
                category=EntityCategory.MATERIAL,
                span=(0, 2),
            )

    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            Entity(
                id="ent_001",
                text="碳钢",
                category=EntityCategory.MATERIAL,
                span=(0, 2),
                confidence=1.5,
            )

    def test_all_categories_valid(self):
        """Verify all EntityCategory.ALL values are valid categories."""
        for cat in EntityCategory.ALL:
            e = Entity(id="ent_001", text="test", category=cat, span=(0, 4))
            assert e.category == cat


# ---------------------------------------------------------------------------
# Relation Tests
# ---------------------------------------------------------------------------

class TestRelation:
    def test_valid_relation(self):
        r = Relation(
            id="rel_001",
            subject="碳钢",
            predicate=RelationPredicate.IS_A,
            object="钢",
            evidence="碳钢是钢的一种",
            evidence_span=(10, 17),
            confidence=0.98,
            source="dep/cop",
        )
        assert r.predicate == "IS_A"
        assert r.confidence == 0.98

    def test_invalid_predicate(self):
        with pytest.raises(ValidationError):
            Relation(
                id="rel_001",
                subject="碳钢",
                predicate="INVALID",
                object="钢",
                evidence="...",
                evidence_span=(0, 3),
            )

    def test_all_predicates_valid(self):
        """Verify all RelationPredicate.ALL values are valid predicates."""
        for pred in RelationPredicate.ALL:
            r = Relation(
                id="rel_001",
                subject="X",
                predicate=pred,
                object="Y",
                evidence="X ... Y",
                evidence_span=(0, 5),
            )
            assert r.predicate == pred

    def test_optional_ent_ids(self):
        r = Relation(
            id="rel_001",
            subject="碳钢",
            subject_ent_id="ent_002",
            predicate=RelationPredicate.IS_A,
            object="钢",
            object_ent_id="ent_003",
            evidence="...",
            evidence_span=(0, 3),
        )
        assert r.subject_ent_id == "ent_002"
        assert r.object_ent_id == "ent_003"

    def test_empty_evidence(self):
        with pytest.raises(ValidationError):
            Relation(
                id="rel_001",
                subject="X",
                predicate=RelationPredicate.IS_A,
                object="Y",
                evidence="",
                evidence_span=(0, 0),
            )


# ---------------------------------------------------------------------------
# NarrativeDocument Tests
# ---------------------------------------------------------------------------

class TestNarrativeDocument:
    def test_empty_document(self):
        doc = NarrativeDocument(
            meta=NarrativeMeta(text_length=0),
            content=NarrativeContent(),
        )
        assert doc.meta.source == "hanlp_v2"
        assert doc.meta.version == "1.0"
        assert doc.content.tokens == []
        assert doc.content.entities == []
        assert doc.content.relations == []

    def test_full_document(self):
        doc = NarrativeDocument(
            meta=NarrativeMeta(text_length=10),
            content=NarrativeContent(
                tokens=[Token(id=0, text="碳钢", pos="NN", span=(0, 2))],
                entities=[
                    Entity(
                        id="ent_001",
                        text="碳钢",
                        category=EntityCategory.MATERIAL,
                        span=(0, 2),
                    )
                ],
                relations=[
                    Relation(
                        id="rel_001",
                        subject="碳钢",
                        predicate=RelationPredicate.IS_A,
                        object="钢",
                        evidence="碳钢是钢",
                        evidence_span=(0, 4),
                    )
                ],
                structural={"raw": "debug"},
            ),
        )
        assert len(doc.content.tokens) == 1
        assert len(doc.content.entities) == 1
        assert len(doc.content.relations) == 1

    def test_json_schema_compliance(self):
        """Verify output JSON matches narrative.schema.json."""
        schema_path = (
            Path(__file__).resolve().parent.parent
            / "schemas"
            / "narrative.schema.json"
        )
        with open(schema_path) as f:
            schema = json.load(f)

        doc = NarrativeDocument(
            meta=NarrativeMeta(text_length=5),
            content=NarrativeContent(
                tokens=[Token(id=0, text="测试", pos="NN", span=(0, 2))],
            ),
        )

        import jsonschema

        # Pydantic serializes tuple span as list, which matches JSON Schema
        data = doc.model_dump()
        # Ensure span is a list (Pydantic v2 serialization)
        data["content"]["tokens"][0]["span"] = list(data["content"]["tokens"][0]["span"])
        jsonschema.validate(data, schema)
