"""
NSP (Narrative Schema Protocol) Pydantic Data Models.

Defines the canonical Python representation of Narrative Atoms:
Token, Entity, Relation, and the top-level NarrativeDocument.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EntityCategory:
    """NSP standard entity categories."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    FACILITY = "FACILITY"
    PRODUCT = "PRODUCT"
    DATE = "DATE"
    NUMBER = "NUMBER"
    MATERIAL = "MATERIAL"
    STANDARD = "STANDARD"
    PARAMETER = "PARAMETER"
    UNKNOWN = "UNKNOWN"

    ALL = frozenset({
        PERSON, ORGANIZATION, LOCATION, FACILITY, PRODUCT,
        DATE, NUMBER, MATERIAL, STANDARD, PARAMETER, UNKNOWN,
    })


class RelationPredicate:
    """NSP standard relation types."""
    IS_A = "IS_A"
    PART_OF = "PART_OF"
    PROPERTY_OF = "PROPERTY_OF"
    HAS_PROPERTY = "HAS_PROPERTY"
    LOCATED_AT = "LOCATED_AT"
    TEMPORAL_AT = "TEMPORAL_AT"
    CAUSES = "CAUSES"
    DEPENDS_ON = "DEPENDS_ON"
    EQUIVALENT_TO = "EQUIVALENT_TO"
    REFERENCE_OF = "REFERENCE_OF"
    CONSTRAINT_OF = "CONSTRAINT_OF"

    ALL = frozenset({
        IS_A, PART_OF, PROPERTY_OF, HAS_PROPERTY,
        LOCATED_AT, TEMPORAL_AT, CAUSES, DEPENDS_ON,
        EQUIVALENT_TO, REFERENCE_OF, CONSTRAINT_OF,
    })


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

class Token(BaseModel):
    """A single token (word) in the analyzed text."""
    id: int = Field(..., ge=0, description="Sequential token index, starting from 0")
    text: str = Field(..., min_length=1, description="Token text")
    pos: str = Field(..., min_length=1, description="Part-of-speech (Universal POS standard)")
    span: tuple[int, int] = Field(..., description="Character offset [start, end) in original text")
    source: str = Field(default="", description="NLP model source (e.g., hanlp_v2, hanlp_lzh)")

    @field_validator("span")
    @classmethod
    def span_valid(cls, v: tuple[int, int]) -> tuple[int, int]:
        if len(v) != 2 or v[0] < 0 or v[1] < v[0]:
            raise ValueError(f"span must be [start, end) with 0 <= start <= end, got {v}")
        return v


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """A named entity recognized in the text."""
    id: str = Field(..., pattern=r"^ent_\d+$", description="Unique entity ID, e.g. ent_001")
    text: str = Field(..., min_length=1, description="Entity surface text")
    category: str = Field(..., description="NSP entity category")
    span: tuple[int, int] = Field(..., description="Character offset [start, end)")
    normalized: str = Field(default="", description="Normalized/canonical form")
    source: str = Field(default="", description="Source NER model, e.g. ner/ontonotes")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score [0, 1]")

    @field_validator("span")
    @classmethod
    def span_valid(cls, v: tuple[int, int]) -> tuple[int, int]:
        if len(v) != 2 or v[0] < 0 or v[1] < v[0]:
            raise ValueError(f"span must be [start, end) with 0 <= start <= end, got {v}")
        return v

    @field_validator("category")
    @classmethod
    def category_valid(cls, v: str) -> str:
        if v not in EntityCategory.ALL:
            raise ValueError(f"Unknown entity category: {v}. Must be one of {sorted(EntityCategory.ALL)}")
        return v


# ---------------------------------------------------------------------------
# Relation
# ---------------------------------------------------------------------------

class Relation(BaseModel):
    """A semantic relation triple extracted from the text."""
    id: str = Field(..., pattern=r"^rel_\d+$", description="Unique relation ID, e.g. rel_001")
    subject: str = Field(..., min_length=1, description="Subject text")
    subject_ent_id: Optional[str] = Field(default=None, description="Linked entity ID if available")
    predicate: str = Field(..., description="Relation type from predefined set")
    object: str = Field(..., min_length=1, description="Object text")
    object_ent_id: Optional[str] = Field(default=None, description="Linked entity ID if available")
    evidence: str = Field(..., min_length=1, description="Original text fragment supporting this relation")
    evidence_span: tuple[int, int] = Field(..., description="Character offset of evidence in original text")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score [0, 1]")
    source: str = Field(default="", description="Extraction source, e.g. dep/nsubj")

    @field_validator("evidence_span")
    @classmethod
    def span_valid(cls, v: tuple[int, int]) -> tuple[int, int]:
        if len(v) != 2 or v[0] < 0 or v[1] < v[0]:
            raise ValueError(f"evidence_span must be [start, end) with 0 <= start <= end, got {v}")
        return v

    @field_validator("predicate")
    @classmethod
    def predicate_valid(cls, v: str) -> str:
        if v not in RelationPredicate.ALL:
            raise ValueError(
                f"Unknown relation predicate: {v}. Must be one of {sorted(RelationPredicate.ALL)}"
            )
        return v


# ---------------------------------------------------------------------------
# NarrativeMeta
# ---------------------------------------------------------------------------

class NarrativeMeta(BaseModel):
    """Metadata for a narrative analysis result."""
    source: str = Field(default="hanlp_v2", description="Original NLP engine identifier")
    version: str = Field(default="1.0", description="NSP schema version")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 analysis timestamp",
    )
    text_length: int = Field(..., ge=0, description="Original text character count")


# ---------------------------------------------------------------------------
# NarrativeDocument (top-level wrapper)
# ---------------------------------------------------------------------------

class NarrativeDocument(BaseModel):
    """
    Top-level NSP output wrapper.

    This is the canonical return type for all operator analyze() calls.
    """
    meta: NarrativeMeta = Field(..., description="Analysis metadata")
    content: NarrativeContent = Field(..., description="Analysis content")


class NarrativeContent(BaseModel):
    """Content payload of a NarrativeDocument."""
    tokens: list[Token] = Field(default_factory=list, description="Normalized token list")
    entities: list[Entity] = Field(default_factory=list, description="Unified entity list")
    relations: list[Relation] = Field(default_factory=list, description="Extracted relation triples")
    structural: dict = Field(
        default_factory=dict,
        description="Raw NLP engine output (for debugging only)",
    )
