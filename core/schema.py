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
# Language detection result per sentence
# ---------------------------------------------------------------------------

class SentenceLanguage(BaseModel):
    """Language detection result for a single sentence."""
    text: str = Field(..., description="Sentence text")
    span: tuple[int, int] = Field(..., description="Character offset [start, end)")
    label: str = Field(..., description="Detected language: modern|classical|english")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0,1]")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EntityCategory:
    """NSP standard entity categories.
    
    Standard categories:
    - PERSON: 人名
    - ORGANIZATION: 组织、机构
    - LOCATION: 地名
    - FACILITY: 设施
    - PRODUCT: 产品、作品
    - DATE: 日期
    - NUMBER: 数字
    - MATERIAL: 材料
    - STANDARD: 标准
    - PARAMETER: 参数
    
    Classical Chinese categories (古汉语扩展):
    - TITLE: 官职、爵位
    - ERA: 朝代、时代
    - INSTITUTION: 典章制度
    - ASTRONOMY: 天文历法
    """
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
    # Classical Chinese extensions
    TITLE = "TITLE"          # 官职、爵位
    ERA = "ERA"             # 朝代、时代
    INSTITUTION = "INSTITUTION"  # 典章制度
    ASTRONOMY = "ASTRONOMY"  # 天文历法

    ALL = frozenset({
        PERSON, ORGANIZATION, LOCATION, FACILITY, PRODUCT,
        DATE, NUMBER, MATERIAL, STANDARD, PARAMETER, UNKNOWN,
        TITLE, ERA, INSTITUTION, ASTRONOMY,
    })


class RelationPredicate:
    """NSP standard relation types."""
    # Core
    IS_A = "IS_A"
    PART_OF = "PART_OF"
    HAS_PROPERTY = "HAS_PROPERTY"
    PROPERTY_OF = "PROPERTY_OF"
    # Spatial / temporal
    LOCATED_AT = "LOCATED_AT"
    TEMPORAL_AT = "TEMPORAL_AT"
    # Causal / influence
    CAUSES = "CAUSES"
    AFFECTS = "AFFECTS"
    # Production / composition
    PRODUCES = "PRODUCES"
    COMPOSED_OF = "COMPOSED_OF"
    # Transfer / exchange
    TRANSFERS_TO = "TRANSFERS_TO"
    DEPENDS_ON = "DEPENDS_ON"
    # Movement
    MOVED_TO = "MOVED_TO"
    DEPARTED_FROM = "DEPARTED_FROM"
    # Interaction
    INTERACTS_WITH = "INTERACTS_WITH"
    # Control
    CONTROLS = "CONTROLS"
    # Benefit
    BENEFITS = "BENEFITS"
    # Reference
    EQUIVALENT_TO = "EQUIVALENT_TO"
    REFERENCE_OF = "REFERENCE_OF"
    CONSTRAINT_OF = "CONSTRAINT_OF"
    # Classical Chinese extensions
    HAS_STYLE_NAME = "HAS_STYLE_NAME"   # 表字/号: 陈胜 字 涉
    HAS_TITLE = "HAS_TITLE"             # 官职任命: 拜为 郎中
    SAYS = "SAYS"                       # 对话引述: X曰...
    DIED_AT = "DIED_AT"                 # 死亡: X卒于Y

    # Generic fallback — raw predicate preserved in predicate_verb
    RELATES_TO = "RELATES_TO"

    ALL = frozenset({
        IS_A, PART_OF, HAS_PROPERTY, PROPERTY_OF,
        LOCATED_AT, TEMPORAL_AT, CAUSES, AFFECTS,
        PRODUCES, COMPOSED_OF, TRANSFERS_TO, DEPENDS_ON,
        MOVED_TO, DEPARTED_FROM, INTERACTS_WITH, CONTROLS,
        BENEFITS, EQUIVALENT_TO, REFERENCE_OF, CONSTRAINT_OF,
        RELATES_TO,
        # Classical Chinese
        HAS_STYLE_NAME, HAS_TITLE, SAYS, DIED_AT,
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
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Token confidence from softmax")

    @field_validator("span")
    @classmethod
    def span_valid(cls, v: tuple[int, int]) -> tuple[int, int]:
        if len(v) != 2 or v[0] < 0 or v[1] < v[0]:
            raise ValueError(f"span must be [start, end) with 0 <= start <= end, got {v}")
        return v


# ── Entity Attribute ──

class EntityAttribute(BaseModel):
    """A key-value property assigned to an entity (e.g. 强度=高)."""
    key: str = Field(..., min_length=1, description="Attribute name, e.g. '强度'")
    value: str = Field(default="", description="Attribute value, e.g. '高'")
    predicate_verb: str = Field(default="", description="Original SRL predicate, e.g. '具有'")
    confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    source_relation_id: Optional[str] = Field(
        default=None,
        description="Source relation ID for traceability back to the original relation"
    )


# ── Relation Modifier ──

class RelationModifier(BaseModel):
    """A modifier/qualifier on a relation (e.g. degree, negation, scope).

    Extracted from the evidence text using built-in or custom dictionaries.
    The matched_dict field indicates the source for transparency.
    """
    text: str = Field(..., min_length=1, description="Modifier surface text, e.g. '非常', 'not'")
    type: str = Field(
        ...,
        description="Modifier type: degree|scope|negation|quantity|condition|temporal|comparison|emphasis"
    )
    span: tuple[int, int] = Field(..., description="Character offset [start, end) in original text")
    matched_dict: str = Field(
        default="builtin",
        description="Dictionary source: builtin (built-in) or custom (user-provided)"
    )

    @field_validator("span")
    @classmethod
    def span_valid(cls, v: tuple[int, int]) -> tuple[int, int]:
        if len(v) != 2 or v[0] < 0 or v[1] < v[0]:
            raise ValueError(f"span must be [start, end) with 0 <= start <= end, got {v}")
        return v

    # Note: type validator is intentionally permissive to allow custom modifier types
    # Users can define their own types (e.g., "certainty", "frequency") via custom dictionaries
    # The built-in types are: degree, scope, negation, quantity, condition, temporal, comparison, emphasis


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

class EntityEvidence(BaseModel):
    """Evidence sources for entity recognition.
    
    Tracks which data sources contributed to recognizing this entity,
    enabling transparency and downstream optimization.
    """
    tokenizer: bool = Field(
        default=False,
        description="Whether the entity was in the tokenizer's dict_combine (forced tokenization)"
    )
    seed_dict: bool = Field(
        default=False,
        description="Whether the entity was found in YAML seed dictionaries"
    )
    seed_dict_source: str = Field(
        default="",
        description="Which seed dictionary file matched: persons.yaml, locations.yaml, etc."
    )
    cbdb: bool = Field(
        default=False,
        description="Whether the entity was found in CBDB database"
    )
    cbdb_category: str = Field(
        default="",
        description="CBDB category matched: PERSON, LOCATION, TITLE, ERA"
    )


class Entity(BaseModel):
    """A named entity recognized in the text."""
    id: str = Field(..., pattern=r"^ent_\d+$", description="Unique entity ID, e.g. ent_001")
    text: str = Field(..., min_length=1, description="Entity surface text")
    category: str = Field(..., description="NSP entity category")
    span: tuple[int, int] = Field(..., description="Character offset [start, end)")
    normalized: str = Field(default="", description="Normalized/canonical form")
    source: str = Field(default="", description="Source NER model, e.g. ner/ontonotes")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score [0, 1]")
    attributes: list[EntityAttribute] = Field(default_factory=list, description="Key-value properties of this entity")
    parent_entity_id: Optional[str] = Field(
        default=None,
        description="Parent entity ID for containment hierarchy (entity ⊃ entity)"
    )
    evidence: EntityEvidence = Field(
        default_factory=EntityEvidence,
        description="Evidence sources for entity recognition (tokenizer, seed_dict, cbdb)"
    )

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
    """A semantic relation triple extracted from the text.

    subject/object: the resolved/canonical entity text after merging.
    subject_raw/object_raw: the original surface mention from the text (for traceability).
    When the mention is already a canonical entity, raw == canonical.
    """
    id: str = Field(..., pattern=r"^rel_\d+$", description="Unique relation ID, e.g. rel_001")
    subject: str = Field(..., min_length=1, description="Subject text (canonical form after entity resolution)")
    subject_raw: str = Field(default="", description="Original subject surface mention (for traceability)")
    subject_ent_id: Optional[str] = Field(default=None, description="Linked entity ID if available")
    predicate: str = Field(..., description="Relation type from predefined set")
    predicate_verb: str | None = Field(default=None, description="Original SRL predicate verb (e.g. '生产', '抛光')")
    object: str = Field(..., min_length=1, description="Object text (canonical form after entity resolution)")
    object_raw: str = Field(default="", description="Original object surface mention (for traceability)")
    object_ent_id: Optional[str] = Field(default=None, description="Linked entity ID if available")
    evidence: str = Field(..., min_length=1, description="Original text fragment supporting this relation")
    evidence_span: tuple[int, int] = Field(..., description="Character offset of evidence in original text")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score [0, 1]")
    source: str = Field(default="", description="Extraction source, e.g. dep/nsubj")
    modifiers: list["RelationModifier"] = Field(
        default_factory=list,
        description="Relation modifiers/qualifiers (degree, negation, scope, etc.)"
    )

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
# Coreference Resolution
# ---------------------------------------------------------------------------

class Mention(BaseModel):
    """A single mention (reference) in a coreference chain."""
    text: str = Field(..., min_length=1, description="Mention surface text")
    span: tuple[int, int] = Field(..., description="Character offset [start, end)")
    mention_type: str = Field(
        default="entity",
        description="pronoun|nominal|entity — type of mention"
    )
    entity_id: Optional[str] = Field(
        default=None,
        description="Linked entity ID if this mention is a named entity"
    )
    is_principal: bool = Field(
        default=False,
        description="Whether this is the principal/representative mention in the chain"
    )

    @field_validator("span")
    @classmethod
    def span_valid(cls, v: tuple[int, int]) -> tuple[int, int]:
        if len(v) != 2 or v[0] < 0 or v[1] < v[0]:
            raise ValueError(f"span must be [start, end) with 0 <= start <= end, got {v}")
        return v


class CoreferenceChain(BaseModel):
    """A coreference chain: a set of mentions that refer to the same entity.

    quality_flag indicates the confidence level of the resolution:
    - high: modern Chinese/English, high confidence
    - medium: medium confidence
    - low: low confidence
    - degraded: classical Chinese or other degraded mode (placeholder for future optimization)
    """
    chain_id: str = Field(..., pattern=r"^coref_\d+$", description="Unique chain ID")
    mentions: list[Mention] = Field(
        ...,
        min_length=1,
        description="All mentions in this coreference chain, ordered by span"
    )
    representative: str = Field(
        ...,
        min_length=1,
        description="Representative mention text (usually the principal entity)"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall chain confidence score [0, 1]"
    )
    language: str = Field(
        default="modern",
        description="Language of the chain: modern|classical|english"
    )
    quality_flag: str = Field(
        default="medium",
        description="high|medium|low|degraded — quality indicator for downstream processing"
    )
    model_version: str = Field(
        default="rule_based_v1",
        description="Model/algorithm version for traceability"
    )


# ── Sentence Pattern ──

class SentencePattern(BaseModel):
    """Structural pattern of a sentence for statistical aggregation.

    Combines syntactic features (sentence type, polarity, voice, ...)
    with NLP-derived semantic structure (entity sequence, predicates).
    """

    # ── Basic Info ──
    sentence: str = Field(default="", description="Original sentence text")

    # ── Syntactic Structure ──
    sentence_type: str = Field(
        default="declarative",
        description="declarative|interrogative|imperative|exclamatory"
    )
    structural_type: str = Field(
        default="subject_predicate",
        description="subject_predicate|non_subject_predicate|unknown (no SRL frames available)"
    )
    polarity: str = Field(
        default="affirmative",
        description="affirmative|negative"
    )
    voice: str = Field(
        default="active",
        description="active|passive"
    )
    sub_types: list[str] = Field(
        default_factory=list,
        description="ba_construction|bei_construction|serial_verb|pivotal|ellipsis"
    )
    rhetorical_form: str = Field(
        default="unknown",
        description="unknown|none|parallel|loose — NLP cannot reliably determine; hints in limitations"
    )
    sentence_length_tier: str = Field(
        default="medium",
        description="short(<=10)|medium(11-30)|long(>30) — by character count"
    )

    # ── NLP-derived Semantic Structure ──
    template: str = Field(default="", description="Entity-category + predicate template, e.g. 'MATERIAL 是 MATERIAL'")
    entity_sequence: list[str] = Field(default_factory=list, description="Ordered entity categories from SRL ARGs")
    predicates: list[str] = Field(default_factory=list, description="SRL predicate verbs in this sentence")
    relation_summary: list[str] = Field(default_factory=list, description="Relation summaries, e.g. '碳钢→是→钢'")
    attribute_count: int = Field(default=0, description="Number of entity attributes assigned")

    # ── Statistical Metadata ──
    word_count: int = Field(default=0, ge=0, description="Token count")
    clause_count: int = Field(default=1, ge=1, description="Clause count (comma/semicolon segments + 1)")
    punctuation_mark: str = Field(default="", description="Sentence-ending punctuation mark")

    # ── Limitations (留给下游) ──
    limitations: list[str] = Field(
        default_factory=list,
        description="Features NLP could not determine: serial_verb, pivotal, ellipsis, imperative-accuracy, etc."
    )


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
    language_mode: str = Field(default="auto", description="Language mode used: auto|modern|classical|english")


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


class DependencyEdge(BaseModel):
    """A single dependency relation edge between two tokens."""
    child: int = Field(..., ge=0, description="Child token index (0-based)")
    head: int = Field(..., description="Head token index (0-based), -1 for ROOT")
    rel: str = Field(..., description="Dependency relation label, e.g. nsubj, dobj")


# ---------------------------------------------------------------------------
# Event Extraction
# ---------------------------------------------------------------------------

class EventArgument(BaseModel):
    """An argument (participant) in an event.

    Maps SRL roles to standard event argument roles:
    - ARG0 → Agent (施事/主体)
    - ARG1 → Patient (受事/客体)
    - ARGM-TMP → Time (时间)
    - ARGM-LOC → Location (地点)
    - ARGM-MNR → Manner (方式)
    - ARGM-CAU → Cause (原因)
    - ARGM-PRD → Product (产物/结果)
    - ARGM-BEN → Beneficiary (受益者)
    - ARGM-REC → Recipient (接收者)
    - ARGM-SRC → Source (来源)
    - ARGM-ADJ → Adjunct (附加语)
    - ARGM-DIR → Direction (方向)
    - ARGM-PRP → Purpose (目的)
    """
    role: str = Field(..., min_length=1, description="Argument role, e.g. Agent, Patient, Time, Location")
    text: str = Field(..., min_length=1, description="Argument surface text")
    entity_id: Optional[str] = Field(default=None, description="Linked entity ID if available")
    span: tuple[int, int] = Field(..., description="Character offset [start, end) in original text")

    @field_validator("span")
    @classmethod
    def span_valid(cls, v: tuple[int, int]) -> tuple[int, int]:
        if len(v) != 2 or v[0] < 0 or v[1] < v[0]:
            raise ValueError(f"span must be [start, end) with 0 <= start <= end, got {v}")
        return v


class Event(BaseModel):
    """An event instance extracted from the text.

    V3: Entity → Relation → Event pipeline:
    1. Entity: NER identifies "who/what"
    2. Relation: dependency syntax extracts subject-predicate-object triples
    3. Event: relations with the same predicate_verb in the same sentence
       are clustered into a single event instance

    Traceability chain: Event.source_relation_ids → Relation.id
                       Relation.subject_ent_id/object_ent_id → Entity.id
    """
    id: str = Field(..., pattern=r"^evt_\d+$", description="Unique event ID, e.g. evt_001")
    event_type: str = Field(..., min_length=1, description="Event type (trigger verb itself, no fixed taxonomy)")
    trigger: str = Field(..., min_length=1, description="Event trigger word")
    trigger_span: tuple[int, int] = Field(..., description="Character offset of trigger [start, end)")
    arguments: list[EventArgument] = Field(default_factory=list, description="Event arguments (participants)")
    sentence_index: int = Field(default=-1, ge=-1, description="Index of the sentence this event belongs to (-1 if unknown)")
    is_main_event: bool = Field(default=True, description="Whether this is the main event of the sentence (vs. sub-event)")
    sub_events: list[str] = Field(default_factory=list, description="IDs of sub-events under this main event")
    source_relation_ids: list[str] = Field(default_factory=list, description="Source relation IDs that form this event (traceability: Event ← Relation)")
    confidence: float = Field(default=0.75, ge=0.0, le=1.0, description="Event confidence score [0, 1]")
    source: str = Field(default="relation_cluster", description="Extraction source, e.g. relation_cluster, dep, srl")

    @field_validator("trigger_span")
    @classmethod
    def span_valid(cls, v: tuple[int, int]) -> tuple[int, int]:
        if len(v) != 2 or v[0] < 0 or v[1] < v[0]:
            raise ValueError(f"trigger_span must be [start, end) with 0 <= start <= end, got {v}")
        return v


class NarrativeContent(BaseModel):
    """Content payload of a NarrativeDocument."""
    tokens: list[Token] = Field(default_factory=list, description="Normalized token list")
    entities: list[Entity] = Field(default_factory=list, description="Unified entity list")
    relations: list[Relation] = Field(default_factory=list, description="Extracted relation triples")
    events: list[Event] = Field(default_factory=list, description="Extracted event instances from SRL frames")
    deps: list[DependencyEdge] = Field(default_factory=list, description="Dependency syntax edges (UD)")
    patterns: list[SentencePattern] = Field(default_factory=list, description="Sentence-level structural patterns")
    coreferences: list[CoreferenceChain] = Field(
        default_factory=list,
        description="Coreference chains linking mentions to entities",
    )
    sentences: list[SentenceLanguage] = Field(
        default_factory=list,
        description="Per-sentence text, span, and detected language label",
    )
    structural: dict = Field(
        default_factory=dict,
        description="Raw NLP engine output (for debugging only)",
    )
