from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AnalyzeRequest(_message.Message):
    __slots__ = ("text", "source", "policy_json", "noun_signals_json")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    POLICY_JSON_FIELD_NUMBER: _ClassVar[int]
    NOUN_SIGNALS_JSON_FIELD_NUMBER: _ClassVar[int]
    text: str
    source: str
    policy_json: str
    noun_signals_json: str
    def __init__(self, text: _Optional[str] = ..., source: _Optional[str] = ..., policy_json: _Optional[str] = ..., noun_signals_json: _Optional[str] = ...) -> None: ...

class AnalyzeResponse(_message.Message):
    __slots__ = ("meta", "content")
    META_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    meta: NarrativeMeta
    content: NarrativeContent
    def __init__(self, meta: _Optional[_Union[NarrativeMeta, _Mapping]] = ..., content: _Optional[_Union[NarrativeContent, _Mapping]] = ...) -> None: ...

class NarrativeMeta(_message.Message):
    __slots__ = ("source", "version", "timestamp", "text_length")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    TEXT_LENGTH_FIELD_NUMBER: _ClassVar[int]
    source: str
    version: str
    timestamp: str
    text_length: int
    def __init__(self, source: _Optional[str] = ..., version: _Optional[str] = ..., timestamp: _Optional[str] = ..., text_length: _Optional[int] = ...) -> None: ...

class NarrativeContent(_message.Message):
    __slots__ = ("tokens", "entities", "relations", "events", "structural_json", "title", "noun_signals")
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    RELATIONS_FIELD_NUMBER: _ClassVar[int]
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    STRUCTURAL_JSON_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    NOUN_SIGNALS_FIELD_NUMBER: _ClassVar[int]
    tokens: _containers.RepeatedCompositeFieldContainer[Token]
    entities: _containers.RepeatedCompositeFieldContainer[Entity]
    relations: _containers.RepeatedCompositeFieldContainer[Relation]
    events: _containers.RepeatedCompositeFieldContainer[Event]
    structural_json: str
    title: Title
    noun_signals: _containers.RepeatedCompositeFieldContainer[NounSignal]
    def __init__(self, tokens: _Optional[_Iterable[_Union[Token, _Mapping]]] = ..., entities: _Optional[_Iterable[_Union[Entity, _Mapping]]] = ..., relations: _Optional[_Iterable[_Union[Relation, _Mapping]]] = ..., events: _Optional[_Iterable[_Union[Event, _Mapping]]] = ..., structural_json: _Optional[str] = ..., title: _Optional[_Union[Title, _Mapping]] = ..., noun_signals: _Optional[_Iterable[_Union[NounSignal, _Mapping]]] = ...) -> None: ...

class Token(_message.Message):
    __slots__ = ("id", "text", "pos", "span")
    ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    POS_FIELD_NUMBER: _ClassVar[int]
    SPAN_FIELD_NUMBER: _ClassVar[int]
    id: int
    text: str
    pos: str
    span: Span
    def __init__(self, id: _Optional[int] = ..., text: _Optional[str] = ..., pos: _Optional[str] = ..., span: _Optional[_Union[Span, _Mapping]] = ...) -> None: ...

class Span(_message.Message):
    __slots__ = ("start", "end")
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    start: int
    end: int
    def __init__(self, start: _Optional[int] = ..., end: _Optional[int] = ...) -> None: ...

class NounSignal(_message.Message):
    __slots__ = ("text", "pos", "syntactic_role", "score", "span", "evidence_json")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    POS_FIELD_NUMBER: _ClassVar[int]
    SYNTACTIC_ROLE_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    SPAN_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_JSON_FIELD_NUMBER: _ClassVar[int]
    text: str
    pos: str
    syntactic_role: str
    score: float
    span: Span
    evidence_json: str
    def __init__(self, text: _Optional[str] = ..., pos: _Optional[str] = ..., syntactic_role: _Optional[str] = ..., score: _Optional[float] = ..., span: _Optional[_Union[Span, _Mapping]] = ..., evidence_json: _Optional[str] = ...) -> None: ...

class Entity(_message.Message):
    __slots__ = ("id", "text", "category", "span", "normalized", "source", "confidence", "keep", "filter", "filter_reason")
    ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    SPAN_FIELD_NUMBER: _ClassVar[int]
    NORMALIZED_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    KEEP_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    FILTER_REASON_FIELD_NUMBER: _ClassVar[int]
    id: str
    text: str
    category: str
    span: Span
    normalized: str
    source: str
    confidence: float
    keep: bool
    filter: str
    filter_reason: str
    def __init__(self, id: _Optional[str] = ..., text: _Optional[str] = ..., category: _Optional[str] = ..., span: _Optional[_Union[Span, _Mapping]] = ..., normalized: _Optional[str] = ..., source: _Optional[str] = ..., confidence: _Optional[float] = ..., keep: _Optional[bool] = ..., filter: _Optional[str] = ..., filter_reason: _Optional[str] = ...) -> None: ...

class Relation(_message.Message):
    __slots__ = ("id", "subject", "subject_ent_id", "predicate", "object", "object_ent_id", "evidence", "evidence_span", "confidence", "source")
    ID_FIELD_NUMBER: _ClassVar[int]
    SUBJECT_FIELD_NUMBER: _ClassVar[int]
    SUBJECT_ENT_ID_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    OBJECT_FIELD_NUMBER: _ClassVar[int]
    OBJECT_ENT_ID_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_FIELD_NUMBER: _ClassVar[int]
    EVIDENCE_SPAN_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    id: str
    subject: str
    subject_ent_id: str
    predicate: str
    object: str
    object_ent_id: str
    evidence: str
    evidence_span: Span
    confidence: float
    source: str
    def __init__(self, id: _Optional[str] = ..., subject: _Optional[str] = ..., subject_ent_id: _Optional[str] = ..., predicate: _Optional[str] = ..., object: _Optional[str] = ..., object_ent_id: _Optional[str] = ..., evidence: _Optional[str] = ..., evidence_span: _Optional[_Union[Span, _Mapping]] = ..., confidence: _Optional[float] = ..., source: _Optional[str] = ...) -> None: ...

class EventArgument(_message.Message):
    __slots__ = ("role", "text", "entity_id", "span", "syntactic_role", "governing_verb", "token_span", "spatial_role", "verb_spatial_class")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    ENTITY_ID_FIELD_NUMBER: _ClassVar[int]
    SPAN_FIELD_NUMBER: _ClassVar[int]
    SYNTACTIC_ROLE_FIELD_NUMBER: _ClassVar[int]
    GOVERNING_VERB_FIELD_NUMBER: _ClassVar[int]
    TOKEN_SPAN_FIELD_NUMBER: _ClassVar[int]
    SPATIAL_ROLE_FIELD_NUMBER: _ClassVar[int]
    VERB_SPATIAL_CLASS_FIELD_NUMBER: _ClassVar[int]
    role: str
    text: str
    entity_id: str
    span: Span
    syntactic_role: str
    governing_verb: str
    token_span: Span
    spatial_role: str
    verb_spatial_class: str
    def __init__(self, role: _Optional[str] = ..., text: _Optional[str] = ..., entity_id: _Optional[str] = ..., span: _Optional[_Union[Span, _Mapping]] = ..., syntactic_role: _Optional[str] = ..., governing_verb: _Optional[str] = ..., token_span: _Optional[_Union[Span, _Mapping]] = ..., spatial_role: _Optional[str] = ..., verb_spatial_class: _Optional[str] = ...) -> None: ...

class Event(_message.Message):
    __slots__ = ("id", "event_type", "trigger", "trigger_span", "arguments", "sentence_index", "is_main_event", "sub_events", "source_relation_ids", "confidence", "source")
    ID_FIELD_NUMBER: _ClassVar[int]
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    TRIGGER_FIELD_NUMBER: _ClassVar[int]
    TRIGGER_SPAN_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    SENTENCE_INDEX_FIELD_NUMBER: _ClassVar[int]
    IS_MAIN_EVENT_FIELD_NUMBER: _ClassVar[int]
    SUB_EVENTS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_RELATION_IDS_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    id: str
    event_type: str
    trigger: str
    trigger_span: Span
    arguments: _containers.RepeatedCompositeFieldContainer[EventArgument]
    sentence_index: int
    is_main_event: bool
    sub_events: _containers.RepeatedScalarFieldContainer[str]
    source_relation_ids: _containers.RepeatedScalarFieldContainer[str]
    confidence: float
    source: str
    def __init__(self, id: _Optional[str] = ..., event_type: _Optional[str] = ..., trigger: _Optional[str] = ..., trigger_span: _Optional[_Union[Span, _Mapping]] = ..., arguments: _Optional[_Iterable[_Union[EventArgument, _Mapping]]] = ..., sentence_index: _Optional[int] = ..., is_main_event: _Optional[bool] = ..., sub_events: _Optional[_Iterable[str]] = ..., source_relation_ids: _Optional[_Iterable[str]] = ..., confidence: _Optional[float] = ..., source: _Optional[str] = ...) -> None: ...
