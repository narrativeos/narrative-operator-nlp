"""
Narrative Operator NLP — Core Module

Protocol-Free NLP analysis core. Contains:
- schema.py: NSP (Narrative Schema Protocol) Pydantic data models
- analyzer.py: Unified analysis entry point (text → NarrativeDocument)
- mapper.py: HanlpSchemaMapper (HanLP raw output → NSP standard format)
- entity_mapper.py: Entity mapping orchestrator (delegates to components)
- ner_label_mapper.py: NER label → NSP category mapping (config-driven)
- keyword_extractor.py: Domain keyword entity discovery (config-driven)
- entity_merger.py: Entity merging (same-category + cross-category)
- entity_deduplicator.py: Entity deduplication
- entity_id_generator.py: Deterministic entity ID generation
- relation_mapper.py: Relation extraction rules (dependency + SRL → triples)
"""

from core.schema import (
    NarrativeDocument,
    NarrativeMeta,
    Token,
    Entity,
    Relation,
    KeySentence,
    Summary,
)
from core.analyzer import analyze, HanlpSchemaMapper
from core.summarizer import summarize, summarize_text
from core.entity_mapper import EntityMappingRules
from core.ner_label_mapper import NerLabelMapper
from core.keyword_extractor import KeywordExtractor
from core.entity_merger import EntityMerger
from core.entity_deduplicator import EntityDeduplicator
from core.entity_id_generator import EntityIdGenerator

__all__ = [
    # Schema
    "NarrativeDocument",
    "NarrativeMeta",
    "Token",
    "Entity",
    "Relation",
    "KeySentence",
    "Summary",
    # Analysis
    "analyze",
    "HanlpSchemaMapper",
    # Summarization
    "summarize",
    "summarize_text",
    # Entity extraction components
    "EntityMappingRules",
    "NerLabelMapper",
    "KeywordExtractor",
    "EntityMerger",
    "EntityDeduplicator",
    "EntityIdGenerator",
]