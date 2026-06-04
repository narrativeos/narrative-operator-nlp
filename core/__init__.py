"""
Narrative Operator NLP — Core Module

Protocol-Free NLP analysis core. Contains:
- schema.py: NSP (Narrative Schema Protocol) Pydantic data models
- analyzer.py: Unified analysis entry point (text → NarrativeDocument)
- mapper.py: HanlpSchemaMapper (HanLP raw output → NSP standard format)
- entity_mapper.py: Entity mapping rules (NER unification)
- relation_mapper.py: Relation extraction rules (dependency + SRL → triples)
"""

__all__ = [
    "NarrativeDocument",
    "NarrativeMeta",
    "Token",
    "Entity",
    "Relation",
    "analyze",
    "HanlpSchemaMapper",
]
