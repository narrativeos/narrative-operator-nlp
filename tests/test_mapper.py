"""Tests: Schema Mapper with real HanLP output formats."""

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
from core.schema import EntityCategory, RelationPredicate, Token


# --- Fixtures matching real HanLP MTL output ---

TEXT = "碳钢是钢的一种。北京立方庭位于海淀区。"

@pytest.fixture
def mock_raw():
    return {
        "tok/fine": ["碳","钢","是","钢","的","一","种","。","北","京","立","方","庭","位","于","海","淀","区","。"],
        "pos/ctb": ["NN","NN","VC","NN","DEG","CD","M","PU","NR","NR","NR","NR","NR","VV","VV","NR","NR","NR","PU"],
        "ner/pku": [("北京","ns",8,10),("立方庭","ns",10,13),("海淀区","ns",15,18)],
        "ner/msra": [("北京","LOCATION",8,10),("立方庭","LOCATION",10,13),("海淀区","LOCATION",15,18)],
        "ner/ontonotes": [("北京立方庭","FAC",8,13)],
        "dep": [(2,"nn"),(3,"top"),(2,"attr"),(7,"assmod"),(7,"assm"),(7,"nummod"),(3,"attr"),(3,"punct"),
                (10,"nn"),(14,"nsubj"),(9,"nn"),(10,"nn"),(10,"nn"),(0,"root"),(14,"conj"),(14,"dobj"),
                (16,"nn"),(18,"nn"),(14,"punct")],
        "srl": [[("碳钢","ARG0",0,2),("是","PRED",2,3),("钢的一种","ARG1",3,7)],
                 [("北京立方庭","ARG0",8,13),("位于","PRED",13,15),("海淀区","ARG1",15,18)]],
    }

@pytest.fixture
def tokens():
    return [Token(id=i,text=t,pos="X",span=(i,i+1)) for i,t in enumerate(TEXT)]


class TestEntityMappingRules:
    def test_map_pku_location(self, mock_raw, tokens):
        rules = EntityMappingRules()
        entities = rules.map_all(TEXT, mock_raw, tokens)
        locs = [e for e in entities if e.category == EntityCategory.LOCATION]
        assert len(locs) >= 1

    def test_map_msra(self, mock_raw, tokens):
        rules = EntityMappingRules()
        entities = rules.map_all(TEXT, mock_raw, tokens)
        assert len(entities) >= 1

    def test_ontonotes_facility_or_location(self, mock_raw, tokens):
        """Verify that entities from the 北京立方庭 region are correctly extracted.
        PKU produces '北京'(LOC) + '立方庭'(LOC), OntoNotes produces '北京立方庭'(FAC).
        Due to deduplication, the individual LOCATION entities from PKU may survive."""
        rules = EntityMappingRules()
        entities = rules.map_all(TEXT, mock_raw, tokens)
        # At minimum, we should have LOCATION entities for 北京 and 立方庭
        locs = [e for e in entities if e.category == EntityCategory.LOCATION]
        loc_texts = {e.text for e in locs}
        assert "北京" in loc_texts or "北京立方庭" in loc_texts, \
            f"Expected 北京 or 北京立方庭 as LOCATION. Got: {[(e.text, e.category, e.span) for e in entities]}"

    def test_deduplication(self, mock_raw, tokens):
        """北京 appears in pku, msra and ontonotes → should deduplicate."""
        rules = EntityMappingRules()
        entities = rules.map_all(TEXT, mock_raw, tokens)
        beijing = [e for e in entities if "北京" in e.text]
        assert len(beijing) <= 2  # "北京" and "北京立方庭" are different texts

    def test_unique_ids(self, mock_raw, tokens):
        rules = EntityMappingRules()
        entities = rules.map_all(TEXT, mock_raw, tokens)
        ids = [e.id for e in entities]
        assert len(ids) == len(set(ids))

    def test_keyword_material(self, tokens):
        rules = EntityMappingRules()
        e = rules.map(("碳钢","UNK",0,2), "ner/msra", tokens, "碳钢测试")
        assert e is not None
        assert e.category == EntityCategory.MATERIAL


class TestRelationExtractionRules:
    def test_extract_from_dep(self, mock_raw, tokens):
        rules = RelationExtractionRules()
        rels = rules.extract_all(TEXT, mock_raw, tokens)
        assert len(rels) > 0

    def test_dep_has_evidence(self, mock_raw, tokens):
        rules = RelationExtractionRules()
        rels = rules.extract_all(TEXT, mock_raw, tokens)
        for r in rels:
            assert len(r.evidence) > 0

    def test_unique_ids(self, mock_raw, tokens):
        rules = RelationExtractionRules()
        rels = rules.extract_all(TEXT, mock_raw, tokens)
        ids = [r.id for r in rels]
        assert len(ids) == len(set(ids))


class TestHanlpSchemaMapper:
    def test_map_produces_document(self, mock_raw):
        mapper = HanlpSchemaMapper()
        doc = mapper.map(TEXT, mock_raw)
        assert doc.meta.source == "hanlp_v2"
        assert doc.meta.text_length == len(TEXT)
        assert len(doc.content.tokens) > 0
        assert isinstance(doc.content.entities, list)
        assert isinstance(doc.content.relations, list)

    def test_map_preserves_structural(self, mock_raw):
        mapper = HanlpSchemaMapper()
        doc = mapper.map(TEXT, mock_raw)
        assert doc.content.structural == mock_raw

    def test_serialization(self, mock_raw):
        mapper = HanlpSchemaMapper()
        doc = mapper.map(TEXT, mock_raw)
        data = doc.model_dump()
        assert "meta" in data
        assert "content" in data
        assert len(data["content"]["tokens"]) > 0
