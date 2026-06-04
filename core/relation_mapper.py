"""
Relation Extraction — Dependency & SRL to NSP Triples.

HanLP MTL output:
    dep: list[(head_1based, deprel)]
    srl: list[list[(text, role, tok_start, tok_end)]]
"""

from __future__ import annotations
from typing import Optional
from .schema import Relation, RelationPredicate, Token

_DEP_MAP = {
    "top":("IS_A",0.80),"attr":("IS_A",0.85),
    "nn":("PART_OF",0.75),"assmod":("PART_OF",0.75),"assm":("PART_OF",0.75),
    "amod":("HAS_PROPERTY",0.85),"rcmod":("HAS_PROPERTY",0.75),"advmod":("HAS_PROPERTY",0.70),
    "lobj":("LOCATED_AT",0.90),"plmod":("LOCATED_AT",0.85),
    "tmod":("TEMPORAL_AT",0.90),
    "advcl":("CAUSES",0.70),
    "pobj":("DEPENDS_ON",0.75),"prep":("DEPENDS_ON",0.70),
    "appos":("EQUIVALENT_TO",0.90),
    "nsubj":("PROPERTY_OF",0.80),"nsubjpass":("PROPERTY_OF",0.80),"dobj":("PROPERTY_OF",0.75),
    "nmod":("PART_OF",0.65),"conj":("PART_OF",0.60),"cc":("PART_OF",0.55),
    "nummod":("HAS_PROPERTY",0.60),"clf":("HAS_PROPERTY",0.55),"det":("HAS_PROPERTY",0.50),
    "pass":("DEPENDS_ON",0.70),"etmp":("TEMPORAL_AT",0.80),
    "punct":(None,0),"root":(None,0),
}


class RelationExtractionRules:
    def __init__(self):
        self._counter = 0

    def extract_from_dep(self, child_idx: int, deprel: str, head_idx: int,
                         tokens: list[Token], text: str = "") -> Optional[Relation]:
        deprel = str(deprel).strip().lower()
        mapping = _DEP_MAP.get(deprel)
        if not mapping or mapping[1] <= 0:
            return None
        pred, conf = mapping
        if pred is None:
            return None
        head_0 = head_idx - 1
        if not (0 <= child_idx < len(tokens) and 0 <= head_0 < len(tokens)):
            return None
        ct, ht = tokens[child_idx], tokens[head_0]
        es, ee = min(ct.span[0], ht.span[0]), max(ct.span[1], ht.span[1])
        evidence = text[es:ee] if text else f"{ht.text}...{ct.text}"
        self._counter += 1
        return Relation(id=f"rel_{self._counter:03d}", subject=ht.text, predicate=pred,
                        object=ct.text, evidence=evidence, evidence_span=(es, ee),
                        confidence=conf, source=f"dep/{deprel}")

    def extract_from_srl(self, frame: list, tokens: list[Token], text: str = "") -> Optional[Relation]:
        a0_t, a1_t = "", ""
        a0_s, a1_s = (0,0), (0,0)
        for item in frame:
            if len(item) < 4:
                continue
            role = str(item[1]).upper()
            itxt = str(item[0])
            ts, te = int(item[2]), int(item[3])
            cs, ce = (tokens[ts].span[0], tokens[te-1].span[1]) if 0<=ts<len(tokens) and 0<te<=len(tokens) else (0,0)
            if "ARG0" in role:
                a0_t, a0_s = itxt, (cs, ce)
            elif "ARG1" in role:
                a1_t, a1_s = itxt, (cs, ce)
        if not a0_t or not a1_t:
            return None
        es, ee = a0_s[0], a1_s[1]
        evidence = text[es:ee] if text else f"{a0_t}...{a1_t}"
        self._counter += 1
        return Relation(id=f"rel_{self._counter:03d}", subject=a0_t, predicate="PROPERTY_OF",
                        object=a1_t, evidence=evidence, evidence_span=(es, ee),
                        confidence=0.75, source="srl/ARG0-ARG1")

    def extract_all(self, text: str, raw: dict, tokens: list[Token]) -> list[Relation]:
        relations = []
        for i, d in enumerate(raw.get("dep", [])):
            if isinstance(d, (list, tuple)) and len(d) >= 2:
                rel = self.extract_from_dep(i, str(d[1]), int(d[0]), tokens, text)
                if rel:
                    relations.append(rel)
        for f in raw.get("srl", []):
            if isinstance(f, list):
                rel = self.extract_from_srl(f, tokens, text)
                if rel:
                    relations.append(rel)
        return relations
