"""
Relation Extraction — SRL-first with DEP supplementary.

Strategy (redesigned):
1. SRL (semantic role labeling) is the primary source — it gives clean
   (ARG0, PRED, ARG1) triples that directly map to NSP predicates.
2. DEP (dependency parsing) is used only for supplementary relations
   that SRL doesn't capture: adjective→noun properties, etc.
3. Entity-aware: relations must involve at least one entity.
4. Function words (是, 的, 和, 一, ...) are excluded as endpoints.
5. Compound-internal dependencies (nn, assmod, assm) are skipped —
   they're entity merging hints, not cross-entity relations.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import Token

from .schema import Relation

# ── Compound-internal dep rels (entity merging, not cross-entity) ──
_COMPOUND_INTERNAL = {"nn", "assmod", "assm", "nummod", "clf", "det", "punct", "cc", "conj", "root", "top", "attr", "pass", "etmp", "prep", "pobj", "appos", "nmod", "lobj", "plmod", "tmod", "advcl", "rcmod", "nsubjpass"}


def _span_between(sp1: tuple, sp2: tuple, text: str) -> str:
    """Extract text covering two spans."""
    start = min(sp1[0], sp2[0])
    end = max(sp1[1], sp2[1])
    if start < 0 or end > len(text):
        return ""
    return text[start:end]


class RelationExtractionRules:
    """SRL-first, DEP-supplementary relation extractor."""

    def __init__(self):
        self._counter = 0

    # ── SRL Extraction (primary) ──

    def extract_from_srl(self, frame: list, tokens: list,
                         text: str = "") -> Optional[Relation]:
        """Extract relation from SRL frame (ARG0, PRED, ARG1)."""
        a0_text, a1_text = "", ""
        a0_span, a1_span = (0, 0), (0, 0)
        pred_text = ""

        for item in frame:
            if len(item) < 4:
                continue
            role = str(item[1]).upper()
            itxt = str(item[0])
            ts, te = int(item[2]), int(item[3])
            if 0 <= ts < len(tokens) and 0 < te <= len(tokens):
                cs = tokens[ts].span[0]
                ce = tokens[te - 1].span[1]
            else:
                cs, ce = 0, 0
            if "ARG0" in role:
                a0_text, a0_span = itxt, (cs, ce)
            elif "ARG1" in role:
                a1_text, a1_span = itxt, (cs, ce)
            elif role == "PRED":
                pred_text = itxt

        if not a0_text or not a1_text:
            return None

        pred_text = pred_text

        evidence = _span_between(a0_span, a1_span, text)
        if not evidence:
            evidence = f"{a0_text} {pred_text} {a1_text}"

        self._counter += 1
        return Relation(
            id=f"rel_{self._counter:03d}",
            subject=a0_text.strip(),
            predicate="RELATES_TO",
            predicate_verb=pred_text,  # raw SRL predicate for downstream reasoning
            object=a1_text.strip(),
            evidence=evidence,
            evidence_span=(a0_span[0], a1_span[1]),
            confidence=0.60,
            source=f"srl/{pred_text}",
        )

    # ── DEP Extraction (supplementary, amod only) ──

    def extract_from_dep(self, child_idx: int, deprel: str, head_idx: int,
                         tokens: list, text: str = "") -> Optional[Relation]:
        """Extract adjective-property relations from DEP (amod only)."""
        deprel = str(deprel).strip().lower()

        if deprel in _COMPOUND_INTERNAL:
            return None

        if deprel != "amod":
            return None

        head_0 = head_idx - 1
        if not (0 <= child_idx < len(tokens) and 0 <= head_0 < len(tokens)):
            return None

        ct, ht = tokens[child_idx], tokens[head_0]

        if ct.text == ht.text:
            return None

        evidence = _span_between(ct.span, ht.span, text)
        if not evidence:
            evidence = f"{ht.text}{ct.text}"

        self._counter += 1
        return Relation(
            id=f"rel_{self._counter:03d}",
            subject=ht.text,
            predicate="HAS_PROPERTY",
            object=ct.text,
            evidence=evidence,
            evidence_span=(min(ct.span[0], ht.span[0]), max(ct.span[1], ht.span[1])),
            confidence=0.80,
            source="dep/amod",
        )

    # ── Aggregate ──

    def extract_all(self, text: str, raw: dict, tokens: list,
                    entities: list | None = None) -> list:
        """Extract relations: SRL first, then supplementary DEP (amod).

        When entities are provided, relations are normalized and filtered:
        - Endpoints are matched to recognized entities
        - Non-entity endpoints cause the relation to be dropped
        - Bare adjectives (高, 大, ...) as objects are suppressed
        """
        # Build entity index: entity text → canonical text
        entity_texts: set[str] = set()
        if entities:
            for e in entities:
                entity_texts.add(e.text)

        # Extract SRL relations
        relations: list = []
        for f in raw.get("srl", []):
            if isinstance(f, list):
                rel = self.extract_from_srl(f, tokens, text)
                if rel:
                    # Normalize endpoints to entities
                    rel = self._entity_normalize(rel, entity_texts)
                    if rel:
                        relations.append(rel)

        # Extract DEP relations (suppress if covered by SRL or bare adjective)
        for i, d in enumerate(raw.get("dep", [])):
            if isinstance(d, (list, tuple)) and len(d) >= 2:
                rel = self.extract_from_dep(i, str(d[1]), int(d[0]), tokens, text)
                if rel:
                    rel = self._entity_normalize(rel, entity_texts)
                    if rel:
                        relations.append(rel)

        return relations

    def _entity_normalize(self, rel: Relation,
                          entity_texts: set[str]) -> Relation | None:
        """Only keep relations where both endpoints are recognized entities."""
        if not entity_texts:
            return rel
        if rel.subject not in entity_texts or rel.object not in entity_texts:
            return None
        return rel
