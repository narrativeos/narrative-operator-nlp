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

Optimizations (v2):
- Extended SRL pairs: ARG0→ARG2, ARG1→ARGM-TMP, etc.
- Predicate refinement: verb-based mapping (位于→LOCATED_AT, etc.)
- Relaxed entity filter: at least ONE endpoint must be an entity
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import Token

from .schema import Relation, EntityAttribute

# ── Compound-internal dep rels (entity merging, not cross-entity) ──
_COMPOUND_INTERNAL = {"nn", "assmod", "assm", "nummod", "clf", "det", "punct", "cc", "conj", "root", "top", "attr", "pass", "etmp", "prep", "pobj", "appos", "lobj", "plmod", "tmod", "advcl", "rcmod", "nsubjpass"}

# POS that are NOT valid relation endpoints (function words / quantities)
_NON_ENDPOINT_POS: frozenset[str] = frozenset({
    "NUM", "ADV", "PART", "PUNCT", "PRON", "DET", "AUX", "SCONJ", "CCONJ",
    "INTJ", "SYM", "X",
})

# ── Verb-based predicate mapping ──
# Maps predicate_verb → more specific NSP predicate
_VERB_PREDICATE_MAP: dict[str, str] = {
    # Location
    "位于": "LOCATED_AT",
    "在": "LOCATED_AT",
    "坐落于": "LOCATED_AT",
    "地处": "LOCATED_AT",
    # Production
    "生产": "PRODUCES",
    "制造": "PRODUCES",
    "制造出": "PRODUCES",
    # Composition
    "组成": "COMPOSED_OF",
    "构成": "COMPOSED_OF",
    "由...组成": "COMPOSED_OF",
    # Causal
    "导致": "CAUSES",
    "引起": "CAUSES",
    "造成": "CAUSES",
    # Control
    "控制": "CONTROLS",
    "管理": "CONTROLS",
    "领导": "CONTROLS",
    # Transfer
    "转移到": "TRANSFERS_TO",
    "运往": "TRANSFERS_TO",
    # Movement
    "移动到": "MOVED_TO",
    "迁往": "MOVED_TO",
    "出发": "DEPARTED_FROM",
    # Interaction
    "与...合作": "INTERACTS_WITH",
    "和...合作": "INTERACTS_WITH",
}


def _is_relation_endpoint(token) -> bool:
    """A token is a valid relation endpoint unless it is a function word."""
    if token is None:
        return False
    return token.pos not in _NON_ENDPOINT_POS


def _find_entity(entities: list, text: str):
    """Find an entity object by its text."""
    for e in entities:
        if e.text == text:
            return e
    return None


def _attach_attribute(
    entities: list, owner_text: str, prop_text: str,
    confidence: float = 0.70, source: str = "",
) -> None:
    """Attach a property to an entity as EntityAttribute (deduplicated)."""
    owner = _find_entity(entities, owner_text)
    if owner is None:
        return
    if any(a.key == prop_text for a in owner.attributes):
        return
    attr = EntityAttribute(
        key=prop_text,
        value="",
        predicate_verb="",
        confidence=confidence,
    )
    owner.attributes.append(attr)


# Classical Chinese deprel types that indicate semantic relations
_CLASSICAL_DEP_RELS: dict[str, str] = {
    "nsubj": "RELATES_TO",
    "obj": "RELATES_TO",
    "dobj": "RELATES_TO",
    "nmod": "HAS_PROPERTY",
    "amod": "HAS_PROPERTY",
    "iobj": "RELATES_TO",
}


def _span_between(sp1: tuple, sp2: tuple, text: str) -> str:
    """Extract text covering two spans."""
    start = min(sp1[0], sp2[0])
    end = max(sp1[1], sp2[1])
    if start < 0 or end > len(text):
        return ""
    return text[start:end]


def _resolve_predicate(pred_text: str, arg_role: str = "") -> str:
    """Resolve predicate_verb to a more specific NSP predicate.

    Args:
        pred_text: The verb/predicate text (e.g., "位于", "生产")
        arg_role: The argument role for context (e.g., "argm-loc")
    """
    # Direct verb mapping
    if pred_text in _VERB_PREDICATE_MAP:
        return _VERB_PREDICATE_MAP[pred_text]

    # Role-based mapping
    role_map = {
        "argm-loc": "LOCATED_AT",
        "argm-tmp": "TEMPORAL_AT",
        "argm-mnr": "HAS_PROPERTY",
        "argm-cau": "CAUSES",
        "argm-prd": "PRODUCES",
        "argm-ben": "BENEFITS",
        "argm-rec": "TRANSFERS_TO",
        "argm-src": "DEPARTED_FROM",
    }
    if arg_role in role_map:
        return role_map[arg_role]

    return "RELATES_TO"


class RelationExtractionRules:
    """SRL-first, DEP-supplementary relation extractor."""

    def __init__(self):
        self._counter = 0

    def reset(self):
        """Reset per-request state. Must be called before each new analysis."""
        self._counter = 0

    # ── SRL Extraction (primary) ──

    def extract_from_srl(self, frame: list, tokens: list,
                         text: str = "") -> list[Relation]:
        """Extract relations from SRL frame.

        Extended pairs:
        - ARG0→ARG1 (primary)
        - ARG0→ARG2 (secondary object)
        - ARG0→ARGM-LOC (location)
        - ARG0→ARGM-TMP (time)
        - ARG1→ARGM-LOC (object location)
        - ARG1→ARG2 (object→secondary)
        """
        args: dict[str, tuple[str, tuple]] = {}
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
            if role == "PRED":
                pred_text = itxt
            elif role.startswith("ARG"):
                args.setdefault(role.lower(), (itxt, (cs, ce)))

        a0 = args.get("arg0")
        a1 = args.get("arg1")
        a2 = args.get("arg2")
        loc = args.get("argm-loc")
        tmp = args.get("argm-tmp")
        mnr = args.get("argm-mnr")  # Manner
        cau = args.get("argm-cau")  # Cause
        prd = args.get("argm-prd")  # Product/Result
        ben = args.get("argm-ben")  # Beneficiary
        rec = args.get("argm-rec")  # Recipient
        src = args.get("argm-src")  # Source

        # ── Generate all applicable relation pairs ──
        pairs: list[tuple[tuple, tuple, str]] = []

        # Primary: ARG0→ARG1
        if a0 and a1:
            pairs.append((a0, a1, "arg1"))

        # ARG0→ARG2
        if a0 and a2:
            pairs.append((a0, a2, "arg2"))

        # ARG0→ARGM-LOC (location)
        if a0 and loc:
            pairs.append((a0, loc, "argm-loc"))

        # ARG0→ARGM-TMP (time)
        if a0 and tmp:
            pairs.append((a0, tmp, "argm-tmp"))

        # ARG0→ARGM-MNR (manner)
        if a0 and mnr:
            pairs.append((a0, mnr, "argm-mnr"))

        # ARG0→ARGM-CAU (cause)
        if a0 and cau:
            pairs.append((a0, cau, "argm-cau"))

        # ARG0→ARGM-PRD (product/result)
        if a0 and prd:
            pairs.append((a0, prd, "argm-prd"))

        # ARG0→ARGM-BEN (beneficiary)
        if a0 and ben:
            pairs.append((a0, ben, "argm-ben"))

        # ARG0→ARGM-REC (recipient)
        if a0 and rec:
            pairs.append((a0, rec, "argm-rec"))

        # ARG0→ARGM-SRC (source)
        if a0 and src:
            pairs.append((a0, src, "argm-src"))

        # ARG1→ARGM-LOC
        if a1 and loc:
            pairs.append((a1, loc, "argm-loc"))

        # ARG1→ARG2
        if a1 and a2:
            pairs.append((a1, a2, "arg2"))

        # ARG1→ARGM-PRD
        if a1 and prd:
            pairs.append((a1, prd, "argm-prd"))

        relations = []
        for subj, obj, role in pairs:
            predicate = _resolve_predicate(pred_text, role)
            evidence = _span_between(subj[1], obj[1], text)
            if not evidence:
                evidence = f"{subj[0]} {pred_text} {obj[0]}"

            self._counter += 1
            relations.append(Relation(
                id=f"rel_{self._counter:03d}",
                subject=subj[0].strip(),
                predicate=predicate,
                predicate_verb=pred_text,
                object=obj[0].strip(),
                evidence=evidence,
                evidence_span=(min(subj[1][0], obj[1][0]), max(subj[1][1], obj[1][1])),
                confidence=0.60,
                source=f"srl/{pred_text}",
            ))
        return relations

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

    # ── Classical DEP Extraction (nsubj, dobj, nmod) ──

    def extract_from_dep_classical(
        self, child_idx: int, deprel: str, head_idx: int,
        tokens: list, text: str = "",
    ) -> Optional[Relation]:
        """Extract semantic relations from classical Chinese dep output."""
        deprel = str(deprel).strip().lower()

        if deprel in _COMPOUND_INTERNAL:
            return None

        mapped_pred = _CLASSICAL_DEP_RELS.get(deprel)
        if mapped_pred is None:
            return None

        head_0 = head_idx - 1
        if not (0 <= child_idx < len(tokens) and 0 <= head_0 < len(tokens)):
            return None

        ct, ht = tokens[child_idx], tokens[head_0]

        if ct.text == ht.text:
            return None

        if deprel in ("nsubj", "nmod", "amod"):
            subj, obj = ct, ht
        elif deprel in ("dobj", "obj", "iobj"):
            subj, obj = ht, ct
        else:
            return None

        evidence = _span_between(subj.span, obj.span, text)
        if not evidence:
            evidence = f"{subj.text}{obj.text}"

        self._counter += 1
        return Relation(
            id=f"rel_{self._counter:03d}",
            subject=subj.text,
            predicate=mapped_pred,
            object=obj.text,
            evidence=evidence,
            evidence_span=(min(subj.span[0], obj.span[0]), max(subj.span[1], obj.span[1])),
            confidence=0.55,
            source=f"dep/{deprel}",
        )

    # ── Token helpers ──

    @staticmethod
    def _merge_det_compound(
        idx: int, dep: list, tokens: list,
    ) -> tuple[str, tuple[int, int]]:
        """Merge a token with its det children into a compound."""
        merged_indices = {idx}
        for ci, d in enumerate(dep):
            if not isinstance(d, (list, tuple)) or len(d) < 2:
                continue
            if int(d[0]) - 1 == idx and str(d[1]).strip().lower() == "det":
                merged_indices.add(ci)
        sorted_idx = sorted(merged_indices)
        text = "".join(tokens[i].text for i in sorted_idx)
        span = (
            tokens[sorted_idx[0]].span[0],
            tokens[sorted_idx[-1]].span[1],
        )
        return text, span

    # ── Aggregate ──

    def extract_all(self, text: str, raw: dict, tokens: list,
                    entities: list | None = None) -> list:
        """Extract relations: SRL first, then supplementary DEP (amod).

        Relaxed entity filter: at least ONE endpoint must be an entity.
        """
        entity_texts: set[str] = set()
        if entities:
            for e in entities:
                entity_texts.add(e.text)

        relations: list = []
        has_srl = bool(raw.get("srl"))
        if has_srl:
            for f in raw.get("srl", []):
                if isinstance(f, list):
                    rels = self.extract_from_srl(f, tokens, text)
                    for rel in rels:
                        rel = self._entity_normalize(rel, entity_texts, strict=False)
                        if rel:
                            relations.append(rel)

        if not has_srl:
            bridges = self._bridge_subj_obj(
                raw.get("dep", []), tokens, text, entity_texts,
            )
            cop_bridges = self._bridge_cop(
                raw.get("dep", []), tokens, text, entity_texts,
            )
            relations.extend(bridges)
            relations.extend(cop_bridges)

            bridged_verbs: set[str] = set()
            for r in bridges + cop_bridges:
                if r.predicate_verb:
                    bridged_verbs.add(r.predicate_verb)

            for i, d in enumerate(raw.get("dep", [])):
                if not isinstance(d, (list, tuple)) or len(d) < 2:
                    continue
                deprel = str(d[1]).strip().lower()
                head_1based = int(d[0])
                head_0 = head_1based - 1

                if deprel in ("nsubj", "obj", "dobj", "iobj"):
                    head_tok = tokens[head_0] if 0 <= head_0 < len(tokens) else None
                    head_text = head_tok.text if head_tok else ""
                    if head_text in bridged_verbs:
                        continue
                    child_tok = tokens[i] if 0 <= i < len(tokens) else None
                    if not _is_relation_endpoint(head_tok) or not _is_relation_endpoint(child_tok):
                        continue
                    if head_tok and child_tok:
                        if (head_tok.pos == "VERB" and child_tok.pos == "NOUN" and len(child_tok.text) == 1):
                            continue
                        if (child_tok.pos == "VERB" and head_tok.pos == "NOUN" and len(head_tok.text) == 1):
                            continue
                    rel = self.extract_from_dep_classical(
                        i, deprel, head_1based, tokens, text,
                    )
                    if rel:
                        rel = self._entity_normalize(rel, entity_texts, strict=False)
                        if rel:
                            relations.append(rel)

                elif deprel in ("nmod", "amod"):
                    if not (0 <= i < len(tokens) and 0 <= head_0 < len(tokens)):
                        continue
                    child_tok, head_tok = tokens[i], tokens[head_0]
                    if head_tok.text in entity_texts:
                        owner_text, prop_text = head_tok.text, child_tok.text
                    elif child_tok.text in entity_texts:
                        owner_text, prop_text = child_tok.text, head_tok.text
                    else:
                        continue
                    if entities:
                        _attach_attribute(entities, owner_text, prop_text,
                                          confidence=0.70, source=f"dep/{deprel}")

            # 之-interpolation + NUM/measure heuristic
            for e_text in entity_texts:
                occurrences = [t for t in tokens if t.text == e_text]
                for e_tok in occurrences:
                    e_start, e_end = e_tok.span
                    found_attr = False
                    for i, t in enumerate(tokens):
                        if t.span[0] < e_end:
                            continue
                        if t.text == "之" and t.pos in ("SCONJ", "PART"):
                            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
                            if nxt and nxt.pos in ("VERB", "ADJ", "ADV"):
                                if entities:
                                    _attach_attribute(
                                        entities, e_text, nxt.text,
                                        confidence=0.50, source="heuristic/之",
                                    )
                                found_attr = True
                                j = i + 2
                                while j < len(tokens):
                                    tt = tokens[j]
                                    if tt.pos in ("PUNCT",) and tt.text in ("。", "！", "？", "；"):
                                        break
                                    if tt.pos == "NUM":
                                        parts = [tt.text]
                                        k = j + 1
                                        while k < len(tokens) and tokens[k].pos in ("NOUN", "NUM"):
                                            xp = (raw.get("pos/xpos", [])[k]
                                                  if raw and k < len(raw.get("pos/xpos", []))
                                                  else "")
                                            if "助数詞" in xp or "度量衡" in xp or "数詞" in xp:
                                                parts.append(tokens[k].text)
                                                k += 1
                                            elif tokens[k].pos == "NUM":
                                                parts.append(tokens[k].text)
                                                k += 1
                                            else:
                                                if tokens[k].pos == "NOUN":
                                                    parts.append(tokens[k].text)
                                                    k += 1
                                                break
                                        compound = "".join(parts)
                                        if entities:
                                            _attach_attribute(
                                                entities, e_text, compound,
                                                confidence=0.45, source="heuristic/num_measure",
                                            )
                                        j = k
                                        continue
                                    if tt.pos == "ADJ":
                                        if entities:
                                            _attach_attribute(
                                                entities, e_text, tt.text,
                                                confidence=0.40, source="heuristic/cont",
                                            )
                                    j += 1
                            break
                        break
                        if found_attr:
                            break

        return relations

    # ── Classical nsubj+obj bridging ──

    def _bridge_subj_obj(
        self, dep: list, tokens: list, text: str, entity_texts: set[str],
    ) -> list[Relation]:
        """Bridge nsubj→verb←obj into subject→object relations."""
        head_to_children: dict[int, list[tuple[int, str]]] = {}
        for child_idx, d in enumerate(dep):
            if not isinstance(d, (list, tuple)) or len(d) < 2:
                continue
            head_1based = int(d[0])
            deprel = str(d[1]).strip().lower()
            if deprel in ("nsubj", "obj", "dobj", "iobj"):
                head_0based = head_1based - 1
                head_to_children.setdefault(head_0based, []).append(
                    (child_idx, deprel),
                )

        bridged: list[Relation] = []
        for head_idx, children in head_to_children.items():
            nsubjs = [(ci, dr) for ci, dr in children if dr == "nsubj"]
            objs = [(ci, dr) for ci, dr in children if dr in ("obj", "dobj", "iobj")]
            if not nsubjs or not objs:
                continue

            head_token = tokens[head_idx] if 0 <= head_idx < len(tokens) else None
            verb_text = head_token.text if head_token else ""

            for subj_idx, _ in nsubjs:
                for obj_idx, _ in objs:
                    if not (0 <= subj_idx < len(tokens) and 0 <= obj_idx < len(tokens)):
                        continue

                    subj_text, subj_span = self._merge_det_compound(
                        subj_idx, dep, tokens,
                    )
                    obj_text, obj_span = self._merge_det_compound(
                        obj_idx, dep, tokens,
                    )

                    subj_is_ent = subj_text in entity_texts
                    obj_is_ent = obj_text in entity_texts
                    if not subj_is_ent and not obj_is_ent:
                        continue

                    evidence = _span_between(subj_span, obj_span, text)
                    if not evidence:
                        evidence = f"{subj_text}{verb_text}{obj_text}"

                    self._counter += 1
                    bridged.append(Relation(
                        id=f"rel_{self._counter:03d}",
                        subject=subj_text,
                        predicate="RELATES_TO",
                        predicate_verb=verb_text,
                        object=obj_text,
                        evidence=evidence,
                        evidence_span=(
                            min(subj_span[0], obj_span[0]),
                            max(subj_span[1], obj_span[1]),
                        ),
                        confidence=0.60,
                        source=f"bridge/{verb_text}" if verb_text else "bridge/nsubj_obj",
                    ))

        return bridged

    def _bridge_cop(
        self, dep: list, tokens: list, text: str, entity_texts: set[str],
    ) -> list[Relation]:
        """Bridge copula relations: 为(cop)→鲲 + 名(obj)→知 → 名→鲲(为)."""
        head_children: dict[int, list[tuple[int, str]]] = {}
        child_head: dict[int, int] = {}
        for ci, d in enumerate(dep):
            if not isinstance(d, (list, tuple)) or len(d) < 2:
                continue
            head = int(d[0]) - 1
            deprel = str(d[1]).strip().lower()
            head_children.setdefault(head, []).append((ci, deprel))
            child_head[ci] = head

        bridged: list[Relation] = []
        for ci, d in enumerate(dep):
            if not isinstance(d, (list, tuple)) or len(d) < 2:
                continue
            deprel = str(d[1]).strip().lower()
            if deprel != "cop":
                continue
            cop_verb = tokens[ci]
            pred_idx = int(d[0]) - 1

            if not (0 <= pred_idx < len(tokens)):
                continue
            pred_tok = tokens[pred_idx]

            pred_head = child_head.get(pred_idx)
            if pred_head is None:
                continue

            for sib_idx, sib_rel in head_children.get(pred_head, []):
                if sib_idx == pred_idx or sib_idx == ci:
                    continue
                if sib_rel not in ("nsubj", "obj", "dobj"):
                    continue
                sib_tok = tokens[sib_idx]

                sib_text, sib_span = self._merge_det_compound(
                    sib_idx, dep, tokens,
                )

                if (sib_text not in entity_texts
                        and pred_tok.text not in entity_texts):
                    continue

                evidence = _span_between(sib_span, pred_tok.span, text)
                if not evidence:
                    evidence = f"{sib_text}{cop_verb.text}{pred_tok.text}"

                self._counter += 1
                bridged.append(Relation(
                    id=f"rel_{self._counter:03d}",
                    subject=sib_text,
                    predicate="RELATES_TO",
                    predicate_verb=cop_verb.text,
                    object=pred_tok.text,
                    evidence=evidence,
                    evidence_span=(
                        min(sib_tok.span[0], pred_tok.span[0]),
                        max(sib_tok.span[1], pred_tok.span[1]),
                    ),
                    confidence=0.55,
                    source=f"bridge/{cop_verb.text}",
                ))

        return bridged

    def _entity_normalize(self, rel: Relation,
                          entity_texts: set[str],
                          strict: bool = True) -> Relation | None:
        """Normalize endpoints: exact match → sub-entity match → reject.

        Relaxed: at least ONE endpoint must match an entity.
        """
        if not entity_texts:
            return rel

        hit_count = 0

        for attr in ("subject", "object"):
            text = getattr(rel, attr)
            if text in entity_texts:
                hit_count += 1
                continue

            matches = [e for e in entity_texts if e in text]
            if len(matches) == 1:
                setattr(rel, attr, matches[0])
                hit_count += 1
                continue

        # Relaxed: at least one endpoint must match
        if hit_count == 0:
            if not entity_texts:
                return rel
            return None

        return rel