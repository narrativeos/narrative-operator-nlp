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

from .schema import Relation, EntityAttribute

# ── Compound-internal dep rels (entity merging, not cross-entity) ──
_COMPOUND_INTERNAL = {"nn", "assmod", "assm", "nummod", "clf", "det", "punct", "cc", "conj", "root", "top", "attr", "pass", "etmp", "prep", "pobj", "appos", "lobj", "plmod", "tmod", "advcl", "rcmod", "nsubjpass"}

# POS that are NOT valid relation endpoints (function words / quantities)
_NON_ENDPOINT_POS: frozenset[str] = frozenset({
    "NUM", "ADV", "PART", "PUNCT", "PRON", "DET", "AUX", "SCONJ", "CCONJ",
    "INTJ", "SYM", "X",
})


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
    # Deduplicate: skip if same key already exists
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
    "nsubj": "RELATES_TO",    # subject → verb: 鲲→有 → RELATES_TO
    "obj": "RELATES_TO",      # verb → object: 有→鱼 → RELATES_TO
    "dobj": "RELATES_TO",     # verb → object (Peking UD)
    "nmod": "HAS_PROPERTY",   # noun modifier: 名→鲲 → HAS_PROPERTY
    "amod": "HAS_PROPERTY",   # adj modifier: 大→鲲 → HAS_PROPERTY
    "iobj": "RELATES_TO",     # indirect object
}


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

    def reset(self):
        """Reset per-request state. Must be called before each new analysis."""
        self._counter = 0

    # ── SRL Extraction (primary) ──

    def extract_from_srl(self, frame: list, tokens: list,
                         text: str = "") -> list[Relation]:
        """Extract relations from SRL frame. One frame can yield
        multiple relations (e.g. ARG0→ARG1 + ARG1→ARGM-LOC)."""
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

        # ── Generate all applicable relation pairs ──
        pairs: list[tuple[tuple, str]] = []
        if a0 and a1:
            pairs.append((a0, a1, "RELATES_TO"))
        if a1 and a2:
            pairs.append((a1, a2, "RELATES_TO"))
        if a0 and loc:
            pairs.append((a0, loc, "RELATES_TO"))
        if a1 and loc:
            pairs.append((a1, loc, "RELATES_TO"))

        relations = []
        for subj, obj, predicate in pairs:
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
        """Extract semantic relations from classical Chinese dep output.

        Unlike modern Chinese (which relies on SRL), classical Chinese
        uses dependency relations as the primary relation source:
        - nsubj → subject-verb → RELATES_TO
        - dobj → verb-object → RELATES_TO
        - nmod → noun-modifier → HAS_PROPERTY
        """
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

        # Determine subject and object based on deprel direction
        if deprel in ("nsubj", "nmod", "amod"):
            # child modifies head: child→subject, head→verb/noun
            subj, obj = ct, ht
        elif deprel in ("dobj", "obj", "iobj"):
            # head is verb, child is object
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
        """Merge a token with its det children into a compound.

        ``其(det)→名`` becomes ``其名`` with combined span.
        Returns (merged_text, merged_span).
        """
        # Collect det children that point to this token
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

        # Extract SRL relations (when available — classical Chinese has no SRL)
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

        # ── Classical Chinese: 3-step pipeline ──
        # Step 1: Entities (already identified by entity_mapper).
        # Step 2: Entity-to-entity relations via nsubj+obj / cop bridges.
        # Step 3: Entity properties via nmod/amod (HAS_PROPERTY).
        if not has_srl:
            # Step 2 — Build entity-to-entity bridges
            bridges = self._bridge_subj_obj(
                raw.get("dep", []), tokens, text, entity_texts,
            )
            cop_bridges = self._bridge_cop(
                raw.get("dep", []), tokens, text, entity_texts,
            )
            relations.extend(bridges)
            relations.extend(cop_bridges)

            # Collect verbs already covered by bridges (for nsubj/obj suppression)
            bridged_verbs: set[str] = set()
            for r in bridges + cop_bridges:
                if r.predicate_verb:
                    bridged_verbs.add(r.predicate_verb)

            # Step 3 — Entity properties: nmod/amod → HAS_PROPERTY
            # Also keep lone nsubj/obj not covered by any bridge.
            for i, d in enumerate(raw.get("dep", [])):
                if not isinstance(d, (list, tuple)) or len(d) < 2:
                    continue
                deprel = str(d[1]).strip().lower()
                head_1based = int(d[0])
                head_0 = head_1based - 1

                if deprel in ("nsubj", "obj", "dobj", "iobj"):
                    # Suppress if covered by bridge; keep lone survivors
                    head_tok = tokens[head_0] if 0 <= head_0 < len(tokens) else None
                    head_text = head_tok.text if head_tok else ""
                    if head_text in bridged_verbs:
                        continue
                    # Drop if the non-entity endpoint is a function word
                    # (NUM, ADV, PART, ...) — these are property material,
                    # not relation endpoints.
                    child_tok = tokens[i] if 0 <= i < len(tokens) else None
                    if not _is_relation_endpoint(head_tok) or not _is_relation_endpoint(child_tok):
                        continue
                    # Suppress VERB→single-char-NOUN (syntactic noise, e.g. 知→名)
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
                    # Attach property to entity as EntityAttribute
                    if not (0 <= i < len(tokens) and 0 <= head_0 < len(tokens)):
                        continue
                    child_tok, head_tok = tokens[i], tokens[head_0]

                    # Determine which token is the entity (property owner)
                    if head_tok.text in entity_texts:
                        owner_text, prop_text = head_tok.text, child_tok.text
                    elif child_tok.text in entity_texts:
                        owner_text, prop_text = child_tok.text, head_tok.text
                    else:
                        continue

                    # Attach to entity object
                    _attach_attribute(entities, owner_text, prop_text,
                                      confidence=0.70, source=f"dep/{deprel}")

            # ── Post: 之-interpolation + NUM/measure heuristic ──
            # "X之Y，…Z" where Y and Z are attributes of X.
            # Also collects trailing NUM+CLF compounds as attributes.
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
                                _attach_attribute(
                                    entities, e_text, nxt.text,
                                    confidence=0.50, source="heuristic/之",
                                )
                                found_attr = True
                                # Continue scanning for NUM+CLF compound
                                j = i + 2
                                while j < len(tokens):
                                    tt = tokens[j]
                                    if tt.pos in ("PUNCT",) and tt.text in ("。", "！", "？", "；"):
                                        break  # stop at sentence-ending punctuation
                                    if tt.pos == "NUM":
                                        # Collect NUM + following CLF/measure tokens
                                        parts = [tt.text]
                                        k = j + 1
                                        while k < len(tokens) and tokens[k].pos in ("NOUN", "NUM"):
                                            # Stop if POS isn't measure-like (check xpos)
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
                                                # Try one more if it's NOUN (e.g., classifier)
                                                if tokens[k].pos == "NOUN":
                                                    parts.append(tokens[k].text)
                                                    k += 1
                                                break
                                        compound = "".join(parts)
                                        _attach_attribute(
                                            entities, e_text, compound,
                                            confidence=0.45, source="heuristic/num_measure",
                                        )
                                        j = k
                                        continue
                                    if tt.pos == "ADJ":
                                        _attach_attribute(
                                            entities, e_text, tt.text,
                                            confidence=0.40, source="heuristic/cont",
                                        )
                                    j += 1
                            break
                        break  # stop at first token after entity
                        if found_attr:
                            break

        return relations

    # ── Classical nsubj+obj bridging ──

    def _bridge_subj_obj(
        self, dep: list, tokens: list, text: str, entity_texts: set[str],
    ) -> list[Relation]:
        """Bridge nsubj→verb←obj into subject→object relations.

        In classical Chinese DEP, ``北冥(nsubj)→有`` and ``鱼(obj)→有``
        are separate edges. We merge them into ``北冥→鱼`` with
        predicate_verb ``有``, producing entity-to-entity relations.

        Only keeps relations where at least one endpoint is an entity.
        """
        # Index: head_verb → list of (child_idx, deprel)
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

                    # Merge det compounds: 其名 → 其名
                    subj_text, subj_span = self._merge_det_compound(
                        subj_idx, dep, tokens,
                    )
                    obj_text, obj_span = self._merge_det_compound(
                        obj_idx, dep, tokens,
                    )

                    # At least one endpoint must be an entity
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
        """Bridge copula relations: 为(cop)→鲲 + 名(obj)→知 → 名→鲲(为).

        A copula (为) points to a predicate nominal (鲲). The predicate's
        head (知) also governs the copula's subject (名,obj). We bridge
        the subject and predicate into a single relation.
        """
        # Build: head → list of (child_idx, deprel)
        head_children: dict[int, list[tuple[int, str]]] = {}
        # Also track: child → its head
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
            cop_verb = tokens[ci]  # e.g. "为"
            pred_idx = int(d[0]) - 1  # predicate nominal, e.g. "鲲"

            if not (0 <= pred_idx < len(tokens)):
                continue
            pred_tok = tokens[pred_idx]

            # The predicate's head governs both the predicate and the
            # copula's subject — find sibling nsubj/obj there.
            pred_head = child_head.get(pred_idx)
            if pred_head is None:
                continue

            for sib_idx, sib_rel in head_children.get(pred_head, []):
                if sib_idx == pred_idx or sib_idx == ci:
                    continue
                if sib_rel not in ("nsubj", "obj", "dobj"):
                    continue
                sib_tok = tokens[sib_idx]

                # Merge det compound: 其名 → 其名
                sib_text, sib_span = self._merge_det_compound(
                    sib_idx, dep, tokens,
                )

                # At least one endpoint must be an entity
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

        If an SRL argument is a phrase containing a known entity
        (e.g., '钢的一种' contains entity '钢'), normalize to that entity.
        Only single-entity matches are accepted to avoid ambiguity.

        When ``strict=False`` (classical Chinese), at least ONE endpoint
        must match an entity; purely verb–noun relations are dropped.
        """
        if not entity_texts:
            return rel

        hit_count = 0  # how many endpoints matched an entity

        for attr in ("subject", "object"):
            text = getattr(rel, attr)
            if text in entity_texts:
                hit_count += 1
                continue  # exact match

            # Find entities that are substrings of this argument
            matches = [e for e in entity_texts if e in text]
            if len(matches) == 1:
                setattr(rel, attr, matches[0])  # normalize to single matched entity
                hit_count += 1
                continue

            # No match
            if strict:
                # Modern SRL: require BOTH endpoints to be entities
                return None
            # Classical DEP: allow non-entity as long as the OTHER endpoint
            # is an entity (checked after loop)

        if not strict and hit_count == 0:
            # Classical: neither endpoint is an entity
            if not entity_texts:
                # No entities found at all → keep relations as-is
                return rel
            # Entities exist but neither endpoint matches → noise, drop
            return None

        return rel
