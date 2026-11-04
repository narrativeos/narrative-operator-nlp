"""Annotation contract for the corpus-scale labeled data (Phase 2).

This module defines the *data contract* only. It deliberately contains no
agent and no LLM client: the whole repo is a deterministic NLP operator (zero
model-API dependency), and the labeling *executor* lives outside — an MCP
client agent drives this service through adapters/mcp_server.py, while the
automatic pre-labeling is done by the deterministic engines already in-tree
(domain dictionary, optional GLiNER, optional HanLP MTL).

Why a contract in-tree and the executor outside: analyze() must stay
reproducible, and labeling is offline batch work that never belongs on the
request path.

Layout (one file per book, JSON Lines so labeled data is reviewable in a diff
— unlike the CBDB precedent, which is a 576MB SQLite blob):

    data/annotations/manifest.json          # CorpusManifest: books, licenses, splits
    data/annotations/<book_id>.jsonl        # one SectionRecord per line

Three levels: book (batch, license, split) -> section (annotation and review
unit) -> span (the label itself). Books are batches, not annotation units:
MTL/GLiNER have length limits and a reviewer needs an addressable unit.

Triage reuses the signals the pipeline already emits (EntityEvidence,
ner_disputed, the F1/F2 quality demotion) instead of inventing a new
confidence score, so the human workload is the disagreement set, not the
whole text.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Optional

from pydantic import BaseModel, Field, field_validator

from core.schema import EntityCategory

_project_root = Path(__file__).resolve().parent.parent

# Overridable so tests and dry runs never write into the repo.
ANNOTATION_DIR = Path(os.environ.get("NARRATIVE_ANNOTATION_DIR",
                                     _project_root / "data" / "annotations"))
MANIFEST_PATH = ANNOTATION_DIR / "manifest.json"

# A section must stay inside what the engines can actually read and what a
# human can review in one pass.
DEFAULT_MAX_SECTION_CHARS = 2000
MIN_SECTION_CHARS = 200

# Share of AGREE sections that still get a human re-check, to catch a
# systematically wrong prelabeler.
AGREE_SAMPLE_RATE = 0.05

# Agreement needs at least this many independent engines to be agreement.
MIN_AGREE_ENGINES = 2


class AgreementTier:
    """How much the independent sources agree on a span.

    Drives the review queue: AGREE is auto-accepted (sampled for audit),
    PARTIAL is the bulk of the human workload, CONFLICT is highest priority
    and also the most informative data for later training.
    """

    AGREE = "agree"
    PARTIAL = "partial"
    CONFLICT = "conflict"

    ALL = frozenset({AGREE, PARTIAL, CONFLICT})


class GoldSpan(BaseModel):
    """A human-confirmed span. Mirrors the span tuples already used by
    tests/datasets/*.json ("ner/pku": [["立方庭", "ns", 14, 15]])."""

    text: str = Field(..., min_length=1)
    category: str = Field(..., description="NSP EntityCategory")
    span: tuple[int, int] = Field(..., description="Character offsets [start, end) into the section text")

    @field_validator("category")
    @classmethod
    def _category_valid(cls, v: str) -> str:
        if v not in EntityCategory.ALL:
            raise ValueError(f"Unknown entity category: {v}")
        return v


class Prelabel(BaseModel):
    """A candidate span produced by one of the deterministic engines."""

    text: str = Field(..., min_length=1)
    category: str = Field(..., description="NSP EntityCategory")
    span: tuple[int, int]
    sources: dict[str, bool] = Field(
        default_factory=dict,
        description="Which engines proposed this span+category: dictionary / gliner / mtl",
    )
    tier: str = Field(default=AgreementTier.PARTIAL)
    demoted: bool = Field(default=False, description="True when the F1/F2 quality layer flagged it")
    disputed: bool = Field(default=False, description="True when NER models disagree on the label")
    review: str = Field(default="", description="Reviewer verdict: accepted / rejected / ''")

    @field_validator("tier")
    @classmethod
    def _tier_valid(cls, v: str) -> str:
        if v not in AgreementTier.ALL:
            raise ValueError(f"Unknown agreement tier: {v}")
        return v


class SectionRecord(BaseModel):
    """One annotation/review unit: a contiguous slice of one book."""

    section_id: str = Field(..., description="Stable id: <book_id>#<index>")
    book_id: str
    text: str = Field(..., min_length=1)
    offset: tuple[int, int] = Field(..., description="Character offsets [start, end) into the whole book")
    labels: list[GoldSpan] = Field(default_factory=list, description="Human-confirmed gold spans")
    prelabels: list[Prelabel] = Field(default_factory=list)
    status: str = Field(default="pending", description="pending | in_review | labeled")

    @field_validator("status")
    @classmethod
    def _status_valid(cls, v: str) -> str:
        if v not in {"pending", "in_review", "labeled"}:
            raise ValueError(f"Unknown section status: {v}")
        return v


class BookRecord(BaseModel):
    """A book is the batch unit: license and dataset split are book-level."""

    book_id: str
    title: str = ""
    license: str = Field(..., description="Book-level license; required, never inferred")
    split: str = Field(default="train", description="train | dev | test — split by book to avoid leakage")
    n_sections: int = 0
    n_chars: int = 0
    sha256: str = Field(default="", description="Content hash, so a relabel run can detect drift")


class CorpusManifest(BaseModel):
    version: int = 1
    books: list[BookRecord] = Field(default_factory=list)

    def book(self, book_id: str) -> Optional[BookRecord]:
        for b in self.books:
            if b.book_id == book_id:
                return b
        return None


# ---------------------------------------------------------------------------
# Splitting: book -> sections
# ---------------------------------------------------------------------------

def split_book(text: str, *, max_chars: int = DEFAULT_MAX_SECTION_CHARS,
               min_chars: int = MIN_SECTION_CHARS) -> list[tuple[int, int]]:
    """Cut a book into [start, end) sections on paragraph boundaries.

    Never cuts mid-paragraph unless a single paragraph exceeds max_chars, in
    which case it falls back to sentence end marks. Sections shorter than
    min_chars are merged forward so review units stay worth reviewing.
    """
    if not text:
        return []

    paras: list[tuple[int, int]] = []
    pos = 0
    for chunk in text.split("\n"):
        end = pos + len(chunk)
        if chunk.strip():
            paras.append((pos, end))
        pos = end + 1
    if not paras:
        return []

    raw: list[tuple[int, int]] = []
    cur_start, cur_end = paras[0]
    for start, end in paras[1:]:
        if end - cur_start > max_chars:
            raw.extend(_split_long(text, cur_start, cur_end, max_chars))
            cur_start, cur_end = start, end
        else:
            cur_end = end
    raw.extend(_split_long(text, cur_start, cur_end, max_chars))

    merged: list[tuple[int, int]] = []
    for start, end in raw:
        if merged and (end - merged[-1][0] <= max_chars
                       and merged[-1][1] - merged[-1][0] < min_chars):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _split_long(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Split one paragraph, preferring sentence ends, hard-cutting as a last resort.

    The hard cut matters: a book saved as a single line with no sentence-end
    marks would otherwise come back as one oversized section, silently
    defeating the length budget the engines depend on.
    """
    if end - start <= max_chars:
        return [(start, end)]
    out: list[tuple[int, int]] = []
    cur = start
    for i in range(start, end):
        if text[i] in "。！？；\n" and i + 1 - cur >= max_chars:
            out.append((cur, i + 1))
            cur = i + 1
    if cur < end:
        if end - cur > max_chars:  # no usable boundary: cut at the budget
            while end - cur > max_chars:
                out.append((cur, cur + max_chars))
                cur += max_chars
        if cur < end:
            out.append((cur, end))
    return out


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()



# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------

def triage(prelabel: Prelabel,
           engines: tuple[str, ...] = ("dictionary", "gliner", "mtl")) -> str:
    """Classify a prelabel into an agreement tier.

    CONFLICT when the quality layer demoted it or the NER models disagree on
    the label; AGREE when at least MIN_AGREE_ENGINES engines that actually ran
    all proposed this exact span+category; PARTIAL otherwise.

    The engine-count floor matters: with one engine enabled, "the engine said
    it" would otherwise read as "the engines agree" and auto-accept the whole
    corpus. A single-source span is by definition unverified, so it is PARTIAL.
    """
    if prelabel.demoted or prelabel.disputed:
        return AgreementTier.CONFLICT
    ran = [e for e in engines if e in prelabel.sources]
    if len(ran) >= MIN_AGREE_ENGINES and all(prelabel.sources[e] for e in ran):
        return AgreementTier.AGREE
    return AgreementTier.PARTIAL


def retriage(record: SectionRecord,
             engines: tuple[str, ...] = ("dictionary", "gliner", "mtl")) -> SectionRecord:
    for p in record.prelabels:
        p.tier = triage(p, engines)
    return record


# ---------------------------------------------------------------------------
# Storage: one JSONL per book
# ---------------------------------------------------------------------------

def book_path(book_id: str) -> Path:
    return ANNOTATION_DIR / f"{book_id}.jsonl"


def write_sections(book_id: str, records: list[SectionRecord]) -> Path:
    path = book_path(book_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.model_dump(), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def iter_sections(book_id: str) -> Iterator[SectionRecord]:
    path = book_path(book_id)
    if not path.exists():
        raise FileNotFoundError(f"No annotation file for book '{book_id}': {path}")
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield SectionRecord.model_validate(json.loads(line))


def load_manifest() -> CorpusManifest:
    if not MANIFEST_PATH.exists():
        return CorpusManifest()
    return CorpusManifest.model_validate(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))


def save_manifest(manifest: CorpusManifest) -> Path:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return MANIFEST_PATH


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_section(record: SectionRecord) -> list[str]:
    """Return human-readable integrity errors for one section."""
    errors: list[str] = []
    text_len = len(record.text)
    groups = (("gold", [(g.text, g.span) for g in record.labels]),
              ("prelabel", [(p.text, p.span) for p in record.prelabels]))
    for kind, spans in groups:
        seen: set[tuple[int, int]] = set()
        for s_text, (start, end) in spans:
            if start < 0 or end > text_len or end <= start:
                errors.append(f"{kind} span {(start, end)} out of section bounds {text_len}")
            elif record.text[start:end] != s_text:
                errors.append(f"{kind} span {(start, end)} text mismatch: offset gives "
                              f"{record.text[start:end]!r}, label says {s_text!r}")
            if (start, end) in seen:
                errors.append(f"{kind} duplicate span {(start, end)}")
            seen.add((start, end))
    if record.status == "labeled" and not record.labels:
        errors.append("status is 'labeled' but no gold spans")
    return errors


# ---------------------------------------------------------------------------
# Pre-labeling: the one implementation shared by the CLI and the MCP tool
# ---------------------------------------------------------------------------

# Only the categories the domain dictionary actually covers. The classical
# labels need their own eval set before they are worth proposing.
DICT_CATEGORIES = {"MATERIAL": "material", "STANDARD": "standard", "PARAMETER": "parameter"}

DICT_PATH = _project_root / "config" / "domain_keywords.yaml"


def dictionary_spans(text: str,
                     categories: tuple[str, ...] = tuple(DICT_CATEGORIES)) -> list[tuple[str, str, tuple[int, int]]]:
    """(surface, category, span) for every dictionary term present in text.

    Read straight from the YAML KeywordExtractor loads, so the pre-labeler and
    the online dictionary cannot drift apart silently.
    """
    import yaml

    config = yaml.safe_load(DICT_PATH.read_text(encoding="utf-8")) or {}
    found: list[tuple[str, str, tuple[int, int]]] = []
    for category in categories:
        key = DICT_CATEGORIES.get(category)
        if not key:
            continue
        for term in sorted(set(config.get(key, []) or [])):
            if not term:
                continue
            start = text.find(term)
            while start != -1:
                found.append((term, category, (start, start + len(term))))
                start = text.find(term, start + 1)
    return found


def merge_proposals(proposals: list[tuple[str, list[tuple[str, str, tuple[int, int]]]]],
                    text_len: int) -> list[Prelabel]:
    """Fold (engine, spans) pairs into one Prelabel per span, then triage.

    Two engines proposing the same span with different categories mark it
    disputed, which triage turns into CONFLICT — that is the signal worth
    sending to a human first.
    """
    merged: dict[tuple[int, int], Prelabel] = {}
    for source, spans in proposals:
        for surface, category, span in spans:
            if span[1] > text_len:
                continue
            existing = merged.get(span)
            if existing is None:
                merged[span] = Prelabel(text=surface, category=category, span=span,
                                        sources={source: True})
                continue
            existing.sources[source] = True
            if existing.category != category:
                # UNKNOWN means an engine could not classify it, not that it
                # disagrees — only two concrete labels are a real conflict.
                if "UNKNOWN" not in (existing.category, category):
                    existing.disputed = True
                if existing.category == "UNKNOWN":
                    existing.category = category
    out = [merged[k] for k in sorted(merged)]
    for p in out:
        p.tier = triage(p)
    return out


def suggest_spans(text: str, categories: Optional[tuple[str, ...]] = None,
                  gliner_engine=None,
                  gliner_labels: Optional[list[str]] = None) -> list[Prelabel]:
    """Deterministic pre-labels for one text, with review tiers attached.

    This is the read-only perception channel for an external labeling agent:
    it proposes and classifies, it never writes. Committing gold labels stays
    an explicit offline act in scripts/label_corpus.py, so no network-facing
    surface can mutate the labeled corpus as a side effect of a query.
    """
    proposals: list[tuple[str, list[tuple[str, str, tuple[int, int]]]]] = [
        ("dictionary", dictionary_spans(text, categories or tuple(DICT_CATEGORIES)))]
    if gliner_engine is not None:
        labels = gliner_labels or ["材料", "material", "标准", "standard", "参数", "parameter"]
        spans = []
        for e in gliner_engine.predict_entities(text, labels, threshold=0.5):
            label = (e.get("label") or "").upper()
            category = label if label in EntityCategory.ALL else "UNKNOWN"
            spans.append((e["text"], category, (e["start"], e["end"])))
        proposals.append(("gliner", spans))
    return merge_proposals(proposals, len(text))



# ---------------------------------------------------------------------------
# Export for the gate, and the review workload estimate
# ---------------------------------------------------------------------------

def to_eval_jsonl(records: list[SectionRecord]) -> list[dict[str, Any]]:
    """Emit the format scripts/bench_gliner_cpu.py --eval consumes.

    Only labeled sections count: a pending section has no gold, and feeding
    one in would silently inflate recall against an empty answer set.
    """
    return [{"text": r.text,
             "entities": [{"text": g.text, "category": g.category} for g in r.labels]}
            for r in records if r.status == "labeled" and r.labels]


def review_workload(records: list[SectionRecord]) -> dict[str, float]:
    """How much human work this corpus actually needs, by tier.

    `engines_seen` is reported because the auto-accept share is meaningless
    without it: one engine cannot produce agreement at all.
    """
    counts = {t: 0 for t in AgreementTier.ALL}
    engines: set[str] = set()
    for r in records:
        for p in r.prelabels:
            counts[p.tier] += 1
            engines.update(p.sources)
    total = sum(counts.values()) or 1
    sample = math.ceil(counts[AgreementTier.AGREE] * AGREE_SAMPLE_RATE)
    return {
        "spans_agree": counts[AgreementTier.AGREE],
        "spans_partial": counts[AgreementTier.PARTIAL],
        "spans_conflict": counts[AgreementTier.CONFLICT],
        "human_review_items": counts[AgreementTier.PARTIAL] + counts[AgreementTier.CONFLICT] + sample,
        "auto_accept_share": counts[AgreementTier.AGREE] / total,
        "engines_seen": len(engines),
    }

