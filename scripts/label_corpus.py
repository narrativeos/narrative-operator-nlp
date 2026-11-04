#!/usr/bin/env python3
"""Corpus labeling CLI — the contract side of the annotation pipeline.

The labeling *executor* is not in this repo: an MCP client agent (or a human)
drives it, and this CLI provides the deterministic parts — cut a book into
reviewable sections, pre-label them with the engines already in-tree, triage
the result so the human only looks at disagreements, validate offsets, and
export what scripts/bench_gliner_cpu.py --eval needs.

Unit: a book is the batch unit (license and train/dev/test split are
book-level, and splitting by book prevents same-book leakage). Inside a book
the annotation unit is a section, because the engines have length limits and a
reviewer needs an addressable span.

Usage:
    python scripts/label_corpus.py split --book book.txt --book-id add-manufacturing \\
        --title "增材制造工艺" --license CC-BY-4.0 --split dev
    python scripts/label_corpus.py prelabel --book-id add-manufacturing [--with-gliner] [--with-mtl]
    python scripts/label_corpus.py validate --book-id add-manufacturing
    python scripts/label_corpus.py stats --book-id add-manufacturing
    python scripts/label_corpus.py export --book-id add-manufacturing --out /tmp/eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import yaml  # noqa: E402

from core.annotation import (  # noqa: E402
    AgreementTier,
    BookRecord,
    CorpusManifest,
    GoldSpan,
    Prelabel,
    SectionRecord,
    content_sha256,
    book_path,
    iter_sections,
    load_manifest,
    retriage,
    save_manifest,
    split_book,
    to_eval_jsonl,
    validate_section,
    review_workload,
)

DICT_PATH = _project_root / "config" / "domain_keywords.yaml"
# Only the categories the dictionary actually covers; the classical labels
# need their own eval set before they are worth prompting for.
DICT_CATEGORIES = {"MATERIAL": "material", "STANDARD": "standard", "PARAMETER": "parameter"}

GLINER_LABELS = ["材料", "material", "标准", "标准规范", "standard", "参数", "性能指标", "parameter"]


def _dictionary_spans(text: str) -> list[tuple[str, str, tuple[int, int]]]:
    """(text, category, span) for every dictionary term present in text."""
    config = yaml.safe_load(DICT_PATH.read_text(encoding="utf-8")) or {}
    found = []
    for category, key in DICT_CATEGORIES.items():
        for term in sorted(set(config.get(key, []) or [])):
            if not term:
                continue
            start = text.find(term)
            while start != -1:
                found.append((term, category, (start, start + len(term))))
                start = text.find(term, start + 1)
    return found


def _gliner_spans(text: str, engine) -> list[tuple[str, str, tuple[int, int]]]:
    out = []
    for e in engine.predict_entities(text, GLINER_LABELS, threshold=0.5):
        label = (e.get("label") or "").upper()
        category = label if label in GoldSpan.model_fields else "UNKNOWN"
        out.append((e["text"], category, (e["start"], e["end"])))
    return out


def _mtl_spans(text: str, pipeline) -> list[tuple[str, str, tuple[int, int]]]:
    """HanLP MTL NER, mapped to NSP categories through NerLabelMapper."""
    from core.ner_label_mapper import NerLabelMapper

    raw = getattr(pipeline(text), "ner", None) or []
    mapper = NerLabelMapper()
    sources = mapper.ner_sources or ["ner/pku"]
    out = []
    for item in raw:
        surface, label, start, end = item[0], item[1], item[2], item[3]
        category = next((c for s in sources if (c := mapper.map_label(label, s))), None)
        out.append((surface, category or "UNKNOWN", (start, end)))
    return out


def cmd_split(args) -> int:
    text = Path(args.book).read_text(encoding="utf-8")
    spans = split_book(text, max_chars=args.max_chars)
    if not spans:
        print("no splittable content", file=sys.stderr)
        return 1
    records = [SectionRecord(section_id=f"{args.book_id}#{i}", book_id=args.book_id,
                              text=text[s:e], offset=(s, e))
               for i, (s, e) in enumerate(spans)]
    path = _write(args.book_id, records)

    manifest = load_manifest()
    manifest.books = [b for b in manifest.books if b.book_id != args.book_id]
    manifest.books.append(BookRecord(book_id=args.book_id, title=args.title,
                                     license=args.license, split=args.split,
                                     n_sections=len(records), n_chars=len(text),
                                     sha256=content_sha256(text)))
    save_manifest(manifest)
    print(f"book={args.book_id} chars={len(text)} sections={len(records)} -> {path}")
    print(f"license={args.license} split={args.split} (both are book-level)")
    return 0


def _write(book_id: str, records: list[SectionRecord]) -> Path:
    from core.annotation import write_sections

    return write_sections(book_id, records)


def cmd_prelabel(args) -> int:
    """Merge the engines' proposals per section and assign review tiers.

    Same span + same category from every engine that ran  -> AGREE
    Different category for the same span                  -> CONFLICT (disputed)
    Anything else                                         -> PARTIAL
    """
    engine = None
    if args.with_gliner:
        sys.path.insert(0, str(_project_root / "scripts"))
        from bench_gliner_cpu import _load_gliner

        engine, reason = _load_gliner(args.model)
        if engine is None:
            print(f"--with-gliner unavailable: {reason}", file=sys.stderr)
            return 1
    pipeline = None
    if args.with_mtl:
        from core.analyzer import _get_modern_pipeline

        pipeline = _get_modern_pipeline()

    records = list(iter_sections(args.book_id))
    for rec in records:
        merged: dict[tuple[int, int], Prelabel] = {}
        proposals = [("dictionary", _dictionary_spans(rec.text)),
                     ("gliner", _gliner_spans(rec.text, engine) if engine else []),
                     ("mtl", _mtl_spans(rec.text, pipeline) if pipeline else [])]
        for source, spans in proposals:
            for surface, category, span in spans:
                if span[1] > len(rec.text):
                    continue
                p = merged.get(span)
                if p is None:
                    merged[span] = Prelabel(text=surface, category=category, span=span,
                                            sources={source: True})
                    continue
                p.sources[source] = True
                if p.category != category:
                    p.disputed = True
                    if p.category == "UNKNOWN":
                        p.category = category
        rec.prelabels = [merged[k] for k in sorted(merged)]
        retriage(rec)
        if rec.prelabels and rec.status == "pending":
            rec.status = "in_review"
    _write(args.book_id, records)
    stats = review_workload(records)
    print(f"book={args.book_id} sections={len(records)} engines_seen={stats['engines_seen']}")
    print(f"agree={stats['spans_agree']} partial={stats['spans_partial']} "
          f"conflict={stats['spans_conflict']}")
    print(f"human review items={stats['human_review_items']} "
          f"(auto-accept covers {stats['auto_accept_share']:.0%} of spans)")
    if stats["engines_seen"] < 2:
        print("note: one engine cannot produce agreement — nothing here is "
              "auto-acceptable. Add --with-gliner or --with-mtl.")
    return 0


def cmd_validate(args) -> int:
    bad = 0
    for rec in iter_sections(args.book_id):
        for e in validate_section(rec):
            print(f"{rec.section_id}: {e}")
            bad += 1
    print("OK — every span offset matches its text" if bad == 0 else f"{bad} problem(s)")
    return 1 if bad else 0


def cmd_stats(args) -> int:
    records = list(iter_sections(args.book_id))
    manifest = load_manifest().book(args.book_id)
    if manifest:
        print(f"title={manifest.title!r} license={manifest.license} split={manifest.split}")
    for k, v in review_workload(records).items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
    by_status: dict[str, int] = {}
    for r in records:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    print("status:", ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
    return 0


def cmd_export(args) -> int:
    rows = to_eval_jsonl(list(iter_sections(args.book_id)))
    if not rows:
        print("no labeled sections yet — export would feed the gate an empty "
              "answer set, which inflates recall", file=sys.stderr)
        return 1
    Path(args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} labeled sections -> {args.out}")
    print(f"next: python scripts/bench_gliner_cpu.py --eval {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("split", help="cut a book file into reviewable sections")
    s.add_argument("--book", required=True, type=Path)
    s.add_argument("--book-id", required=True)
    s.add_argument("--title", default="")
    s.add_argument("--license", required=True, help="book-level license, never inferred")
    s.add_argument("--split", default="train", choices=["train", "dev", "test"])
    s.add_argument("--max-chars", type=int, default=2000)
    s.set_defaults(func=cmd_split)

    for name, fn in (("prelabel", cmd_prelabel), ("validate", cmd_validate),
                     ("stats", cmd_stats)):
        p = sub.add_parser(name)
        p.add_argument("--book-id", required=True)
        if name == "prelabel":
            p.add_argument("--with-gliner", action="store_true")
            p.add_argument("--model", default="urchade/gliner_multi-v2.1")
            p.add_argument("--with-mtl", action="store_true",
                           help="also propose spans from the HanLP MTL NER task")
        p.set_defaults(func=fn)

    e = sub.add_parser("export", help="emit the --eval format for bench_gliner_cpu.py")
    e.add_argument("--book-id", required=True)
    e.add_argument("--out", required=True, type=Path)
    e.set_defaults(func=cmd_export)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
