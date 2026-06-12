"""
Large-scale test using 《史记》(Records of the Grand Historian) dataset.

Tests ALL files and outputs detailed results for manual evaluation.

Usage:
    cd narrative-operator-nlp
    uv run python tests/test_shiji_scale.py

Output:
    tests/shiji_results.txt  - Human-readable evaluation report
    tests/shiji_results.json - Machine-readable JSON
"""

import sys
import os
import json
import warnings
import re
import time
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.WARNING)

sys.path.insert(0, '.')

from core.analyzer import analyze

SHIJI_DIR = "/tmp/shiji/zh-tw"
OUTPUT_DIR = "tests"


def extract_sentences(text):
    """Split text into sentences by Chinese punctuation."""
    sentences = re.split(r'[。！？]', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]


def clean_md(text):
    """Remove markdown headers and formatting."""
    text = re.sub(r'^#+\s.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'[*_]{1,3}', '', text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return '\n'.join(lines)


def run_shiji_test():
    import glob

    md_files = sorted(glob.glob(os.path.join(SHIJI_DIR, "*.md")))
    if not md_files:
        print(f"Error: No .md files found in {SHIJI_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    text_report_path = os.path.join(OUTPUT_DIR, "shiji_results.txt")
    json_report_path = os.path.join(OUTPUT_DIR, "shiji_results.json")

    print("=" * 80)
    print(f"《史记》Full Test ({len(md_files)} files) - Manual Evaluation Report")
    print("=" * 80)

    # Global statistics
    total_sentences = 0
    total_entities = 0
    total_relations = 0
    total_pattern_relations = 0
    pattern_types = {}
    error_count = 0
    file_results = []

    all_results = []

    t0 = time.time()

    for fidx, fpath in enumerate(md_files):
        fname = os.path.basename(fpath)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                raw = f.read()
        except Exception as e:
            error_count += 1
            continue

        text = clean_md(raw)
        sentences = extract_sentences(text)

        file_entities = 0
        file_relations = 0
        file_pattern = 0
        file_errors = 0
        sentence_results = []

        for sidx, sent in enumerate(sentences):
            try:
                result = analyze(sent, language='classical')
                total_sentences += 1

                entities = result.content.entities
                relations = result.content.relations
                pattern_rels = [r for r in relations if 'classical_pattern' in r.source]

                file_entities += len(entities)
                file_relations += len(relations)
                file_pattern += len(pattern_rels)

                for r in pattern_rels:
                    pattern_types[r.predicate] = pattern_types.get(r.predicate, 0) + 1

                # Record for manual evaluation
                sent_record = {
                    "sentence": sent,
                    "entities": [(e.text, e.category) for e in entities],
                    "pattern_relations": [
                        f"{r.subject} --[{r.predicate}]--> {r.object}"
                        for r in pattern_rels
                    ],
                    "all_relations": [
                        f"{r.subject} --[{r.predicate}]--> {r.object} ({r.source})"
                        for r in relations
                    ],
                }
                sentence_results.append(sent_record)

            except Exception as e:
                error_count += 1
                file_errors += 1
                sentence_results.append({
                    "sentence": sent,
                    "error": str(e),
                })

        total_entities += file_entities
        total_relations += file_relations
        total_pattern_relations += file_pattern

        elapsed = time.time() - t0
        print(f"  [{fidx+1:3d}/{len(md_files)}] {fname:15s} "
              f"sents={len(sentences):3d} ents={file_entities:4d} "
              f"rels={file_relations:3d} pattern={file_pattern:3d} "
              f"errs={file_errors} ({elapsed:.1f}s)")

        file_results.append({
            "file": fname,
            "sentences": len(sentences),
            "entities": file_entities,
            "relations": file_relations,
            "pattern_relations": file_pattern,
            "errors": file_errors,
        })

        all_results.append({
            "file": fname,
            "sentences": sentence_results,
        })

    # Write text report
    with open(text_report_path, 'w', encoding='utf-8') as f:
        f.write("《史记》古汉语NLP测试报告\n")
        f.write("=" * 80 + "\n\n")

        for fr in file_results:
            f.write(f"\n--- {fr['file']} ---\n")
            f.write(f"  句子: {fr['sentences']}  实体: {fr['entities']}  "
                    f"关系: {fr['relations']}  模式: {fr['pattern_relations']}  "
                    f"错误: {fr['errors']}\n")

        # Detailed sentence-by-sentence output
        for ar in all_results:
            fname = ar['file']
            f.write(f"\n{'=' * 80}\n")
            f.write(f"文件: {fname}\n")
            f.write(f"{'=' * 80}\n")
            for sr in ar['sentences']:
                f.write(f"\n原文: {sr['sentence']}\n")
                if 'error' in sr:
                    f.write(f"  错误: {sr['error']}\n")
                    continue
                if sr['entities']:
                    f.write("  实体:\n")
                    for text, cat in sr['entities']:
                        f.write(f"    [{cat:15s}] {text}\n")
                if sr['pattern_relations']:
                    f.write("  模式关系:\n")
                    for rel in sr['pattern_relations']:
                        f.write(f"    {rel}\n")
                elif sr['all_relations']:
                    f.write("  全部关系:\n")
                    for rel in sr['all_relations']:
                        f.write(f"    {rel}\n")

        # Summary
        f.write(f"\n\n{'=' * 80}\n")
        f.write("汇总\n")
        f.write(f"{'=' * 80}\n")
        f.write(f"文件数:        {len(md_files)}\n")
        f.write(f"句子数:        {total_sentences}\n")
        f.write(f"实体数:        {total_entities}\n")
        f.write(f"关系数:        {total_relations}\n")
        f.write(f"模式关系数:    {total_pattern_relations}\n")
        f.write(f"错误数:        {error_count}\n\n")
        f.write("模式分布:\n")
        for pred, count in sorted(pattern_types.items(), key=lambda x: -x[1]):
            f.write(f"  {pred:20s} {count}\n")

    # Write JSON report
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "files": len(md_files),
                "sentences": total_sentences,
                "entities": total_entities,
                "relations": total_relations,
                "pattern_relations": total_pattern_relations,
                "errors": error_count,
                "pattern_distribution": pattern_types,
            },
            "files": file_results,
            "details": all_results,
        }, f, ensure_ascii=False, indent=2)

    elapsed_total = time.time() - t0

    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(f"Files processed:      {len(md_files)}")
    print(f"Sentences analyzed:   {total_sentences}")
    print(f"Total entities:       {total_entities}")
    print(f"Total relations:      {total_relations}")
    print(f"Pattern relations:    {total_pattern_relations}")
    print(f"Errors:               {error_count}")
    print(f"Time:                 {elapsed_total:.1f}s")
    print()
    print("Pattern distribution:")
    for pred, count in sorted(pattern_types.items(), key=lambda x: -x[1]):
        print(f"  {pred:20s} {count}")
    print(f"{'=' * 80}")
    print()
    print(f"Reports saved to:")
    print(f"  Text:  {text_report_path}")
    print(f"  JSON:  {json_report_path}")


if __name__ == '__main__':
    run_shiji_test()