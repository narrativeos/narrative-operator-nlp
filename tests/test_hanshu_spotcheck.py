"""
Spot-check test for 《汉书》(Book of Han) dataset.

Usage:
    cd narrative-operator-nlp
    uv run python tests/test_hanshu_spotcheck.py
"""

import sys
import os
import warnings
import re
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.WARNING)

sys.path.insert(0, '.')

from core.analyzer import analyze

HANSHU_DIR = "/tmp/hanshu/zh-tw"


def extract_sentences(text):
    sentences = re.split(r'[。！？]', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]


def clean_md(text):
    text = re.sub(r'^#+\s.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'[*_]{1,3}', '', text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return '\n'.join(lines)


def run_spotcheck():
    import glob

    md_files = sorted(glob.glob(os.path.join(HANSHU_DIR, "*.md")))

    print("=" * 80)
    print(f"《汉书》Spot Check ({len(md_files)} files)")
    print("=" * 80)

    total_sentences = 0
    total_entities = 0
    total_relations = 0
    total_pattern = 0
    pattern_types = {}

    for fpath in md_files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                raw = f.read()
        except:
            continue

        text = clean_md(raw)
        sentences = extract_sentences(text)

        # Sample 5 sentences from different parts
        indices = []
        n = len(sentences)
        if n >= 5:
            indices = [0, n//4, n//2, 3*n//4, n-1]
        else:
            indices = list(range(n))

        file_ents = 0
        file_rels = 0
        file_pattern = 0

        for idx in indices:
            sent = sentences[idx]
            try:
                result = analyze(sent, language='classical')
                total_sentences += 1

                ents = result.content.entities
                rels = result.content.relations
                pattern_rels = [r for r in rels if 'classical_pattern' in r.source]

                file_ents += len(ents)
                file_rels += len(rels)
                file_pattern += len(pattern_rels)

                for r in pattern_rels:
                    pattern_types[r.predicate] = pattern_types.get(r.predicate, 0) + 1

                print(f"\n--- {fname} 句子 {idx+1} ---")
                print(f"原文: {sent[:80]}{'...' if len(sent) > 80 else ''}")

                if ents:
                    print("实体:")
                    for e in ents:
                        print(f"  [{e.category:15s}] {e.text:10s} conf={e.confidence:.2f}")

                if pattern_rels:
                    print("模式关系:")
                    for r in pattern_rels:
                        print(f"  {r.subject:10s} --[{r.predicate:20s}]--> {r.object:10s}  ({r.source})")

                if not ents and not pattern_rels:
                    print("  (无实体/模式关系)")

            except Exception as e:
                print(f"\n--- {fname} 句子 {idx+1} ---")
                print(f"原文: {sent[:80]}...")
                print(f"  ERROR: {e}")

        total_entities += file_ents
        total_relations += file_rels
        total_pattern += file_pattern

        print(f"\n[{fname}] ents={file_ents} rels={file_rels} pattern={file_pattern}")

    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(f"Sentences analyzed: {total_sentences}")
    print(f"Total entities:     {total_entities}")
    print(f"Total relations:    {total_relations}")
    print(f"Pattern relations:  {total_pattern}")
    print(f"\nPattern distribution:")
    for pred, count in sorted(pattern_types.items(), key=lambda x: -x[1]):
        print(f"  {pred:20s} {count}")


if __name__ == '__main__':
    run_spotcheck()