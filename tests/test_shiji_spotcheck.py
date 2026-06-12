"""
Spot-check test: sample sentences from various 《史记》 chapters for manual evaluation.

Usage:
    cd narrative-operator-nlp
    uv run python tests/test_shiji_spotcheck.py
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

SHIJI_DIR = "/tmp/shiji/zh-tw"


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

    md_files = sorted(glob.glob(os.path.join(SHIJI_DIR, "*.md")))

    # Sample from different chapters
    sample_indices = [0, 10, 20, 40, 60, 80, 100, 120]
    samples = [md_files[i] for i in sample_indices if i < len(md_files)]

    print("=" * 80)
    print("《史记》Spot Check - Manual Evaluation")
    print("=" * 80)

    for fpath in samples:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                raw = f.read()
        except:
            continue

        text = clean_md(raw)
        sentences = extract_sentences(text)

        # Take 5 random sentences from the middle
        start = min(10, len(sentences) // 2)
        batch = sentences[start:start+5]

        print(f"\n{'=' * 80}")
        print(f"文件: {fname} (句子 {start+1}-{start+len(batch)})")
        print(f"{'=' * 80}")

        for i, sent in enumerate(batch):
            print(f"\n--- 句子 {start+i+1} ---")
            print(f"原文: {sent}")

            try:
                result = analyze(sent, language='classical')

                entities = result.content.entities
                pattern_rels = [r for r in result.content.relations if 'classical_pattern' in r.source]
                all_rels = result.content.relations

                if entities:
                    print("实体:")
                    for e in entities:
                        print(f"  [{e.category:15s}] {e.text:10s} conf={e.confidence:.2f}")

                if pattern_rels:
                    print("模式关系:")
                    for r in pattern_rels:
                        print(f"  {r.subject:10s} --[{r.predicate:20s}]--> {r.object:10s}  ({r.source})")

                if all_rels and not pattern_rels:
                    print("关系:")
                    for r in all_rels[:5]:
                        print(f"  {r.subject:10s} --[{r.predicate:20s}]--> {r.object:10s}  ({r.source})")

                if not entities and not all_rels:
                    print("  (无实体/关系)")

            except Exception as e:
                print(f"  ERROR: {e}")


if __name__ == '__main__':
    run_spotcheck()