"""
Test script for Classical Pattern Extractor.

Usage:
    cd narrative-operator-nlp
    uv run python tests/test_classical_pattern.py
"""

import sys
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.WARNING)

sys.path.insert(0, '.')

from core.analyzer import analyze

def test_main():
    text = '陈胜者，阳城人也，字涉。吴广者，阳夏人也，字叔。陈涉少时，尝与人佣耕。'
    print(f'=== Testing: {text}')
    result = analyze(text, language='classical')

    print()
    print('Entities:')
    for e in result.content.entities:
        print(f'  [{e.category:15s}] {e.text:8s} conf={e.confidence:.2f}')

    print()
    print('Relations (classical_pattern only):')
    for r in result.content.relations:
        if 'classical_pattern' in r.source:
            pv = r.predicate_verb or ''
            print(f'  {r.subject:8s} --[{r.predicate:20s}/{pv}]--> {r.object:8s} ({r.source})')

    print()
    print('Coreferences (high quality):')
    for c in result.content.coreferences:
        if c.quality_flag == 'high':
            print(f'  {c.representative} -> {[m.text for m in c.mentions]}')

    print()
    print('=== Test 2: 沛公军霸上 ===')
    text2 = '沛公军霸上'
    result2 = analyze(text2, language='classical')
    for e in result2.content.entities:
        print(f'  [{e.category:15s}] {e.text}')
    for r in result2.content.relations:
        if 'classical_pattern' in r.source:
            print(f'  {r.subject} --[{r.predicate}]--> {r.object} ({r.source})')

    print()
    print('=== Test 3: 拜为郎中 ===')
    text3 = '张良拜为郎中'
    result3 = analyze(text3, language='classical')
    for e in result3.content.entities:
        print(f'  [{e.category:15s}] {e.text}')
    for r in result3.content.relations:
        if 'classical_pattern' in r.source:
            print(f'  {r.subject} --[{r.predicate}]--> {r.object} ({r.source})')

    print()
    print('DONE')


if __name__ == '__main__':
    test_main()