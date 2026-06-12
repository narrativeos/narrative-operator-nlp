"""
Systematic benchmark for Classical Pattern Extractor.

Tests various classical Chinese sentence patterns to evaluate coverage and accuracy.

Usage:
    cd narrative-operator-nlp
    uv run python tests/test_classical_benchmark.py
"""

import sys
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.WARNING)

sys.path.insert(0, '.')

from core.analyzer import analyze

# Test cases: (text, expected_relations)
# Each expected_relation is (subject_contains, predicate, object_contains)
TEST_CASES = [
    # 1. 判断句
    {
        "name": "判断句_者也",
        "text": "陈胜者，阳城人也，字涉。",
        "expect": [
            ("陈胜", "LOCATED_AT", "阳城"),
            ("陈胜", "HAS_STYLE_NAME", "涉"),
        ]
    },
    # 2. 命名
    {
        "name": "命名_其名为",
        "text": "北冥有鱼，其名为鲲。",
        "expect": [
            ("北冥", "HAS_PROPERTY", "鱼"),
        ]
    },
    # 3. 变化 (requires full context for pronoun resolution)
    {
        "name": "变化_化而为",
        "text": "北冥有鱼，其名为鲲。鲲之大，不知其几千里也。化而为鸟，其名为鹏。",
        "expect": [
            ("北冥", "HAS_PROPERTY", "鱼"),
            ("鲲", "TRANSFERS_TO", "鸟"),
            ("鲲", "HAS_PROPERTY", "鹏"),
        ]
    },
    # 4. 官职
    {
        "name": "官职_拜为",
        "text": "张良拜为郎中。",
        "expect": [
            ("张良", "HAS_TITLE", "郎中"),
        ]
    },
    # 5. 对话
    {
        "name": "对话_曰",
        "text": "孔子曰：学而时习之，不亦说乎。",
        "expect": [
            ("孔子", "SAYS", "学而时习之"),
        ]
    },
    # 6. 移动
    {
        "name": "移动_至",
        "text": "沛公至霸上。",
        "expect": [
            ("沛公", "MOVED_TO", "霸上"),
        ]
    },
    # 7. 地理位置
    {
        "name": "位置_军",
        "text": "沛公军霸上。",
        "expect": [
            ("沛公", "LOCATED_AT", "霸上"),
        ]
    },
    # 8. 被动
    {
        "name": "被动_为所",
        "text": "吾属今为之所虏。",
        "expect": [
            # This is a passive construction
        ]
    },
    # 9. 死亡
    {
        "name": "死亡_卒",
        "text": "廉颇卒于寿春。",
        "expect": [
            ("廉颇", "DIED_AT", "寿春"),
        ]
    },
    # 10. 综合测试
    {
        "name": "综合_史记陈胜传",
        "text": "陈胜者，阳城人也，字涉。吴广者，阳夏人也，字叔。陈涉少时，尝与人佣耕。",
        "expect": [
            ("陈胜", "LOCATED_AT", "阳城"),
            ("吴广", "LOCATED_AT", "阳夏"),
            ("陈胜", "HAS_STYLE_NAME", "涉"),
            ("吴广", "HAS_STYLE_NAME", "叔"),
        ]
    },
    # 11. 逍遥游全文片段
    {
        "name": "综合_逍遥游",
        "text": "北冥有鱼，其名为鲲。鲲之大，不知其几千里也。化而为鸟，其名为鹏。",
        "expect": [
            ("北冥", "HAS_PROPERTY", "鱼"),
            ("鲲", "TRANSFERS_TO", "鸟"),
            ("鲲", "HAS_PROPERTY", "鹏"),
        ]
    },
]


def run_benchmark():
    total_cases = len(TEST_CASES)
    total_expectations = 0
    total_passed = 0
    total_failed = 0

    print("=" * 80)
    print("Classical Pattern Extractor Benchmark")
    print("=" * 80)

    for case in TEST_CASES:
        name = case["name"]
        text = case["text"]
        expect = case["expect"]

        print(f"\n{'─' * 60}")
        print(f"Test: {name}")
        print(f"Text: {text}")
        print()

        result = analyze(text, language='classical')

        # Get classical_pattern relations
        pattern_rels = [r for r in result.content.relations if 'classical_pattern' in r.source]

        print("Extracted relations:")
        for r in pattern_rels:
            print(f"  {r.subject:10s} --[{r.predicate:20s}]--> {r.object}")
        print()

        # Check expectations
        case_passed = True
        for exp_subj, exp_pred, exp_obj in expect:
            total_expectations += 1
            found = False
            for r in pattern_rels:
                if (exp_subj in r.subject and r.predicate == exp_pred and exp_obj in r.object):
                    found = True
                    break
            if found:
                print(f"  ✅ PASS: ({exp_subj}, {exp_pred}, {exp_obj})")
                total_passed += 1
            else:
                print(f"  ❌ FAIL: ({exp_subj}, {exp_pred}, {exp_obj}) not found")
                total_failed += 1
                case_passed = False

        if not expect:
            print(f"  (No specific expectations - visual inspection only)")

    # Summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(f"Test cases:     {total_cases}")
    print(f"Expectations:   {total_expectations}")
    print(f"Passed:         {total_passed} ({100*total_passed//max(total_expectations,1)}%)")
    print(f"Failed:         {total_failed}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    run_benchmark()