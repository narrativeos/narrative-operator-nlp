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

# Test cases: (text, expected_relations, forbidden_relations)
# Each expected/forbidden relation is (subject_contains, predicate, object_contains)
# "expect": relations that MUST be extracted
# "forbid": relations that MUST NOT be extracted (false-positive guards)
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
            # "化而为鸟，其名为鹏" — 其 refers to 鸟 (the bird), whose name is 鹏
            ("鸟", "HAS_PROPERTY", "鹏"),
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
            # "化而为鸟，其名为鹏" — 其 refers to 鸟 (the bird), whose name is 鹏
            ("鸟", "HAS_PROPERTY", "鹏"),
        ]
    },
    # 12. 被动 (见V于X): agent CAUSES patient
    {
        "name": "被动_见于",
        "text": "廉颇见欺于秦王。",
        "expect": [
            ("秦王", "CAUSES", "廉颇"),
        ]
    },
    # 13. 表字: 字后接逗号
    {
        "name": "表字_字后接逗号",
        "text": "孔子字仲尼，鲁人也。",
        "expect": [
            ("孔子", "HAS_STYLE_NAME", "仲尼"),
        ]
    },
    # 14. 负例: 拜送 = 跪拜相送, 非官职任命
    {
        "name": "负例_拜送",
        "text": "燕王拜送于庭。",
        "expect": [],
        "forbid": [
            ("燕王", "HAS_TITLE", "送"),
        ]
    },
    # 15. 负例: 军 = 名词(军队), 长句不应产生 LOCATED_AT
    {
        "name": "负例_夜驰沛公军",
        "text": "项伯乃夜驰之沛公军，私见张良，具告以事。",
        "expect": [],
        "forbid": [
            ("项伯", "LOCATED_AT", "张良"),
        ]
    },
    # 16. 负例: 跨句 者也 不应把 陈胜 绑定到 吴广/阳夏
    {
        "name": "负例_跨句者也",
        "text": "陈胜者，阳城人。吴广者，阳夏人也。",
        "expect": [
            ("吴广", "LOCATED_AT", "阳夏"),
        ],
        "forbid": [
            ("陈胜", "LOCATED_AT", "吴广"),
            ("陈胜", "LOCATED_AT", "阳夏"),
        ]
    },
    # 17. 负例: 见 = 看见, 非被动
    {
        "name": "负例_见渔人",
        "text": "晋人见渔人于潭中。",
        "expect": [],
        "forbid": [
            ("晋人", "CAUSES", "渔人"),
        ]
    },
    # 18. 负例: 年号 = 年代名, 非表号
    {
        "name": "负例_年号",
        "text": "时年号建武，天下初定。",
        "expect": [],
        "forbid": [
            ("年", "HAS_STYLE_NAME", "建武"),
        ]
    },
]


def run_benchmark():
    total_cases = len(TEST_CASES)
    total_expectations = 0
    total_passed = 0
    total_failed = 0
    total_forbids = 0
    id_conflicts = 0

    print("=" * 80)
    print("Classical Pattern Extractor Benchmark")
    print("=" * 80)

    for case in TEST_CASES:
        name = case["name"]
        text = case["text"]
        expect = case["expect"]
        forbid = case.get("forbid", [])

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

        # Check forbidden relations (false-positive guards)
        for fb_subj, fb_pred, fb_obj in forbid:
            total_forbids += 1
            leaked = False
            for r in pattern_rels:
                if (fb_subj in r.subject and r.predicate == fb_pred and fb_obj in r.object):
                    leaked = True
                    break
            if leaked:
                print(f"  ❌ FAIL: forbidden ({fb_subj}, {fb_pred}, {fb_obj}) was extracted")
                total_failed += 1
            else:
                print(f"  ✅ PASS: ({fb_subj}, {fb_pred}, {fb_obj}) correctly absent")
                total_passed += 1

        if not expect and not forbid:
            print(f"  (No specific expectations - visual inspection only)")

        # Check relation ID uniqueness across the whole document
        # (pattern extractor IDs must not collide with the main pipeline)
        all_rel_ids = [r.id for r in result.content.relations]
        dupes = sorted({i for i in all_rel_ids if all_rel_ids.count(i) > 1})
        bad_prefix = [r.id for r in pattern_rels if not r.id.startswith("rel_cp_")]
        if dupes or bad_prefix:
            id_conflicts += 1
            if dupes:
                print(f"  ❌ FAIL: duplicate relation IDs: {dupes}")
            if bad_prefix:
                print(f"  ❌ FAIL: pattern relations without rel_cp_ prefix: {bad_prefix}")

    # Summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(f"Test cases:     {total_cases}")
    print(f"Expectations:   {total_expectations + total_forbids} (incl. {total_forbids} forbidden)")
    print(f"Passed:         {total_passed} ({100*total_passed//max(total_expectations + total_forbids,1)}%)")
    print(f"Failed:         {total_failed}")
    print(f"ID conflicts:   {id_conflicts}")
    print(f"{'=' * 80}")

    return total_failed == 0 and id_conflicts == 0


if __name__ == '__main__':
    import sys
    ok = run_benchmark()
    sys.exit(0 if ok else 1)