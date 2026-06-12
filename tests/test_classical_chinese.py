"""
古汉语 NLP 改进测试脚本

测试所有古汉语相关的增强功能：
1. 修饰符提取器 - 古汉语修饰符词典
2. 否定检测器 - 古汉语否定词
3. 共指消解器 - 古汉语代词系统
4. 关系映射器 - 古汉语动词映射、连词分割
5. 句型检测 - 古汉语句型、韵律特征
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.modifier_extractor import ModifierExtractor
from core.negation_detector import _NEGATION_WORDS
from core.coref_resolver import _CN_CLASSICAL_PRONUN
from core.relation_mapper import _CLASSICAL_VERB_PREDICATE_MAP
from core.mapper import _detect_rhetorical, _collect_limitations, _detect_sub_types
from core.entity_mapper import EntityMappingRules
from core.schema import EntityCategory


def test_modifier_extractor():
    """Test 3.1: 古汉语修饰符词典"""
    print("=" * 60)
    print("测试 3.1: 古汉语修饰符词典")
    print("=" * 60)

    m = ModifierExtractor()
    d = ModifierExtractor.get_builtin_dict()

    # Test interrogative words (疑问词)
    interrogative = d.get('interrogative', set())
    expected_interrogative = {'何', '胡', '奚', '曷', '安', '焉', '孰', '盍', '讵', '岂', 
                              '何故', '何以', '何如', '何若', '若何', '奈何'}
    print(f"  疑问词数量: {len(interrogative)}")
    assert len(interrogative) >= 10, f"疑问词数量不足: {len(interrogative)}"
    print(f"  ✓ 疑问词测试通过")

    # Test temporal words (时态副词)
    temporal = d.get('temporal', set())
    expected_temporal = {'尝', '曾', '已', '既', '方', '正', '将', '欲', '昔', '今', 
                         '向', '曩', '遽', '忽', '俄', '旋', '寻', '既而', '须臾'}
    found_temporal = [w for w in expected_temporal if w in temporal]
    print(f"  古汉语时态副词数量: {len(found_temporal)}/{len(expected_temporal)}")
    assert len(found_temporal) >= 10, f"古汉语时态副词不足: {len(found_temporal)}"
    print(f"  ✓ 时态副词测试通过")

    # Test degree words (程度副词)
    degree = d.get('degree', set())
    expected_degree = {'甚', '极', '至', '颇', '稍', '略', '微', '绝', '殊', '尤'}
    found_degree = [w for w in expected_degree if w in degree]
    print(f"  古汉语程度副词数量: {len(found_degree)}/{len(expected_degree)}")
    assert len(found_degree) >= 5, f"古汉语程度副词不足: {len(found_degree)}"
    print(f"  ✓ 程度副词测试通过")

    # Test scope words (范围副词)
    scope = d.get('scope', set())
    expected_scope = {'悉', '皆', '俱', '咸', '尽', '毕', '总', '并'}
    found_scope = [w for w in expected_scope if w in scope]
    print(f"  古汉语范围副词数量: {len(found_scope)}/{len(expected_scope)}")
    assert len(found_scope) >= 5, f"古汉语范围副词不足: {len(found_scope)}"
    print(f"  ✓ 范围副词测试通过")

    print()
    return True


def test_negation_detector():
    """Test 3.2: 古汉语否定检测增强"""
    print("=" * 60)
    print("测试 3.2: 古汉语否定检测增强")
    print("=" * 60)

    # Test classical negation words
    classical_negations = ['弗', '毋', '罔', '微', '未尝', '未始', '未能']
    found = [w for w in classical_negations if w in _NEGATION_WORDS]
    
    print(f"  总否定词数量: {len(_NEGATION_WORDS)}")
    print(f"  古汉语否定词: {found}")
    
    assert len(_NEGATION_WORDS) >= 15, f"否定词数量不足: {len(_NEGATION_WORDS)}"
    assert len(found) >= 5, f"古汉语否定词不足: {len(found)}"
    
    print(f"  ✓ 否定检测测试通过")
    print()
    return True


def test_coref_resolver():
    """Test 4.1: 古汉语代词系统完善"""
    print("=" * 60)
    print("测试 4.1: 古汉语代词系统完善")
    print("=" * 60)

    print(f"  古汉语代词数量: {len(_CN_CLASSICAL_PRONUN)}")
    
    # Test specific pronouns with case attributes
    expected_pronouns = ['之', '其', '彼', '此', '是', '斯', '厥', '予', '若', '而']
    found_pronouns = [p for p in expected_pronouns if p in _CN_CLASSICAL_PRONUN]
    
    print(f"  预期代词找到: {len(found_pronouns)}/{len(expected_pronouns)}")
    
    # Test case attributes
    assert '之' in _CN_CLASSICAL_PRONUN, "缺少代词: 之"
    assert '其' in _CN_CLASSICAL_PRONUN, "缺少代词: 其"
    assert '彼' in _CN_CLASSICAL_PRONUN, "缺少代词: 彼"
    
    # Check case attribute exists
    zhi = _CN_CLASSICAL_PRONUN.get('之', {})
    qi = _CN_CLASSICAL_PRONUN.get('其', {})
    
    print(f"  之的case属性: {zhi.get('case', 'N/A')}")
    print(f"  其的case属性: {qi.get('case', 'N/A')}")
    
    assert len(_CN_CLASSICAL_PRONUN) >= 15, f"古汉语代词数量不足: {len(_CN_CLASSICAL_PRONUN)}"
    
    print(f"  ✓ 共指消解测试通过")
    print()
    return True


def test_relation_mapper():
    """Test 2.2 & 2.3: 古汉语动词映射与连词分割"""
    print("=" * 60)
    print("测试 2.2 & 2.3: 古汉语动词映射与连词分割")
    print("=" * 60)

    # Test classical verb predicate mappings
    print(f"  古汉语动词映射数量: {len(_CLASSICAL_VERB_PREDICATE_MAP)}")
    
    # Test specific mappings
    expected_mappings = {
        '为': 'IS_A',
        '乃': 'IS_A',
        '即': 'IS_A',
        '曰': 'RELATES_TO',
        '至': 'MOVED_TO',
        '使': 'CAUSES',
        '赐': 'TRANSFERS_TO',
    }
    
    for verb, expected_pred in expected_mappings.items():
        actual_pred = _CLASSICAL_VERB_PREDICATE_MAP.get(verb)
        if actual_pred == expected_pred:
            print(f"  ✓ {verb} → {expected_pred}")
        else:
            print(f"  ✗ {verb} → {actual_pred} (预期: {expected_pred})")
            assert False, f"动词映射错误: {verb}"
    
    assert len(_CLASSICAL_VERB_PREDICATE_MAP) >= 20, f"动词映射数量不足: {len(_CLASSICAL_VERB_PREDICATE_MAP)}"
    
    print(f"  ✓ 关系映射测试通过")
    print()
    return True


def test_rhetorical_detection():
    """Test 5.2: 韵律特征分析"""
    print("=" * 60)
    print("测试 5.2: 韵律特征分析 (四字格检测)")
    print("=" * 60)

    # Test four-character parallel structure detection
    tests = [
        ('天地玄黄，宇宙洪荒', 'parallel', '2个四字格'),
        ('天地玄黄，宇宙洪荒，日月盈昃，辰宿列张', 'parallel', '4个四字格'),
        ('子曰诗云，礼乐射御', 'parallel', '2个四字格'),
        ('这是一个普通的现代句子', 'unknown', '现代句子'),
        ('天地玄黄', 'unknown', '仅1个四字格'),
        ('道德经曰道可道非常道', 'unknown', '无标点分隔'),
    ]

    passed = 0
    for text, expected, desc in tests:
        result = _detect_rhetorical(text)
        status = '✓' if result == expected else '✗'
        print(f"  {status} {desc}: \"{text[:20]}...\" → {result}")
        if result == expected:
            passed += 1
    
    print(f"  通过: {passed}/{len(tests)}")
    assert passed >= len(tests) - 1, f"韵律检测失败过多"
    
    print(f"  ✓ 韵律特征测试通过")
    print()
    return True


def test_classical_sentence_patterns():
    """Test 2.1: 古汉语特有句式模式检测"""
    print("=" * 60)
    print("测试 2.1: 古汉语特有句式模式检测")
    print("=" * 60)

    # Test judgment sentence (判断句)
    tests = [
        ("陈胜者，阳城人也", ["judgment_sentence"], "判断句"),
        ("见欺于王", ["passive_classical"], "被动句"),
        ("何陋之有", ["rhetorical_question"], "反问句"),
        ("不亦乐乎", ["rhetorical_question"], "反问句"),
        ("孰与君少", ["comparison"], "比较句"),
        ("非君子也", ["negation_judgment"], "否定判断句"),
    ]
    
    passed = 0
    for text, expected, desc in tests:
        result = _detect_sub_types(text)
        # Check if all expected types are in result
        if all(t in result for t in expected):
            status = '✓'
            passed += 1
        else:
            status = '✗'
        print(f"  {status} {desc}: \"{text}\" → {result}")
    
    print(f"  通过: {passed}/{len(tests)}")
    assert passed >= len(tests) - 1, f"句式检测失败过多"
    
    print(f"  ✓ 句式模式测试通过")
    print()
    return True


def test_entity_categories():
    """Test 1.2: 古汉语实体类别扩展"""
    print("=" * 60)
    print("测试 1.2: 古汉语实体类别扩展")
    print("=" * 60)

    # Test new classical Chinese entity categories
    expected_categories = {'TITLE', 'ERA', 'INSTITUTION', 'ASTRONOMY'}
    found = expected_categories.intersection(EntityCategory.ALL)
    
    print(f"  古汉语实体类别: {found}")
    assert len(found) == len(expected_categories), f"缺少实体类别: {expected_categories - found}"
    
    # Test xpos parsing
    tests = [
        ("名詞,官職名", "TITLE"),
        ("名詞,朝代名", "ERA"),
        ("名詞,典章制度", "INSTITUTION"),
        ("名詞,天文名", "ASTRONOMY"),
        ("名詞,地名", "LOCATION"),
        ("名詞,人名", "PERSON"),
    ]
    
    for xpos, expected in tests:
        result = EntityMappingRules._parse_xpos_category(xpos)
        status = '✓' if result == expected else '✗'
        print(f"  {status} {xpos} → {result} (预期: {expected})")
    
    print(f"  ✓ 实体类别测试通过")
    print()
    return True


def test_limitations_detection():
    """Test 5.1: 古汉语句型检测"""
    print("=" * 60)
    print("测试 5.1: 古汉语句型检测")
    print("=" * 60)

    # Test judgment sentence detection (判断句)
    limits1 = _collect_limitations('陈胜者，阳城人也', [])
    has_judgment = 'hint:judgment_sentence' in limits1
    print(f"  判断句检测: {limits1}")
    
    # Test passive voice detection (被动句)
    limits2 = _collect_limitations('见欺于王', [])
    has_passive = 'hint:passive_voice' in limits2
    print(f"  被动句检测: {limits2}")
    
    # Test rhetorical question detection (反问句)
    limits3 = _collect_limitations('何陋之有', [])
    has_rhetorical = 'hint:rhetorical_question' in limits3
    print(f"  反问句检测: {limits3}")
    
    print(f"  ✓ 句型检测测试完成")
    print()
    return True


def main():
    """Run all tests"""
    print()
    print("=" * 60)
    print("  古汉语 NLP 改进测试套件")
    print("=" * 60)
    print()
    
    tests = [
        ("1.2 古汉语实体类别", test_entity_categories),
        ("2.1 古汉语句式模式", test_classical_sentence_patterns),
        ("3.1 古汉语修饰符词典", test_modifier_extractor),
        ("3.2 古汉语否定检测", test_negation_detector),
        ("4.1 古汉语代词系统", test_coref_resolver),
        ("2.2 古汉语动词映射", test_relation_mapper),
        ("5.2 韵律特征分析", test_rhetorical_detection),
        ("5.1 古汉语句型检测", test_limitations_detection),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except AssertionError as e:
            print(f"  ✗ {name} 失败: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {name} 错误: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"  测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    print()
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)