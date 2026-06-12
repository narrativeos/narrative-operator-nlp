"""
Demo: Classical Chinese NLP Processing

Demonstrates the improvements made to classical Chinese (文言文) processing:
- Entity categories: TITLE, ERA, INSTITUTION, ASTRONOMY
- Sentence pattern detection
- Modifier extraction
- Negation detection
- Coreference resolution
- Rhythm analysis

Usage:
    python examples/demo_classical_chinese.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.modifier_extractor import ModifierExtractor
from core.negation_detector import _NEGATION_WORDS, find_negation_spans
from core.coref_resolver import _CN_CLASSICAL_PRONUN
from core.relation_mapper import _CLASSICAL_VERB_PREDICATE_MAP
from core.mapper import _detect_rhetorical, _collect_limitations, _detect_sub_types
from core.entity_mapper import EntityMappingRules
from core.schema import EntityCategory


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def demo_entity_categories():
    """Demo: Extended entity categories for classical Chinese."""
    print_section("1. 古汉语实体类别扩展")
    
    # Test new classical Chinese entity categories
    expected_categories = {'TITLE', 'ERA', 'INSTITUTION', 'ASTRONOMY'}
    found = expected_categories.intersection(EntityCategory.ALL)
    
    print(f"  古汉语实体类别: {found}")
    
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
        status = "✓" if result == expected else "✗"
        print(f"  {status} {xpos} → {result} (预期: {expected})")


def demo_sentence_patterns():
    """Demo: Classical Chinese sentence pattern detection."""
    print_section("2. 古汉语特有句式模式检测")
    
    tests = [
        ("陈胜者，阳城人也", ["judgment_sentence"], "判断句"),
        ("见欺于王", ["passive_classical"], "被动句"),
        ("何陋之有", ["rhetorical_question"], "反问句"),
        ("不亦乐乎", ["rhetorical_question"], "反问句"),
        ("孰与君少", ["comparison"], "比较句"),
        ("非君子也", ["negation_judgment"], "否定判断句"),
    ]
    
    for text, expected, desc in tests:
        result = _detect_sub_types(text)
        if all(t in result for t in expected):
            status = "✓"
        else:
            status = "✗"
        print(f"  {status} {desc}: \"{text}\" → {result}")


def demo_modifiers():
    """Demo: Classical Chinese modifier extraction."""
    print_section("3. 古汉语修饰符词典")
    
    m = ModifierExtractor()
    d = ModifierExtractor.get_builtin_dict()
    
    # Test interrogative words (疑问词)
    interrogative = d.get('interrogative', set())
    print(f"  疑问词数量: {len(interrogative)}")
    
    # Test temporal words (时态副词)
    temporal = d.get('temporal', set())
    expected_temporal = {'尝', '曾', '已', '既', '方', '正', '将', '欲', '昔', '今'}
    found_temporal = [w for w in expected_temporal if w in temporal]
    print(f"  时态副词: {len(found_temporal)}/{len(expected_temporal)} 找到")
    
    # Test degree words (程度副词)
    degree = d.get('degree', set())
    expected_degree = {'甚', '极', '至', '颇', '稍', '略', '微', '绝', '殊', '尤'}
    found_degree = [w for w in expected_degree if w in degree]
    print(f"  程度副词: {len(found_degree)}/{len(expected_degree)} 找到")
    
    # Test scope words (范围副词)
    scope = d.get('scope', set())
    expected_scope = {'悉', '皆', '俱', '咸', '尽', '毕', '总', '并'}
    found_scope = [w for w in expected_scope if w in scope]
    print(f"  范围副词: {len(found_scope)}/{len(expected_scope)} 找到")


def demo_negation():
    """Demo: Classical Chinese negation detection."""
    print_section("4. 古汉语否定检测增强")
    
    # Test classical negation words
    classical_negations = ['弗', '毋', '罔', '微', '未尝', '未始', '未能']
    found = [w for w in classical_negations if w in _NEGATION_WORDS]
    
    print(f"  总否定词数量: {len(_NEGATION_WORDS)}")
    print(f"  古汉语否定词: {found}")
    
    # Test negation detection in text
    test_text = "未尝不叹息而痛恨于"
    spans = find_negation_spans(test_text)
    print(f"  文本: \"{test_text}\"")
    print(f"  否定区间: {spans}")


def demo_coreference():
    """Demo: Classical Chinese coreference resolution."""
    print_section("5. 古汉语代词系统完善")
    
    print(f"  古汉语代词数量: {len(_CN_CLASSICAL_PRONUN)}")
    
    # Test specific pronouns with case attributes
    expected_pronouns = ['之', '其', '彼', '此', '是', '斯', '厥', '予', '若', '而']
    found_pronouns = [p for p in expected_pronouns if p in _CN_CLASSICAL_PRONUN]
    
    print(f"  预期代词找到: {len(found_pronouns)}/{len(expected_pronouns)}")
    
    # Check case attributes
    zhi = _CN_CLASSICAL_PRONUN.get('之', {})
    qi = _CN_CLASSICAL_PRONUN.get('其', {})
    
    print(f"  之的case属性: {zhi.get('case', 'N/A')}")
    print(f"  其的case属性: {qi.get('case', 'N/A')}")


def demo_relations():
    """Demo: Classical Chinese verb-to-predicate mapping."""
    print_section("6. 古汉语动词映射与关系提取")
    
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


def demo_rhythm():
    """Demo: Rhythm feature analysis (four-character patterns)."""
    print_section("7. 韵律特征分析 (四字格检测)")
    
    tests = [
        ('天地玄黄，宇宙洪荒', 'parallel', '2个四字格'),
        ('天地玄黄，宇宙洪荒，日月盈昃，辰宿列张', 'parallel', '4个四字格'),
        ('子曰诗云，礼乐射御', 'parallel', '2个四字格'),
        ('这是一个普通的现代句子', 'unknown', '现代句子'),
    ]
    
    for text, expected, desc in tests:
        result = _detect_rhetorical(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {desc}: \"{text[:20]}...\" → {result}")


def demo_limitations():
    """Demo: Sentence type detection."""
    print_section("8. 古汉语句型检测")
    
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


def demo_full_analysis():
    """Demo: Full analysis of a classical Chinese text."""
    print_section("9. 完整分析示例")
    
    # Sample classical Chinese text
    text = "陈胜者，阳城人也，字涉。吴广者，阳夏人也，字叔。陈涉少时，尝与人佣耕，辍耕之垄上，怅恨久之，曰：'苟富贵，无相忘。'"
    
    print(f"  原文: {text}")
    print(f"  长度: {len(text)} 字符")
    
    # Show what would be extracted
    print(f"\n  预期提取:")
    print(f"    - 人物: 陈胜, 吴广")
    print(f"    - 地名: 阳城, 阳夏")
    print(f"    - 句式: 判断句 ('...者，...也')")
    print(f"    - 修饰符: 尝 (时态副词)")
    print(f"    - 动词: 佣耕, 辍耕, 怅恨")


def main():
    print("="*60)
    print("  Narrative Operator NLP — 古汉语处理 Demo")
    print("="*60)
    
    # Run all demos
    demo_entity_categories()
    demo_sentence_patterns()
    demo_modifiers()
    demo_negation()
    demo_coreference()
    demo_relations()
    demo_rhythm()
    demo_limitations()
    demo_full_analysis()
    
    print(f"\n{'='*60}")
    print("  Demo 完成")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()