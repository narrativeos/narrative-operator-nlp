"""End-to-end validation: GB50150-style sample through the quality pipeline.

Runs analyze() with the default policy (F1+F2 enabled) and reports the
kept/filtered breakdown by category and filter reason, plus the noise ratio.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.analyzer import analyze

# Representative GB50150 (电气装置安装工程 电气设备交接试验标准) style text.
# Mixes meaningful values (kept) with noise (low-conf NNP, generic terms).
SAMPLE = (
    "第4章 变压器\n"
    "4.3.2 变压器交接试验应包括下列项目：\n"
    "1 绝缘电阻测量\n"
    "2 直流电阻测量\n"
    "3 绕组变形测试\n"
    "4.3.3 绝缘电阻测量应符合下列规定：\n"
    "1 采用2500V或5000V兆欧表测量，绝缘电阻值不应低于出厂值的70%。\n"
    "2 测量时，绕组温度应在10℃～40℃之间。\n"
    "3 吸收比不应低于1.3。\n"
    "油浸式变压器的绝缘电阻不应低于1000MΩ。\n"
    "干式变压器的直流电阻三相不平衡度不应大于2%。\n"
    "第5章 断路器\n"
    "5.2.1 断路器交接试验应包括下列项目：\n"
    "1 测量绝缘电阻\n"
    "2 测量导电回路电阻\n"
    "3 测量分合闸时间\n"
    "4 测量分合闸同期性\n"
    "5.2.2 断路器导电回路电阻不应大于出厂值的120%。\n"
    "断路器分合闸时间应符合制造厂规定。\n"
    "真空断路器的真空度不应低于标准规定的限值。\n"
    "SF6断路器的SF6气体湿度不应大于150μL/L。\n"
    "第6章 互感器\n"
    "6.1.1 互感器交接试验应包括下列项目：\n"
    "1 测量绝缘电阻\n"
    "2 测量绕组直流电阻\n"
    "3 测量变比\n"
    "4 测量极性\n"
    "6.1.2 电流互感器一次绕组直流电阻不应大于出厂值。\n"
    "电压互感器二次绕组绝缘电阻不应低于10MΩ。\n"
    "互感器变比误差应符合GB/T 20840.1的规定。\n"
    "本标准依据GB50150-2006编制，试验设备应符合相关规范要求。\n"
    "编号63906433的设备应单独记录，试验数据应存档备查。\n"
)


def _cat(e) -> str:
    return e.category.value if hasattr(e.category, "value") else str(e.category)


def main() -> None:
    doc = analyze(
        SAMPLE,
        policy={"f1_shape": {"enabled": True}, "f2_confidence": {"enabled": True}},
        noun_signals={"enabled": True, "pos_whitelist": ["NN", "NR", "NT"], "max_per_block": 8},
    )
    ents = doc.content.entities
    kept = [e for e in ents if e.keep]
    filtered = [e for e in ents if not e.keep]

    print(f"TOTAL entities : {len(ents)}")
    print(f"KEPT           : {len(kept)}")
    print(f"FILTERED       : {len(filtered)}")
    if kept:
        ratio = len(filtered) / len(kept)
        print(f"FILTERED/KEPT  : {ratio:.2%}  (target < 30% means noise is a small minority)")
    print()

    print("=== KEPT by category ===")
    for cat, n in Counter(_cat(e) for e in kept).most_common():
        print(f"  {cat:14s} {n}")
    print()
    print("=== FILTERED by (filter, category) ===")
    for (flt, cat), n in Counter((e.filter, _cat(e)) for e in filtered).most_common():
        print(f"  {str(flt):14s} {cat:14s} {n}")
    print()
    print("=== FILTERED details ===")
    for e in filtered:
        print(f"  [{e.filter}] {e.text!r:20s} {_cat(e):10s} conf={e.confidence:.2f} src={e.source} :: {e.filter_reason}")
    print()
    print("=== KEPT details ===")
    for e in kept:
        print(f"  {e.text!r:20s} {_cat(e):10s} conf={e.confidence:.2f} src={e.source}")

    ns = doc.content.noun_signals
    print()
    print(f"=== NOUN SIGNALS ({len(ns)}) ===")
    for s in ns:
        print(f"  {s.text!r:16s} pos={s.pos:4s} role={s.syntactic_role:8s} score={s.score:.2f} verb={s.evidence.get('governing_verb')}")


if __name__ == "__main__":
    main()
