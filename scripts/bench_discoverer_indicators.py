#!/usr/bin/env python3
"""Benchmark: do CNG / BM25 belong in the new-word-discovery score?

Borrowed from the unsupervised keyphrase literature (termweight: CNG +
TF-IDF + BCBM25 + MI). This script decides, on a realistic single-document
domain corpus, whether folding them into ``core.discoverer`` scoring improves
candidate quality. It compares four scorings over the same candidate pool:

    base     the shipped score (pmi/llr/entropy/cohesion)
    boost    base + (1 - base) * (cng_norm + bm25) / 2   -- additive
    convex   six-way weighted average                    -- diluted
    penalty  base * (1 - 0.15 * (1 - cng_norm))          -- demote only

and additionally sweeps a CNG floor used as a hard filter.

Usage:
    python scripts/bench_discoverer_indicators.py
    python scripts/bench_discoverer_indicators.py --text path/to/corpus.txt

Verdict recorded in core/discoverer.py: none of the four forms is a net win
on a single document, so CNG and BM25 stay diagnostics only.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from core.discoverer import (  # noqa: E402
    _adaptive_min_freq, _adaptive_min_score, _clean_text, _compute_cng,
    _compute_bm25, _compute_cohesion, _compute_entropy_multisentence,
    _compute_llr, _compute_pmi, _contains_function_char, _count_ngrams,
    _effective_entropy, _is_locked_fragment, _norm_cng, _norm_entropy,
    _norm_llr, _norm_pmi, _split_sentences,
)

MAX_LEN = 6

# Terms a domain reader would accept, used only for the top-25 quality probe.
KNOWN_TERMS = [
    "碳纤维", "复合材料", "预浸料", "树脂基体", "高温合金", "涡轮叶片", "玻璃化",
    "层间剪切", "屈服强度", "腐蚀疲劳", "增材制造", "钛合金", "铝合金", "热塑性",
    "孔隙率", "成型工艺", "无损检测", "蠕变",
]


def build_stats(text: str):
    """Mirror discover()'s statistics: sentences, n-grams, char counts."""
    sents = [s for s in (_clean_text(x) for x in _split_sentences(text))
             if len(s) >= 2]
    total = sum(len(s) for s in sents)
    ngram_counts: dict[int, Counter] = {}
    for n in range(2, MAX_LEN + 1):
        counts: Counter = Counter()
        for sent in sents:
            counts.update(_count_ngrams(sent, n))
        ngram_counts[n] = counts
    char_counts: Counter = Counter()
    for sent in sents:
        char_counts.update(sent)
    pos_totals = {
        n: sum(max(len(s) - n + 1, 0) for s in sents)
        for n in range(2, MAX_LEN + 1)
    }
    return sents, ngram_counts, char_counts, pos_totals, total


def score_variants(word, ngram_counts, char_counts, sents, pos_totals, total):
    """Return (base, boost, convex, penalty, cng_norm, freq) for one n-gram."""
    n = len(word)
    freq = ngram_counts.get(n, Counter()).get(word, 0)
    pmi = _compute_pmi(word, freq, ngram_counts, char_counts, total)
    llr = _compute_llr(word, freq, char_counts, total, pos_totals.get(n, 0))
    le, le_n = _compute_entropy_multisentence(word, sents, "left")
    re, re_n = _compute_entropy_multisentence(word, sents, "right")
    coh = _compute_cohesion(word, ngram_counts, char_counts, total)
    ent = _effective_entropy(le, le_n, re, re_n)
    base = (_norm_pmi(pmi) * 0.3 + _norm_llr(llr) * 0.3
            + _norm_entropy(ent) * 0.2 + coh * 0.2)
    cng_norm = _norm_cng(_compute_cng(word, freq, char_counts), total)
    bm25 = _compute_bm25(freq)
    boost = base + (1.0 - base) * (0.5 * cng_norm + 0.5 * bm25)
    convex = (_norm_pmi(pmi) * 0.25 + _norm_llr(llr) * 0.25
              + _norm_entropy(ent) * 0.15 + coh * 0.15
              + cng_norm * 0.075 + bm25 * 0.075)
    penalty = base * (1.0 - 0.15 * (1.0 - cng_norm))
    return base, boost, convex, penalty, cng_norm, freq


def collect(text: str) -> tuple[list[tuple], float, int]:
    """Build the candidate pool the same way discover() filters it."""
    sents, ng, cc, pos, total = build_stats(text)
    min_freq = _adaptive_min_freq(total, 2)
    min_score = _adaptive_min_score(total, 0.5)
    pool = []
    for n in range(2, MAX_LEN + 1):
        for word, freq in ng[n].items():
            if freq < min_freq or _contains_function_char(word):
                continue
            if _is_locked_fragment(word, freq, ng, sents, MAX_LEN):
                continue
            pool.append((word,) + score_variants(word, ng, cc, sents, pos, total))
    return pool, min_score, total


# ~1050 chars of connected domain prose: terms recur naturally and no
# sentence is repeated (repetition would inflate every frequency signal at
# once and make the frequency indicators look better than they are).
DEFAULT_TEXT = (
    "碳纤维复合材料是航空航天结构件的主要承载材料，其比强度和比模量远高于传统金属材料。"
    "预浸料成型工艺中，树脂基体的流变特性决定了浸渍率和孔隙率两个核心参数。"
    "热塑性预浸料可以通过二次加热重复成型，而热固性预浸料一旦固化便不可逆。"
    "自动铺丝设备能够按照纤维轨迹铺设预浸带，适合制造复杂曲面的整体构件。"
    "高温合金仍然广泛用于航空发动机的涡轮盘和涡轮叶片，因为其在高温下的蠕变抗力优异。"
    "陶瓷基复合材料的密度只有高温合金的三分之一，但韧性不足限制了它的工程应用。"
    "玻璃纤维复合材料在风电叶片和汽车车身中的应用增长迅速，主要优势是成本较低。"
    "树脂基体的玻璃化转变温度决定了复合材料构件的最高使用温度。"
    "层间剪切强度是评价碳纤维复合材料制件质量的重要指标，通常由超声检测评估。"
    "铝合金属于轻质金属材料，通过添加锌镁铜等合金元素可以显著提高其屈服强度。"
    "腐蚀疲劳是海洋环境下金属结构件的主要失效形式，需要涂层防护和阴极保护联合措施。"
    "增材制造技术可以直接成形复杂结构的钛合金零件，缩短研制周期并降低材料去除量。"
    "复合材料的固化收缩会引起残余应力，进而影响构件的尺寸精度和层间性能。"
    "树脂基体的增韧改性通常采用热塑性颗粒或核壳橡胶粒子，代价是耐温性有所下降。"
    "预浸料的储存寿命受环境温度和湿度影响显著，需要在低温冷库中保存并严格控制回温时间。"
    "热压罐成型是目前复合材料制件最常用的工艺，通过温度和压力的协同曲线控制孔隙率。"
    "缠绕成型工艺主要用于制造压力容器和管道，纤维缠绕角度直接影响容器的承压能力。"
    "复合材料的损伤形式包括基体开裂、纤维断裂、分层和脱粘，其中分层最隐蔽也最危险。"
    "无损检测手段中，超声C扫描对分层缺陷敏感，X射线计算机断层成像适合观察内部孔隙。"
    "高温合金的冶炼通常采用真空感应加电渣重熔的双联工艺，以保证合金的纯净度和组织均匀性。"
    "单晶高温合金通过定向凝固消除晶界，使涡轮叶片的承温能力提高了约一百摄氏度。"
    "钛合金的比强度在金属材料中名列前茅，但它的切削加工困难且摩擦系数偏高。"
    "金属基复合材料以铝或镁为基体，加入碳化硅颗粒或碳纤维后耐磨性和刚度明显改善。"
    "增材制造的微观组织呈明显的外延生长特征，后续热处理可以消除部分各向异性。"
    "粉末冶金的流程包括制粉、筛分、装套、热等静压和锻造，粉末氧含量是关键的工艺参数。"
    "复合材料的连接方式有机械连接和胶接两种，钻孔会切断纤维并造成局部强度损失。"
    "结构胶的固化工艺对表面处理质量敏感，通常要求接触角小于某个临界值以保证润湿。"
    "航空结构件的疲劳试验通常在谱载条件下进行，用疲劳寿命分散系数评价结构安全性。"
    "蠕变试验数据是高温合金部件寿命评估的基础，Larson-Miller参数法在工程中应用广泛。"
    "复合材料的雷击防护需要在表面铺设铜网或金属网，否则树脂基体会因放电而分层剥落。"
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure CNG/BM25 integration variants for discoverer scoring.")
    ap.add_argument("--text", help="UTF-8 file to use as the corpus")
    args = ap.parse_args()

    text = Path(args.text).read_text(encoding="utf-8") if args.text else DEFAULT_TEXT
    pool, min_score, total = collect(text)
    if not pool:
        print("No candidates: corpus too short for freq >= min_freq "
              "(use >= ~1000 chars).")
        return 1

    base_set = {t[0] for t in pool if t[1] >= min_score}
    print(f"corpus={total} chars | pool={len(pool)} | min_score={min_score} "
          f"| base passes={len(base_set)}\n")

    for label, idx in (("boost", 2), ("convex", 3), ("penalty", 4)):
        passed = {t[0] for t in pool if t[idx] >= min_score}
        print(f"[{label:7s}] pass={len(passed):4d}  "
              f"+{len(passed - base_set):3d} {sorted(passed - base_set)[:6]}  "
              f"-{len(base_set - passed):3d} {sorted(base_set - passed)[:6]}")

    print()
    for floor in (0.10, 0.15, 0.20, 0.25):
        kept = {t[0] for t in pool if t[1] >= min_score and t[5] >= floor}
        lost = sorted(base_set - kept)
        print(f"[cng floor {floor:.2f}] keep={len(kept):4d} reject={len(lost):3d} "
              f"{lost[:8]}")

    print()
    for label, idx in (("base", 1), ("boost", 2), ("convex", 3), ("penalty", 4)):
        ranked = sorted(pool, key=lambda t: -t[idx])
        top = [t[0] for t in ranked[:25]]
        hit = [w for w in KNOWN_TERMS if w in top]
        print(f"[{label:7s}] top25 known-term hits {len(hit):2d}/{len(KNOWN_TERMS)}: "
              f"{' '.join(hit)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
