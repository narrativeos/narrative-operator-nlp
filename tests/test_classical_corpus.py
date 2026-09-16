"""
Tests: Classical Chinese corpus — whole-document language detection.

Regression guard for the two defects found while auditing the demo's
confidence distribution:

  A. ``_apply_mixed_penalty`` misfired on classical texts because 我/是/过/
     可以/所以/其中/所谓 are also everyday classical words, so a strong
     classical document (raw 0.90+) got halved below the threshold and
     routed to the modern model.
  B. ``classical_confidence`` under-scored 骈文/赋体 (parallel prose / fu),
     which is built from balanced equal-length phrases with very few
     function words (阿房宫赋 raw 0.17, 滕王阁序 raw 0.12 before the fix).

The corpus spans pre-Qin -> Ming/Qing and many genres (论/记/序/赋/说/铭/
表/传/书/辞). Every document must be classified as ``classical`` at the
whole-document level, and its raw classical score must clear the threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from core.language_detector import (
    CLASSICAL_THRESHOLD,
    classical_confidence,
    detect_language,
)

# (title, era/author, text) — extended excerpts, one per document.
CLASSICAL_CORPUS: list[tuple[str, str, str]] = [
    ("论语·学而", "先秦", "子曰：学而时习之，不亦说乎？有朋自远方来，不亦乐乎？人不知而不愠，不亦君子乎？曾子曰：吾日三省吾身：为人谋而不忠乎？与朋友交而不信乎？传不习乎？"),
    ("孟子·鱼我所欲也", "先秦", "鱼，我所欲也；熊掌，亦我所欲也。二者不可得兼，舍鱼而取熊掌者也。生，亦我所欲也；义，亦我所欲也。二者不可得兼，舍生而取义者也。生亦我所欲，所欲有甚于生者，故不为苟得也。"),
    ("荀子·劝学", "先秦", "君子曰：学不可以已。青，取之于蓝，而青于蓝；冰，水为之，而寒于水。木直中绳，輮以为轮，其曲中规。虽有槁暴，不复挺者，輮使之然也。"),
    ("大学", "先秦", "大学之道，在明明德，在亲民，在止于至善。知止而后有定，定而后能静，静而后能安，安而后能虑，虑而后能得。物格而后知至，知至而后意诚，意诚而后心正。"),
    ("中庸", "先秦", "天命之谓性，率性之谓道，修道之谓教。道也者，不可须臾离也；可离，非道也。是故君子戒慎乎其所不睹，恐惧乎其所不闻。"),
    ("过秦论", "汉·贾谊", "秦孝公据崤函之固，拥雍州之地，君臣固守以窥周室，有席卷天下，包举宇内，囊括四海之意，并吞八荒之心。当是时也，商君佐之，内立法度，务耕织。"),
    ("报任安书", "汉·司马迁", "盖文王拘而演周易；仲尼厄而作春秋；屈原放逐，乃赋离骚；左丘失明，厥有国语。孙子膑脚，兵法修列；不韦迁蜀，世传吕览。"),
    ("兰亭集序", "晋·王羲之", "永和九年，岁在癸丑，暮春之初，会于会稽山阴之兰亭，修禊事也。群贤毕至，少长咸集。此地有崇山峻岭，茂林修竹；又有清流激湍，映带左右。"),
    ("归去来兮辞", "晋·陶渊明", "归去来兮，田园将芜胡不归？既自以心为形役，奚惆怅而独悲？悟已往之不谏，知来者之可追。实迷途其未远，觉今是而昨非。"),
    ("桃花源记", "晋·陶渊明", "晋太元中，武陵人捕鱼为业。缘溪行，忘路之远近。忽逢桃花林，夹岸数百步，中无杂树，芳草鲜美，落英缤纷。渔人甚异之，复前行，欲穷其林。"),
    ("师说", "唐·韩愈", "古之学者必有师。师者，所以传道受业解惑也。人非生而知之者，孰能无惑？惑而不从师，其为惑也，终不解矣。"),
    ("陋室铭", "唐·刘禹锡", "山不在高，有仙则名。水不在深，有龙则灵。斯是陋室，惟吾德馨。苔痕上阶绿，草色入帘青。谈笑有鸿儒，往来无白丁。"),
    ("岳阳楼记", "宋·范仲淹", "庆历四年春，滕子京谪守巴陵郡。越明年，政通人和，百废具兴，乃重修岳阳楼，增其旧制。予观夫巴陵胜状，在洞庭一湖。衔远山，吞长江，浩浩汤汤，横无际涯。"),
    ("醉翁亭记", "宋·欧阳修", "环滁皆山也。其西南诸峰，林壑尤美，望之蔚然而深秀者，琅琊也。山行六七里，渐闻水声潺潺，而泻出于两峰之间者，酿泉也。"),
    ("前赤壁赋", "宋·苏轼", "壬戌之秋，七月既望，苏子与客泛舟游于赤壁之下。清风徐来，水波不兴。举酒属客，诵明月之诗，歌窈窕之章。"),
    ("阿房宫赋", "唐·杜牧", "六王毕，四海一，蜀山兀，阿房出。覆压三百余里，隔离天日。骊山北构而西折，直走咸阳。二川溶溶，流入宫墙。五步一楼，十步一阁；廊腰缦回，檐牙高啄；各抱地势，钩心斗角。"),
    ("滕王阁序", "唐·王勃", "豫章故郡，洪都新府。星分翼轸，地接衡庐。襟三江而带五湖，控蛮荆而引瓯越。物华天宝，龙光射牛斗之墟；人杰地灵，徐孺下陈蕃之榻。"),
    ("项脊轩志", "明·归有光", "项脊轩，旧南阁子也。室仅方丈，可容一人居。百年老屋，尘泥渗漉，雨泽下注；每移案，顾视无可置者。"),
    ("核舟记", "明·魏学洢", "明有奇巧人曰王叔远，能以径寸之木，为宫室、器皿、人物，以至杯盘、碗碟，罔不因势象形，各具情态。"),
    ("出师表", "三国·诸葛亮", "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。然侍卫之臣不懈于内，忠志之士忘身于外者，盖追先帝之殊遇，欲报之于陛下也。"),
    ("爱莲说", "宋·周敦颐", "水陆草木之花，可爱者甚蕃。晋陶渊明独爱菊。自李唐来，世人甚爱牡丹。予独爱莲之出淤泥而不染，濯清涟而不妖，中通外直，不蔓不枝。"),
    ("逍遥游", "先秦·庄子", "北冥有鱼，其名为鲲。鲲之大，不知其几千里也。化而为鸟，其名为鹏。鹏之背，不知其几千里也。怒而飞，其翼若垂天之云。"),
    ("曹刿论战", "先秦·左传", "十年春，齐师伐我。公将战，曹刿请见。其乡人曰：肉食者谋之，又何间焉？刿曰：肉食者鄙，未能远谋。乃入见。"),
    ("陈涉世家", "汉·司马迁", "陈胜者，阳城人也，字涉。吴广者，阳夏人也，字叔。陈涉少时，尝与人佣耕，辍耕之垄上，怅恨久之，曰：苟富贵，无相忘。"),
]


def _ids() -> list[str]:
    return [title for title, _, _ in CLASSICAL_CORPUS]


# ---------------------------------------------------------------------------
# Per-document: whole-document verdict must be classical
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title,era,text", CLASSICAL_CORPUS, ids=_ids())
def test_whole_document_is_classical(title: str, era: str, text: str) -> None:
    label, conf = detect_language(text)
    raw = classical_confidence(text)
    assert label == "classical", (
        f"{title} ({era}) misclassified as {label!r} "
        f"(label_conf={conf:.3f}, raw_classical={raw:.3f})"
    )
    assert conf >= CLASSICAL_THRESHOLD


@pytest.mark.parametrize("title,era,text", CLASSICAL_CORPUS, ids=_ids())
def test_raw_classical_clears_threshold(title: str, era: str, text: str) -> None:
    raw = classical_confidence(text)
    assert raw >= CLASSICAL_THRESHOLD, (
        f"{title} ({era}) raw classical {raw:.3f} below threshold "
        f"{CLASSICAL_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Aggregate guard: no classical document may be dragged to 'modern'
# ---------------------------------------------------------------------------

def test_no_classical_document_misjudged_modern() -> None:
    misjudged = [
        f"{title}({era})"
        for title, era, text in CLASSICAL_CORPUS
        if detect_language(text)[0] != "classical"
    ]
    assert not misjudged, f"misjudged as non-classical: {misjudged}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
