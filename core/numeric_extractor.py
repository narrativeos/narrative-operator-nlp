"""
Numeric Entity Extraction — Detect dates, amounts, percentages as NUMBER entities.

Uses regex patterns to find numeric expressions in text that HanLP NER
may miss. This is a lightweight, rule-based approach that complements
the NER pipeline.

Patterns supported:
- Dates: 2024年, 2024-01-15, 2024/01/15, 2024.01.15
- Amounts: 100元, 5000美元, 3.5万元, ¥100, $500
- Percentages: 50%, 3.14%
- Quantities: 100个, 500吨, 2000人, 1000万
"""

from __future__ import annotations

import re
from typing import Optional

from .schema import Entity


# ── Pattern definitions ──

# Date patterns
_DATE_PATTERNS = [
    # 2024年1月15日 / 2024年1月 / 2024年
    re.compile(r"\d{4}年(?:\d{1,2}月)?(?:\d{1,2}日)?"),
    # 2024-01-15 / 2024-1-5
    re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"),
    # 2024.01.15
    re.compile(r"\d{4}\.\d{1,2}\.\d{1,2}"),
    # 20240115 (8-digit date) — validated: month 01-12, day 01-31, and the
    # 8 digits must be standalone (not part of a longer digit run). This fixes
    # the bug where any 8-digit string (e.g. an ID like 63906433) was treated
    # as a date.
    re.compile(r"(?<!\d)\d{4}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)"),
]

# Currency patterns
_CURRENCY_PATTERNS = [
    # 100元 / 5000美元 / 3.5万元 / 100欧元
    re.compile(r"\d+(?:\.\d+)?(?:万|亿|千万)?(?:元|美元|欧元|英镑|日元|人民币|万元|亿美元)"),
    # ¥100 / $500 / €300 / £200
    re.compile(r"[¥$€£]\d+(?:\.\d+)?(?:万|亿|千万)?"),
]

# Percentage patterns
_PERCENTAGE_PATTERNS = [
    # 50% / 3.14% / 100%
    re.compile(r"\d+(?:\.\d+)?%"),
]

# Quantity patterns (number + measure word)
_QUANTITY_PATTERNS = [
    # 100个 / 500吨 / 2000人 / 1000万 / 3.5亿
    re.compile(r"\d+(?:\.\d+)?(?:万|亿|千万|个|件|台|辆|艘|张|把|支|根|条|块|片|份|次|回|年|月|天|小时|分钟|秒|米|公里|厘米|毫米|克|千克|吨|升|毫升|度|摄氏度|人|口|家|所|座|栋|层|楼|页|章|节|篇|首|首|幅|张|部|集|卷|册|本|卷|集|期|号|班|组|队|团|军|师|团|营|连|排|班|人|名|位|位|位|位)"),
]


def _normalize_date(text: str) -> str:
    """Convert Chinese date expression to ISO format (YYYY-MM-DD / YYYY-MM / YYYY).
    
    Supports:
    - 2024年1月15日 → 2024-01-15
    - 2024年1月 → 2024-01
    - 2024年 → 2024
    - 2024-01-15 → 2024-01-15 (already ISO)
    - 2024/01/15 → 2024-01-15
    - 2024.01.15 → 2024-01-15
    - 20240115 → 2024-01-15
    """
    # 2024年1月15日 (must match 日 to avoid group(3) being None)
    m = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    
    # 2024年1月
    m = re.match(r"(\d{4})年(\d{1,2})月", text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    
    # 2024年
    m = re.match(r"(\d{4})年", text)
    if m:
        return m.group(1)
    
    # 2024-01-15 (already ISO)
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    
    # 2024/01/15 → 2024-01-15
    m = re.match(r"(\d{4})[/.](\d{1,2})[/.](\d{1,2})", text)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    
    # 20240115 → 2024-01-15 (with validation)
    m = re.match(r"(\d{4})(\d{2})(\d{2})", text)
    if m:
        month = int(m.group(2))
        day = int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        # Invalid date: fall through to return original
    
    return text  # Fallback: return original


def extract_numeric_entities(
    text: str,
    existing_spans: set[tuple[int, int]],
    id_gen,
) -> list[Entity]:
    """Extract numeric entities from text.

    Args:
        text: Original text.
        existing_spans: Set of (start, end) spans already covered by other entities.
        id_gen: EntityIdGenerator instance.

    Returns:
        List of NUMBER entities.
    """
    entities: list[Entity] = []

    for pattern_group, source_name in [
        (_DATE_PATTERNS, "numeric/date"),
        (_CURRENCY_PATTERNS, "numeric/currency"),
        (_PERCENTAGE_PATTERNS, "numeric/percentage"),
        (_QUANTITY_PATTERNS, "numeric/quantity"),
    ]:
        for pattern in pattern_group:
            for match in pattern.finditer(text):
                span = (match.start(), match.end())
                # Skip if already covered by another entity
                if _overlaps(span, existing_spans):
                    continue
                ent_text = match.group()
                ent_id = id_gen.generate(ent_text, span, "NUMBER")
                if ent_id is None:
                    continue
                # Step 7: Normalize date expressions
                normalized = _normalize_date(ent_text) if source_name == "numeric/date" else ent_text
                entities.append(Entity(
                    id=ent_id,
                    text=ent_text,
                    category="NUMBER",
                    span=span,
                    normalized=normalized,
                    source=source_name,
                    confidence=0.75,
                ))
                existing_spans.add(span)

    entities.sort(key=lambda e: e.span[0])
    return entities


def _overlaps(span: tuple[int, int], existing: set[tuple[int, int]]) -> bool:
    """Check if a span overlaps with any existing span."""
    s, e = span
    for es, ee in existing:
        if s < ee and e > es:
            return True
    return False