"""
Entity Quality Pipeline — F0 / F1 / F2 soft-demotion layers.

Purpose
-------
Reduce entity-extraction noise (NUMBER / STANDARD / PRODUCT / UNKNOWN) without
physically removing anything. Every layer only *soft-demotes*: it sets
``keep=False`` plus ``filter`` / ``filter_reason`` so downstream can audit and
the caller can override.

Layers
------
- **F0** (input cleaning): strip HTML tags from raw text. Always runs at the
  entry point (it is input hygiene, not policy-dependent). TraceView pre-cleans,
  but other callers may send raw HTML.
- **F1** (shape/structure): relabel section references to ``SECTION_REF`` and
  demote bare numbers (no unit/context).
- **F2** (confidence/evidence): per-source confidence thresholds, category
  corroboration gates, and a keyword-injection requirement.

Backward compatibility
----------------------
When ``policy`` is ``None`` (the default for existing callers), F1/F2 are
skipped entirely and every entity keeps ``keep=True`` — behavior is unchanged.
F0 always runs because it only removes markup, never content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .schema import Entity, EntityCategory

# ---------------------------------------------------------------------------
# F0: Input cleaning
# ---------------------------------------------------------------------------

# HTML tags: <tag>, </tag>, <tag attr="...">, self-closing <tag/>
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
# A small set of common HTML entities (avoid a full unescape dependency).
_HTML_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
    "&#39;": "'", "&apos;": "'", "&nbsp;": " ",
    "&mdash;": "—", "&ndash;": "–", "&hellip;": "…",
}


def clean_html(text: str) -> str:
    """F0: Strip HTML tags from raw text, keeping inner text.

    Examples: ``<sub>6</sub>`` → ``6``, ``<sup>1</sup>`` → ``1``,
    ``<br>`` → ```` (removed). This is a safety net for callers that send raw
    HTML; it never removes real content, only markup.
    """
    if not text:
        return text
    cleaned = _HTML_TAG_RE.sub("", text)
    for entity, char in _HTML_ENTITIES.items():
        cleaned = cleaned.replace(entity, char)
    return cleaned


# ---------------------------------------------------------------------------
# Policy data structures
# ---------------------------------------------------------------------------

# Default section-reference patterns (F1). Match chapter/section/clause refs
# such as "4.3.2", "第5章", "12条".
_DEFAULT_SECTION_REF_PATTERNS = [
    r"\d+\.\d+",        # 4.3.2 / 12.1
    r"第\s*\d+",        # 第5章 / 第3条
    r"\d+\s*章",        # 5章
    r"\d+\s*节",        # 3节
    r"\d+\s*条",        # 12条
]


@dataclass
class F1ShapePolicy:
    """F1 shape/structure policy."""
    section_ref_enabled: bool = True
    section_ref_patterns: list[str] = field(
        default_factory=lambda: list(_DEFAULT_SECTION_REF_PATTERNS)
    )
    demote_bare_number: bool = True


@dataclass
class F2ConfidencePolicy:
    """F2 confidence/evidence policy."""
    enabled: bool = True
    min_confidence_default: float = 0.6
    min_confidence_by_source: dict[str, float] = field(default_factory=dict)
    # Categories that require corroboration (evidence) to be kept. PERSON /
    # ORGANIZATION / LOCATION are intentionally NOT here (high-value, never
    # demoted by the confidence/evidence gates).
    demote_categories: list[str] = field(default_factory=lambda: [
        EntityCategory.NUMBER,
        EntityCategory.STANDARD,
        EntityCategory.PRODUCT,
        EntityCategory.UNKNOWN,
    ])
    keyword_require_injected: bool = True


@dataclass
class QualityPolicy:
    """Top-level entity quality policy (F1 + F2)."""
    version: int = 1
    f1_shape: F1ShapePolicy = field(default_factory=F1ShapePolicy)
    f2_confidence: F2ConfidencePolicy = field(default_factory=F2ConfidencePolicy)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["QualityPolicy"]:
        """Parse a policy dict. Returns ``None`` when ``d`` is None/empty so the
        caller can treat "no policy" as "skip F1/F2" (backward compatible)."""
        if not d:
            return None
        policy = cls()
        if "version" in d:
            try:
                policy.version = int(d["version"])
            except (TypeError, ValueError):
                pass

        f1 = d.get("f1_shape") or {}
        if f1:
            sr = f1.get("section_ref") or {}
            if "enabled" in sr:
                policy.f1_shape.section_ref_enabled = bool(sr["enabled"])
            if "patterns" in sr and isinstance(sr["patterns"], list):
                policy.f1_shape.section_ref_patterns = [str(p) for p in sr["patterns"]]
            if "demote_bare_number" in f1:
                policy.f1_shape.demote_bare_number = bool(f1["demote_bare_number"])

        f2 = d.get("f2_confidence") or {}
        if f2:
            if "enabled" in f2:
                policy.f2_confidence.enabled = bool(f2["enabled"])
            mc = f2.get("min_confidence") or {}
            if "default" in mc:
                policy.f2_confidence.min_confidence_default = float(mc["default"])
            if isinstance(mc.get("by_source"), dict):
                policy.f2_confidence.min_confidence_by_source = {
                    str(k): float(v) for k, v in mc["by_source"].items()
                }
            if "demote_categories" in f2 and isinstance(f2["demote_categories"], list):
                policy.f2_confidence.demote_categories = [str(c) for c in f2["demote_categories"]]
            if "keyword_require_injected" in f2:
                policy.f2_confidence.keyword_require_injected = bool(f2["keyword_require_injected"])
        return policy


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BARE_NUMBER_RE = re.compile(r"^\d{1,4}$")

# Standard-code prefixes for STANDARD corroboration (GB/ISO/etc.).
_STANDARD_CODE_RE = re.compile(
    r"^(GB|GB/T|ISO|IEC|ASTM|DIN|JIS|EN|ANSI|IEEE|HG|SH|DL|Q/|T/|JB)",
    re.IGNORECASE,
)


def _is_valid_date(text: str) -> bool:
    """Return True if ``text`` is a plausible date (for NUMBER corroboration)."""
    # 8-digit YYYYMMDD with valid month/day
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", text)
    if m:
        return 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31
    # YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD
    if re.fullmatch(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", text):
        return True
    # 2024年 / 2024年1月 / 2024年1月15日
    if re.fullmatch(r"\d{4}年(?:\d{1,2}月)?(?:\d{1,2}日)?", text):
        return True
    return False


def _has_unit(text: str) -> bool:
    """True when a number carries a unit suffix (%, V, ℃, MΩ, 倍, ...).

    A number with a unit is a meaningful measured value, not a bare number.
    The text must start with a digit (i.e. be a number) to qualify.
    """
    t = text.strip()
    if not t or not t[0].isdigit():
        return False
    stripped = re.sub(r"^[\d.,\s\-+]+", "", t)
    return len(stripped) > 0


def _number_corroborated(ent: Entity) -> bool:
    """NUMBER is corroborated when it is a meaningful measured value:
    produced by the numeric extractor (percentage / quantity / currency), a
    *valid* date, or carrying a unit suffix (%, V, ℃, MΩ, ...). Bare NER
    numbers (no unit) are the primary noise source and are not corroborated."""
    if _has_unit(ent.text):
        return True
    if not ent.source.startswith("numeric/"):
        return False
    if ent.source == "numeric/date":
        return _is_valid_date(ent.text)
    return True


def _standard_corroborated(ent: Entity, injected: set[str]) -> bool:
    """STANDARD is corroborated when it looks like a real standard code
    (GB/ISO/...) or was explicitly injected by the caller."""
    if _STANDARD_CODE_RE.match(ent.text.strip()):
        return True
    return ent.text in injected


def _product_corroborated(ent: Entity, injected: set[str]) -> bool:
    """PRODUCT is corroborated when high-confidence or injected."""
    if ent.confidence >= 0.85:
        return True
    return ent.text in injected


def _unknown_corroborated(ent: Entity, injected: set[str]) -> bool:
    """UNKNOWN is corroborated when high-confidence, injected, a standard code
    (e.g. 'GB50150' mis-tagged as NNP), or carrying a unit suffix (a mis-tagged
    measured value, e.g. '2500V')."""
    if _has_unit(ent.text):
        return True
    if _STANDARD_CODE_RE.match(ent.text.strip()):
        return True
    if ent.confidence >= 0.75:
        return True
    return ent.text in injected


# ---------------------------------------------------------------------------
# F1: Shape / structure
# ---------------------------------------------------------------------------

# Categories that a section-reference relabel may apply to (numeric-ish only).
_SECTION_REF_CANDIDATE_CATEGORIES = frozenset({
    EntityCategory.NUMBER,
    EntityCategory.STANDARD,
    EntityCategory.PARAMETER,
    EntityCategory.UNKNOWN,
})


def apply_f1(entities: list[Entity], policy: QualityPolicy) -> None:
    """F1: shape/structure checks. Modifies entities in-place.

    - section_ref: relabel numeric-ish entities matching a section pattern to
      ``SECTION_REF`` (so they are not treated as value noise).
    - bare_number: demote bare numbers (no unit/context) to ``keep=False``.
    """
    f1 = policy.f1_shape
    compiled = (
        [re.compile(p) for p in f1.section_ref_patterns]
        if f1.section_ref_enabled and f1.section_ref_patterns
        else []
    )

    for ent in entities:
        if not ent.keep:
            continue

        # Section-reference relabeling (only for numeric-ish categories).
        if compiled and ent.category in _SECTION_REF_CANDIDATE_CATEGORIES:
            if any(p.search(ent.text) for p in compiled):
                ent.category = EntityCategory.SECTION_REF
                continue

        # Bare-number demotion.
        if f1.demote_bare_number and ent.category == EntityCategory.NUMBER:
            if _BARE_NUMBER_RE.match(ent.text.strip()):
                ent.keep = False
                ent.filter = "F1_shape"
                ent.filter_reason = f"bare number '{ent.text}' (no unit/context)"


# ---------------------------------------------------------------------------
# F2: Confidence / evidence gates
# ---------------------------------------------------------------------------

def apply_f2(
    entities: list[Entity],
    policy: QualityPolicy,
    injected_keywords: Optional[set[str]] = None,
) -> None:
    """F2: confidence/evidence gates. Modifies entities in-place.

    For categories in ``demote_categories``:
      1. category corroboration gate (NUMBER/STANDARD/PRODUCT/UNKNOWN), then
      2. per-source confidence gate.
    Additionally, when ``keyword_require_injected`` is set, keyword-source
    entities are kept only if their text was injected by the caller.

    PERSON / ORGANIZATION / LOCATION are never touched (not in demote_categories).
    """
    if not policy.f2_confidence.enabled:
        return
    f2 = policy.f2_confidence
    injected = injected_keywords or set()
    demote_cats = set(f2.demote_categories)

    for ent in entities:
        if not ent.keep:
            continue  # already demoted by F1

        # Keyword-injection requirement (applies to any keyword-source entity).
        if f2.keyword_require_injected and ent.source.startswith("keyword"):
            if ent.text not in injected:
                ent.keep = False
                ent.filter = "F2_confidence"
                ent.filter_reason = (
                    f"keyword '{ent.text}' not in injected entity_categories"
                )
                continue

        if ent.category not in demote_cats:
            continue  # high-value / non-demoted category

        # 1. Category corroboration gate.
        if ent.category == EntityCategory.NUMBER:
            if not _number_corroborated(ent):
                ent.keep = False
                ent.filter = "F2_confidence"
                ent.filter_reason = f"NUMBER not corroborated (source={ent.source})"
                continue
        elif ent.category == EntityCategory.STANDARD:
            if not _standard_corroborated(ent, injected):
                ent.keep = False
                ent.filter = "F2_confidence"
                ent.filter_reason = (
                    "STANDARD not corroborated (no standard-code match / not injected)"
                )
                continue
        elif ent.category == EntityCategory.PRODUCT:
            if not _product_corroborated(ent, injected):
                ent.keep = False
                ent.filter = "F2_confidence"
                ent.filter_reason = (
                    f"PRODUCT not corroborated (confidence {ent.confidence:.2f} < 0.85)"
                )
                continue
        elif ent.category == EntityCategory.UNKNOWN:
            if not _unknown_corroborated(ent, injected):
                ent.keep = False
                ent.filter = "F2_confidence"
                ent.filter_reason = (
                    f"UNKNOWN not corroborated (confidence {ent.confidence:.2f} < 0.75)"
                )
                continue

        # 2. Per-source confidence gate.
        #    Measured values with a unit (%, V, ℃, MΩ, ...) and standard codes
        #    (GB50150, ISO ...) carry strong intrinsic evidence, so the
        #    confidence gate is skipped for them.
        if _has_unit(ent.text) or _STANDARD_CODE_RE.match(ent.text.strip()):
            continue
        threshold = f2.min_confidence_by_source.get(
            ent.source, f2.min_confidence_default
        )
        if ent.confidence < threshold:
            ent.keep = False
            ent.filter = "F2_confidence"
            ent.filter_reason = (
                f"confidence {ent.confidence:.2f} < {threshold:.2f} ({ent.source})"
            )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def apply_entity_quality(
    entities: list[Entity],
    policy: Optional[QualityPolicy],
    injected_keywords: Optional[set[str]] = None,
) -> None:
    """Apply F1 + F2 to ``entities`` in-place.

    ``policy`` of ``None`` is a no-op (backward compatible). F0 (``clean_html``)
    is applied to the raw text separately, at the entry point.
    """
    if policy is None:
        return
    apply_f1(entities, policy)
    apply_f2(entities, policy, injected_keywords)


