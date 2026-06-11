"""
Keyword Entity Extractor — Domain keyword-based entity discovery.

Loads keyword configuration from YAML files, supporting both built-in and
user-defined domain keywords passed at runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from .schema import Entity, EntityCategory, Token

logger = logging.getLogger(__name__)


class KeywordExtractor:
    """Extracts entities from domain keywords not caught by NER models.

    Loads keywords from config/domain_keywords.yaml for built-in categories
    (MATERIAL, STANDARD, PARAMETER), and supports runtime injection of
    user-defined domain keywords via add_keywords().

    Usage:
        extractor = KeywordExtractor()
        extractor.add_keywords({"DISEASE": ["乳腺癌", "肿瘤"]})  # user-defined
        entities = extractor.extract(tokens, existing_entities, text, id_gen)
    """

    # Built-in YAML key → EntityCategory mapping
    _CATEGORY_MAP = {
        "material": "MATERIAL",
        "standard": "STANDARD",
        "parameter": "PARAMETER",
    }

    def __init__(self, config_path: Optional[str] = None):
        self._keywords: dict[str, frozenset] = {
            k: frozenset() for k in self._CATEGORY_MAP
        }
        self._keyword_to_category: dict[str, str] = {}
        self._parameter_keywords: frozenset = frozenset()

        if config_path is None:
            config_path = self._default_config_path()

        self._load_config(config_path)

    @staticmethod
    def _default_config_path() -> str:
        config_dir = Path(__file__).parent.parent / "config"
        return str(config_dir / "domain_keywords.yaml")

    @staticmethod
    def _from_dir(config_dir: str) -> "KeywordExtractor":
        """Create instance from a config directory path."""
        return KeywordExtractor(str(Path(config_dir) / "domain_keywords.yaml"))

    def _load_config(self, config_path: str):
        """Load keyword configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning("Keyword config not found: %s, using empty keywords", config_path)
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.error("Failed to load keyword config: %s", exc)
            return

        # Load built-in keywords (material/standard/parameter only)
        for category_key, category_name in self._CATEGORY_MAP.items():
            keywords = config.get(category_key, [])
            self._keywords[category_key] = frozenset(keywords)
            for kw in keywords:
                self._keyword_to_category[kw] = category_name

        # Cache parameter keywords for suffix matching
        self._parameter_keywords = self._keywords.get("parameter", frozenset())

        # Merge user_keywords from config file (for backward compat)
        user_keywords = config.get("user_keywords", {})
        for category, kws in user_keywords.items():
            if category not in EntityCategory.ALL:
                logger.warning("Skipping unknown keyword category: %s", category)
                continue
            for kw in kws:
                self._keyword_to_category[kw] = category
            category_key = category.lower()
            if category_key in self._keywords:
                self._keywords[category_key] = self._keywords[category_key] | frozenset(kws)

        counts = ", ".join(
            f"{k}={len(v)}" for k, v in self._keywords.items() if v
        )
        logger.info("Loaded keywords: %s", counts)

    def add_keywords(self, entity_categories: dict[str, list[str]]):
        """Add user-defined domain keywords at runtime.

        Args:
            entity_categories: {category_name: [keyword1, keyword2, ...]}
                e.g. {"DISEASE": ["乳腺癌", "肿瘤"], "ANATOMY": ["乳腺"]}
                Category names must be in EntityCategory.ALL, OR the user can
                use custom category names that will be mapped to UNKNOWN.
        """
        for category, keywords in entity_categories.items():
            for kw in keywords:
                if category in EntityCategory.ALL:
                    self._keyword_to_category[kw] = category
                    cat_key = category.lower()
                    if cat_key not in self._keywords:
                        self._keywords[cat_key] = frozenset()
                    self._keywords[cat_key] = self._keywords[cat_key] | frozenset([kw])
                    if category == "PARAMETER":
                        self._parameter_keywords = self._parameter_keywords | frozenset([kw])
                else:
                    # Custom category not in schema → map to UNKNOWN
                    self._keyword_to_category[kw] = "UNKNOWN"
                    if "unknown" not in self._keywords:
                        self._keywords["unknown"] = frozenset()
                    self._keywords["unknown"] = self._keywords["unknown"] | frozenset([kw])
                logger.debug("Added keyword: %s -> %s", kw, category)

    def lookup_category(self, text: str) -> Optional[str]:
        """Look up the category for a keyword (public method).

        Multi-char tokens: exact match against all categories.
        Single-char tokens: only match MATERIAL (e.g., 钢, 铁, 铜).
        Parameter suffix match: "高强度" ends with "强度" -> PARAMETER.
        """
        # Exact match
        if text in self._keyword_to_category:
            return self._keyword_to_category[text]

        # Single-char: only match materials
        if len(text) == 1:
            if text in self._keywords.get("material", frozenset()):
                return "MATERIAL"
            return None

        # Multi-char: try parameter suffix match
        if len(text) >= 3:
            for kw in self._parameter_keywords:
                if text.endswith(kw) and len(kw) >= 2:
                    return "PARAMETER"

        return None

    # Backward compatibility alias (used internally by entity_mapper)
    _keyword_category = lookup_category

    @property
    def parameter_keywords(self) -> frozenset:
        """Return parameter keywords for use by mapper.py _assign_attributes."""
        return self._parameter_keywords

    def extract(
        self,
        tokens: list[Token],
        existing_entities: list[Entity],
        text: str,
        id_generator,
    ) -> list[Entity]:
        """Scan tokens for domain keywords not caught by NER.

        Args:
            tokens: Token list.
            existing_entities: Already-found entities (to avoid duplicates).
            text: Original text.
            id_generator: EntityIdGenerator instance for ID generation.

        Returns:
            List of new Entity objects found.
        """
        result: list[Entity] = []
        entity_spans = {(e.span[0], e.span[1]) for e in existing_entities}

        for t in tokens:
            span_key = (t.span[0], t.span[1])
            if span_key in entity_spans:
                continue

            cat = self._keyword_category(t.text)
            if cat and cat != "UNKNOWN":
                ent_id = id_generator.generate(t.text, t.span, cat)
                if ent_id is None:
                    continue
                result.append(Entity(
                    id=ent_id,
                    text=t.text,
                    category=cat,
                    span=t.span,
                    normalized=t.text,
                    source="keyword",
                    confidence=0.95,
                ))
                entity_spans.add(span_key)

        result.sort(key=lambda e: e.span[0])
        return result

    def add_keyword(self, keyword: str, category: str):
        """Add a single keyword at runtime.

        Args:
            keyword: Keyword text.
            category: NSP category (e.g., "MATERIAL").
        """
        self.add_keywords({category: [keyword]})