"""
NER Label Mapper — Maps NER model labels to NSP standard categories.

Loads mapping configuration from YAML files, supporting both built-in and
user-defined mappings.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml

from .schema import EntityCategory

logger = logging.getLogger(__name__)


class NerLabelMapper:
    """Maps NER model labels to NSP standard entity categories.

    Usage:
        mapper = NerLabelMapper()
        category = mapper.map_label("nr", "ner/pku")  # -> "PERSON"
    """

    def __init__(self, config_path: Optional[str] = None):
        self._source_maps: dict[str, dict[str, str]] = {}
        self._ner_sources: list[str] = []

        if config_path is None:
            # Default: look for entity_mapping.yaml in the config directory
            config_path = self._default_config_path()

        self._load_config(config_path)

    @staticmethod
    def _default_config_path() -> str:
        """Find the default config path relative to this file."""
        config_dir = Path(__file__).parent.parent / "config"
        return str(config_dir / "entity_mapping.yaml")

    @staticmethod
    def _from_dir(config_dir: str) -> "NerLabelMapper":
        """Create instance from a config directory path."""
        return NerLabelMapper(str(Path(config_dir) / "entity_mapping.yaml"))

    def _load_config(self, config_path: str):
        """Load mapping configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning("Entity mapping config not found: %s, using defaults", config_path)
            self._load_defaults()
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.error("Failed to load entity mapping config: %s", exc)
            self._load_defaults()
            return

        # Load source maps
        self._source_maps = config.get("source_maps", {})

        # Load ner sources order
        self._ner_sources = config.get("ner_sources", [
            "ner/pku", "ner/msra", "ner/ontonotes", "ner", "ner/conll2003"
        ])

        # Merge custom maps
        custom_maps = config.get("custom_maps", {})
        for source, mappings in custom_maps.items():
            if source not in self._source_maps:
                self._source_maps[source] = {}
            self._source_maps[source].update(mappings)

        logger.info(
            "Loaded entity mapping config: %d sources, %s",
            len(self._source_maps),
            self._ner_sources,
        )

    def _load_defaults(self):
        """Load built-in default mappings (fallback)."""
        self._source_maps = {
            "ner/pku": {"nr": "PERSON", "ns": "LOCATION", "nt": "ORGANIZATION", "nz": "PRODUCT"},
            "ner/msra": {"PERSON": "PERSON", "LOCATION": "LOCATION", "ORGANIZATION": "ORGANIZATION", "DATE": "DATE"},
            "ner/ontonotes": {
                "PERSON": "PERSON", "NORP": "ORGANIZATION", "FAC": "FACILITY",
                "ORG": "ORGANIZATION", "GPE": "LOCATION", "LOC": "LOCATION",
                "PRODUCT": "PRODUCT", "DATE": "DATE", "TIME": "DATE",
                "PERCENT": "NUMBER", "MONEY": "NUMBER", "QUANTITY": "NUMBER",
                "CARDINAL": "NUMBER", "ORDINAL": "NUMBER", "LAW": "STANDARD",
                "EVENT": "UNKNOWN", "WORK_OF_ART": "UNKNOWN", "LANGUAGE": "UNKNOWN",
            },
            "ner": {
                "PER": "PERSON", "PERSON": "PERSON", "LOC": "LOCATION", "GPE": "LOCATION",
                "ORG": "ORGANIZATION", "ORGANIZATION": "ORGANIZATION", "MISC": "UNKNOWN",
                "DATE": "DATE", "TIME": "DATE", "MONEY": "NUMBER", "PERCENT": "NUMBER",
                "QUANTITY": "NUMBER", "CARDINAL": "NUMBER", "ORDINAL": "NUMBER",
                "FAC": "FACILITY", "PRODUCT": "PRODUCT", "EVENT": "UNKNOWN",
                "WORK_OF_ART": "UNKNOWN", "LAW": "STANDARD", "LANGUAGE": "UNKNOWN",
                "NORP": "ORGANIZATION",
            },
            "ner/conll2003": {
                "PER": "PERSON", "PERSON": "PERSON", "LOC": "LOCATION", "GPE": "LOCATION",
                "ORG": "ORGANIZATION", "ORGANIZATION": "ORGANIZATION", "MISC": "UNKNOWN",
                "DATE": "DATE", "TIME": "DATE", "MONEY": "NUMBER", "PERCENT": "NUMBER",
                "QUANTITY": "NUMBER", "CARDINAL": "NUMBER", "ORDINAL": "NUMBER",
                "FAC": "FACILITY", "PRODUCT": "PRODUCT", "EVENT": "UNKNOWN",
                "WORK_OF_ART": "UNKNOWN", "LAW": "STANDARD", "LANGUAGE": "UNKNOWN",
                "NORP": "ORGANIZATION",
            },
        }
        self._ner_sources = [
            "ner/pku", "ner/msra", "ner/ontonotes", "ner", "ner/conll2003"
        ]

    @property
    def source_maps(self) -> dict[str, dict[str, str]]:
        """Read-only access to source maps."""
        return self._source_maps

    @property
    def ner_sources(self) -> list[str]:
        """Ordered list of NER sources to check."""
        return self._ner_sources

    def map_label(self, label: str, source: str) -> Optional[str]:
        """Map a single NER label to an NSP category.

        Args:
            label: Raw NER label from the model (e.g., "nr", "PERSON").
            source: NER source key (e.g., "ner/pku", "ner/ontonotes").

        Returns:
            NSP category string or None if not mapped.
        """
        if not label:
            return None
        mappings = self._source_maps.get(source, {})
        category = mappings.get(label.strip())
        if category and category in EntityCategory.ALL:
            return category
        return None

    def add_custom_mapping(self, source: str, label: str, category: str):
        """Add a custom mapping at runtime.

        Args:
            source: NER source key.
            label: Raw NER label.
            category: NSP category.
        """
        if category not in EntityCategory.ALL:
            raise ValueError(f"Invalid category: {category}. Must be in {EntityCategory.ALL}")
        if source not in self._source_maps:
            self._source_maps[source] = {}
        self._source_maps[source][label] = category
        logger.debug("Added custom mapping: %s/%s -> %s", source, label, category)