# -*- coding: utf-8 -*-
"""
Dictionary Loader — Engineering-grade dictionary management for classical Chinese.

Provides:
- Three-layer architecture (seed/custom/user)
- Version management
- Metadata tracking
- Validation and error handling
- Caching and lazy loading

Usage:
    loader = DictionaryLoader(config_dir)
    loader.load_all()
    keywords = loader.get_keywords("TITLE")
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .schema import EntityCategory

logger = logging.getLogger(__name__)


@dataclass
class DictionaryMetadata:
    """Metadata for a dictionary file."""
    source: str              # Source (e.g., "CLNER", "manual")
    version: str             # Version string (e.g., "1.0.0")
    last_updated: float     # Unix timestamp
    entry_count: int        # Number of entries
    checksum: str           # MD5 checksum for integrity check
    category: str           # Entity category (e.g., "TITLE", "ERA")


@dataclass
class DictionaryEntry:
    """A single dictionary entry with metadata."""
    keyword: str
    category: str
    source: str
    confidence: float = 0.95


class DictionaryLoader:
    """Engineering-grade dictionary loader for classical Chinese.
    
    Three-layer architecture:
    1. Seed dictionaries (read-only, synced from open source datasets)
    2. Custom dictionaries (manually maintained)
    3. User dictionaries (injected at runtime via API)
    
    Features:
    - Version management with checksums
    - Metadata tracking (source, update time, etc.)
    - Validation and error handling
    - Caching for performance
    """
    
    # File to category mapping
    FILE_TO_CATEGORY = {
        "titles.yaml": "TITLE",
        "eras.yaml": "ERA",
        "locations.yaml": "LOCATION",
        "persons.yaml": "PERSON",
        "classics.yaml": "PRODUCT",
        "institutions.yaml": "INSTITUTION",
        "organizations.yaml": "ORGANIZATION",
        "events.yaml": "EVENT",
    }
    
    def __init__(self, config_dir: str):
        self._config_dir = Path(config_dir)
        self._seed_dir = self._config_dir / "classical" / "seed"
        self._custom_dir = self._config_dir / "classical" / "custom"
        self._user_dir = self._config_dir / "classical" / "user"
        self._db_dir = self._config_dir / "classical" / "databases"
        
        # Keyword storage: category -> set of keywords
        self._keywords: dict[str, set[str]] = {}
        
        # Metadata storage: filepath -> DictionaryMetadata
        self._metadata: dict[str, DictionaryMetadata] = {}
        
        # Entry storage: keyword -> DictionaryEntry
        self._entries: dict[str, DictionaryEntry] = {}
        
        # SQLite database (lazy loaded)
        self._sqlite_conn: Optional[sqlite3.Connection] = None
        self._sqlite_cache: dict[str, Optional[DictionaryEntry]] = {}
        
        # Cache
        self._cache_enabled = True
        self._cache: dict[str, Any] = {}
    
    @property
    def seed_dir(self) -> Path:
        """Seed dictionary directory."""
        return self._seed_dir
    
    @property
    def custom_dir(self) -> Path:
        """Custom dictionary directory."""
        return self._custom_dir
    
    @property
    def user_dir(self) -> Path:
        """User dictionary directory."""
        return self._user_dir
    
    def load_all(self) -> int:
        """Load all dictionaries from all layers.
        
        Returns:
            Total number of entries loaded.
        """
        total = 0
        
        # Load seed dictionaries (read-only)
        total += self._load_layer(self._seed_dir, "seed")
        
        # Load custom dictionaries (manually maintained)
        total += self._load_layer(self._custom_dir, "custom")
        
        # Load user dictionaries (runtime injection)
        total += self._load_layer(self._user_dir, "user")
        
        logger.info("DictionaryLoader: Total entries loaded: %d", total)
        return total
    
    def _load_layer(self, directory: Path, layer_name: str) -> int:
        """Load dictionaries from a single layer.
        
        Args:
            directory: Directory path
            layer_name: Layer name for logging
            
        Returns:
            Number of entries loaded.
        """
        if not directory.exists():
            logger.warning("Dictionary layer not found: %s", directory)
            return 0
        
        total = 0
        for filename, category in self.FILE_TO_CATEGORY.items():
            filepath = directory / filename
            if not filepath.exists():
                continue
            
            try:
                count = self._load_file(filepath, category, layer_name)
                total += count
            except Exception as e:
                logger.error("Failed to load %s: %s", filepath, e)
        
        logger.info("DictionaryLoader: %s layer loaded %d entries", layer_name, total)
        return total
    
    def _load_file(self, filepath: Path, category: str, source: str) -> int:
        """Load a single dictionary file.
        
        Args:
            filepath: Path to YAML file
            category: Entity category
            source: Source name for metadata
            
        Returns:
            Number of entries loaded.
        """
        # Read file
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Parse YAML
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {filepath}: {e}")
        
        if not data or not isinstance(data, dict):
            raise ValueError(f"Invalid dictionary format in {filepath}")
        
        # Get keyword list
        key = list(data.keys())[0]
        keywords = data.get(key, [])
        
        if not keywords:
            return 0
        
        # Validate category
        if category not in EntityCategory.ALL:
            logger.warning("Skipping unknown category: %s", category)
            return 0
        
        # Calculate checksum
        checksum = hashlib.md5(content.encode("utf-8")).hexdigest()
        
        # Store metadata
        self._metadata[str(filepath)] = DictionaryMetadata(
            source=source,
            version=self._extract_version(content),
            last_updated=time.time(),
            entry_count=len(keywords),
            checksum=checksum,
            category=category,
        )
        
        # Store keywords
        if category not in self._keywords:
            self._keywords[category] = set()
        
        for kw in keywords:
            if isinstance(kw, str):
                self._keywords[category].add(kw)
                self._entries[kw] = DictionaryEntry(
                    keyword=kw,
                    category=category,
                    source=source,
                )
        
        return len(keywords)
    
    @staticmethod
    def _extract_version(content: str) -> str:
        """Extract version from file comments.
        
        Looks for comments like: # version: 1.0.0
        """
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("#") and "version:" in line.lower():
                parts = line.split(":")
                if len(parts) >= 2:
                    return parts[-1].strip()
        return "1.0.0"
    
    def get_keywords(self, category: str) -> set[str]:
        """Get all keywords for a category.
        
        Args:
            category: Entity category
            
        Returns:
            Set of keywords.
        """
        return self._keywords.get(category, set())
    
    def get_entry(self, keyword: str) -> Optional[DictionaryEntry]:
        """Get dictionary entry for a keyword.
        
        Args:
            keyword: Keyword to look up
            
        Returns:
            DictionaryEntry or None.
        """
        return self._entries.get(keyword)
    
    def get_metadata(self, filepath: str) -> Optional[DictionaryMetadata]:
        """Get metadata for a dictionary file.
        
        Args:
            filepath: Path to dictionary file
            
        Returns:
            DictionaryMetadata or None.
        """
        return self._metadata.get(filepath)
    
    def add_keywords(self, category: str, keywords: list[str], source: str = "custom") -> None:
        """Add keywords at runtime.
        
        Args:
            category: Entity category
            keywords: List of keywords
            source: Source name
        """
        if category not in EntityCategory.ALL:
            logger.warning("Skipping unknown category: %s", category)
            return
        
        if category not in self._keywords:
            self._keywords[category] = set()
        
        for kw in keywords:
            if isinstance(kw, str):
                self._keywords[category].add(kw)
                self._entries[kw] = DictionaryEntry(
                    keyword=kw,
                    category=category,
                    source=source,
                )
    
    def export_metadata(self) -> dict:
        """Export all metadata as a dictionary.
        
        Returns:
            Dictionary of metadata.
        """
        result = {}
        for filepath, meta in self._metadata.items():
            result[filepath] = {
                "source": meta.source,
                "version": meta.version,
                "last_updated": meta.last_updated,
                "entry_count": meta.entry_count,
                "checksum": meta.checksum,
                "category": meta.category,
            }
        return result
    
    def _get_sqlite_connection(self) -> Optional[sqlite3.Connection]:
        """Get SQLite database connection (lazy loaded).
        
        Returns:
            sqlite3.Connection or None if database not found.
        """
        if self._sqlite_conn is not None:
            return self._sqlite_conn
        
        db_path = self._db_dir / "cbdb.sqlite"
        if not db_path.exists():
            logger.warning("CBDB database not found: %s", db_path)
            return None
        
        try:
            self._sqlite_conn = sqlite3.connect(str(db_path))
            logger.info("CBDB database loaded: %s", db_path)
        except sqlite3.Error as e:
            logger.error("Failed to connect to CBDB: %s", e)
            return None
        
        return self._sqlite_conn

    def lookup(self, keyword: str) -> Optional[DictionaryEntry]:
        """Unified lookup across all dictionary sources.
        
        Search order:
        1. Memory cache (YAML dictionaries)
        2. SQLite cache (previously queried)
        3. SQLite database (on-demand query)
        
        Args:
            keyword: Keyword to look up
            
        Returns:
            DictionaryEntry or None.
        """
        # 1. Check memory (YAML dictionaries)
        if keyword in self._entries:
            return self._entries[keyword]
        
        # 2. Check SQLite cache
        if keyword in self._sqlite_cache:
            return self._sqlite_cache[keyword]
        
        # 3. Query SQLite database (on-demand)
        conn = self._get_sqlite_connection()
        if conn is None:
            self._sqlite_cache[keyword] = None
            return None
        
        entry = self._lookup_sqlite(keyword, conn)
        self._sqlite_cache[keyword] = entry
        
        return entry

    def _lookup_sqlite(self, keyword: str, conn: sqlite3.Connection) -> Optional[DictionaryEntry]:
        """Look up a keyword in the CBDB SQLite database.
        
        Search order (by priority):
        1. ERA (dynasties) - highest priority, most specific
        2. TITLE (offices) - specific to CBDB
        3. LOCATION (places) - common entities
        4. PERSON (people) - largest table, lowest priority
        
        Args:
            keyword: Keyword to look up
            conn: SQLite connection
            
        Returns:
            DictionaryEntry or None.
        """
        found = []
        
        # Check ERA (DYNASTIES.c_dynasty_chn) - highest priority
        cursor = conn.execute(
            "SELECT c_dynasty_chn FROM DYNASTIES WHERE c_dynasty_chn = ? LIMIT 1",
            (keyword,)
        )
        if cursor.fetchone():
            return DictionaryEntry(
                keyword=keyword,
                category="ERA",
                source="CBDB",
                confidence=0.95,
            )
        
        # Check TITLE (OFFICE_CODES.c_office_chn)
        cursor = conn.execute(
            "SELECT c_office_chn FROM OFFICE_CODES WHERE c_office_chn = ? LIMIT 1",
            (keyword,)
        )
        if cursor.fetchone():
            return DictionaryEntry(
                keyword=keyword,
                category="TITLE",
                source="CBDB",
                confidence=0.90,
            )
        
        # Check LOCATION (ADDR_CODES.c_name_chn)
        cursor = conn.execute(
            "SELECT c_name_chn FROM ADDR_CODES WHERE c_name_chn = ? LIMIT 1",
            (keyword,)
        )
        if cursor.fetchone():
            return DictionaryEntry(
                keyword=keyword,
                category="LOCATION",
                source="CBDB",
                confidence=0.90,
            )
        
        # Check PERSON (BIOG_MAIN.c_name_chn) - lowest priority
        # Only return PERSON if not found in other categories
        cursor = conn.execute(
            "SELECT c_name_chn FROM BIOG_MAIN WHERE c_name_chn = ? LIMIT 1",
            (keyword,)
        )
        if cursor.fetchone():
            return DictionaryEntry(
                keyword=keyword,
                category="PERSON",
                source="CBDB",
                confidence=0.85,
            )
        
        return None

    def lookup_tokens(
        self,
        tokens: list[tuple[str, int, int]],  # (text, start, end)
        text: str,
    ) -> list[DictionaryEntry]:
        """Scientific token-based CBDB lookup.
        
        Method:
        1. Exact match for each token
        2. N-gram combination (2-4 gram) with contiguous span check
        3. Precise CBDB query for each candidate
        
        This avoids fuzzy matching which produces high false positive rates.
        Instead, we rely on the tokenizer's output as candidate boundaries
        and explore adjacent combinations.
        
        Args:
            tokens: List of (text, start_offset, end_offset) tuples
            text: Original text for span validation
            
        Returns:
            List of DictionaryEntry objects for discovered entities.
        """
        if not tokens:
            return []
        
        results = []
        seen_texts = set()
        
        # Step 1: Exact match for each token
        for token_text, start, end in tokens:
            if token_text in seen_texts:
                continue
            
            entry = self.lookup(token_text)
            if entry is not None:
                results.append(entry)
                seen_texts.add(token_text)
        
        # Step 2: N-gram combinations with contiguous span check
        # Only combine adjacent tokens that form a contiguous span
        for n in range(2, min(5, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                # Check for contiguous span (no gaps between tokens)
                expected_end = tokens[i][1]
                is_contiguous = True
                for j in range(i, i + n):
                    if tokens[j][1] != expected_end:
                        is_contiguous = False
                        break
                    expected_end = tokens[j][2]
                
                if not is_contiguous:
                    continue
                
                combined = "".join(tokens[j][0] for j in range(i, i + n))
                if combined in seen_texts:
                    continue
                
                entry = self.lookup(combined)
                if entry is not None:
                    results.append(entry)
                    seen_texts.add(combined)
        
        return results
    
    def validate_integrity(self) -> bool:
        """Validate integrity of all loaded dictionaries.
        
        Returns:
            True if all checksums match.
        """
        for filepath, meta in self._metadata.items():
            filepath_obj = Path(filepath)
            if not filepath_obj.exists():
                logger.error("Dictionary file missing: %s", filepath)
                return False
            
            with open(filepath_obj, "r", encoding="utf-8") as f:
                content = f.read()
            
            current_checksum = hashlib.md5(content.encode("utf-8")).hexdigest()
            if current_checksum != meta.checksum:
                logger.error("Checksum mismatch for %s", filepath)
                return False
        
        return True
