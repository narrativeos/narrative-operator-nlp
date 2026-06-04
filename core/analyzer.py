"""
Unified NLP Analysis Entry Point.

This is the SINGLE entry point for all NLP analysis. Every protocol
adapter (MCP, gRPC, FastAPI) calls this function. The core module
has ZERO protocol dependencies.
"""

from __future__ import annotations

import logging
from typing import Optional

from .mapper import HanlpSchemaMapper
from .schema import NarrativeDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded HanLP pipeline
# ---------------------------------------------------------------------------

_hanlp_pipeline: Optional[object] = None
_mapper: Optional[HanlpSchemaMapper] = None


def _get_pipeline():
    """Lazily load the HanLP multi-task pipeline."""
    global _hanlp_pipeline
    if _hanlp_pipeline is None:
        try:
            import hanlp
            # Use the compact ELECTRA-small model (faster to download, lighter)
            _hanlp_pipeline = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
            logger.info("HanLP pipeline loaded successfully (ELECTRA-small).")
        except Exception as exc:
            logger.error("Failed to load HanLP pipeline: %s", exc)
            raise RuntimeError(
                "HanLP pipeline could not be loaded. "
                "Ensure hanlp is installed: pip install hanlp"
            ) from exc
    return _hanlp_pipeline


def _get_mapper() -> HanlpSchemaMapper:
    """Lazily create the schema mapper."""
    global _mapper
    if _mapper is None:
        _mapper = HanlpSchemaMapper()
    return _mapper


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(text: str, source: str = "hanlp_v2") -> NarrativeDocument:
    """
    Analyze raw text and return NSP-standardized narrative atoms.

    This is the canonical entry point used by ALL protocol adapters.
    It wraps HanLP's multi-task model and maps the raw output through
    the HanlpSchemaMapper to produce a NarrativeDocument.

    Args:
        text: Raw input text to analyze.
        source: Engine identifier recorded in meta.source.

    Returns:
        NarrativeDocument containing tokens, entities, and relations.

    Raises:
        ValueError: If text is empty.
        RuntimeError: If HanLP pipeline fails to load or run.

    Example::

        >>> from core.analyzer import analyze
        >>> doc = analyze("碳钢是钢的一种。")
        >>> print(len(doc.content.entities))
    """
    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")

    pipeline = _get_pipeline()
    mapper = _get_mapper()

    try:
        raw = pipeline(text)
    except Exception as exc:
        logger.error("HanLP analysis failed for text length=%d: %s", len(text), exc)
        raise RuntimeError(f"HanLP analysis failed: {exc}") from exc

    return mapper.map(text, raw, source=source)
