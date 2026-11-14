"""
Noun Signal Extraction — Step A (POS gating) + Step B (syntactic-role weighting).

Surfaces block-level noun-phrase candidates for the caller to aggregate,
cluster, and audit (optionally via LLM). These are NOT entities — they are
linguistic signals with evidence.

Step A: keep tokens whose POS is in the whitelist (default ``NN``/``NR``),
        dropping any candidate that is not one contiguous word (a token
        merged across a space or punctuation mark has a span that no longer
        describes its text).
Step B: for each candidate, determine its syntactic role from the dependency
        tree and find the governing verb, then compute a block-level salience
        score (POS base + role bonus). Global frequency/clustering is the
        caller's responsibility (see main spec §6).

The governing verb is resolved by ``dep_walk.find_governing_verb`` — the same
implementation used by ``EventExtractor._assign_syntactic_roles``, so the two
call sites cannot drift apart again. It is strict: the verb is either one
single-token verb from the head chain, or ``None`` with a
``governing_verb_status`` explaining why (``dep`` / ``coarse_token`` /
``not_found``). The key is always present in ``evidence``, so the output
shape does not depend on which worker or which parse produced it.

"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from .dep_walk import build_parent_index, find_governing_verb
from .schema import DependencyEdge, NounSignal, Token

logger = logging.getLogger(__name__)

# Default POS whitelist for Step A (Chinese CTB/UD-style tags).
_DEFAULT_POS_WHITELIST = ["NN", "NR"]

# Unicode general categories that can never appear inside one well-formed
# word: Punctuation, Separators (space, NBSP, ideographic space) and
# Control/Cf. A token whose surface text contains one of these was merged
# by the tokenizer across a gap in the source text, so its character span
# covers characters that are not part of the word.
_WELLFORMED_REJECTED_CATEGORIES = ("P", "Z", "C")


def _is_wellformed_token(text: str, span: tuple[int, int]) -> bool:
    """Reject candidates that are not a single contiguous word.

    Guards against upstream tokenizer merges (dictionary forcing, coarse
    model output) that produce a "noun" whose span crosses whitespace or
    punctuation — the emitted ``text`` and ``span`` then disagree, and a
    caller highlighting ``text[span[0]:span[1]]`` sees a whole clause.
    """
    if span[1] - span[0] != len(text):
        return False
    return not any(
        unicodedata.category(ch)[0] in _WELLFORMED_REJECTED_CATEGORIES
        for ch in text
    )


# Dependency rel → syntactic role (mirrors EventExtractor._DEP_TO_SYNTACTIC).
_DEP_TO_SYNTACTIC: dict[str, str] = {
    "nsubj": "Subject", "nsubjpass": "Subject",
    "dobj": "Object", "iobj": "Object",
    "loc": "Adverbial", "lobj": "Adverbial",
    "nn": "Attributive", "amod": "Attributive",
}

# Base salience by POS (Step A).
_POS_BASE_SCORE: dict[str, float] = {
    "NN": 0.5,
    "NR": 0.4,
    "NT": 0.4,
    "JJ": 0.3,
}

# Syntactic-role bonus (Step B).
_ROLE_BONUS: dict[str, float] = {
    "Subject": 0.2,
    "Object": 0.2,
    "Adverbial": 0.1,
    "Attributive": 0.05,
    "unknown": 0.0,
}


@dataclass
class NounSignalConfig:
    """Noun-signal extraction config (request ``noun_signals`` block)."""
    enabled: bool = False
    min_score: float = 0.3
    max_per_block: int = 30
    pos_whitelist: list[str] = field(default_factory=lambda: list(_DEFAULT_POS_WHITELIST))

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "NounSignalConfig":
        """Parse a config dict. ``None``/empty yields a disabled default config."""
        cfg = cls()
        if not d:
            return cfg
        if "enabled" in d:
            cfg.enabled = bool(d["enabled"])
        if "min_score" in d:
            cfg.min_score = float(d["min_score"])
        if "max_per_block" in d:
            cfg.max_per_block = int(d["max_per_block"])
        if "pos_whitelist" in d and isinstance(d["pos_whitelist"], list):
            cfg.pos_whitelist = [str(p) for p in d["pos_whitelist"]]
        return cfg


def extract_noun_signals(
    tokens: list[Token],
    deps: list[DependencyEdge],
    config: Optional[NounSignalConfig] = None,
) -> list[NounSignal]:
    """Extract block-level noun signals (Step A + Step B).

    Returns an empty list when ``config`` is None or ``config.enabled`` is
    False (backward compatible — no noun signals by default).

    ``tokens`` must be globally indexed (``tokens[i].id == i``) and ``deps``
    must use global token indices, as produced by ``analyzer.analyze``.
    """
    if config is None or not config.enabled:
        return []
    if not tokens:
        return []

    # Build parent_of: child_idx -> (head_idx, rel)
    parent_of = build_parent_index(deps)

    whitelist = set(config.pos_whitelist)
    candidates: list[NounSignal] = []

    for tok in tokens:
        # Step A: POS gating (skip single-char fragments / function words).
        if tok.pos not in whitelist:
            continue
        if len(tok.text) < 2:
            continue
        # Step A guard: a candidate must be one contiguous word. Tokens
        # merged by the tokenizer across a space or a punctuation mark are
        # dropped rather than surfaced — their span no longer describes
        # their text, so a caller highlighting the span gets a clause.
        if not _is_wellformed_token(tok.text, tok.span):
            logger.debug(
                "noun signal dropped (not a contiguous word): %r span=%s",
                tok.text, tok.span,
            )
            continue

        # Step B: syntactic role from the dependency tree.
        head_idx, dep_rel = parent_of.get(tok.id, (-1, ""))
        dep_rel_lower = (dep_rel or "").lower()
        syn_role = _DEP_TO_SYNTACTIC.get(dep_rel_lower, "unknown")

        # Governing verb: the first *single-token* verb ancestor. Strict —
        # never guessed, never truncated; a coarse verb phrase on the
        # chain is reported as evidence instead of being passed off as the
        # verb. The key is always present so the output shape does not
        # depend on which worker (or which parse) produced it.
        governing_verb, gv_status, gv_raw = find_governing_verb(
            tokens, parent_of, tok.id,
        )

        # Score: POS base + role bonus (block-level salience, not frequency).
        score = min(_POS_BASE_SCORE.get(tok.pos, 0.3) + _ROLE_BONUS.get(syn_role, 0.0), 1.0)
        if score < config.min_score:
            continue

        evidence: dict = {
            "head_rel": dep_rel or None,
            "governing_verb": governing_verb,
            "governing_verb_status": gv_status,
        }
        if gv_raw is not None:
            evidence["governing_verb_raw"] = gv_raw

        candidates.append(NounSignal(
            text=tok.text,
            pos=tok.pos,
            syntactic_role=syn_role,
            score=round(score, 4),
            span=tok.span,
            evidence=evidence,
        ))

    # Sort by score desc, then by position; cap at max_per_block.
    candidates.sort(key=lambda s: (-s.score, s.span[0]))
    return candidates[:config.max_per_block]
