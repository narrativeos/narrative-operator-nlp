"""
Noun Signal Extraction — Step A (POS gating) + Step B (syntactic-role weighting).

Surfaces block-level noun-phrase candidates for the caller to aggregate,
cluster, and audit (optionally via LLM). These are NOT entities — they are
linguistic signals with evidence.

Step A: keep tokens whose POS is in the whitelist (default ``NN``/``NR``).
Step B: for each candidate, determine its syntactic role from the dependency
        tree and find the governing verb, then compute a block-level salience
        score (POS base + role bonus). Global frequency/clustering is the
        caller's responsibility (see main spec §6).

The rel→role mapping and governing-verb walk-up mirror
``EventExtractor._assign_syntactic_roles`` for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .schema import DependencyEdge, NounSignal, Token

# Default POS whitelist for Step A (Chinese CTB/UD-style tags).
_DEFAULT_POS_WHITELIST = ["NN", "NR"]

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
    parent_of: dict[int, tuple[int, str]] = {}
    for dep in deps:
        if dep.child >= 0:
            parent_of[dep.child] = (dep.head, dep.rel)

    whitelist = set(config.pos_whitelist)
    candidates: list[NounSignal] = []

    for tok in tokens:
        # Step A: POS gating (skip single-char fragments / function words).
        if tok.pos not in whitelist:
            continue
        if len(tok.text) < 2:
            continue

        # Step B: syntactic role from the dependency tree.
        head_idx, dep_rel = parent_of.get(tok.id, (-1, ""))
        dep_rel_lower = (dep_rel or "").lower()
        syn_role = _DEP_TO_SYNTACTIC.get(dep_rel_lower, "unknown")

        # Walk up to the first V-pos ancestor for the governing verb.
        governing_verb = ""
        current = tok.id
        visited: set[int] = set()
        while current in parent_of and current not in visited:
            visited.add(current)
            h, _rel = parent_of[current]
            if h < 0:
                break
            if 0 <= h < len(tokens):
                h_pos = getattr(tokens[h], "pos", "")
                if h_pos and h_pos[0] == "V":
                    governing_verb = getattr(tokens[h], "text", "")
                    break
            current = h

        # Score: POS base + role bonus (block-level salience, not frequency).
        score = min(_POS_BASE_SCORE.get(tok.pos, 0.3) + _ROLE_BONUS.get(syn_role, 0.0), 1.0)
        if score < config.min_score:
            continue

        evidence: dict = {}
        if dep_rel:
            evidence["head_rel"] = dep_rel
        if governing_verb:
            evidence["governing_verb"] = governing_verb

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
