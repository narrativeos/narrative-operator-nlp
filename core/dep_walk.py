"""Shared dependency-tree helpers — the single governing-verb walk-up.

Two call sites resolve "which verb governs this token":
``noun_signal.extract_noun_signals`` (Step B) and
``EventExtractor._assign_syntactic_roles``. They used to carry private
copies of the same loop, and the copies drifted: on the same document one
run emitted ``governing_verb: null`` while another emitted a whole verb
phrase (``电阻应大于``), because a duplicated walk-up silently accepts
whatever token happens to sit on the head chain.

Strict mode (no guessing): a governing verb is always exactly one
tokenizer token. A verb ancestor that is too long to be a single lexical
verb — a coarse multi-word token like ``电阻应大于`` — is *not* used and
not truncated; the walk keeps looking upward, and if nothing better is
found the resolved verb stays ``None`` with a status that says why. The
raw coarse text is returned separately so the caller can keep it as
audit evidence.
"""

from __future__ import annotations

from typing import Optional

# ── governing_verb_status values (stable, machine-readable) ──
# A single-token verb was found on the head chain.
GOVERNING_VERB_DEP = "dep"
# Only coarse multi-word verb tokens were found; nothing was guessed.
GOVERNING_VERB_COARSE = "coarse_token"
# No verb ancestor at all (noun-noun chain, flat parse, no dep data).
GOVERNING_VERB_NOT_FOUND = "not_found"

# Longest ancestor token still considered one lexical verb. Chinese verb
# tokens are 1-2 chars; 4 leaves room for a bound modal/degree prefix
# while rejecting clause-sized tokens such as 「电阻应大于」 (5 chars).
MAX_GOVERNING_VERB_CHARS = 4


def build_parent_index(deps) -> dict[int, tuple[int, str]]:
    """Index dependency edges as ``child_index -> (head_index, rel)``.

    ``deps`` must already use the same (global) token indices as the
    ``tokens`` list the caller walks.
    """
    parent_of: dict[int, tuple[int, str]] = {}
    for dep in deps:
        if dep.child >= 0:
            parent_of[dep.child] = (dep.head, dep.rel)
    return parent_of


def find_governing_verb(
    tokens: list,
    parent_of: dict[int, tuple[int, str]],
    start_idx: int,
) -> tuple[Optional[str], str, Optional[str]]:
    """Walk up the head chain to the first single-token verb ancestor.

    Returns ``(verb, status, coarse_raw)``:

    - ``(text, GOVERNING_VERB_DEP, None)`` — a verb token of at most
      ``MAX_GOVERNING_VERB_CHARS`` characters was found.
    - ``(None, GOVERNING_VERB_COARSE, raw)`` — the chain only offered
      coarse verb tokens; ``raw`` is the first one seen, for evidence.
    - ``(None, GOVERNING_VERB_NOT_FOUND, None)`` — no verb ancestor
      (including the case where the token has no parent at all).

    The walk stops at the sentence root (``head < 0``) and is cycle-safe.
    """
    current = start_idx
    visited: set[int] = set()
    coarse_raw: Optional[str] = None

    while current in parent_of and current not in visited:
        visited.add(current)
        head_idx, _rel = parent_of[current]
        if head_idx < 0:
            break
        if 0 <= head_idx < len(tokens):
            head_pos = getattr(tokens[head_idx], "pos", "") or ""
            if head_pos[:1] == "V":
                head_text = getattr(tokens[head_idx], "text", "") or ""
                if len(head_text) <= MAX_GOVERNING_VERB_CHARS:
                    return head_text, GOVERNING_VERB_DEP, None
                if coarse_raw is None:
                    coarse_raw = head_text
        current = head_idx

    if coarse_raw is not None:
        return None, GOVERNING_VERB_COARSE, coarse_raw
    return None, GOVERNING_VERB_NOT_FOUND, None
