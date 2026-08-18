"""
Extractive Summarizer — NSP Semantic Features + TextRank Fusion.

Modes: "chars" (default 30) or "ratio" (default 0.2).
"""
from __future__ import annotations
import re
from typing import Optional
from .schema import (
    Entity, Event, KeySentence, NarrativeContent,
    NarrativeDocument, Relation, SentenceLanguage, Summary,
)

_DEFAULT_WEIGHTS: dict[str, tuple[float, float]] = {
    "classical": (0.85, 0.15),
    "modern": (0.60, 0.40),
    "english": (0.50, 0.50),
}
_LOW_VALUE_PATTERNS: list[str] = [r"^来源", r"^作者", r"^编辑", r"^转载", r"^本文", r"^据"]
_EMPTY_RE = re.compile(r"^[^\w\s]*$")


def _compute_nsp_score(sent: SentenceLanguage, content: NarrativeContent) -> tuple[float, list[str]]:
    reasons: list[str] = []
    raw_score = 0.0
    s0, s1 = sent.span
    entities_in = [e for e in content.entities if e.span[0] >= s0 and e.span[1] <= s1]
    n_ent = len(entities_in)
    if n_ent > 0:
        raw_score += min(n_ent / 3.0, 1.0) * 0.35
        reasons.append(f"entity_density({n_ent})")
    events_in = [e for e in content.events if e.trigger_span[0] >= s0 and e.trigger_span[1] <= s1]
    n_evt = len(events_in)
    if n_evt > 0:
        raw_score += min(n_evt / 2.0, 1.0) * 0.30
        reasons.append(f"event_density({n_evt})")
    rels_in = [r for r in content.relations if r.evidence_span[0] >= s0 and r.evidence_span[1] <= s1]
    n_rel = len(rels_in)
    if n_rel > 0:
        raw_score += min(n_rel / 4.0, 1.0) * 0.20
        reasons.append(f"relation_density({n_rel})")
    text_stripped = sent.text.strip()
    if _EMPTY_RE.match(text_stripped):
        raw_score = 0.0
        reasons.append("empty_filtered")
    for pat in _LOW_VALUE_PATTERNS:
        if re.match(pat, text_stripped):
            raw_score *= 0.1
            reasons.append("low_value_pattern")
            break
    return min(raw_score, 1.0), reasons


def _compute_textrank(
    sentences: list[SentenceLanguage],
    max_iter: int = 30, damping: float = 0.85, threshold: float = 1e-6,
) -> dict[int, float]:
    n = len(sentences)
    if n == 0:
        return {}
    if n == 1:
        return {0: 1.0}
    token_sets = [set(re.findall(r'\w+', s.text)) for s in sentences]
    sim: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            uni = len(token_sets[i] | token_sets[j])
            if uni > 0:
                v = len(token_sets[i] & token_sets[j]) / uni
                sim[i][j] = v
                sim[j][i] = v
    scores = [1.0 / n] * n
    for _ in range(max_iter):
        new_scores = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i != j and sim[j][i] > 0:
                    out = sum(sim[j])
                    if out > 0:
                        s += sim[j][i] / out * scores[j]
            new_scores[i] = (1 - damping) + damping * s
        diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores
        if diff < threshold:
            break
    mx = max(scores) if scores else 1.0
    if mx > 0:
        scores = [s / mx for s in scores]
    return {i: scores[i] for i in range(n)}


def _detect_dominant_language(sentences: list[SentenceLanguage]) -> str:
    if not sentences:
        return "modern"
    counts: dict[str, int] = {}
    for s in sentences:
        counts[s.label] = counts.get(s.label, 0) + 1
    return max(counts, key=counts.get)


def _select_sentences(
    scored: list[tuple[int, float, list[str], SentenceLanguage]],
    mode: str, target_chars: int, target_ratio: float, total_len: int,
) -> list[tuple[int, float, list[str], SentenceLanguage]]:
    if not scored:
        return []
    budget = max(1, int(total_len * target_ratio)) if mode == "ratio" else target_chars
    if total_len <= budget:
        return sorted(scored, key=lambda x: x[0])
    selected: dict[int, tuple[int, float, list[str], SentenceLanguage]] = {}
    cur = 0
    for idx, sc, reasons, sent in scored:
        sl = len(sent.text)
        if cur + sl <= budget or len(selected) == 0:
            selected[idx] = (idx, sc, reasons, sent)
            cur += sl
    return sorted(selected.values(), key=lambda x: x[0])


def summarize(
    doc: NarrativeDocument,
    mode: str = "chars",
    target_chars: int = 30,
    target_ratio: float = 0.2,
    nsp_weight: Optional[float] = None,
    textrank_weight: Optional[float] = None,
) -> Summary:
    """Generate extractive summary from a NarrativeDocument."""
    content = doc.content
    sentences = content.sentences
    if not sentences:
        return Summary(
            key_sentences=[], summary_text="", method="nsp_textrank",
            mode=mode, target_chars=target_chars, target_ratio=target_ratio,
            fusion_weights={}, total_sentences=0,
        )
    dominant_lang = _detect_dominant_language(sentences)
    default_nsp, default_tr = _DEFAULT_WEIGHTS.get(dominant_lang, _DEFAULT_WEIGHTS["modern"])
    eff_nsp = nsp_weight if nsp_weight is not None else default_nsp
    eff_tr = textrank_weight if textrank_weight is not None else default_tr
    tw = eff_nsp + eff_tr
    if tw > 0:
        eff_nsp /= tw
        eff_tr /= tw
    nsp_scores: dict[int, tuple[float, list[str]]] = {}
    for i, sent in enumerate(sentences):
        nsp_scores[i] = _compute_nsp_score(sent, content)
    tr_scores = _compute_textrank(sentences)
    n_total = len(sentences)
    scored: list[tuple[int, float, list[str], SentenceLanguage]] = []
    for i, sent in enumerate(sentences):
        ns, nr = nsp_scores[i]
        ts = tr_scores.get(i, 0.0)
        pos = 0.15 if i == 0 else (0.10 if i == n_total - 1 and n_total > 1 else 0.0)
        fused = min(eff_nsp * ns + eff_tr * ts + pos, 1.0)
        reasons = list(nr)
        if ts > 0.3:
            reasons.append(f"textrank({ts:.2f})")
        if pos > 0:
            reasons.append(f"position_prior({pos:.2f})")
        scored.append((i, fused, reasons, sent))
    scored.sort(key=lambda x: x[1], reverse=True)
    total_len = sum(len(s.text) for s in sentences)
    selected = _select_sentences(scored, mode, target_chars, target_ratio, total_len)
    key_sentences = [
        KeySentence(
            sentence_index=idx, text=sent.text, span=sent.span,
            score=round(sc, 4), reasons=reasons, source="summarizer",
        )
        for idx, sc, reasons, sent in selected
    ]
    summary_text = "".join(ks.text for ks in key_sentences)
    fusion_weights: dict[str, float] = {
        "nsp_weight": round(eff_nsp, 4),
        "textrank_weight": round(eff_tr, 4),
        "default_nsp": round(default_nsp, 4),
        "default_textrank": round(default_tr, 4),
    }
    for lang, (nw, tw2) in _DEFAULT_WEIGHTS.items():
        fusion_weights[f"default_{lang}_nsp"] = nw
        fusion_weights[f"default_{lang}_textrank"] = tw2
    return Summary(
        key_sentences=key_sentences, summary_text=summary_text,
        method="nsp_textrank", mode=mode,
        target_chars=target_chars, target_ratio=target_ratio,
        fusion_weights=fusion_weights, total_sentences=n_total,
    )



def summarize_text(
    text: str,
    mode: str = "chars",
    target_chars: int = 30,
    target_ratio: float = 0.2,
    language: str = "auto",
) -> Summary:
    """Summarize raw text without full NLP analysis (TextRank only)."""
    from .analyzer import _split_sentences
    from .language_detector import detect_language
    raw_sents = _split_sentences(text)
    if not raw_sents:
        return Summary(
            key_sentences=[], summary_text="", method="textrank_only",
            mode=mode, target_chars=target_chars, target_ratio=target_ratio,
            fusion_weights={}, total_sentences=0,
        )
    sentences: list[SentenceLanguage] = []
    for st, off in raw_sents:
        lbl = language if language != "auto" else detect_language(st)[0]
        sentences.append(SentenceLanguage(
            text=st, span=(off, off + len(st)), label=lbl, confidence=0.5,
        ))
    content = NarrativeContent(sentences=sentences)
    tr_scores = _compute_textrank(sentences)
    n_total = len(sentences)
    scored: list[tuple[int, float, list[str], SentenceLanguage]] = []
    for i, sent in enumerate(sentences):
        ts = tr_scores.get(i, 0.0)
        pos = 0.15 if i == 0 else (0.10 if i == n_total - 1 and n_total > 1 else 0.0)
        fused = min(ts + pos, 1.0)
        reasons = [f"textrank({ts:.2f})"]
        if pos > 0:
            reasons.append(f"position_prior({pos:.2f})")
        scored.append((i, fused, reasons, sent))
    scored.sort(key=lambda x: x[1], reverse=True)
    total_len = sum(len(s.text) for s in sentences)
    selected = _select_sentences(scored, mode, target_chars, target_ratio, total_len)
    key_sentences = [
        KeySentence(
            sentence_index=idx, text=sent.text, span=sent.span,
            score=round(sc, 4), reasons=reasons, source="summarizer",
        )
        for idx, sc, reasons, sent in selected
    ]
    summary_text = "".join(ks.text for ks in key_sentences)
    dominant_lang = _detect_dominant_language(sentences)
    dn, dt = _DEFAULT_WEIGHTS.get(dominant_lang, _DEFAULT_WEIGHTS["modern"])
    fusion_weights: dict[str, float] = {
        "nsp_weight": 0.0, "textrank_weight": 1.0,
        "default_nsp": round(dn, 4), "default_textrank": round(dt, 4),
    }
    return Summary(
        key_sentences=key_sentences, summary_text=summary_text,
        method="textrank_only", mode=mode,
        target_chars=target_chars, target_ratio=target_ratio,
        fusion_weights=fusion_weights, total_sentences=n_total,
    )

