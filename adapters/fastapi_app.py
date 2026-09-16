"""
FastAPI Adapter — Dev/Debug HTTP Interface.

Provides a human-readable REST API with Swagger UI for manual testing
and Studio UI prototyping. NOT for production traffic or inter-operator
communication.

Start::

    python adapters/fastapi_app.py

Then open http://localhost:8000/docs for the Swagger UI.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from pydantic import BaseModel, Field

from core.analyzer import analyze
from core.schema import NarrativeDocument


def _apply_dict(pipeline, dict_words: list[str]):
    """Apply custom dictionary to pipeline's tokenizer if words provided."""
    if dict_words and hasattr(pipeline, '__getitem__'):
        try:
            pipeline['tok/fine'].dict_combine = set(dict_words)
        except (KeyError, AttributeError):
            pass


def _discover_true_new_words(text: str, doc: NarrativeDocument, dict_combine: list[str]) -> list[str]:
    """Run new word discovery and filter to true new words.

    A "true new word" is a candidate that is NOT already a single token
    in the MTL output — i.e., MTL would split it but it should be merged.
    """
    from core.discoverer import discover as run_discover, discover_convseg
    from core.schema import Token

    # MTL tokens as-is (single tokens)
    mtl_token_set = {t.text for t in doc.content.tokens}

    # PMI + MTL discovery
    pmi_result = run_discover(
        text,
        mtl_tokens=doc.content.tokens,
        min_freq=1,
        max_candidates=30,
    )
    pmi_words = {c.word for c in pmi_result.candidates}

    # ConvSeg discovery
    convseg_result = discover_convseg(text)
    convseg_words = set(convseg_result["candidates"]) if convseg_result else set()

    # Merge all candidates
    all_candidates = pmi_words | convseg_words

    # True new words: not already in MTL output
    true_new = sorted(w for w in all_candidates if w not in mtl_token_set)

    return true_new

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Narrative Operator NLP",
    description="NLP analysis operator for NarrativeOS — FastAPI dev/debug interface.",
    version="1.0.0",
    docs_url=None,  # Disabled — using custom /docs with local Swagger UI assets
    redoc_url="/redoc",
    license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
)

# Mount local Swagger UI static assets (offline-friendly, no CDN dependency)
app.mount(
    "/static/swagger-ui",
    StaticFiles(directory=str(Path(__file__).resolve().parent.parent / "static" / "swagger-ui")),
    name="swagger-ui-static",
)


# ---------------------------------------------------------------------------
# Custom Swagger UI (offline, no CDN dependency)
# ---------------------------------------------------------------------------

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    """Custom Swagger UI with local assets (no CDN dependency)."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url or "/openapi.json",
        title=app.title + " - Swagger UI",
        swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/swagger-ui.css",
        oauth2_redirect_url="/docs/oauth2-redirect",
        init_oauth=False,
    )


@app.get("/docs/oauth2-redirect", include_in_schema=False)
async def custom_swagger_ui_oauth2_redirect():
    """OAuth2 redirect page for Swagger UI."""
    return get_swagger_ui_oauth2_redirect_html()


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Raw text to analyze",
        examples=["碳钢是钢的一种，具有高强度和高韧性。"],
    )
    source: str = Field(
        default="hanlp_v2",
        description="NLP engine identifier",
    )
    dict_combine: list[str] = Field(
        default=[],
        description="Custom dictionary words to force-combine during tokenization (e.g. ['碳钢','高强度'])",
        examples=[["碳钢", "高强度", "高韧性", "立方庭", "海淀区"]],
    )
    discover: bool = Field(
        default=False,
        description="Enable new word discovery (PMI + ConvSeg comparison)",
    )
    enhance: bool = Field(
        default=False,
        description="Auto-apply discovered new words as dict_combine and re-analyze",
    )
    language: str = Field(
        default="auto",
        description="Language mode: auto (per-sentence detection), modern, classical, english",
        examples=["auto", "modern", "classical", "english"],
    )
    entity_dict: dict[str, str] = Field(
        default_factory=dict,
        description="Optional entity dictionary for classical Chinese: {term: NSP_category}. "
                    "Example: {'北冥': 'LOCATION', '鲲': 'PERSON'}. "
                    "No hardcoded dictionaries — the caller supplies this.",
        examples=[{"北冥": "LOCATION", "鲲": "PERSON"}],
    )
    entity_categories: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Domain-specific keyword injection. The caller supplies domain "
                    "keywords at runtime — no hardcoded domain knowledge in the tool. "
                    "Format: {category: [keyword1, keyword2, ...]}. "
                    "Categories in EntityCategory.ALL are used as-is; unknown categories "
                    "are mapped to UNKNOWN. "
                    "Example: {'MATERIAL': ['石墨烯'], 'DISEASE': ['乳腺癌', '肿瘤']}. "
                    "Note: DISEASE/ANATOMY/FINDING etc. are not in the default schema — "
                    "use existing categories (PERSON/ORGANIZATION/LOCATION/FACILITY/PRODUCT/"
                    "MATERIAL/STANDARD/UNKNOWN) or your custom names (mapped to UNKNOWN).",
        examples=[{"MATERIAL": ["石墨烯"], "UNKNOWN": ["乳腺癌", "肿块"]}],
    )
    auto_discover_entities: bool = Field(
        default=False,
        description="If True, run new word discovery (PMI+MTL hybrid) and promote "
                    "high-score candidates to UNKNOWN entities. Uses statistical "
                    "methods (PMI mutual information + left/right entropy) for "
                    "unsupervised entity discovery. Default False — opt-in so users "
                    "control the recall/precision tradeoff. "
                    "Score threshold: 3.0+ (PMI). Source field: 'discover'.",
    )
    # Entity quality pipeline (F0/F1/F2) — optional, backward compatible.
    policy: dict | None = Field(
        default=None,
        description="Optional entity quality policy (F1 shape + F2 confidence/evidence). "
                    "When absent, F1/F2 are skipped (backward compatible). "
                    "Schema: {version, f1_shape: {section_ref: {enabled, patterns}, "
                    "demote_bare_number}, f2_confidence: {enabled, min_confidence: "
                    "{default, by_source}, demote_categories, keyword_require_injected}}.",
    )
    # Noun signal extraction (Step A/B) — optional, off by default.
    noun_signals: dict | None = Field(
        default=None,
        description="Optional noun-signal extraction config (Step A POS gating + "
                    "Step B syntactic role). When absent or enabled=false, no noun "
                    "signals are produced. Schema: {enabled, min_score, max_per_block, "
                    "pos_whitelist}.",
    )
    # Summary options
    summarize: bool = Field(
        default=False,
        description="If True, generate an extractive summary attached to content.summary",
    )
    summary_mode: str = Field(
        default="chars",
        description="Summary budget mode: 'chars' (fixed length) or 'ratio' (percentage of text)",
    )
    summary_chars: int = Field(
        default=30,
        description="Target character count for summary (used when summary_mode='chars')",
    )
    summary_ratio: float = Field(
        default=0.2,
        description="Target ratio of original text for summary (used when summary_mode='ratio')",
    )
    nsp_weight: float | None = Field(
        default=None,
        description="Override NSP feature weight for summary scoring (0.0-1.0)",
    )
    textrank_weight: float | None = Field(
        default=None,
        description="Override TextRank weight for summary scoring (0.0-1.0)",
    )
    # Title options
    generate_title: bool = Field(
        default=False,
        description="If True, generate a title attached to content.title",
    )
    title_mode: str = Field(
        default="chars",
        description="Title budget mode: 'chars' (fixed length) or 'ratio' (percentage of text)",
    )
    title_chars: int = Field(
        default=14,
        description="Target character count for title (used when title_mode='chars')",
    )
    title_ratio: float = Field(
        default=0.05,
        description="Target ratio of original text for title (used when title_mode='ratio')",
    )


class AnalyzeResponse(BaseModel):
    """Wraps NarrativeDocument for API serialization."""
    meta: dict
    content: dict
    true_new_words: list[str] | None = None

    @classmethod
    def from_doc(cls, doc: NarrativeDocument, true_new_words: list[str] | None = None) -> "AnalyzeResponse":
        return cls(
            meta=doc.meta.model_dump(),
            content=doc.content.model_dump(),
            true_new_words=true_new_words,
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Redirect to interactive demo page."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/demo")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "narrative-operator-nlp"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(request: AnalyzeRequest):
    """
    Analyze text and return NSP-standardized narrative atoms.

    - discover=True: returns true_new_words alongside normal results.
    - enhance=True:  auto-applies discovered new words as dict_combine,
                      re-analyzes, and returns enhanced results.
    """
    try:
        user_dict = set(request.dict_combine) if request.dict_combine else set()
        user_entity_dict = request.entity_dict if request.entity_dict else None
        user_entity_categories = (
            request.entity_categories if request.entity_categories else None
        )

        # Baseline analysis (with language mode)
        doc = analyze(
            request.text,
            dict_combine=user_dict if user_dict else None,
            language=request.language,
            entity_dict=user_entity_dict,
            entity_categories=user_entity_categories,
            auto_discover_entities=request.auto_discover_entities,
            summarize=request.summarize,
            summary_mode=request.summary_mode,
            summary_chars=request.summary_chars,
            summary_ratio=request.summary_ratio,
            nsp_weight=request.nsp_weight,
            textrank_weight=request.textrank_weight,
            generate_title=request.generate_title,
            title_mode=request.title_mode,
            title_chars=request.title_chars,
            title_ratio=request.title_ratio,
            policy=request.policy,
            noun_signals=request.noun_signals,
        )

        true_new_words: list[str] | None = None

        if request.enhance:
            # Discover true new words from baseline
            true_new_words = _discover_true_new_words(request.text, doc, request.dict_combine)
            if true_new_words:
                # Re-analyze with enhanced dict (user + discovered)
                enhanced_dict = user_dict | set(true_new_words)
                doc = analyze(
                    request.text,
                    dict_combine=enhanced_dict if enhanced_dict else None,
                    language=request.language,
                    entity_dict=user_entity_dict,
                    entity_categories=user_entity_categories,
                    auto_discover_entities=request.auto_discover_entities,
                    summarize=request.summarize,
                    summary_mode=request.summary_mode,
                    summary_chars=request.summary_chars,
                    summary_ratio=request.summary_ratio,
                    nsp_weight=request.nsp_weight,
                    textrank_weight=request.textrank_weight,
                    generate_title=request.generate_title,
                    title_mode=request.title_mode,
                    title_chars=request.title_chars,
                    title_ratio=request.title_ratio,
                    policy=request.policy,
                    noun_signals=request.noun_signals,
                )
        elif request.discover:
            true_new_words = _discover_true_new_words(request.text, doc, request.dict_combine)

        return AnalyzeResponse.from_doc(doc, true_new_words=true_new_words)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Extractive Summary Endpoint (standalone, no full analysis required)
# ---------------------------------------------------------------------------

class SummaryRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to summarize")
    mode: str = Field(default="chars", description="Budget mode: 'chars' or 'ratio'")
    target_chars: int = Field(default=30, description="Target character count (mode=chars)")
    target_ratio: float = Field(default=0.2, description="Target ratio (mode=ratio)")
    language: str = Field(default="auto", description="Language: auto, modern, classical, english")


@app.post("/analyze/summary")
async def analyze_summary(request: SummaryRequest):
    """Generate an extractive summary without full NLP analysis.

    Uses TextRank + position prior on raw text (no entities/events required).
    For NSP-enhanced summarization, use /analyze with summarize=True.
    """
    try:
        from core.summarizer import summarize_text
        summary = summarize_text(
            request.text,
            mode=request.mode,
            target_chars=request.target_chars,
            target_ratio=request.target_ratio,
            language=request.language,
        )
        return summary.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Title Generation Endpoint (standalone, no full analysis required)
# ---------------------------------------------------------------------------

class TitleRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text to generate title for")
    mode: str = Field(default="chars", description="Budget mode: 'chars' or 'ratio'")
    target_chars: int = Field(default=14, description="Target character count (mode=chars)")
    target_ratio: float = Field(default=0.05, description="Target ratio (mode=ratio)")
    language: str = Field(default="auto", description="Language: auto, modern, classical, english")


@app.post("/analyze/title")
async def analyze_title(request: TitleRequest):
    """Generate a title from raw text without full NLP analysis.

    Uses TextRank to find the most important sentence, then truncates.
    For entity-composition titles, use /analyze with generate_title=True.
    """
    try:
        from core.titler import generate_title_text
        title = generate_title_text(
            request.text,
            mode=request.mode,
            target_chars=request.target_chars,
            target_ratio=request.target_ratio,
            language=request.language,
        )
        return title.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Dependency Tree Data (for SVG visualization)
# ---------------------------------------------------------------------------

@app.post("/analyze/dep")
async def analyze_dep(request: AnalyzeRequest):
    """Return token list + dependency edges for frontend SVG rendering."""
    try:
        from core.analyzer import _get_modern_pipeline
        pipeline = _get_modern_pipeline()
        _apply_dict(pipeline, request.dict_combine)
        raw = pipeline(request.text)
        tokens = [{"id": i, "text": t, "pos": raw.get("pos/ctb", [""] * len(raw["tok/fine"]))[i] if i < len(raw.get("pos/ctb", [])) else "X"}
                  for i, t in enumerate(raw.get("tok/fine", []))]
        deps = [{"child": i, "head": int(d[0]) - 1, "rel": str(d[1])}
                for i, d in enumerate(raw.get("dep", [])) if isinstance(d, (list, tuple)) and len(d) >= 2]
        return {"tokens": tokens, "deps": deps, "text": request.text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# New Word Discovery
# ---------------------------------------------------------------------------

@app.post("/analyze/discover")
async def analyze_discover(request: AnalyzeRequest):
    """Discover potential new words using PMI + MTL hybrid + ConvSeg comparison.

    Returns candidates from three engines for comparison.
    """
    try:
        from core.analyzer import _get_modern_pipeline
        from core.discoverer import discover as discover_words, discover_convseg
        from core.schema import Token

        pipeline = _get_modern_pipeline()
        _apply_dict(pipeline, request.dict_combine)
        raw = pipeline(request.text)

        # Build MTL tokens
        pos_list = raw.get("pos/ctb", [])
        tok_list = raw.get("tok/fine", [])
        mtl_tokens = []
        cursor = 0
        for i, t in enumerate(tok_list):
            start = request.text.find(t, cursor)
            if start < 0:
                start = cursor
            mtl_tokens.append(Token(
                id=i, text=t,
                pos=pos_list[i] if i < len(pos_list) else "X",
                span=(start, start + len(t)),
                source="hanlp_v2",
            ))
            cursor = start + len(t)

        result = discover_words(
            request.text,
            mtl_tokens=mtl_tokens,
            min_freq=1,
            max_candidates=30,
        )

        # Try ConvSeg comparison
        convseg_result = discover_convseg(request.text)

        response = result.to_dict()
        if convseg_result:
            response["convseg"] = convseg_result
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------

@app.post("/analyze/pretty")
async def analyze_pretty(request: AnalyzeRequest):
    """Return raw HanLP Document pretty-print (same as demo notebooks)."""
    try:
        from core.analyzer import _get_modern_pipeline
        pipeline = _get_modern_pipeline()
        _apply_dict(pipeline, request.dict_combine)
        raw = pipeline(request.text)
        from hanlp_common.document import Document
        doc = Document(raw)
        return {"pretty": doc.to_pretty(), "text": request.text}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Demo Page — Interactive Visualization (static frontend in static/demo/)
# ---------------------------------------------------------------------------

app.mount(
    "/demo",
    StaticFiles(directory=str(Path(__file__).resolve().parent.parent / "static" / "demo"), html=True),
    name="demo-static",
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "adapters.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
