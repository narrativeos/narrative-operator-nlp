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
    docs_url="/docs",
    redoc_url="/redoc",
)


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

        # Baseline analysis (with language mode)
        doc = analyze(
            request.text,
            dict_combine=user_dict if user_dict else None,
            language=request.language,
            entity_dict=user_entity_dict,
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
# Demo Page — Interactive Visualization
# ---------------------------------------------------------------------------

DEMO_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Narrative Operator NLP — Demo</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0d1117;color:#c9d1d9;min-height:100vh}
header{background:#161b22;border-bottom:1px solid #30363d;padding:16px 24px;display:flex;align-items:center;gap:12px}
header h1{font-size:18px;color:#58a6ff}
header span{font-size:12px;color:#8b949e;background:#21262d;padding:2px 8px;border-radius:12px}
main{max-width:1200px;margin:0 auto;padding:24px}
.input-area{display:flex;gap:12px;margin-bottom:24px}
.input-area textarea{flex:1;min-height:60px;padding:12px;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#c9d1d9;font-size:15px;font-family:inherit;resize:vertical}
.input-area textarea:focus{outline:none;border-color:#58a6ff}
button{background:#238636;color:#fff;border:none;border-radius:6px;padding:10px 24px;font-size:14px;cursor:pointer;white-space:nowrap;transition:background .2s}
button:hover{background:#2ea043}
button:disabled{background:#21262d;color:#484f58;cursor:not-allowed}
.tabs{display:flex;gap:0;margin-bottom:16px;border-bottom:1px solid #30363d}
.tab{padding:8px 16px;font-size:13px;color:#8b949e;cursor:pointer;border-bottom:2px solid transparent;transition:all .2s}
.tab:hover{color:#c9d1d9}
.tab.active{color:#58a6ff;border-bottom-color:#58a6ff}
.panel{display:none}
.panel.active{display:block}
.card{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:20px;margin-bottom:16px}
.card h3{font-size:14px;color:#58a6ff;margin-bottom:12px}
.stats{display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap}
.stat{background:#21262d;padding:8px 16px;border-radius:6px;font-size:13px}
.stat b{color:#58a6ff}
.token{display:inline-block;padding:2px 6px;margin:2px;border-radius:3px;font-size:13px;cursor:default}
.token.NN{background:#1a3a1a}
.token.VC,.token.VV{background:#3a1a1a}
.token.NR{background:#1a1a3a}
.token.CD{background:#3a3a1a}
.token.DEG,.token.DEC{background:#2a2a2a}
.token.PU{color:#484f58}
.entity-card{background:#21262d;padding:8px 12px;margin:4px 0;border-radius:4px;font-size:13px;display:inline-block;margin-right:8px}
.entity-card .cat{background:#58a6ff;color:#0d1117;padding:1px 6px;border-radius:3px;font-size:11px;margin-right:6px}
.relation-row{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid #21262d;font-size:13px}
.relation-row .subj{color:#79c0ff}
.relation-row .pred{background:#1a3a1a;color:#7ee787;padding:1px 8px;border-radius:10px;font-size:11px}
.relation-row .obj{color:#79c0ff}
.relation-row .src{color:#484f58;font-size:11px;margin-left:auto}
pre.pretty{background:#0d1117;padding:16px;border-radius:6px;overflow-x:auto;font-size:11px;line-height:1.15;color:#c9d1d9}
.discover-item{display:inline-flex;align-items:center;gap:6px;background:#21262d;padding:6px 12px;margin:4px;border-radius:6px;font-size:13px;cursor:pointer;border:1px solid #30363d;transition:all .2s}
.discover-item:hover{border-color:#58a6ff}
.discover-item.selected{background:#1a3a1a;border-color:#238636}
.discover-item .score{font-size:10px;color:#484f58;margin-left:4px}
.discover-item .freq{font-size:10px;color:#e3b341;margin-left:2px}
.loading{text-align:center;padding:40px;color:#8b949e}
.error{color:#f85149;padding:12px;background:#3a1a1a;border-radius:6px}
.sample-btn{font-size:11px;padding:4px 8px;background:#21262d;color:#8b949e;border:1px solid #30363d;border-radius:4px;cursor:pointer;margin:2px}
.sample-btn:hover{color:#c9d1d9;border-color:#58a6ff}
.pattern-table{width:100%;border-collapse:collapse;font-size:12px}
.pattern-table th{text-align:left;padding:6px 8px;color:#8b949e;border-bottom:1px solid #30363d;font-weight:normal}
.pattern-table td{padding:6px 8px;border-bottom:1px solid #21262d}
.pattern-table .sent{max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pattern-table .tpl{color:#7ee787;font-family:monospace}
.pattern-table .pred{color:#e3b341}
.pattern-table .rel{color:#58a6ff}
.tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;margin:1px 2px}
.tag.negative{background:#3a1a1a;color:#f85149}
.tag.passive{background:#1a1a3a;color:#79c0ff}
.tag.active{background:#1a3a1a;color:#7ee787}
.tag.declarative{background:#21262d;color:#8b949e}
.tag.structural{background:#1a1a3a;color:#a5b4fc}
.tag.tier{background:#1a2e1a;color:#7ee787}
.tag.limit{background:#3a2a1a;color:#f0883e;cursor:help}
.tag.interrogative{background:#2a1a3a;color:#c084fc}
.tag.exclamatory{background:#3a1a2a;color:#f472b6}
.tag.imperative{background:#1a3a2a;color:#6ee7b7}
/* API Documentation Styles */
.api-endpoint{background:#21262d;border:1px solid #30363d;border-radius:6px;margin-bottom:16px;overflow:hidden}
.api-endpoint .method{padding:12px 16px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #30363d}
.api-endpoint .method .badge{display:inline-block;padding:2px 10px;border-radius:4px;font-size:12px;font-weight:600;font-family:monospace}
.api-endpoint .method .badge.get{background:#1a3a5c;color:#58a6ff}
.api-endpoint .method .badge.post{background:#1a3a1a;color:#7ee787}
.api-endpoint .method .path{font-family:monospace;font-size:14px;color:#c9d1d9}
.api-endpoint .method .summary{font-size:13px;color:#8b949e;margin-left:auto}
.api-endpoint .body{padding:12px 16px;display:none}
.api-endpoint .body.open{display:block}
.api-endpoint .body h4{font-size:12px;color:#58a6ff;margin:8px 0 4px}
.api-endpoint .body h4:first-child{margin-top:0}
.api-endpoint .body p, .api-endpoint .body li{font-size:12px;color:#c9d1d9;line-height:1.6}
.api-endpoint .body .field{display:flex;gap:8px;padding:2px 0;font-size:12px}
.api-endpoint .body .field .fname{color:#e3b341;font-family:monospace;min-width:100px}
.api-endpoint .body .field .ftype{color:#79c0ff;font-family:monospace;min-width:60px}
.api-endpoint .body .field .fdesc{color:#8b949e}
.api-endpoint .expand{background:transparent;border:none;color:#58a6ff;cursor:pointer;font-size:12px;padding:4px 8px;border-radius:4px}
.api-endpoint .expand:hover{background:#1a3a5c}
.api-doc-intro{font-size:13px;color:#8b949e;margin-bottom:16px;line-height:1.6}
.api-doc-intro code{background:#21262d;padding:1px 6px;border-radius:3px;font-size:12px;color:#7ee787}
/* Language Selector Styles */
.lang-selector{display:flex;gap:6px;margin-bottom:16px;align-items:center}
.lang-selector label{font-size:12px;color:#8b949e}
.lang-option{display:flex;align-items:center;gap:4px;padding:4px 10px;border:1px solid #30363d;border-radius:6px;font-size:12px;cursor:pointer;transition:all .2s;background:#21262d;color:#c9d1d9}
.lang-option:hover{border-color:#58a6ff}
.lang-option.active{background:#1a3a5c;border-color:#58a6ff;color:#58a6ff}
.lang-option input{display:none}
.lang-badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;margin:1px 2px}
.lang-badge.modern{background:#1a3a1a;color:#7ee787}
.lang-badge.classical{background:#3a2a1a;color:#e3b341}
.lang-badge.english{background:#1a1a3a;color:#79c0ff}
.lang-table{width:100%;border-collapse:collapse;font-size:12px}
.lang-table th{text-align:left;padding:6px 8px;color:#8b949e;border-bottom:1px solid #30363d;font-weight:normal}
.lang-table td{padding:6px 8px;border-bottom:1px solid #21262d}
.lang-table .sent{max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lang-table .classical{color:#e3b341}
.lang-table .modern{color:#7ee787}
.lang-table .english{color:#79c0ff}
.source-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
.source-dot.hanlp_v2{background:#58a6ff}
.source-dot.hanlp_lzh{background:#e3b341}
.source-dot.en_modernbert{background:#79c0ff}
/* Model Status Indicators */
.model-status{display:flex;gap:8px;margin:8px 0 12px;flex-wrap:wrap}
.model-status .stat-item{display:flex;align-items:center;gap:4px;padding:3px 8px;border-radius:5px;font-size:11px;background:#161b22;border:1px solid #30363d}
.model-status .stat-item .dot{width:6px;height:6px;min-width:6px;border-radius:50%;display:inline-block}
.model-status .stat-item .dot.idle{background:#484f58}
.model-status .stat-item .dot.loading{background:#e3b341;animation:pulse 1s infinite}
.model-status .stat-item .dot.ready{background:#7ee787}
.model-status .stat-item .dot.error{background:#f85149}
.model-status .stat-item .label{color:#8b949e;font-size:11px}
.model-status .stat-item .title{color:#c9d1d9;font-size:11px}
@keyframes pulse{0%,100%{opacity:0.4}50%{opacity:1}}
/* Sample Language Cards */
.sample-cards{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
.sample-card{display:flex;align-items:center;gap:8px;padding:8px 14px;background:#21262d;border:1px solid #30363d;border-radius:8px;cursor:pointer;transition:all .2s;font-size:12px;color:#c9d1d9;flex:1;min-width:160px}
.sample-card:hover{border-color:#58a6ff;background:#1a3a5c;color:#58a6ff}
.sample-card .lang-tag{font-size:10px;padding:1px 6px;border-radius:4px;background:#1a3a1a;color:#7ee787}
.sample-card .lang-tag.classical{background:#3a2a1a;color:#e3b341}
.sample-card .lang-tag.english{background:#1a1a3a;color:#79c0ff}
.sample-card .preview{color:#8b949e;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
</style>
</head>
<body>
<header>
<h1>🧠 Narrative Operator NLP</h1>
<span>HanLP MTL Demo</span>
</header>
<main>
<div class="model-status" id="modelStatus">
<div class="stat-item" id="statusModern"><span class="dot idle" id="dotModern"></span><span class="title">🌐 现代</span><span class="label" id="labelModern">等待中</span></div>
<div class="stat-item" id="statusClassical"><span class="dot idle" id="dotClassical"></span><span class="title">🏯 古汉语</span><span class="label" id="labelClassical">等待中</span></div>
<div class="stat-item" id="statusEnglish"><span class="dot idle" id="dotEnglish"></span><span class="title">🇬🇧 英文</span><span class="label" id="labelEnglish">等待中</span></div>
</div>
<div class="sample-cards" id="sampleCards">
<div class="sample-card" onclick="setLanguageAndAnalyze('碳钢是钢的一种，具有高强度和高韧性。北京立方庭位于海淀区。','auto')">
  <span class="lang-tag">📄 现代</span>
  <span class="preview">碳钢是钢的一种，具有高强度和高韧性。北京立方庭位于海淀区。</span>
</div>
<div class="sample-card" onclick="setLanguageAndAnalyze('北冥有鱼，其名为鲲。鲲之大，不知其几千里也。','classical')">
  <span class="lang-tag classical">🏯 古汉语</span>
  <span class="preview">北冥有鱼，其名为鲲。鲲之大，不知其几千里也。</span>
</div>
<div class="sample-card" onclick="setLanguageAndAnalyze('Apple was founded by Steve Jobs in California. Microsoft is based in Redmond.','english')">
  <span class="lang-tag english">🇬🇧 英文</span>
  <span class="preview">Apple was founded by Steve Jobs in California. Microsoft is based in Redmond.</span>
</div>
</div>
<div class="input-area">
<textarea id="input" placeholder="输入中文文本进行分析...">碳钢是钢的一种，具有高强度和高韧性。北京立方庭位于海淀区。</textarea>
<div style="display:flex;flex-direction:column;gap:6px">
<button id="analyzeBtn" onclick="analyze()">🔍 分析</button>
<select id="discoverMode" onchange="analyze()" style="padding:4px 8px;background:#0d1117;border:1px solid #30363d;border-radius:4px;color:#c9d1d9;font-size:11px;cursor:pointer">
<option value="">🔍 新词发现：关闭</option>
<option value="discover">🔍 新词发现：仅发现</option>
<option value="enhance">⚡ 新词发现：强化模式</option>
</select>
</div>
</div>
<div class="input-area" style="margin-bottom:16px">
<input id="dictInput" value="碳钢 高强度 高韧性 立方庭" placeholder="自定义词典（用空格/逗号/换行分隔，如：碳钢 高强度 立方庭）" style="flex:1;padding:8px 12px;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#c9d1d9;font-size:13px;font-family:inherit">
</div>
<div class="lang-selector">
<label>📖 语言模式:</label>
<label class="lang-option active" id="langAuto" onclick="setLanguage('auto')"><input type="radio" name="lang" value="auto" checked>🔄 自动识别</label>
<label class="lang-option" id="langModern" onclick="setLanguage('modern')"><input type="radio" name="lang" value="modern">📄 现代汉语</label>
<label class="lang-option" id="langClassical" onclick="setLanguage('classical')"><input type="radio" name="lang" value="classical">🏯 古汉语</label>
<label class="lang-option" id="langEnglish" onclick="setLanguage('english')"><input type="radio" name="lang" value="english">🇬🇧 English</label>
</div>
<div class="tabs">
<div class="tab active" onclick="switchTab('nsp')">📊 NSP 结构化</div>
<div class="tab" onclick="switchTab('pretty')">🎨 HanLP 原生可视化</div>
<div class="tab" onclick="switchTab('depsvg')">🧬 依存树 SVG</div>
<div class="tab" onclick="switchTab('discover')">🔍 新词发现</div>
<div class="tab" onclick="switchTab('patterns')">📊 句式模式</div>
<div class="tab" onclick="switchTab('langdetect')">🏯 语言检测</div>
<div class="tab" onclick="switchTab('json')">{ } JSON Raw</div>
<div class="tab" onclick="switchTab('api')">📋 API 接口</div>
</div>
<div id="nsp" class="panel active"></div>
<div id="pretty" class="panel"></div>
<div id="depsvg" class="panel"></div>
<div id="discover" class="panel"></div>
<div id="patterns" class="panel"></div>
<div id="langdetect" class="panel"></div>
<div id="json" class="panel"></div>
<div id="api" class="panel"></div>
</main>
<script>
let _currentLang='auto';
function setLanguage(lang){
    _currentLang=lang;
    document.querySelectorAll('.lang-option').forEach(el=>el.classList.toggle('active', el.id==='lang'+lang.charAt(0).toUpperCase()+lang.slice(1)));
    analyze();
}

async function analyze(){
    const text=document.getElementById('input').value.trim();
    if(!text) return;
    const dictRaw=document.getElementById('dictInput').value.trim();
    const dictCombine=dictRaw?dictRaw.split(/[\s,，;；]+/).filter(w=>w) :[];
    const mode=document.getElementById('discoverMode').value;
    const discover=mode==='discover';
    const enhance=mode==='enhance';
    const language=_currentLang;
    const body={text, dict_combine: dictCombine, discover, enhance, language, entity_dict: {}};
    const btn=document.getElementById('analyzeBtn');
    btn.disabled=true; btn.textContent='分析中...';
    ['nsp','pretty','depsvg','discover','patterns','langdetect','json'].forEach(id=>document.getElementById(id).innerHTML='<div class=\"loading\">⏳ 分析中...</div>');

    // Independent fetches — one failure doesn't block others
    const post=(url,body)=>fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json()).catch(e=>({_error:e.message}));
    const [r1,r2,r3,r4]=await Promise.all([
        post('/analyze',body),
        post('/analyze/pretty',body),
        post('/analyze/dep',body),
        post('/analyze/discover',body)
    ]);
    if(r1._error) document.getElementById('nsp').innerHTML='<div class="error">分析失败: '+r1._error+'</div>';
    else renderNSP(r1);
    if(r2._error) document.getElementById('pretty').innerHTML='<div class="error">分析失败: '+r2._error+'</div>';
    else renderPretty(r2);
    if(r3._error) document.getElementById('depsvg').innerHTML='<div class="error">分析失败: '+r3._error+'</div>';
    else renderDepSVG(r3);    if(r4._error) document.getElementById('discover').innerHTML='<div class=\"error\">新词发现失败: '+r4._error+'</div>';
    else renderDiscover(r4);    if(!r1._error) renderJSON(r1);
    const patData=r1.content&&r1.content.patterns;
    if(patData&&patData.length) renderPatterns(patData); else document.getElementById('patterns').innerHTML='<div class="card"><h3>📊 句式模式</h3><span style="color:#484f58">无模式数据</span></div>';
    renderLangDetect(r1);
    btn.disabled=false; btn.textContent='🔍 分析';
}

function renderNSP(data){
    if(!data||!data.content){ document.getElementById('nsp').innerHTML='<div class="error">分析失败：服务器未响应</div>'; return; }
    const c=data.content;
    const meta=data.meta||{};
    const trueNew=data.true_new_words||[];

    // Language mode badge
    const langMode=meta.language_mode||'auto';
    const langBadge=langMode==='classical'?'<span class="lang-badge classical">🏯 古汉语</span>'
        :langMode==='modern'?'<span class="lang-badge modern">📄 现代汉语</span>'
        :langMode==='english'?'<span class="lang-badge english">🇬🇧 English</span>'
        :'<span class="lang-badge" style="background:#1a1a3a;color:#79c0ff">🔄 自动识别</span>';

    // Language stats
    const langSents=c.sentences||[];
    const classicalCount=langSents.filter(s=>s.label==='classical').length;
    const modernCount=langSents.filter(s=>s.label==='modern').length;
    const englishCount=langSents.filter(s=>s.label==='english').length;

    const tokens=c.tokens.map(t=>{
        const pct=Math.round((t.confidence||1)*100);
        const color=pct>=95?'#7ee787':pct>=80?'#e3b341':'#f85149';
        const sourceDot=t.source==='hanlp_lzh'?'<span class="source-dot hanlp_lzh" title="古汉语模型"></span>'
            :t.source==='en_modernbert'?'<span class="source-dot en_modernbert" title="英文模型"></span>'
            :'<span class="source-dot hanlp_v2" title="现代汉语模型"></span>';
        return `<span class="token ${t.pos}" title="POS:${t.pos} span:${t.span} conf:${pct}% source:${t.source}">${sourceDot}${t.text}<sub style="color:${color};font-size:0.65em">${pct}</sub></span>`;
    }).join('');

    const entities=c.entities.map(e=>{
        const pct=Math.round((e.confidence||1)*100);
        const color=pct>=95?'#7ee787':pct>=80?'#e3b341':'#f85149';
        const attrs=(e.attributes||[]).map(a=>`<small style="color:#e3b341;margin-left:2px">${a.key}=${a.value||'?'}</small>`).join('');
        return `<span class="entity-card"><span class="cat">${e.category}</span>${e.text}${attrs} <small style="color:#484f58">[${e.span[0]}:${e.span[1]}]</small> <small style="color:${color}">${pct}%</small></span>`;
    }).join('')||'<span style="color:#484f58">-</span>';

    const relations=c.relations.map(r=>`<div class="relation-row"><span class="subj">${r.subject}</span> &rarr; <span class="pred">${r.predicate}</span> &rarr; <span class="obj">${r.object}</span><span class="src">${r.source}</span></div>`).join('')||'<span style="color:#484f58">-</span>';

    let newWordsHtml='';
    const mode=document.getElementById('discoverMode').value;
    if(trueNew.length>0){
        const label=mode==='enhance'?'⚡ 强化模式已应用':'🆕 发现真新词';
        const borderColor=mode==='enhance'?'#238636':'#e3b341';
        const titleColor=mode==='enhance'?'#7ee787':'#e3b341';
        const badges=trueNew.map(w=>`<span class="discover-item" onclick="toggleDictWord(this,'${esc(w)}');event.stopPropagation()" style="border-color:${borderColor}" title="点击加入自定义词典">${mode==='enhance'?'✅':'🆕'} ${esc(w)}</span>`).join('');
        newWordsHtml=`<div class="card" style="border-color:${borderColor}"><h3 style="color:${titleColor}">${label} <small style="color:#8b949e;font-weight:normal">(点击加入自定义词典)</small></h3><div>${badges}</div><button onclick="applyDict()" style="margin-top:8px;font-size:12px;padding:6px 16px">📋 一键应用并重新分析</button></div>`;
    }else if(data.true_new_words!==undefined&&data.true_new_words!==null){
        newWordsHtml='<div class="card"><h3>🔍 新词发现</h3><span style="color:#484f58">未发现真新词（所有候选词已在分词结果中）</span></div>';
    }

    const langStats=langSents.length>0?`
    <div class="stat">${langBadge}</div>
    <div class="stat">🏯 <b>${classicalCount}</b> 古汉语句</div>
    <div class="stat">📄 <b>${modernCount}</b> 现代语句</div>
    <div class="stat">🇬🇧 <b>${englishCount}</b> 英文句</div>`:'';

    document.getElementById('nsp').innerHTML=`
    <div class="stats">
    <div class="stat"><b>${c.tokens.length}</b> tokens</div>
    <div class="stat"><b>${c.entities.length}</b> entities</div>
    <div class="stat"><b>${c.relations.length}</b> relations</div>
    <div class="stat">source: <b>${data.meta.source}</b></div>
    ${langStats}
    </div>
    ${newWordsHtml}
    <div class="card"><h3>📝 分词 & POS <span style="font-size:11px;color:#484f58">●蓝=现代 · ●黄=古汉语 · ●浅蓝=英文</span></h3><div style="line-height:2">${tokens}</div></div>
    <div class="card"><h3>🏷️ 实体 Entities</h3><div>${entities}</div></div>
    <div class="card"><h3>🔗 关系 Relations</h3><div>${relations}</div></div>`;
}

function renderPretty(data){
    if(!data||!data.pretty){
        const err=data&&data.detail?data.detail:(data&&data._error?data._error:'无 pretty 字段');
        document.getElementById('pretty').innerHTML=`<div class="error">分析失败: ${esc(err)}</div>`;
        return;
    }
    document.getElementById('pretty').innerHTML=`<div class="card"><pre class="pretty">${escapeHtml(data.pretty)}</pre></div>`;
}

function renderDepSVG(data){
    if(!data||!data.tokens){ document.getElementById('depsvg').innerHTML='<div class="card">分析失败</div>'; return; }
    const tokens=data.tokens, deps=data.deps;
    if(!tokens.length){ document.getElementById('depsvg').innerHTML='<div class="card">无数据</div>'; return; }

    const W=Math.max(800,tokens.length*56), H=280, bottomY=H-30, charW=12;
    const colors={'nsubj':'#58a6ff','dobj':'#79c0ff','root':'#f78166','nn':'#7ee787','amod':'#d2a8ff',
        'advmod':'#ffa657','conj':'#e3b341','top':'#f78166','attr':'#56d364','assmod':'#a5d6ff','assm':'#a5d6ff',
        'punct':'#484f58','nummod':'#ff7b72','cc':'#8b949e','prep':'#c9d1d9','tmod':'#79c0ff','plmod':'#79c0ff',
        'rcmod':'#d2a8ff','nsubjpass':'#58a6ff'};

    let svg=`<svg width="${W}" height="${H}" style="background:#0d1117;border-radius:6px;font-family:monospace;max-width:100%">`;

    // Token nodes at bottom
    const spacing=(W-40)/(tokens.length-1||1);
    tokens.forEach((t,i)=>{
        const x=20+i*spacing, y=bottomY, tw=t.text.length*charW+16;
        svg+=`<rect x="${x-tw/2}" y="${y-14}" width="${tw}" height="24" rx="4" fill="#21262d" stroke="#30363d"/>`;
        svg+=`<text x="${x}" y="${y+3}" text-anchor="middle" fill="#c9d1d9" font-size="13">${esc(t.text)}</text>`;
        svg+=`<text x="${x}" y="${y-18}" text-anchor="middle" fill="#484f58" font-size="9">${esc(t.pos)}</text>`;
    });

    // Dependency arrows
    const usedHeights={};
    deps.forEach(d=>{
        if(d.head<0||d.head>=tokens.length) return;
        const cx=20+d.child*spacing, hx=20+d.head*spacing;
        const key=Math.min(d.child,d.head)+'_'+Math.max(d.child,d.head);
        usedHeights[key]=(usedHeights[key]||0)+1;
        const arcH=Math.max(30,50-usedHeights[key]*12);
        const mid=(cx+hx)/2, dir=d.head>d.child?1:-1;
        const color=colors[d.rel]||'#8b949e';
        const sy=bottomY-24;

        // Curved arrow
        const path=`M${cx},${sy} C${cx},${sy-arcH} ${hx},${sy-arcH} ${hx},${sy}`;
        svg+=`<path d="${path}" fill="none" stroke="${color}" stroke-width="1.5" marker-end="url(#arrow${d.child}_${d.head})" opacity="0.8"/>`;
        // Arrowhead
        svg+=`<defs><marker id="arrow${d.child}_${d.head}" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="${color}"/></marker></defs>`;
        // Label
        svg+=`<text x="${mid}" y="${sy-arcH-4}" text-anchor="middle" fill="${color}" font-size="9">${esc(d.rel)}</text>`;
    });

    // Root marker
    const rootDep=deps.find(d=>d.head<0);
    if(rootDep){
        const rx=20+rootDep.child*spacing;
        svg+=`<text x="${rx}" y="${bottomY-40}" text-anchor="middle" fill="#f78166" font-size="10">ROOT</text>`;
    }

    svg+=`</svg>`;
    document.getElementById('depsvg').innerHTML=`<div class="card"><h3>🧬 依存句法树 (SVG)</h3><div style="overflow-x:auto">${svg}</div></div>`;
}

function renderDiscover(data){
    if(!data||!data.candidates){ document.getElementById('discover').innerHTML='<div class="card">未发现候选新词</div>'; return; }
    const candidates=data.candidates;
    const convseg=data.convseg;

    // PMI+MTL candidates
    const pmiItems=candidates.length?candidates.map(c=>{
        const color=c.score>=5?'#7ee787':c.score>=3?'#e3b341':'#f85149';
        return `<span class="discover-item" onclick="toggleDictWord(this,'${esc(c.word)}')" title="点击添加到自定义词典">${esc(c.word)}<span class="score" style="color:${color}">${c.score.toFixed(1)}</span></span>`;
    }).join(''):'<span style="color:#484f58">-</span>';

    // Coarse ELECTRA comparison (PyTorch, no TF needed)
    let convsegHtml='';
    if(convseg){
        if(convseg.candidates&&convseg.candidates.length){
            convsegHtml=convseg.candidates.map(w=>{
                const inPmi=candidates.some(c=>c.word===w);
                const style=inPmi?'background:#1a3a1a;border-color:#238636':'';
                return `<span class=\"discover-item\" style=\"${style}\" onclick=\"toggleDictWord(this,'${esc(w)}')\" title=\"点击添加到自定义词典\">${esc(w)}<span class=\"score\" style=\"color:#58a6ff\">粗分</span></span>`;
            }).join('')||'<span style=\"color:#484f58\">-</span>';
        } else {
            convsegHtml='<span style=\"color:#484f58;font-size:11px\">粗分模型未加载</span>';
        }
    } else {
        convsegHtml='<span style=\"color:#484f58;font-size:11px\">粗分模型未安装</span>';
    }

    document.getElementById('discover').innerHTML=`
    <div class="card">
      <h3>🔍 新词发现 <small style="color:#484f58;font-weight:normal">(点击候选词加入自定义词典)</small></h3>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
        <div>
          <h4 style="font-size:13px;color:#e3b341;margin-bottom:8px">🧬 PMI + MTL 混合</h4>
          <div style="margin-bottom:8px">${pmiItems}</div>
        </div>
        <div>
          <h4 style="font-size:13px;color:#58a6ff;margin-bottom:8px">🧠 粗分 ELECTRA 对比</h4>
          <div style="margin-bottom:8px">${convsegHtml}</div>
        </div>
      </div>
      <button onclick="applyDict()" style="margin-top:8px">📋 应用选中词典并重新分析</button>
      <button onclick="clearDictSelection()" style="margin-top:8px;margin-left:8px;background:#21262d;color:#c9d1d9">清除选择</button>
    </div>`;
    window._discoverCandidates=candidates;
}

let selectedDictWords=[];
function toggleDictWord(el,word){
    el.classList.toggle('selected');
    if(el.classList.contains('selected')){
        if(!selectedDictWords.includes(word)) selectedDictWords.push(word);
    }else{
        selectedDictWords=selectedDictWords.filter(w=>w!==word);
    }
    const dictInput=document.getElementById('dictInput');
    dictInput.value=selectedDictWords.join(' ');
}

function clearDictSelection(){
    selectedDictWords=[];
    document.getElementById('dictInput').value='';
    document.querySelectorAll('.discover-item.selected').forEach(el=>el.classList.remove('selected'));
}

function applyDict(){
    if(selectedDictWords.length) document.getElementById('dictInput').value=selectedDictWords.join(' ');
    analyze();
}

function renderPatterns(patterns){
    if(!patterns||!patterns.length){ document.getElementById('patterns').innerHTML='<div class="card"><span style="color:#484f58">无句式数据</span></div>'; return; }

    const rows=patterns.map(p=>{
        const rels=p.relation_summary.join(', ')||'-';
        const structTag = p.structural_type==='unknown' 
            ? '<span class="tag structural">?</span>'
            : p.structural_type==='subject_predicate'
                ? '<span class="tag active">主谓</span>'
                : '<span class="tag structural">非主谓</span>';
        const sentAbbr = p.sentence.length > 28 ? p.sentence.slice(0,26)+'…' : p.sentence;
        return `<tr>
          <td class="sent" title="${esc(p.sentence)}">${esc(sentAbbr)}</td>
          <td class="tpl">${esc(p.template)||'<span style="color:#484f58">—</span>'}</td>
          <td class="pred">${esc(p.predicates.join(', '))||'<span style="color:#484f58">—</span>'}</td>
          <td class="rel">${esc(rels)}</td>
          <td style="text-align:center">${structTag}</td>
          <td style="text-align:center;color:#484f58">${p.attribute_count>0?'⚡'+p.attribute_count:'0'}</td>
          <td style="text-align:center;color:#484f58">${p.word_count||'-'}</td>
          <td style="text-align:center;color:#484f58">${p.clause_count||'-'}</td>
          <td style="text-align:center"><span class="tag tier">${esc(p.sentence_length_tier)}</span></td>
        </tr>`;
    }).join('');

    document.getElementById('patterns').innerHTML=`
    <div class="card">
      <h3>📊 句式模式 <small style="color:#484f58;font-weight:normal">(可按模板聚合统计)</small></h3>
      <div style="overflow-x:auto">
      <table class="pattern-table">
        <thead><tr>
          <th>句子</th>
          <th>模板 Template</th>
          <th>谓词</th>
          <th>关系</th>
          <th>结构</th>
          <th>属性</th>
          <th>词数</th>
          <th>分句</th>
          <th>句长</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
      </div>
    </div>`;
}

function renderLangDetect(data){
    if(!data||!data.content||!data.content.sentences){
        document.getElementById('langdetect').innerHTML='<div class="card"><span style="color:#484f58">无语言检测数据</span></div>';
        return;
    }
    const sents=data.content.sentences;
    const mode=data.meta.language_mode||'auto';
    const modeLabel=mode==='classical'?'🏯 古汉语 (强制)':mode==='modern'?'📄 现代汉语 (强制)':mode==='english'?'🇬🇧 English (强制)':'🔄 自动识别';

    const rows=sents.map((s,i)=>{
        const label=s.label==='classical'?'🏯 古汉语':s.label==='english'?'🇬🇧 English':'📄 现代汉语';
        const cls=s.label==='classical'?'classical':s.label==='english'?'english':'modern';
        const pct=Math.round(s.confidence*100);
        const barWidth=Math.round(s.confidence*100);
        const barColor=s.label==='classical'?'#e3b341':s.label==='english'?'#79c0ff':'#7ee787';
        const sentShort=s.text.length>40?s.text.slice(0,38)+'…':s.text;
        return `<tr>
            <td>${i+1}</td>
            <td class="sent" title="${esc(s.text)}">${esc(sentShort)}</td>
            <td class="${cls}"><b>${label}</b></td>
            <td>
                <div style="background:#21262d;border-radius:4px;overflow:hidden;width:80px;display:inline-block;vertical-align:middle">
                    <div style="width:${barWidth}%;height:12px;background:${barColor};border-radius:4px"></div>
                </div>
                <span style="font-size:11px;color:#484f58;margin-left:4px">${pct}%</span>
            </td>
            <td style="font-size:11px;color:#484f58">[${s.span[0]}:${s.span[1]}]</td>
        </tr>`;
    }).join('');

    document.getElementById('langdetect').innerHTML=`
    <div class="card">
      <h3>🏯 语言检测 <small style="color:#484f58;font-weight:normal">${modeLabel}</small></h3>
      <p style="font-size:12px;color:#8b949e;margin-bottom:12px">
        基于多特征加权评分（虚词密度、人称代词、否定模式、句末语气词等）对每个句子进行文言/现代文判定。
      </p>
      <div style="overflow-x:auto">
      <table class="lang-table">
        <thead><tr>
          <th>#</th>
          <th>句子</th>
          <th>判定结果</th>
          <th>置信度</th>
          <th>位置</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
      </div>
    </div>`;
}

function renderJSON(data){
    document.getElementById('json').innerHTML=`<div class="card"><pre class="pretty">${escapeHtml(JSON.stringify(data,null,2))}</pre></div>`;
}

function switchTab(id){
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
    document.querySelector(`.tab[onclick="switchTab('${id}')"]`).classList.add('active');
    document.getElementById(id).classList.add('active');
    if(id==='api') renderAPIDocs();
}

function setSample(text){
    document.getElementById('input').value=text;
    analyze();
}

function setLanguageAndAnalyze(text, lang){
    document.getElementById('input').value=text;
    setLanguage(lang);
}

function renderAPIDocs(){
    if(document.querySelector('#api .api-endpoint')) return; // already rendered
    const endpoints=[
        {method:'GET',path:'/',badge:'get',summary:'重定向到交互式 Demo 页面'},
        {method:'GET',path:'/health',badge:'get',summary:'健康检查端点',desc:'<p>返回服务运行状态，用于 Docker 健康检查和监控。</p>',resp:'<pre class="pretty">{"status": "ok", "service": "narrative-operator-nlp"}</pre>',ex:'curl http://localhost:8000/health'},
        {method:'POST',path:'/analyze',badge:'post',summary:'NLP 分析主端点 — 返回 NSP 标准化的叙事原子',
            desc:'<p>对输入文本执行完整 NLP 流水线（分词、词性标注、命名实体识别、依存分析、关系抽取、句式模式分析），返回 NSP 标准化结果。</p>',
            fields:[
                {name:'text',type:'string',desc:'(必填) 待分析的中文文本',ex:'"碳钢是钢的一种，具有高强度和高韧性。"'},
                {name:'source',type:'string',desc:'NLP 引擎标识',default:'"hanlp_v2"'},
                {name:'dict_combine',type:'string[]',desc:'自定义词典 — 强制合并的分词单元',default:'[]',ex:'["碳钢","高强度"]'},
                {name:'discover',type:'boolean',desc:'启用新词发现（PMI + ConvSeg 比对）',default:'false'},
                {name:'enhance',type:'boolean',desc:'强化模式 — 自动应用发现的新词并重新分析',default:'false'},
                {name:'language',type:'string',desc:'语言模式: auto(自动检测), modern(现代汉语), classical(古汉语), english(英文)',default:'"auto"',ex:'"classical"'},
            ],
            resp:'<pre class="pretty">{\\n  "meta": {"source": "hanlp_v2", "timestamp": "..."},\\n  "content": {\\n    "tokens": [...],\\n    "entities": [...],\\n    "relations": [...],\\n    "patterns": [...]\\n  },\\n  "true_new_words": [...] | null\\n}</pre>',
            ex:'curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d \\'{"text":"北冥有鱼，其名为鲲。","language":"classical"}\\''},
        {method:'POST',path:'/analyze/dep',badge:'post',summary:'依存句法分析 — 返回 token 列表 + 依存边',
            desc:'<p>专为前端 SVG 渲染优化的端点，返回扁平化的 token 列表和依存关系边。</p>',
            fields:[
                {name:'text',type:'string',desc:'(必填) 待分析的中文文本'},
                {name:'dict_combine',type:'string[]',desc:'自定义词典',default:'[]'},
            ],
            resp:'<pre class="pretty">{\\n  "tokens": [{"id": 0, "text": "碳钢", "pos": "NN"}, ...],\\n  "deps": [{"child": 0, "head": 1, "rel": "nsubj"}, ...],\\n  "text": "碳钢是钢的一种..."\\n}</pre>',
            ex:'curl -X POST http://localhost:8000/analyze/dep -H "Content-Type: application/json" -d \\'{"text":"碳钢是钢的一种。"}\\''},
        {method:'POST',path:'/analyze/discover',badge:'post',summary:'新词发现 — PMI + MTL 混合 + ConvSeg 比对',
            desc:'<p>使用 PMI 逐点互信息、MTL 分词对比和 ConvSeg（PKU_NAME）三种引擎发现潜在的未登录词。</p>',
            fields:[
                {name:'text',type:'string',desc:'(必填) 待分析的中文文本'},
                {name:'dict_combine',type:'string[]',desc:'自定义词典',default:'[]'},
            ],
            resp:'<pre class="pretty">{\\n  "candidates": [{"word": "高强度", "score": 8.5, "freq": 3}, ...],\\n  "convseg": {"candidates": ["立方庭", ...]}\\n}</pre>',
            ex:'curl -X POST http://localhost:8000/analyze/discover -H "Content-Type: application/json" -d \\'{"text":"碳钢是钢的一种，具有高强度。"}\\''},
        {method:'POST',path:'/analyze/pretty',badge:'post',summary:'HanLP 原生可视化 — 返回 pretty-print 文本',
            desc:'<p>返回 HanLP Document 的 <code>to_pretty()</code> 文本输出，与 Jupyter Notebook 中的展示效果一致。</p>',
            fields:[
                {name:'text',type:'string',desc:'(必填) 待分析的中文文本'},
                {name:'dict_combine',type:'string[]',desc:'自定义词典',default:'[]'},
            ],
            resp:'<pre class="pretty">  tok/fine: [碳钢, 是, 钢, 的, 一种, ，, 具有, 高强度, 和, 高韧性, 。]\\n  pos/ctb: [NN, VC, NN, DEG, CD, PU, VV, NN, CC, NN, PU]\\n  ...</pre>',
            ex:'curl -X POST http://localhost:8000/analyze/pretty -H "Content-Type: application/json" -d \\'{"text":"碳钢是钢的一种。"}\\''},
    ];
    let html=`<div class="api-doc-intro">
      <p>Narrative Operator NLP 提供以下 REST API 端点，所有请求均通过 <code>HTTP POST</code> 或 <code>GET</code> 访问。
      自动生成的 API 文档请访问 <a href="/docs" style="color:#58a6ff">Swagger UI</a> 或 <a href="/redoc" style="color:#58a6ff">ReDoc</a>。</p>
    </div>`;
    endpoints.forEach((ep,i)=>{
        let fieldsHtml='';
        if(ep.fields){
            fieldsHtml=`<h4>📥 请求参数</h4>`+ep.fields.map(f=>{
                const extra=f.default!==undefined?` <span style="color:#484f58">默认: ${f.default}</span>`:'';
                const ex=f.ex?`<br><span style="color:#484f58">示例: ${f.ex}</span>`:'';
                return `<div class="field"><span class="fname">${f.name}</span><span class="ftype">${f.type}</span><span class="fdesc">${f.desc}${extra}${ex}</span></div>`;
            }).join('');
        }
        const respHtml=ep.resp?`<h4>📤 响应示例</h4>${ep.resp}`:'';
        const exHtml=ep.ex?`<h4>🔧 cURL 示例</h4><pre class="pretty" style="font-size:11px">${ep.ex}</pre>`:'';
        html+=`<div class="api-endpoint">
          <div class="method" onclick="toggleEndpoint(${i})">
            <span class="badge ${ep.badge}">${ep.method}</span>
            <span class="path">${ep.path}</span>
            <span class="summary">${ep.summary}</span>
            <span class="expand">▼</span>
          </div>
          <div class="body" id="api-body-${i}">
            ${ep.desc||''}
            ${fieldsHtml}
            ${respHtml}
            ${exHtml}
          </div>
        </div>`;
    });
    document.getElementById('api').innerHTML=html;
}
window.toggleEndpoint=function(i){
    const body=document.getElementById('api-body-'+i);
    body.classList.toggle('open');
    const expand=body.parentElement.querySelector('.expand');
    expand.textContent=body.classList.contains('open')?'▲':'▼';
}

function escapeHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
const esc=escapeHtml;

// On load: warm up all three models in parallel, show loading status
window.onload=async function(){
    _setModelStatus('modern','loading','⏳ 加载中…');
    _setModelStatus('classical','loading','⏳ 加载中…');
    _setModelStatus('english','loading','⏳ 加载中…');

    // Modern Chinese
    try{
        const r=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:'碳钢是钢的一种，具有高强度和高韧性。北京立方庭位于海淀区。',language:'auto'})});
        const d=await r.json();
        _setModelStatus('modern','ready','✅ 已就绪');
        _renderResults(d);
    }catch(e){_setModelStatus('modern','error','❌ 失败');}

    // Classical Chinese
    try{
        const r=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:'北冥有鱼，其名为鲲。鲲之大，不知其几千里也。',language:'classical'})});
        await r.json();
        _setModelStatus('classical','ready','✅ 已就绪');
    }catch(e){_setModelStatus('classical','error','❌ 失败');}

    // English
    try{
        const r=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:'The cat sat on the mat. This is a simple test sentence.',language:'english'})});
        await r.json();
        _setModelStatus('english','ready','✅ 已就绪');
    }catch(e){_setModelStatus('english','error','❌ 失败');}

    // Restore input to modern sample
    document.getElementById('input').value='碳钢是钢的一种，具有高强度和高韧性。北京立方庭位于海淀区。';
    setLanguage('auto');
};

function _setModelStatus(model,state,label){
    const dot=document.getElementById('dot'+model.charAt(0).toUpperCase()+model.slice(1));
    const lbl=document.getElementById('label'+model.charAt(0).toUpperCase()+model.slice(1));
    if(dot){dot.className='dot '+state;}
    if(lbl){lbl.textContent=label;}
}

function _renderResults(data){
    renderNSP(data);
    renderJSON(data);
    const patData=data.content&&data.content.patterns;
    if(patData&&patData.length) renderPatterns(patData);
    renderLangDetect(data);
}
</script>
<footer style="margin-top:40px;padding:16px 20px;border-top:1px solid #21262d;text-align:center;font-size:11px;color:#484f58">
  <p style="margin:0 0 6px">Narrative Operator NLP · Apache 2.0</p>
  <p style="margin:0">
    Powered by <a href="https://github.com/hankcs/HanLP" style="color:#58a6ff" target="_blank">HanLP</a> (Apache 2.0) ·
    <a href="https://github.com/google-research/electra" style="color:#58a6ff" target="_blank">ELECTRA</a> (Apache 2.0) ·
    <a href="https://github.com/AnswerDotAI/ModernBERT" style="color:#58a6ff" target="_blank">ModernBERT</a> (Apache 2.0) ·
    <a href="https://github.com/koheiw/kyoto-eva" style="color:#58a6ff" target="_blank">KYOTO-EVAHAN</a> (CC BY 4.0)
  </p>
</footer>
</body>
</html>"""


@app.get("/demo")
async def demo_page():
    """Interactive NLP analysis demo page with visualization."""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=DEMO_HTML)


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
