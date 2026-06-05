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

        # Baseline analysis
        doc = analyze(request.text, dict_combine=user_dict if user_dict else None)

        true_new_words: list[str] | None = None

        if request.enhance:
            # Discover true new words from baseline
            true_new_words = _discover_true_new_words(request.text, doc, request.dict_combine)
            if true_new_words:
                # Re-analyze with enhanced dict (user + discovered)
                enhanced_dict = user_dict | set(true_new_words)
                doc = analyze(request.text, dict_combine=enhanced_dict if enhanced_dict else None)
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
</style>
</head>
<body>
<header>
<h1>🧠 Narrative Operator NLP</h1>
<span>HanLP MTL Demo</span>
<span style="margin-left:auto;font-size:12px;color:#8b949e">Python 3.10 · ELECTRA_SMALL</span>
</header>
<main>
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
<input id="dictInput" placeholder="自定义词典（用空格/逗号/换行分隔，如：碳钢 高强度 立方庭）" style="flex:1;padding:8px 12px;background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#c9d1d9;font-size:13px;font-family:inherit">
</div>
<div style="margin-bottom:16px;display:flex;gap:4px;flex-wrap:wrap">
<span style="font-size:11px;color:#484f58;line-height:24px">示例:</span>
<button class="sample-btn" onclick="setSample('碳钢是钢的一种，具有高强度和高韧性。北京立方庭位于海淀区。')">材料+地点</button>
<button class="sample-btn" onclick="setSample('阿婆主来到北京立方庭参观自然语义科技公司。')">组织机构</button>
<button class="sample-btn" onclick="setSample('2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。')">多任务</button>
</div>
<div class="tabs">
<div class="tab active" onclick="switchTab('nsp')">📊 NSP 结构化</div>
<div class="tab" onclick="switchTab('pretty')">🎨 HanLP 原生可视化</div>
<div class="tab" onclick="switchTab('depsvg')">🧬 依存树 SVG</div>
<div class="tab" onclick="switchTab('discover')">🔍 新词发现</div>
<div class="tab" onclick="switchTab('patterns')">📊 句式模式</div>
<div class="tab" onclick="switchTab('json')">{ } JSON Raw</div>
</div>
<div id="nsp" class="panel active"></div>
<div id="pretty" class="panel"></div>
<div id="depsvg" class="panel"></div>
<div id="discover" class="panel"></div>
<div id="patterns" class="panel"></div>
<div id="json" class="panel"></div>
</main>
<script>
async function analyze(){
    const text=document.getElementById('input').value.trim();
    if(!text) return;
    const dictRaw=document.getElementById('dictInput').value.trim();
    const dictCombine=dictRaw?dictRaw.split(/[\s,，;；]+/).filter(w=>w) :[];
    const mode=document.getElementById('discoverMode').value;
    const discover=mode==='discover';
    const enhance=mode==='enhance';
    const body={text, dict_combine: dictCombine, discover, enhance};
    const btn=document.getElementById('analyzeBtn');
    btn.disabled=true; btn.textContent='分析中...';
    ['nsp','pretty','depsvg','discover','patterns','json'].forEach(id=>document.getElementById(id).innerHTML='<div class=\"loading\">⏳ 分析中...</div>');

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
    btn.disabled=false; btn.textContent='🔍 分析';
}

function renderNSP(data){
    if(!data||!data.content){ document.getElementById('nsp').innerHTML='<div class="error">分析失败：服务器未响应</div>'; return; }
    const c=data.content;
    const trueNew=data.true_new_words||[];
    const tokens=c.tokens.map(t=>{
        const pct=Math.round((t.confidence||1)*100);
        const color=pct>=95?'#7ee787':pct>=80?'#e3b341':'#f85149';
        return `<span class="token ${t.pos}" title="POS:${t.pos} span:${t.span} conf:${pct}%">${t.text}<sub style="color:${color};font-size:0.65em">${pct}</sub></span>`;
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

    document.getElementById('nsp').innerHTML=`
    <div class="stats">
    <div class="stat"><b>${c.tokens.length}</b> tokens</div>
    <div class="stat"><b>${c.entities.length}</b> entities</div>
    <div class="stat"><b>${c.relations.length}</b> relations</div>
    <div class="stat">source: <b>${data.meta.source}</b></div>
    </div>
    ${newWordsHtml}
    <div class="card"><h3>📝 分词 & POS</h3><div style="line-height:2">${tokens}</div></div>
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

    // ConvSeg candidates
    let convsegHtml='<span style="color:#484f58">ConvSeg 不可用（需 TensorFlow）</span>';
    if(convseg&&convseg.candidates){
        convsegHtml=convseg.candidates.map(w=>{
            const inPmi=candidates.some(c=>c.word===w);
            const style=inPmi?'background:#1a3a1a;border-color:#238636':''; // green if also found by PMI
            return `<span class="discover-item" style="${style}" onclick="toggleDictWord(this,'${esc(w)}')" title="点击添加到自定义词典">${esc(w)}<span class="score" style="color:#58a6ff">conv</span></span>`;
        }).join('')||'<span style="color:#484f58">-</span>';
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
          <h4 style="font-size:13px;color:#58a6ff;margin-bottom:8px">🧠 ConvSeg (PKU_NAME)</h4>
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
        const tags=[];
        if(p.sentence_type!=='declarative') tags.push(`<span class="tag">${p.sentence_type}</span>`);
        if(p.structural_type&&p.structural_type!=='subject_predicate') tags.push(`<span class="tag structural">${esc(p.structural_type)}</span>`);
        if(p.sentence_length_tier) tags.push(`<span class="tag tier">${esc(p.sentence_length_tier)}</span>`);
        const rels=p.relation_summary.join(', ')||'-';
        // ── NLP gap signals: all keyword/heuristic detections are in limitations ──
        const L = p.limitations||[];
        if(L.some(l=>l.startsWith('hint:negation'))) tags.push(`<span class="tag negative">⚠否定</span>`);
        if(L.includes('hint:passive'))  tags.push(`<span class="tag passive">⚠被动</span>`);
        if(L.includes('hint:ba-construction')) tags.push(`<span class="tag">⚠把字</span>`);
        if(L.includes('hint:imperative')) tags.push(`<span class="tag imperative">⚠祈使</span>`);
        if(L.includes('hint:parallel')) tags.push(`<span class="tag">⚠排比</span>`);
        if(L.includes('hint:loose')) tags.push(`<span class="tag">⚠松散</span>`);
        if(L.includes('hint:serial_verb')) tags.push(`<span class="tag">⚠连动</span>`);
        if(L.includes('hint:pivotal')) tags.push(`<span class="tag">⚠兼语</span>`);
        const limits=L.length
          ? p.limitations.map(l=>`<span class="tag limit" title="NLP无法确定，留给下游">⚠${esc(l)}</span>`).join(' ')
          : '<span style="color:#484f58">-</span>';
        return `<tr>
          <td class="tpl">${esc(p.template)}</td>
          <td class="pred">${esc(p.predicates.join(', '))}</td>
          <td class="rel">${esc(rels)}</td>
          <td style="color:#484f58">${p.attribute_count>0?'⚡'+p.attribute_count:'-'}</td>
          <td style="color:#484f58">${p.word_count||'-'}</td>
          <td style="color:#484f58">${p.clause_count||'-'}</td>
          <td>${tags.join('')||'<span style="color:#484f58">-</span>'}</td>
          <td>${limits}</td>
        </tr>`;
    }).join('');

    document.getElementById('patterns').innerHTML=`
    <div class="card">
      <h3>📊 句式模式 <small style="color:#484f58;font-weight:normal">(可按模板聚合统计)</small></h3>
      <div style="overflow-x:auto">
      <table class="pattern-table">
        <thead><tr>
          <th>模板 Template</th>
          <th>谓词</th>
          <th>关系</th>
          <th>属性</th>
          <th>词数</th>
          <th>分句</th>
          <th>特征</th>
          <th>限制/下游</th>
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
}

function setSample(text){
    document.getElementById('input').value=text;
    analyze();
}

function escapeHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
const esc=escapeHtml;

// Auto-analyze on load
window.onload=analyze;
</script>
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
