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


class AnalyzeResponse(BaseModel):
    """Wraps NarrativeDocument for API serialization."""
    meta: dict
    content: dict

    @classmethod
    def from_doc(cls, doc: NarrativeDocument) -> "AnalyzeResponse":
        return cls(
            meta=doc.meta.model_dump(),
            content=doc.content.model_dump(),
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

    This is the primary endpoint for manual testing and Studio prototyping.
    """
    try:
        doc = analyze(request.text, dict_combine=set(request.dict_combine) if request.dict_combine else None)
        return AnalyzeResponse.from_doc(doc)
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
.loading{text-align:center;padding:40px;color:#8b949e}
.error{color:#f85149;padding:12px;background:#3a1a1a;border-radius:6px}
.sample-btn{font-size:11px;padding:4px 8px;background:#21262d;color:#8b949e;border:1px solid #30363d;border-radius:4px;cursor:pointer;margin:2px}
.sample-btn:hover{color:#c9d1d9;border-color:#58a6ff}
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
<button id="analyzeBtn" onclick="analyze()">🔍 分析</button>
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
<div class="tab" onclick="switchTab('json')">{ } JSON Raw</div>
</div>
<div id="nsp" class="panel active"></div>
<div id="pretty" class="panel"></div>
<div id="depsvg" class="panel"></div>
<div id="json" class="panel"></div>
</main>
<script>
async function analyze(){
    const text=document.getElementById('input').value.trim();
    if(!text) return;
    const dictRaw=document.getElementById('dictInput').value.trim();
    const dictCombine=dictRaw?dictRaw.split(/[\s,，;；]+/).filter(w=>w) :[];
    const body={text, dict_combine: dictCombine};
    const btn=document.getElementById('analyzeBtn');
    btn.disabled=true; btn.textContent='分析中...';
    ['nsp','pretty','depsvg','json'].forEach(id=>document.getElementById(id).innerHTML='<div class="loading">⏳ 分析中...</div>');

    // Independent fetches — one failure doesn't block others
    const post=(url,body)=>fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json()).catch(e=>({_error:e.message}));
    const [r1,r2,r3]=await Promise.all([
        post('/analyze',body),
        post('/analyze/pretty',body),
        post('/analyze/dep',body)
    ]);
    if(r1._error) document.getElementById('nsp').innerHTML='<div class="error">分析失败: '+r1._error+'</div>';
    else renderNSP(r1);
    if(r2._error) document.getElementById('pretty').innerHTML='<div class="error">分析失败: '+r2._error+'</div>';
    else renderPretty(r2);
    if(r3._error) document.getElementById('depsvg').innerHTML='<div class="error">分析失败: '+r3._error+'</div>';
    else renderDepSVG(r3);
    if(!r1._error) renderJSON(r1);
    btn.disabled=false; btn.textContent='🔍 分析';
}

function renderNSP(data){
    if(!data||!data.content){ document.getElementById('nsp').innerHTML='<div class="error">分析失败：服务器未响应</div>'; return; }
    const c=data.content;
    const tokens=c.tokens.map(t=>{
        const pct=Math.round((t.confidence||1)*100);
        const color=pct>=95?'#7ee787':pct>=80?'#e3b341':'#f85149';
        return `<span class="token ${t.pos}" title="POS:${t.pos} span:${t.span} conf:${pct}%">${t.text}<sub style="color:${color};font-size:0.65em">${pct}</sub></span>`;
    }).join('');

    const entities=c.entities.map(e=>{
        const pct=Math.round((e.confidence||1)*100);
        const color=pct>=95?'#7ee787':pct>=80?'#e3b341':'#f85149';
        return `<span class="entity-card"><span class="cat">${e.category}</span>${e.text} <small style="color:#484f58">[${e.span[0]}:${e.span[1]}]</small> <small style="color:${color}">${pct}%</small></span>`;
    }).join('')||'<span style="color:#484f58">-</span>';

    const relations=c.relations.map(r=>`<div class="relation-row"><span class="subj">${r.subject}</span> &rarr; <span class="pred">${r.predicate}</span> &rarr; <span class="obj">${r.object}</span><span class="src">${r.source}</span></div>`).join('')||'<span style="color:#484f58">-</span>';

    document.getElementById('nsp').innerHTML=`
    <div class="stats">
    <div class="stat"><b>${c.tokens.length}</b> tokens</div>
    <div class="stat"><b>${c.entities.length}</b> entities</div>
    <div class="stat"><b>${c.relations.length}</b> relations</div>
    <div class="stat">source: <b>${data.meta.source}</b></div>
    </div>
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
