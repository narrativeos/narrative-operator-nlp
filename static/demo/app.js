// ── app.js — demo entry: state, analysis orchestration, events ────
import { SAMPLES, TABS } from './data.js';
import { esc, renderOverview, renderNSP, renderPretty, renderDepSVG, renderDiscover, renderPatterns, renderLangDetect, renderCoref, renderSummary, renderTitle, renderJSON, renderAPIDocs, toggleEndpoint, toggleEventRow } from './render.js';

let _currentLang='auto';
let _bootDone=false;
function setLanguage(lang){
    _currentLang=lang;
    ['langAuto','langModern','langClassical','langEnglish'].forEach(id=>{
        const el=document.getElementById(id);
        if(el) el.classList.toggle('active', id==='lang'+lang.charAt(0).toUpperCase()+lang.slice(1));
    });
    analyze();
}

let _summaryMode='chars';
function setSummaryMode(mode){
    _summaryMode=mode;
    document.getElementById('sumChars').classList.toggle('active', mode==='chars');
    document.getElementById('sumRatio').classList.toggle('active', mode==='ratio');
    analyze();
}

let _titleMode='chars';
function setTitleMode(mode){
    _titleMode=mode;
    document.getElementById('ttlChars').classList.toggle('active', mode==='chars');
    document.getElementById('ttlRatio').classList.toggle('active', mode==='ratio');
    analyze();
}

// Build the /analyze request body from current UI state
function buildBody(text){
    const dictRaw=document.getElementById('dictInput').value.trim();
    const dictCombine=dictRaw?dictRaw.split(/[\s,，;；]+/).filter(w=>w) :[];
    const mode=document.getElementById('discoverMode').value;
    return {
        text,
        dict_combine: dictCombine,
        discover: mode==='discover',
        enhance: mode==='enhance',
        language: _currentLang,
        entity_dict: {},
        summarize: true,
        summary_mode: _summaryMode,
        summary_chars: parseInt(document.getElementById('sumCharsVal').value)||30,
        summary_ratio: parseFloat(document.getElementById('sumRatioVal').value)||0.2,
        generate_title: true,
        title_mode: _titleMode,
        title_chars: parseInt(document.getElementById('ttlCharsVal').value)||14,
        title_ratio: parseFloat(document.getElementById('ttlRatioVal').value)||0.05,
    };
}

const post=(url,body)=>fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json()).catch(e=>({_error:e.message}));
const errHtml=msg=>'<div class="error">分析失败: '+esc(msg)+'</div>';

// Render every panel derivable from a single /analyze response
function renderFromAnalyze(data){
    renderOverview(data);
    renderNSP(data);
    if(data.content&&data.content.tokens) renderDepSVG({tokens: data.content.tokens, deps: data.content.deps || []});
    renderJSON(data);
    renderCoref(data);
    renderSummary(data);
    renderTitle(data);
    renderLangDetect(data);
    const patData=data.content&&data.content.patterns;
    if(patData&&patData.length) renderPatterns(patData);
    else document.getElementById('patterns').innerHTML='<div class="card"><h3>📊 句式模式</h3><span class="empty">无模式数据</span></div>';
}

async function analyze(){
    const text=document.getElementById('input').value.trim();
    if(!text) return;
    const body=buildBody(text);
    const btn=document.getElementById('analyzeBtn');
    btn.disabled=true; btn.textContent='分析中...';
    // No-flash: keep previous results visible, dim them while updating
    document.getElementById('panels').classList.add('updating');

    // Independent fetches — one failure doesn't block others
    const [r1,r2,r3]=await Promise.all([
        post('/analyze',body),
        post('/analyze/pretty',body),
        post('/analyze/discover',body)
    ]);
    if(r1._error){
        ['overview','nsp','depsvg','patterns','langdetect','coref','summary','title','json'].forEach(id=>document.getElementById(id).innerHTML=errHtml(r1._error));
    }else{
        renderFromAnalyze(r1);
    }
    if(r2._error) document.getElementById('pretty').innerHTML=errHtml(r2._error);
    else renderPretty(r2);
    if(r3._error) document.getElementById('discover').innerHTML='<div class="error">新词发现失败: '+esc(r3._error)+'</div>';
    else renderDiscover(r3);

    document.getElementById('panels').classList.remove('updating');
    btn.disabled=false; btn.textContent='🔍 分析';
}

// ── UI: render tabs & panels from TABS (grouped) ───────────────────────────
function renderTabs(){
    const tabsEl=document.getElementById('tabs');
    const panelsEl=document.getElementById('panels');
    let lastGroup=null;
    tabsEl.innerHTML=TABS.map((t,i)=>{
        let sep='';
        if(t.group!==lastGroup){ sep=`<span class="tab-group">${t.group}</span>`; lastGroup=t.group; }
        return sep+`<div class="tab${i===0?' active':''}" data-tab="${t.id}" role="tab">${t.label}</div>`;
    }).join('');
    panelsEl.innerHTML=TABS.map((t,i)=>`<div id="${t.id}" class="panel${i===0?' active':''}" role="tabpanel"></div>`).join('');
    tabsEl.querySelectorAll('.tab').forEach(el=>{
        el.addEventListener('click',()=>switchTab(el.dataset.tab));
    });
}

// ── UI: render sample cards from SAMPLES ────────────────────────────────────
function renderSamples(){
    const wrap=document.getElementById('sampleCards');
    SAMPLES.forEach(s=>{
        const card=document.createElement('div');
        card.className='sample-card';
        const tagCls=s.lang==='classical'?' classical':s.lang==='english'?' english':'';
        card.innerHTML=`<span class="lang-tag${tagCls}">${s.label}</span><span class="preview">${esc(s.preview)}</span>`;
        card.addEventListener('click',()=>{
            if(s.dict) setLanguageAndDict(s.text, s.lang, s.dict);
            else setLanguageAndAnalyze(s.text, s.lang);
        });
        wrap.appendChild(card);
    });
}

// ── UI: wire up static controls ─────────────────────────────────────────────
function bindControls(){
    document.getElementById('analyzeBtn').addEventListener('click',analyze);
    document.getElementById('discoverMode').addEventListener('change',analyze);
    // Language mode options
    [['langAuto','auto'],['langModern','modern'],['langClassical','classical'],['langEnglish','english']].forEach(([id,lang])=>{
        document.getElementById(id).addEventListener('click',()=>setLanguage(lang));
    });
    // Summary / title mode options
    [['sumChars','chars'],['sumRatio','ratio']].forEach(([id,mode])=>{
        document.getElementById(id).addEventListener('click',()=>setSummaryMode(mode));
    });
    [['ttlChars','chars'],['ttlRatio','ratio']].forEach(([id,mode])=>{
        document.getElementById(id).addEventListener('click',()=>setTitleMode(mode));
    });
    // Numeric value changes re-run analysis
    ['sumCharsVal','sumRatioVal','ttlCharsVal','ttlRatioVal'].forEach(id=>{
        const el=document.getElementById(id);
        el.addEventListener('change',analyze);
        // Clicking/focusing the number input inside a mode label must not
        // trigger the label's mode-switch (which would re-analyze immediately)
        el.addEventListener('click',e=>e.stopPropagation());
        el.addEventListener('mousedown',e=>e.stopPropagation());
    });
}

// Single delegated listener for all dynamically rendered interactive content
// (discover items, dict action buttons, event-row toggles, API doc expanders).
function bindPanelDelegation(){
    document.getElementById('panels').addEventListener('click',e=>{
        const item=e.target.closest('.discover-item');
        if(item && item.dataset.word!==undefined){ toggleDictWord(item,item.dataset.word); return; }
        const actionBtn=e.target.closest('[data-action]');
        if(actionBtn){
            if(actionBtn.dataset.action==='apply-dict') applyDict();
            else if(actionBtn.dataset.action==='clear-dict') clearDictSelection();
            return;
        }
        const toggle=e.target.closest('.event-toggle');
        if(toggle){ toggleEventRow(toggle,toggle.dataset.eventId); return; }
        const goto=e.target.closest('[data-goto-tab]');
        if(goto){ switchTab(goto.dataset.gotoTab); return; }
        const method=e.target.closest('.api-endpoint .method');
        if(method) toggleEndpoint(method);
    });
}

function switchTab(id){
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
    document.querySelector(`.tab[data-tab="${id}"]`).classList.add('active');
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

function setLanguageAndDict(text, lang, dictWords){
    document.getElementById('input').value=text;
    document.getElementById('dictInput').value=dictWords;
    setLanguage(lang);
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

// ── Boot: parallel model warm-up behind a full-screen overlay ──────────────
// The modern warm-up uses the full first-sample body (summarize/title on),
// so its response can be rendered directly afterwards — no second request,
// no visible re-render "flash".
window.onload=async function(){
    renderTabs();
    renderSamples();
    bindControls();
    bindPanelDelegation();

    const modernBody=buildBody(SAMPLES[0].text);
    const warmups=[
        {model:'modern',  body:modernBody},
        {model:'classical', body:{text:'北冥有鱼，其名为鲲。鲲之大，不知其几千里也。',language:'classical'}},
        {model:'english', body:{text:'The cat sat on the mat. This is a simple test sentence.',language:'english'}},
    ];
    const results={};
    await Promise.all(warmups.map(async w=>{
        _setModelStatus(w.model,'loading','加载中…');
        _setBootState(w.model,'loading','加载中…');
        const t=performance.now();
        try{
            const d=await post('/analyze',w.body);
            const ms=Math.round(performance.now()-t);
            if(d._error) throw new Error(d._error);
            results[w.model]=d;
            _setModelStatus(w.model,'ready','已就绪',ms);
            _setBootState(w.model,'ready','已就绪 · '+(ms/1000).toFixed(1)+'s');
        }catch(e){
            _setModelStatus(w.model,'error','失败');
            _setBootState(w.model,'error','失败');
        }
    }));

    // Pre-fill the first sample and render from the cached warm-up response
    document.getElementById('input').value=SAMPLES[0].text;
    _currentLang='auto';
    if(results.modern){
        renderFromAnalyze(results.modern);
        // Only the two endpoints not covered by the warm-up are fetched now
        const [r2,r3]=await Promise.all([
            post('/analyze/pretty',modernBody),
            post('/analyze/discover',modernBody),
        ]);
        if(r2._error) document.getElementById('pretty').innerHTML=errHtml(r2._error);
        else renderPretty(r2);
        if(r3._error) document.getElementById('discover').innerHTML='<div class="error">新词发现失败: '+esc(r3._error)+'</div>';
        else renderDiscover(r3);
    }else{
        ['overview','nsp','pretty','depsvg','discover','patterns','coref','langdetect','summary','title','json'].forEach(id=>{
            document.getElementById(id).innerHTML='<div class="error">现代汉语模型预热失败，请检查服务后刷新页面</div>';
        });
    }

    _bootDone=true;
    const btn=document.getElementById('analyzeBtn');
    btn.disabled=false;
    _hideBootOverlay();
};

function _setModelStatus(model,state,label,ms){
    const cap=model.charAt(0).toUpperCase()+model.slice(1);
    const dot=document.getElementById('dot'+cap);
    const lbl=document.getElementById('label'+cap);
    const time=document.getElementById('time'+cap);
    if(dot){dot.className='dot '+state;}
    if(lbl){lbl.textContent=label;}
    if(time){time.textContent=ms?ms+'ms':'';}
}

function _setBootState(model,state,text){
    const el=document.getElementById('bootState'+model.charAt(0).toUpperCase()+model.slice(1));
    if(el){el.className='boot-state '+state; el.textContent=text;}
}

function _hideBootOverlay(){
    const ov=document.getElementById('bootOverlay');
    if(!ov) return;
    ov.classList.add('hidden');
    setTimeout(()=>ov.remove(),400);
}
