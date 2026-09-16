// ── app.js — demo entry: state, analysis orchestration, events ────
import { SAMPLES, TABS } from './data.js';
import { esc, renderNSP, renderPretty, renderDepSVG, renderDiscover, renderPatterns, renderLangDetect, renderCoref, renderSummary, renderTitle, renderJSON, renderAPIDocs, toggleEndpoint, toggleEventRow } from './render.js';

let _currentLang='auto';
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

async function analyze(){
    const text=document.getElementById('input').value.trim();
    if(!text) return;
    const dictRaw=document.getElementById('dictInput').value.trim();
    const dictCombine=dictRaw?dictRaw.split(/[\s,，;；]+/).filter(w=>w) :[];
    const mode=document.getElementById('discoverMode').value;
    const discover=mode==='discover';
    const enhance=mode==='enhance';
    const language=_currentLang;
    const summarize=true;
    const summary_mode=_summaryMode;
    const summary_chars=parseInt(document.getElementById('sumCharsVal').value)||30;
    const summary_ratio=parseFloat(document.getElementById('sumRatioVal').value)||0.2;
    const generate_title=true;
    const title_mode=_titleMode;
    const title_chars=parseInt(document.getElementById('ttlCharsVal').value)||14;
    const title_ratio=parseFloat(document.getElementById('ttlRatioVal').value)||0.05;
    const body={text, dict_combine: dictCombine, discover, enhance, language, entity_dict: {}, summarize, summary_mode, summary_chars, summary_ratio, generate_title, title_mode, title_chars, title_ratio};
    const btn=document.getElementById('analyzeBtn');
    btn.disabled=true; btn.textContent='分析中...';
    ['nsp','pretty','depsvg','discover','patterns','langdetect','summary','title','json'].forEach(id=>document.getElementById(id).innerHTML='<div class="loading">⏳ 分析中...</div>');

    // Independent fetches — one failure doesn't block others
    const post=(url,body)=>fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json()).catch(e=>({_error:e.message}));
    const [r1,r2,r3]=await Promise.all([
        post('/analyze',body),
        post('/analyze/pretty',body),
        post('/analyze/discover',body)
    ]);
    if(r1._error) document.getElementById('nsp').innerHTML='<div class="error">分析失败: '+r1._error+'</div>';
    else renderNSP(r1);
    if(r2._error) document.getElementById('pretty').innerHTML='<div class="error">分析失败: '+r2._error+'</div>';
    else renderPretty(r2);
    // depsvg: use deps from /analyze response (now includes deps field)
    if(r1._error || !r1.content) document.getElementById('depsvg').innerHTML='<div class="error">分析失败</div>';
    else renderDepSVG({tokens: r1.content.tokens, deps: r1.content.deps || []});
    if(r3._error) document.getElementById('discover').innerHTML='<div class="error">新词发现失败: '+r3._error+'</div>';
    else renderDiscover(r3);    if(!r1._error) renderJSON(r1);
    const patData=r1.content&&r1.content.patterns;
    if(patData&&patData.length) renderPatterns(patData); else document.getElementById('patterns').innerHTML='<div class="card"><h3>📊 句式模式</h3><span style="color:#484f58">无模式数据</span></div>';
    renderLangDetect(r1);
    renderCoref(r1);
    renderSummary(r1);
    renderTitle(r1);
    btn.disabled=false; btn.textContent='🔍 分析';
}

// ── UI: render tabs & panels from TABS ──────────────────────────────────────
function renderTabs(){
    const tabsEl=document.getElementById('tabs');
    const panelsEl=document.getElementById('panels');
    tabsEl.innerHTML=TABS.map((t,i)=>`<div class="tab${i===0?' active':''}" data-tab="${t.id}" role="tab">${t.label}</div>`).join('');
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
        document.getElementById(id).addEventListener('change',analyze);
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

// On load: warm up all three models in parallel, show loading status
window.onload=async function(){
    renderTabs();
    renderSamples();
    bindControls();
    bindPanelDelegation();
    _setModelStatus('modern','loading','⏳ 加载中…');
    _setModelStatus('classical','loading','⏳ 加载中…');
    _setModelStatus('english','loading','⏳ 加载中…');

    // Modern Chinese
    try{
        const r=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:SAMPLES[0].text,language:'auto'})});
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
    document.getElementById('input').value=SAMPLES[0].text;
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
    renderSummary(data);
    const patData=data.content&&data.content.patterns;
    if(patData&&patData.length) renderPatterns(patData);
    renderLangDetect(data);
}
