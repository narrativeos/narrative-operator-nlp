// Sample texts — extracted 1:1 from the original inline sample cards.
// Each card: text (full sample), lang (language mode), label, preview, dict (optional custom dict).
const SAMPLES = [
  { text: '阿里巴巴集团成立于1999年，由马云等人在杭州创立。公司最初专注于B2B电子商务，为中小企业提供在线交易平台。2003年，阿里巴巴推出淘宝网，进军C2C电商领域，迅速成为中国最大的网上购物平台。2004年，支付宝成立，解决了网络交易的信任问题，后来发展为全球最大的第三方支付平台之一。2009年，阿里云成立，致力于打造云计算基础设施，现已成为中国最大的公有云服务商。2014年，阿里巴巴在纽约证券交易所上市，成为当时全球最大的IPO。此后，阿里巴巴持续拓展业务版图，涵盖电商、云计算、数字媒体、物流和本地生活等多个领域。2023年，阿里巴巴宣布启动"1+6+N"组织变革，将业务拆分为六大业务集团，以提升各业务单元的灵活性和竞争力。', lang: 'auto', label: '📄 企业+摘要', preview: '阿里巴巴集团成立于1999年，由马云等人在杭州创立。公司最初专注于B2B电子商务……' },
  { text: '张三于2020年从清华大学计算机系毕业，获得博士学位。他的研究方向是自然语言处理和知识图谱。毕业后，他加入百度研究院，担任高级算法工程师。在百度工作期间，他主导开发了新一代中文分词系统，准确率提升了15%。2022年，他参与构建了百度知识图谱2.0，覆盖实体超过10亿。2023年，他转投阿里巴巴达摩院，继续深耕大语言模型方向。目前，他负责阿里通义千问模型的优化工作，在推理速度和生成质量方面取得了显著进展。', lang: 'modern', label: '📄 人物+摘要', preview: '张三于2020年从清华大学计算机系毕业，获得博士学位。他的研究方向是自然语言处理……' },
  { text: '2024年5月28日，百度在北京举办了AI开发者大会。大会上，百度正式发布了文心大模型4.0专业版，支持超长上下文窗口，可处理超过25万字的文档。百度CEO李彦宏表示，文心大模型已在金融、医疗、法律等多个行业落地应用。同日，百度还发布了飞桨深度学习框架6.0版本，新增了对多模态任务的全面支持。在自动驾驶领域，百度Apollo宣布累计测试里程突破1000万公里，已在10个城市开展Robotaxi商业化运营。此外，百度还发布了智能云新战略，提出"云智一体"理念，将AI能力深度融入云计算服务。百度还与清华大学联合成立了AI联合实验室，聚焦大模型基础研究和产业应用。', lang: 'modern', label: '📄 事件+摘要', preview: '2024年5月28日，百度在北京举办了AI开发者大会。大会上，百度正式发布了文心大模型……' },
  { text: '苹果公司在加州库比蒂诺的总部发布了新款iPhone 16 Pro。该产品搭载A18 Pro芯片，采用台积电3纳米工艺，CPU性能提升25%，GPU性能提升40%。iPhone 16 Pro配备6.3英寸和6.9英寸两种尺寸的OLED屏幕，支持120Hz自适应刷新率。在AI方面，苹果推出了Apple Intelligence，深度集成于Siri和系统级应用中。该产品起售价为799美元，于9月20日正式开售。分析师预计，iPhone 16系列首季度销量将超过8000万台，同比增长10%。苹果公司总部位于加州库比蒂诺，是全球市值最高的科技公司之一。', lang: 'modern', label: '📄 产品+摘要', preview: '苹果公司在加州库比蒂诺的总部发布了新款iPhone 16 Pro。该产品搭载A18 Pro芯片……', dict: 'iPhone 16 Pro A18 Pro 库比蒂诺 Apple Intelligence' },
  { text: '北京立方庭位于海淀区中关村核心区域，是一栋甲级写字楼。总建筑面积约5万平方米，地上28层，地下3层。建筑高度120米，于2018年竣工投入使用。立方庭采用全玻璃幕墙设计，获得LEED金级认证。该建筑配备智能楼宇管理系统，可实现能耗自动优化。周边交通便利，距离地铁4号线中关村站仅200米。立方庭入驻企业包括多家知名科技公司和金融机构。物业管理由仲量联行负责，提供24小时安保和前台服务。', lang: 'auto', label: '📄 建筑+摘要', preview: '北京立方庭位于海淀区中关村核心区域，是一栋甲级写字楼。总建筑面积约5万平方米……' },
  { text: '陈胜者，阳城人也，字涉。吴广者，阳夏人也，字叔。陈涉少时，尝与人佣耕，辍耕之垄上，怅恨久之，曰：\'苟富贵，无相忘。\'佣者笑而应曰：\'若为佣耕，何富贵也？\'陈涉太息曰：\'嗟乎，燕雀安知鸿鹄之志哉！\'二世元年七月，发闾左適戍渔阳，九百人屯大泽乡。陈胜吴广皆次当行，为屯长。会天大雨，道不通，度已失期。失期，法皆斩。陈胜吴广乃谋曰：\'今亡亦死，举大计亦死，等死，死国可乎？\'陈胜曰：\'天下苦秦久矣。\'卒买鱼烹食，得鱼腹中书，曰\'陈胜王\'。卒皆夜惊恐。又间令吴广之次所旁丛祠中，夜篝火，狐鸣呼曰：\'大楚兴，陈胜王。\'卒皆夜惊恐。旦日，卒中往往语，皆指目陈胜。', lang: 'classical', label: '🏯 陈涉世家+摘要', preview: '陈胜者，阳城人也，字涉。吴广者，阳夏人也，字叔。陈涉少时，尝与人佣耕……' },
  { text: '北冥有鱼，其名为鲲。鲲之大，不知其几千里也。化而为鸟，其名为鹏。鹏之背，不知其几千里也。怒而飞，其翼若垂天之云。是鸟也，海运则将徙于南冥。南冥者，天池也。《齐谐》者，志怪者也。《谐》之言曰：\'鹏之徙于南冥也，水击三千里，抟扶摇而上者九万里，绝云气，负青天，然后图南，且适南冥也。\'斥鴳笑之曰：\'彼且奚适也？我腾跃而上，不过数仞而下，翱翔蓬蒿之间，此亦飞之至也。\'而彼且奚适也？此小大之辩也。', lang: 'classical', label: '🏯 逍遥游+摘要', preview: '北冥有鱼，其名为鲲。鲲之大，不知其几千里也。化而为鸟，其名为鹏……' },
  { text: '十年春，齐师伐我。公将战，曹刿请见。其乡人曰：\'肉食者谋之，又何间焉？\'刿曰：\'肉食者鄙，未能远谋。\'乃入见。问：\'何以战？\'公曰：\'衣食所安，弗敢专也，必以分人。\'对曰：\'小惠未遍，民弗从也。\'公曰：\'牺牲玉帛，弗敢加也，必以信。\'对曰：\'小信未孚，神弗福也。\'公曰：\'小大之狱，虽不能察，必以情。\'对曰：\'忠之属也，可以一战。战则请从。\'公与之乘，战于长勺。公将鼓之。刿曰：\'未可。\'齐人三鼓。刿曰：\'可矣。\'齐师败绩。公将驰之。刿曰：\'未可。\'下视其辙，登轼而望之，曰：\'可矣。\'遂逐齐师。既克，公问其故。对曰：\'夫战，勇气也。一鼓作气，再而衰，三而竭。彼竭我盈，故克之。\' ', lang: 'classical', label: '🏯 曹刿论战+摘要', preview: '十年春，齐师伐我。公将战，曹刿请见。其乡人曰：\'肉食者谋之……' },
  { text: 'Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976, in Cupertino, California. The company initially focused on personal computers, with the Apple I and Apple II becoming commercial successes. In 1984, Apple introduced the Macintosh, the first mass-market computer with a graphical user interface. After Jobs was ousted in 1985, he returned in 1997 and transformed the company. Under his leadership, Apple launched the iMac, iPod, iPhone, and iPad, revolutionizing multiple industries. The iPhone, released in 2007, became the best-selling smartphone worldwide. As of 2024, Apple is the world\'s most valuable company, with a market capitalization exceeding $3 trillion. The company generates over $380 billion in annual revenue, with services including Apple Music, iCloud, and the App Store contributing significantly to its growth.', lang: 'english', label: '🇬🇧 Apple+Summary', preview: 'Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne……' },
  { text: 'Microsoft Corporation was founded by Bill Gates and Paul Allen on April 4, 1975, in Albuquerque, New Mexico. The company\'s first product was an interpreter for the BASIC programming language. In 1981, Microsoft developed MS-DOS for IBM\'s first personal computer, which became the industry standard. Windows 3.0, released in 1990, was the first truly successful graphical operating system. Windows 95 introduced the Start menu and taskbar, becoming a cultural phenomenon. In 2014, Satya Nadella became CEO and shifted Microsoft\'s focus to cloud computing and AI. Azure, launched in 2010, became the second-largest cloud platform globally. Microsoft acquired LinkedIn in 2016 and GitHub in 2018, expanding its developer ecosystem. The company\'s Copilot AI assistant, powered by OpenAI\'s GPT models, has been integrated across its product suite. As of 2024, Microsoft is valued at over $3 trillion.', lang: 'english', label: '🇬🇧 Microsoft+Summary', preview: 'Microsoft Corporation was founded by Bill Gates and Paul Allen……' },
];
// ── Data: tabs & panels ─────────────────────────────────────────────────────
// id must match the panel element id; the first tab/panel is active by default.
const TABS = [
  { id: 'nsp',        label: '📊 NSP 结构化' },
  { id: 'pretty',     label: '🎨 HanLP 原生可视化' },
  { id: 'depsvg',     label: '🧬 依存树 SVG' },
  { id: 'discover',   label: '🔍 新词发现' },
  { id: 'patterns',   label: '📊 句式模式' },
  { id: 'coref',      label: '🔗 指代消解' },
  { id: 'langdetect', label: '🏯 语言检测' },
  { id: 'summary',    label: '📝 摘要' },
  { id: 'title',      label: '🏷️ 标题' },
  { id: 'json',       label: '{ } JSON Raw' },
  { id: 'api',        label: '📋 API 接口' },
];

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

    // Build entity lookup for hierarchy display
    const entityMap={};
    c.entities.forEach(e=>entityMap[e.id]=e);

    // Build hierarchy tree for display
    const rootEntities = c.entities.filter(e => !e.parent_entity_id);
    const childEntities = c.entities.filter(e => e.parent_entity_id);
    
    // Group children by parent
    const childrenByParent = {};
    childEntities.forEach(e => {
        if (!childrenByParent[e.parent_entity_id]) {
            childrenByParent[e.parent_entity_id] = [];
        }
        childrenByParent[e.parent_entity_id].push(e);
    });

    function renderEntityCard(e) {
        const pct=Math.round((e.confidence||1)*100);
        const color=pct>=95?'#7ee787':pct>=80?'#e3b341':'#f85149';
        const attrs=(e.attributes||[]).map(a=>{
            const src=a.source_relation_id?`<small style="color:#484f58">←${a.source_relation_id}</small>`:'';
            return `<small style="color:#e3b341;margin-left:2px">${a.key}=${a.value||'?'}</small>${src}`;
        }).join('');
        const parent=e.parent_entity_id?`<small style="color:#a5b4fc;margin-left:4px">⊂${esc((entityMap[e.parent_entity_id]||{}).text||e.parent_entity_id)}</small>`:'';
        return `<span class="entity-card"><span class="cat">${e.category}</span>${e.text}${attrs}${parent} <small style="color:#484f58">[${e.span[0]}:${e.span[1]}]</small> <small style="color:${color}">${pct}%</small></span>`;
    }

    function renderEntityHierarchy() {
        let html = '';
        // Show entities with hierarchy
        const shown = new Set();
        rootEntities.forEach(e => {
            html += `<div style="margin:4px 0">${renderEntityCard(e)}</div>`;
            shown.add(e.id);
            const children = childrenByParent[e.id] || [];
            if (children.length > 0) {
                html += `<div style="margin-left:24px;border-left:2px solid #30363d;padding-left:12px">`;
                children.forEach(child => {
                    html += `<div style="margin:4px 0">${renderEntityCard(child)}</div>`;
                    shown.add(child.id);
                    // Recursively show grandchildren
                    const grandchildren = childrenByParent[child.id] || [];
                    if (grandchildren.length > 0) {
                        html += `<div style="margin-left:24px;border-left:2px solid #484f58;padding-left:12px">`;
                        grandchildren.forEach(gc => {
                            html += `<div style="margin:4px 0">${renderEntityCard(gc)}</div>`;
                            shown.add(gc.id);
                        });
                        html += `</div>`;
                    }
                });
                html += `</div>`;
            }
        });
        // Show any remaining entities not in hierarchy
        c.entities.filter(e => !shown.has(e.id)).forEach(e => {
            html += `<div style="margin:4px 0">${renderEntityCard(e)}</div>`;
        });
        return html;
    }

    const entities = renderEntityHierarchy();

    // Events rendering — table with collapsible sub-events
    const events = c.events && c.events.length
        ? (() => {
            // Separate main events and sub-events
            const mainEvents = c.events.filter(ev => ev.is_main_event);
            const subEventMap = {};
            c.events.filter(ev => !ev.is_main_event).forEach(sub => {
                for (const main of mainEvents) {
                    if (main.sub_events && main.sub_events.includes(sub.id)) {
                        if (!subEventMap[main.id]) subEventMap[main.id] = [];
                        subEventMap[main.id].push(sub);
                    }
                }
            });

            // Build args HTML for an event
            const buildArgs = (ev) => {
                const agent = (ev.arguments || []).find(a => a.role === 'Agent');
                const patient = (ev.arguments || []).find(a => ['Patient','Result','Product'].includes(a.role));
                const time = (ev.arguments || []).find(a => a.role === 'Time');
                const location = (ev.arguments || []).find(a => a.role === 'Location');
                const others = (ev.arguments || []).filter(a =>
                    a.role !== 'Agent' && !['Patient','Result','Product'].includes(a.role)
                    && a.role !== 'Time' && a.role !== 'Location'
                );
                const spatialBadge = (sr) => {
                    if (!sr || sr === 'STATIC') return '';
                    const colors = {CONTAINER:'#79c0ff',TARGET:'#7ee787',ORIGIN:'#f97583',PATH:'#d2a8ff',ACTOR:'#ffa657',MODIFIER:'#8b949e'};
                    const c = colors[sr] || '#8b949e';
                    return ` <small style="color:${c}">[${sr}]</small>`;
                };
                const argTag = (a) => a
                    ? `<span class="tag" style="background:#e3b341;color:#0d1117">${esc(a.role)}: ${esc(a.text)}${spatialBadge(a.spatial_role)}</span>`
                    : '<span style="color:#484f58">-</span>';
                const othersHtml = others.map(a =>
                    `<span class="tag" style="background:#30363d;color:#8b949e">${esc(a.role)}: ${esc(a.text)}${spatialBadge(a.spatial_role)}</span>`
                ).join(' ') || '<span style="color:#484f58">-</span>';
                // Verb spatial class: pick first non-NONE from any argument
                const vsc = (ev.arguments || []).reduce((acc, a) => acc || (a.verb_spatial_class && a.verb_spatial_class !== 'NONE' ? a.verb_spatial_class : null), null);
                const vscColors = {MOVE_TO:'#7ee787',MOVE_FROM:'#f97583',MOVE_ALONG:'#d2a8ff',STATIC_EXIST:'#79c0ff',STATIC_ACTION:'#ffa657',VIEW:'#e3b341'};
                const verbClassHtml = vsc
                    ? `<span class="tag" style="background:${vscColors[vsc]||'#30363d'};color:#0d1117;font-size:10px">${vsc}</span>`
                    : '<span style="color:#484f58">-</span>';
                // Spatial roles: collect unique non-STATIC roles
                const spatialRoles = [...new Set((ev.arguments||[]).map(a=>a.spatial_role).filter(s=>s&&s!=='STATIC'))];
                const spatialHtml = spatialRoles.length > 0
                    ? spatialRoles.map(sr => `<span class="tag" style="background:#1a3a5c;color:#79c0ff;font-size:10px">${sr}</span>`).join(' ')
                    : '<span style="color:#484f58">-</span>';
                return { agent: argTag(agent), patient: argTag(patient), time: argTag(time), location: argTag(location), othersHtml, spatialHtml, verbClassHtml };
            };

            const rowHtml = (ev, isSub) => {
                const rowStyle = isSub ? 'style="background:#161b22;display:none"' : '';
                const firstCellStyle = isSub ? 'style="padding-left:24px"' : '';
                const arrow = ev.sub_events && ev.sub_events.length > 0
                    ? `<span class="event-toggle" onclick="toggleEventRow(this, '${ev.id}')" style="cursor:pointer;color:#58a6ff;margin-right:4px;user-select:none">▶</span>`
                    : (isSub ? '<span style="margin-right:14px;color:#484f58">└ </span>' : '<span style="margin-right:14px"></span>');
                const mainBadge = ev.is_main_event && !isSub ? '<span class="tag" style="background:#1a3a5c;color:#58a6ff;margin-right:4px">主</span>' : '';
                const subBadge = isSub ? '<span class="tag" style="background:#30363d;color:#8b949e;margin-right:4px">子</span>' : '';
                const args = buildArgs(ev);

                return `<tr ${rowStyle} class="event-row" data-event-id="${ev.id}" ${isSub ? '' : `data-main-event="${ev.id}"`}><td ${firstCellStyle}>${arrow}${mainBadge}${subBadge}<span class="cat" style="background:#58a6ff">${esc(ev.event_type)}</span></td>
                    <td><span style="color:#7ee787;font-weight:600">${esc(ev.trigger)}</span></td>
                    <td><small style="color:#484f58">[${ev.trigger_span[0]}:${ev.trigger_span[1]}]</small></td>
                    <td>${args.agent}</td>
                    <td>${args.patient}</td>
                    <td>${args.time}</td>
                    <td>${args.location}</td>
                    <td style="font-size:11px">${args.othersHtml}</td>
                    <td>${args.spatialHtml}</td>
                    <td>${args.verbClassHtml}</td>
                    <td><small style="color:#7ee787">${Math.round(ev.confidence*100)}%</small></td>
                    <td><small style="color:#484f58">${esc(ev.source)}</small></td>
                </tr>`;
            };

            let html = `<table class="event-table">
                <thead><tr>
                    <th>类型</th><th>触发词</th><th>位置</th>
                    <th>Agent</th><th>Patient/Result</th><th>Time</th><th>Location</th>
                    <th>其他参数</th><th>空间角色</th><th>动词语义类</th><th>置信度</th><th>来源</th>
                </tr></thead><tbody>`;

            // If all events are main events (no sub-events), just render them all
            if (mainEvents.length === 0) {
                c.events.forEach(ev => { html += rowHtml(ev, false); });
            } else {
                mainEvents.forEach(ev => {
                    html += rowHtml(ev, false);
                    const subs = subEventMap[ev.id] || [];
                    subs.forEach(sub => {
                        html += rowHtml(sub, true);
                    });
                });
                // Orphan sub-events (not linked to any main event)
                c.events.filter(ev =>
                    !ev.is_main_event && !mainEvents.some(m => m.sub_events && m.sub_events.includes(ev.id))
                ).forEach(ev => { html += rowHtml(ev, false); });
            }

            html += '</tbody></table>';
            return html;
        })()
        : '<span style="color:#484f58">未提取到事件（此文本暂不支持事件分析，或无明显的动作型关系三元组）</span>';

    const relations=c.relations.map(r=>{
        const subjRaw = r.subject_raw && r.subject_raw !== r.subject ? `<br><small style="color:#484f58">raw: ${esc(r.subject_raw)}</small>` : '';
        const objRaw = r.object_raw && r.object_raw !== r.object ? `<br><small style="color:#484f58">raw: ${esc(r.object_raw)}</small>` : '';
        const mods=(r.modifiers||[]).map(m=>{
            const typeColor=m.type==='negation'?'#f85149':m.type==='degree'?'#e3b341':m.type==='scope'?'#79c0ff':'#8b949e';
            return `<small style="color:${typeColor};margin-right:4px">[${m.text}:${m.type}]</small>`;
        }).join('');
        const modHtml=mods?`<span style="margin-left:8px">${mods}</span>`:'';
        const srcLabel=r.source==='hierarchy/containment'?'<span style="color:#a5b4fc;font-size:10px">⊂hierarchy</span>':esc(r.source);
        return `<div class="relation-row"><span class="subj">${esc(r.subject)}${subjRaw}</span> &rarr; <span class="pred">${esc(r.predicate)}</span> &rarr; <span class="obj">${esc(r.object)}${objRaw}</span>${modHtml}<span class="src">${srcLabel}</span></div>`;
    }).join('')||'<span style="color:#484f58">-</span>';

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
    <div class="stat"><b>${(c.events||[]).length}</b> events</div>
    <div class="stat">source: <b>${data.meta.source}</b></div>
    ${langStats}
    </div>
    ${newWordsHtml}
    <div class="card"><h3>📝 分词 & POS <span style="font-size:11px;color:#484f58">●蓝=现代 · ●黄=古汉语 · ●浅蓝=英文</span></h3><div style="line-height:2">${tokens}</div></div>
    <div class="card"><h3>🏷️ 实体 Entities</h3><div>${entities}</div></div>
    <div class="card"><h3>🔗 关系 Relations</h3><div>${relations}</div></div>
    <div class="card"><h3>⚡ 事件 Events <small style="color:#484f58;font-weight:normal">(从 SRL 语义角色标注派生)</small></h3><div>${events}</div></div>`;
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
                return `<span class="discover-item" style="${style}" onclick="toggleDictWord(this,'${esc(w)}')" title="点击添加到自定义词典">${esc(w)}<span class="score" style="color:#58a6ff">粗分</span></span>`;
            }).join('')||'<span style="color:#484f58">-</span>';
        } else {
            convsegHtml='<span style="color:#484f58;font-size:11px">粗分模型未加载</span>';
        }
    } else {
        convsegHtml='<span style="color:#484f58;font-size:11px">粗分模型未安装</span>';
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

function renderCoref(data){
    if(!data||!data.content){ document.getElementById('coref').innerHTML='<div class="card"><span style="color:#484f58">无指代数据</span></div>'; return; }
    const chains=data.content.coreferences||[];
    if(!chains||!chains.length){
        document.getElementById('coref').innerHTML='<div class="card"><h3>🔗 指代消解</h3><div class="coref-empty">未检测到指代链（可能是简单文本或无指代关系）</div></div>';
        return;
    }
    const html=chains.map(chain=>{
        const mentions=chain.mentions.map(m=>{
            const cls=m.is_principal?'principal':m.mention_type;
            return `<span class="mention ${cls}" title="type:${m.mention_type} span:[${m.span}]">${esc(m.text)}<span class="span-info">[${m.span[0]}-${m.span[1]}]</span></span>`;
        }).join('');
        return `<div class="coref-chain">
            <div class="chain-header">
                <span class="chain-id">${esc(chain.chain_id)}</span>
                <span class="rep">${esc(chain.representative)}</span>
                <span class="quality ${chain.quality_flag}">${chain.quality_flag}</span>
                <span style="font-size:10px;color:#484f58">conf:${(chain.confidence*100).toFixed(0)}% ${chain.language}</span>
            </div>
            <div class="mentions">${mentions}</div>
        </div>`;
    }).join('');
    document.getElementById('coref').innerHTML=`<div class="card"><h3>🔗 指代消解 <small style="color:#484f58;font-weight:normal">${chains.length} 条指代链</small></h3>${html}</div>`;
}

function renderSummary(data){
    if(!data||!data.content||!data.content.summary){
        document.getElementById('summary').innerHTML='<div class="card"><h3>📝 摘要</h3><span style="color:#484f58">未生成摘要（文本过短或无句子结构）</span></div>';
        return;
    }
    const s=data.content.summary;
    const methodLabel=s.method==='nsp+textrank'?'NSP+TextRank 融合':s.method==='textrank'?'TextRank':'未知';
    const modeLabel=s.mode==='chars'?`按字数 (${s.target_chars}字)`:s.mode==='ratio'?`按比例 (${(s.target_ratio*100).toFixed(0)}%)`:'';
    const ratio=s.total_sentences>0?Math.round((s.key_sentences.length/s.total_sentences)*100):0;
    const fusionInfo=s.fusion_weights?`<div style="margin-top:8px;font-size:11px;color:#8b949e">权重: NSP=${(s.fusion_weights.nsp_weight||0).toFixed(2)} / TextRank=${(s.fusion_weights.textrank_weight||0).toFixed(2)}</div>`:'';

    // Key sentences with scores
    const sentencesHtml=(s.key_sentences||[]).map((ks,i)=>{
        const nsp=ks.nsp_score!==undefined?`<span style="color:#e3b341;margin-left:4px">NSP:${ks.nsp_score.toFixed(2)}</span>`:'';
        const tr=ks.textrank_score!==undefined?`<span style="color:#79c0ff;margin-left:4px">TR:${ks.textrank_score.toFixed(2)}</span>`:'';
        const entities=(ks.entities||[]).map(e=>`<span style="background:#1a3a1a;color:#7ee787;padding:1px 4px;border-radius:2px;font-size:10px;margin:1px">${esc(e)}</span>`).join('');
        const events=(ks.events||[]).map(e=>`<span style="background:#1a1a3a;color:#79c0ff;padding:1px 4px;border-radius:2px;font-size:10px;margin:1px">${esc(e)}</span>`).join('');
        return `<div style="padding:8px 12px;margin:4px 0;background:#0d1117;border-radius:4px;border-left:3px solid #58a6ff">
            <div style="font-size:13px;color:#c9d1d9">${esc(ks.text)}</div>
            <div style="font-size:11px;color:#8b949e;margin-top:4px">
                <span style="color:#7ee787">综合:${(ks.score||0).toFixed(3)}</span>${nsp}${tr}
                <span style="margin-left:8px;color:#484f58">[句${ks.sentence_index+1}]</span>
            </div>
            ${entities||events?`<div style="margin-top:4px">${entities}${events}</div>`:''}
        </div>`;
    }).join('');

    document.getElementById('summary').innerHTML=`
    <div class="card">
        <h3>📝 摘要 <small style="color:#484f58;font-weight:normal">${methodLabel} · ${modeLabel} · ${s.key_sentences.length}/${s.total_sentences} 句 (${ratio}%)</small></h3>
        ${fusionInfo}
        ${s.summary_text?`<div style="padding:12px;background:#0d1117;border-radius:4px;margin-bottom:12px;font-size:14px;line-height:1.8;color:#c9d1d9;border:1px solid #30363d">${esc(s.summary_text)}</div>`:''}
        <h4 style="font-size:12px;color:#8b949e;margin-bottom:8px">关键句 (按综合得分排序)</h4>
        ${sentencesHtml||'<span style="color:#484f58">无关键句</span>'}
    </div>`;
}

function renderTitle(data){
    if(!data||!data.content||!data.content.title){
        document.getElementById('title').innerHTML='<div class="card"><h3>🏷️ 标题</h3><span style="color:#484f58">未生成标题（文本过短或无实体）</span></div>';
        return;
    }
    const ti=data.content.title;
    const methodLabel=ti.method==='entity_composition'?'实体组合':ti.method==='textrank_truncate'?'TextRank截断':ti.method==='textrank_fallback'?'TextRank回退':'未知';
    const modeLabel=ti.mode==='chars'?`按字数 (${ti.target_chars}字)`:ti.mode==='ratio'?`按比例 (${(ti.target_ratio*100).toFixed(0)}%)`:'';
    const entitiesHtml=(ti.entities_used||[]).map(e=>`<span style="background:#1a3a1a;color:#7ee787;padding:2px 6px;border-radius:3px;font-size:11px;margin:2px">${esc(e)}</span>`).join('');
    const predsHtml=(ti.predicates_used||[]).map(p=>`<span style="background:#3a1a1a;color:#f97583;padding:2px 6px;border-radius:3px;font-size:11px;margin:2px">${esc(p)}</span>`).join('');
    const reasonsHtml=(ti.reasons||[]).map(r=>`<span style="color:#8b949e;font-size:10px;margin-right:6px">${esc(r)}</span>`).join('');

    document.getElementById('title').innerHTML=`
    <div class="card">
        <h3>🏷️ 标题 <small style="color:#484f58;font-weight:normal">${methodLabel} · ${modeLabel}</small></h3>
        ${ti.title_text?`<div style="padding:16px;background:#0d1117;border-radius:4px;margin-bottom:12px;font-size:20px;font-weight:bold;line-height:1.4;color:#e6edf3;border:1px solid #30363d;text-align:center">${esc(ti.title_text)}</div>`:'<div style="color:#484f58">无标题文本</div>'}
        ${entitiesHtml?`<div style="margin-bottom:8px"><span style="color:#8b949e;font-size:11px">实体: </span>${entitiesHtml}</div>`:''}
        ${predsHtml?`<div style="margin-bottom:8px"><span style="color:#8b949e;font-size:11px">谓词: </span>${predsHtml}</div>`:''}
        ${ti.source_text?`<div style="margin-bottom:8px;font-size:11px;color:#484f58">源句: ${esc(ti.source_text)}</div>`:''}
        ${ti.score>0?`<div style="font-size:11px;color:#8b949e">得分: ${ti.score.toFixed(4)}</div>`:''}
        <div style="margin-top:8px">${reasonsHtml}</div>
    </div>`;
}

function renderJSON(data){
    document.getElementById('json').innerHTML=`<div class="card"><pre class="pretty">${escapeHtml(JSON.stringify(data,null,2))}</pre></div>`;
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
            resp:'<pre class="pretty">{\n  "meta": {"source": "hanlp_v2", "timestamp": "..."},\n  "content": {\n    "tokens": [...],\n    "entities": [...],\n    "relations": [...],\n    "patterns": [...]\n  },\n  "true_new_words": [...] | null\n}</pre>',
            ex:'curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d \'{"text":"北冥有鱼，其名为鲲。","language":"classical"}\''},
        {method:'POST',path:'/analyze/dep',badge:'post',summary:'依存句法分析 — 返回 token 列表 + 依存边',
            desc:'<p>专为前端 SVG 渲染优化的端点，返回扁平化的 token 列表和依存关系边。</p>',
            fields:[
                {name:'text',type:'string',desc:'(必填) 待分析的中文文本'},
                {name:'dict_combine',type:'string[]',desc:'自定义词典',default:'[]'},
            ],
            resp:'<pre class="pretty">{\n  "tokens": [{"id": 0, "text": "碳钢", "pos": "NN"}, ...],\n  "deps": [{"child": 0, "head": 1, "rel": "nsubj"}, ...],\n  "text": "碳钢是钢的一种..."\n}</pre>',
            ex:'curl -X POST http://localhost:8000/analyze/dep -H "Content-Type: application/json" -d \'{"text":"碳钢是钢的一种。"}\''},
        {method:'POST',path:'/analyze/discover',badge:'post',summary:'新词发现 — PMI + MTL 混合 + ConvSeg 比对',
            desc:'<p>使用 PMI 逐点互信息、MTL 分词对比和 ConvSeg（PKU_NAME）三种引擎发现潜在的未登录词。</p>',
            fields:[
                {name:'text',type:'string',desc:'(必填) 待分析的中文文本'},
                {name:'dict_combine',type:'string[]',desc:'自定义词典',default:'[]'},
            ],
            resp:'<pre class="pretty">{\n  "candidates": [{"word": "高强度", "score": 8.5, "freq": 3}, ...],\n  "convseg": {"candidates": ["立方庭", ...]}\n}</pre>',
            ex:'curl -X POST http://localhost:8000/analyze/discover -H "Content-Type: application/json" -d \'{"text":"碳钢是钢的一种，具有高强度。"}\''},
        {method:'POST',path:'/analyze/pretty',badge:'post',summary:'HanLP 原生可视化 — 返回 pretty-print 文本',
            desc:'<p>返回 HanLP Document 的 <code>to_pretty()</code> 文本输出，与 Jupyter Notebook 中的展示效果一致。</p>',
            fields:[
                {name:'text',type:'string',desc:'(必填) 待分析的中文文本'},
                {name:'dict_combine',type:'string[]',desc:'自定义词典',default:'[]'},
            ],
            resp:'<pre class="pretty">  tok/fine: [碳钢, 是, 钢, 的, 一种, ，, 具有, 高强度, 和, 高韧性, 。]\n  pos/ctb: [NN, VC, NN, DEG, CD, PU, VV, NN, CC, NN, PU]\n  ...</pre>',
            ex:'curl -X POST http://localhost:8000/analyze/pretty -H "Content-Type: application/json" -d \'{"text":"碳钢是钢的一种。"}\''},
    ];
    let html=`<div class="api-doc-intro">
      <p>Narrative Operator NLP 提供以下 REST API 端点，所有请求均通过 <code>HTTP POST</code> 或 <code>GET</code> 访问。
      人类可读文档请访问 <a href="/docs" style="color:#58a6ff">Swagger UI</a>（支持在线测试）或 <a href="/redoc" style="color:#58a6ff">ReDoc</a>（三栏布局）。
      智能体可程序化读取 <a href="/openapi.json" style="color:#58a6ff">OpenAPI JSON</a> 规范。</p>
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

// Toggle sub-event rows under a main event
// Sub-events are rendered immediately after their parent main event row,
// and are identified by NOT having data-main-event attribute.
window.toggleEventRow=function(el, eventId){
    const mainRow = document.querySelector(`tr.event-row[data-event-id="${eventId}"]`);
    if (!mainRow) return;
    
    // Collect consecutive sibling rows that are sub-events (no data-main-event attribute)
    let next = mainRow.nextElementSibling;
    const subRows = [];
    while (next && next.classList.contains('event-row') && !next.hasAttribute('data-main-event')) {
        subRows.push(next);
        next = next.nextElementSibling;
    }
    
    if (subRows.length === 0) return;
    
    // Determine current visibility from the first sub-row
    const isCurrentlyVisible = subRows[0].style.display !== 'none';
    
    subRows.forEach(row => {
        row.style.display = isCurrentlyVisible ? 'none' : '';
    });
    
    // Toggle arrow direction
    el.textContent = isCurrentlyVisible ? '▶' : '▼';
};

function escapeHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
const esc=escapeHtml;

// On load: warm up all three models in parallel, show loading status
window.onload=async function(){
    renderTabs();
    renderSamples();
    bindControls();
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
