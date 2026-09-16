// ── render.js — pure rendering functions (no app state) ─────
export function escapeHtml(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
export const esc=escapeHtml;
// Escape a value for use inside a double-quoted HTML attribute.
export function escAttr(s){return esc(String(s)).replace(/"/g,"&quot;").replace(/'/g,"&#39;");}

// ── Overview dashboard (default tab) ───────────────────────────────────────
export function renderOverview(data){
    const el=document.getElementById('overview');
    if(!data||!data.content){ el.innerHTML='<div class="card"><span class="ov-empty">暂无数据 — 输入文本并点击「分析」</span></div>'; return; }
    const c=data.content;
    const meta=data.meta||{};
    const langMode=meta.language_mode||'auto';
    const langBadge=langMode==='classical'?'<span class="lang-badge classical">🏯 古汉语</span>'
        :langMode==='modern'?'<span class="lang-badge modern">📄 现代汉语</span>'
        :langMode==='english'?'<span class="lang-badge english">🇬🇧 English</span>'
        :'<span class="lang-badge" style="background:#1a1a3a;color:#79c0ff">🔄 自动识别</span>';

    const stats=`<div class="ov-stats">
        <div class="ov-stat"><span class="ov-num">${c.tokens.length}</span><span class="ov-cap">Tokens</span></div>
        <div class="ov-stat"><span class="ov-num">${c.entities.length}</span><span class="ov-cap">实体</span></div>
        <div class="ov-stat"><span class="ov-num">${c.relations.length}</span><span class="ov-cap">关系</span></div>
        <div class="ov-stat"><span class="ov-num">${(c.events||[]).length}</span><span class="ov-cap">事件</span></div>
        <div class="ov-stat"><span class="ov-num">${(c.sentences||[]).length}</span><span class="ov-cap">句子</span></div>
        <div class="ov-stat"><span class="ov-num" style="font-size:12px;padding-top:6px">${langBadge}</span><span class="ov-cap">语言模式</span></div>
    </div>`;

    const ti=c.title;
    const titleCard=ti&&ti.title_text?`
        <div class="ov-card" data-goto-tab="title">
            <h4>🏷️ 生成标题 <span class="ov-go">查看 →</span></h4>
            <div class="ov-title-text">${esc(ti.title_text)}</div>
        </div>`:`<div class="ov-card"><h4>🏷️ 生成标题</h4><div class="ov-empty">未生成</div></div>`;

    const s=c.summary;
    const summaryCard=s&&s.summary_text?`
        <div class="ov-card" data-goto-tab="summary">
            <h4>📝 摘要 <span class="ov-go">查看 →</span></h4>
            <div class="ov-summary-text">${esc(s.summary_text)}</div>
        </div>`:`<div class="ov-card"><h4>📝 摘要</h4><div class="ov-empty">未生成</div></div>`;

    const entChips=c.entities.length?`
        <div class="ov-card" data-goto-tab="nsp">
            <h4>🏷️ 实体速览 <span class="ov-go">查看 →</span></h4>
            <div class="ov-ent-chips">${c.entities.slice(0,12).map(e=>`<span class="entity-card ent-${e.category}"><b>${esc(e.text)}</b></span>`).join('')}${c.entities.length>12?`<span class="ov-empty">+${c.entities.length-12}</span>`:''}</div>
        </div>`:`<div class="ov-card"><h4>🏷️ 实体速览</h4><div class="ov-empty">未识别到实体</div></div>`;

    const evs=(c.events||[]).slice().sort((a,b)=>b.confidence-a.confidence).slice(0,3);
    const eventCard=evs.length?`
        <div class="ov-card" data-goto-tab="nsp">
            <h4>⚡ 事件速览 <span class="ov-go">查看 →</span></h4>
            ${evs.map(ev=>{
                const agent=(ev.arguments||[]).find(a=>a.role==='Agent');
                const patient=(ev.arguments||[]).find(a=>['Patient','Result','Product'].includes(a.role));
                const args=[agent&&agent.text,patient&&patient.text].filter(Boolean).join(' → ');
                return `<div class="ov-event-item"><span class="event-type">${esc(ev.event_type)}</span><span class="ov-trigger">${esc(ev.trigger)}</span><span class="ov-args">${esc(args)}</span></div>`;
            }).join('')}
        </div>`:`<div class="ov-card"><h4>⚡ 事件速览</h4><div class="ov-empty">未提取到事件</div></div>`;

    el.innerHTML=`<div class="card">${stats}<div class="ov-grid">${titleCard}${summaryCard}${entChips}${eventCard}</div></div>`;
}

export function renderNSP(data){
    if(!data||!data.content){ document.getElementById('nsp').innerHTML='<div class="error">分析失败：服务器未响应</div>'; return; }
    const c=data.content;
    const trueNew=data.true_new_words||[];

    // Token stream: POS color + confidence bar + entity underline.
    // Longest entity wins on overlapping token spans.
    const entByToken={};
    [...c.entities].sort((a,b)=>(b.span[1]-b.span[0])-(a.span[1]-a.span[0])).forEach(e=>{
        for(let i=e.span[0];i<e.span[1];i++){ if(!entByToken[i]) entByToken[i]=e; }
    });
    const tokens=c.tokens.map((t,i)=>{
        const pct=Math.round((t.confidence||1)*100);
        const confCls=pct>=95?'conf-hi':pct>=80?'conf-mid':'conf-lo';
        const sourceDot=t.source==='hanlp_lzh'?'<span class="source-dot hanlp_lzh" title="古汉语模型"></span>'
            :t.source==='en_modernbert'?'<span class="source-dot en_modernbert" title="英文模型"></span>'
            :'<span class="source-dot hanlp_v2" title="现代汉语模型"></span>';
        const e=entByToken[i];
        const entCls=e?` ent-${e.category}`:'';
        const entTitle=e?` · 实体:${e.text} (${e.category})`:'';
        return `<span class="token ${t.pos}${entCls}" title="POS:${t.pos} span:${t.span} conf:${pct}% source:${t.source}${entTitle}">${sourceDot}${t.text}<i class="conf ${confCls}"></i></span>`;
    }).join('');

    // Entities grouped by category
    const entityMap={};
    c.entities.forEach(e=>entityMap[e.id]=e);
    const byCat={};
    c.entities.forEach(e=>{ (byCat[e.category]=byCat[e.category]||[]).push(e); });

    function entityChip(e){
        const pct=Math.round((e.confidence||1)*100);
        const confCls=pct>=95?'conf-hi':pct>=80?'conf-mid':'conf-lo';
        const attrs=(e.attributes||[]).map(a=>`<span class="ent-attr">${a.key}=${a.value||'?'}</span>`).join(' ');
        const parent=e.parent_entity_id?`<span class="ent-parent">⊂${esc((entityMap[e.parent_entity_id]||{}).text||e.parent_entity_id)}</span>`:'';
        return `<span class="entity-card ent-${e.category}"><b>${esc(e.text)}</b>${attrs}${parent}<span class="ent-span">[${e.span[0]}:${e.span[1]}]</span><span class="ent-conf ${confCls}">${pct}%</span></span>`;
    }

    const entGroups=Object.entries(byCat).map(([cat,list])=>{
        list.sort((a,b)=>a.span[0]-b.span[0]);
        return `<div class="ent-group"><span class="ent-cat">${cat}</span><span class="ent-count">×${list.length}</span><div class="ent-items">${list.map(entityChip).join('')}</div></div>`;
    }).join('')||'<span class="empty">未识别到实体</span>';

    const entities=entGroups;

    // Events rendering — cards with collapsible sub-events
    const events = c.events && c.events.length ? (()=>{
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
        const buildArgs = (ev) => {
            const args = ev.arguments || [];
            const find = role => args.find(a => a.role === role);
            const patient = args.find(a => ['Patient','Result','Product'].includes(a.role));
            const others = args.filter(a => a.role !== 'Agent' && !['Patient','Result','Product'].includes(a.role) && a.role !== 'Time' && a.role !== 'Location');
            const tag = (a, dim) => a ? `<span class="arg-tag${dim?' dim':''}">${esc(a.role)}: ${esc(a.text)}${a.spatial_role && a.spatial_role !== 'STATIC' ? ` <span class="spatial-tag">[${a.spatial_role}]</span>` : ''}</span>` : '<span class="arg-none">-</span>';
            return {
                agent: tag(find('Agent')),
                patient: tag(patient),
                time: tag(find('Time')),
                location: tag(find('Location')),
                others: others.map(a => `<span class="arg-tag dim">${esc(a.role)}: ${esc(a.text)}</span>`).join(' ') || '<span class="arg-none">-</span>',
                vsc: [...new Set(args.map(a => a.verb_spatial_class).filter(s => s && s !== 'NONE'))].map(s => `<span class="vsc-tag">${s}</span>`).join(' ') || '<span class="arg-none">-</span>'
            };
        };
        const cardHtml = (ev, isSub) => {
            const args = buildArgs(ev);
            const pct = Math.round(ev.confidence * 100);
            const confColor = pct >= 95 ? 'var(--green-text)' : pct >= 80 ? 'var(--yellow)' : 'var(--red)';
            const arrow = ev.sub_events && ev.sub_events.length > 0 ? `<span class="event-toggle" data-event-id="${escAttr(ev.id)}">▶</span>` : (isSub ? '<span class="arg-none">└</span>' : '');
            const badge = ev.is_main_event && !isSub ? '<span class="event-badge main">主</span>' : (isSub ? '<span class="event-badge subbadge">子</span>' : '');
            return `<div class="event-card${isSub?' sub':''}" data-event-id="${escAttr(ev.id)}" ${isSub?'style="display:none"':''}>
                <div class="event-head">
                    ${arrow}${badge}
                    <span class="event-type">${esc(ev.event_type)}</span>
                    <span class="event-trigger">${esc(ev.trigger)}</span>
                    <span class="event-span">[${ev.trigger_span[0]}:${ev.trigger_span[1]}]</span>
                    <span class="event-conf"><span class="conf-track"><span class="conf-fill" style="width:${pct}%;background:${confColor}"></span></span>${pct}%</span>
                    <span class="event-src">${esc(ev.source)}</span>
                </div>
                <div class="event-args">
                    <div class="arg"><span class="arg-role">Agent</span>${args.agent}</div>
                    <div class="arg"><span class="arg-role">Patient</span>${args.patient}</div>
                    <div class="arg"><span class="arg-role">Time</span>${args.time}</div>
                    <div class="arg"><span class="arg-role">Location</span>${args.location}</div>
                    <div class="arg"><span class="arg-role">其他</span>${args.others}</div>
                    <div class="arg"><span class="arg-role">语义类</span>${args.vsc}</div>
                </div>
            </div>`;
        };
        let html = '';
        if (mainEvents.length === 0) {
            c.events.forEach(ev => { html += cardHtml(ev, false); });
        } else {
            mainEvents.forEach(ev => {
                html += cardHtml(ev, false);
                (subEventMap[ev.id] || []).forEach(sub => { html += cardHtml(sub, true); });
            });
            c.events.filter(ev => !ev.is_main_event && !mainEvents.some(m => m.sub_events && m.sub_events.includes(ev.id))).forEach(ev => { html += cardHtml(ev, false); });
        }
        return html;
    })() : '<span class="empty">未提取到事件（此文本暂不支持事件分析，或无明显的动作型关系三元组）</span>';

    // Relations — chip rows: subject → predicate → object
    const relations=c.relations.map(r=>{
        const subjRaw = r.subject_raw && r.subject_raw !== r.subject ? `<small>raw: ${esc(r.subject_raw)}</small>` : '';
        const objRaw = r.object_raw && r.object_raw !== r.object ? `<small>raw: ${esc(r.object_raw)}</small>` : '';
        const mods=(r.modifiers||[]).map(m=>`<span class="rel-mod ${m.type}">[${esc(m.text)}:${m.type}]</span>`).join('');
        const srcLabel=r.source==='hierarchy/containment'?'⊂hierarchy':esc(r.source);
        return `<div class="relation-row">
            <span class="rel-chip subj">${esc(r.subject)}${subjRaw}</span>
            <span class="rel-arrow">→</span>
            <span class="rel-chip pred">${esc(r.predicate)}</span>
            <span class="rel-arrow">→</span>
            <span class="rel-chip obj">${esc(r.object)}${objRaw}</span>
            ${mods}
            <span class="rel-src">${srcLabel}</span>
        </div>`;
    }).join('')||'<span class="empty">未抽取到关系</span>';

    let newWordsHtml='';
    const mode=document.getElementById('discoverMode').value;
    if(trueNew.length>0){
        const label=mode==='enhance'?'⚡ 强化模式已应用':'🆕 发现真新词';
        const borderColor=mode==='enhance'?'#238636':'#e3b341';
        const titleColor=mode==='enhance'?'#7ee787':'#e3b341';
        const badges=trueNew.map(w=>`<span class="discover-item" data-word="${escAttr(w)}" style="border-color:${borderColor}" title="点击加入自定义词典">${mode==='enhance'?'✅':'🆕'} ${esc(w)}</span>`).join('');
        newWordsHtml=`<div class="card" style="border-color:${borderColor}"><h3 style="color:${titleColor}">${label} <small style="color:#8b949e;font-weight:normal">(点击加入自定义词典)</small></h3><div>${badges}</div><button data-action="apply-dict" style="margin-top:8px;font-size:12px;padding:6px 16px">📋 一键应用并重新分析</button></div>`;
    }else if(data.true_new_words!==undefined&&data.true_new_words!==null){
        newWordsHtml='<div class="card"><h3>🔍 新词发现</h3><span class="empty">未发现真新词（所有候选词已在分词结果中）</span></div>';
    }

    document.getElementById('nsp').innerHTML=`
    ${newWordsHtml}
    <div class="card"><h3>📝 分词 & POS <span class="empty" style="font-weight:normal">●蓝=现代 · ●黄=古汉语 · ●浅蓝=英文 · 底部色条=置信度 · 下划线=实体</span></h3><div style="line-height:2.1">${tokens}</div></div>
    <div class="card"><h3>🏷️ 实体 Entities <span class="empty" style="font-weight:normal">按类别分组</span></h3>${entities}</div>
    <div class="card"><h3>🔗 关系 Relations</h3>${relations}</div>
    <div class="card"><h3>⚡ 事件 Events <span class="empty" style="font-weight:normal">(从 SRL 语义角色标注派生)</span></h3>${events}</div>`;
}

export function renderPretty(data){
    if(!data||!data.pretty){
        const err=data&&data.detail?data.detail:(data&&data._error?data._error:'无 pretty 字段');
        document.getElementById('pretty').innerHTML=`<div class="error">分析失败: ${esc(err)}</div>`;
        return;
    }
    document.getElementById('pretty').innerHTML=`<div class="card"><pre class="pretty">${escapeHtml(data.pretty)}</pre></div>`;
}

export function renderDepSVG(data){
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

export function renderDiscover(data){
    if(!data||!data.candidates){ document.getElementById('discover').innerHTML='<div class="card">未发现候选新词</div>'; return; }
    const candidates=data.candidates;
    const convseg=data.convseg;

    // PMI+MTL candidates
    const pmiItems=candidates.length?candidates.map(c=>{
        const color=c.score>=5?'#7ee787':c.score>=3?'#e3b341':'#f85149';
        return `<span class="discover-item" data-word="${escAttr(c.word)}" title="点击添加到自定义词典">${esc(c.word)}<span class="score" style="color:${color}">${c.score.toFixed(1)}</span></span>`;
    }).join(''):'<span style="color:#484f58">-</span>';

    // Coarse ELECTRA comparison (PyTorch, no TF needed)
    let convsegHtml='';
    if(convseg){
        if(convseg.candidates&&convseg.candidates.length){
            convsegHtml=convseg.candidates.map(w=>{
                const inPmi=candidates.some(c=>c.word===w);
                const style=inPmi?'background:#1a3a1a;border-color:#238636':'';
                return `<span class="discover-item" style="${style}" data-word="${escAttr(w)}" title="点击添加到自定义词典">${esc(w)}<span class="score" style="color:#58a6ff">粗分</span></span>`;
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
      <button data-action="apply-dict" style="margin-top:8px">📋 应用选中词典并重新分析</button>
      <button data-action="clear-dict" style="margin-top:8px;margin-left:8px;background:#21262d;color:#c9d1d9">清除选择</button>
    </div>`;
}

export function renderPatterns(patterns){
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

export function renderLangDetect(data){
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

export function renderCoref(data){
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

export function renderSummary(data){
    if(!data||!data.content||!data.content.summary){
        document.getElementById('summary').innerHTML='<div class="card"><h3>📝 摘要</h3><span class="empty">未生成摘要（文本过短或无句子结构）</span></div>';
        return;
    }
    const s=data.content.summary;
    const methodLabel=s.method==='nsp+textrank'?'NSP+TextRank 融合':s.method==='textrank'?'TextRank':(s.method||'未知');
    const modeLabel=s.mode==='chars'?`按字数 (${s.target_chars}字)`:s.mode==='ratio'?`按比例 (${(s.target_ratio*100).toFixed(0)}%)`:'';
    const ratio=s.total_sentences>0?Math.round((s.key_sentences.length/s.total_sentences)*100):0;
    const fusionInfo=s.fusion_weights?`<div class="empty" style="margin-bottom:10px">权重: NSP=${(s.fusion_weights.nsp_weight||0).toFixed(2)} / TextRank=${(s.fusion_weights.textrank_weight||0).toFixed(2)}</div>`:'';

    // Full text with key sentences highlighted
    const keyIdx=new Set((s.key_sentences||[]).map(k=>k.sentence_index));
    const sents=data.content.sentences||[];
    const fullTextHtml=sents.length?`<div class="sum-fulltext">${sents.map((st,i)=>`<span class="sum-sent${keyIdx.has(i)?' key':''}" title="${keyIdx.has(i)?'关键句':''}">${esc(st.text)}</span>`).join('')}</div>`:'';

    // Key sentences with fused score bar + selection reasons
    const maxScore=Math.max(...(s.key_sentences||[]).map(k=>k.score||0),0.001);
    const sentencesHtml=(s.key_sentences||[]).map(ks=>{
        const pct=Math.max(4,Math.round(((ks.score||0)/maxScore)*100));
        const reasons=(ks.reasons||[]).map(r=>`<span class="empty">${esc(r)}</span>`).join(' · ');
        return `<div class="sum-key-item">
            <div class="sum-text">${esc(ks.text)}</div>
            <div class="sum-meta">
                <span class="sum-score"><span class="conf-track"><span class="conf-fill" style="width:${pct}%"></span></span>综合:${(ks.score||0).toFixed(3)}</span>
                <span class="empty">[句${(ks.sentence_index||0)+1}]</span>
                ${reasons}
            </div>
        </div>`;
    }).join('');

    document.getElementById('summary').innerHTML=`
    <div class="card">
        <h3>📝 摘要 <span class="empty" style="font-weight:normal">${methodLabel} · ${modeLabel} · ${s.key_sentences.length}/${s.total_sentences} 句 (${ratio}%)</span></h3>
        ${fusionInfo}
        ${s.summary_text?`<div style="padding:12px;background:var(--bg-inset);border-radius:var(--radius);margin-bottom:12px;font-size:14px;line-height:1.8;color:var(--text);border:1px solid var(--border)">${esc(s.summary_text)}</div>`:''}
        ${fullTextHtml}
        <h4 style="font-size:12px;color:var(--text-dim);margin-bottom:8px">关键句 (按综合得分排序)</h4>
        ${sentencesHtml||'<span class="empty">无关键句</span>'}
    </div>`;
}

export function renderTitle(data){
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

export function renderJSON(data){
    document.getElementById('json').innerHTML=`<div class="card"><pre class="pretty">${escapeHtml(JSON.stringify(data,null,2))}</pre></div>`;
}

export function renderAPIDocs(){
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
          <div class="method" data-endpoint="${i}">
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

export function toggleEndpoint(methodEl){
    const body=methodEl.nextElementSibling;
    body.classList.toggle('open');
    const expand=methodEl.querySelector('.expand');
    expand.textContent=body.classList.contains('open')?'▲':'▼';
}

// Toggle sub-event rows under a main event
// Sub-events are rendered immediately after their parent main event row,
// and are identified by NOT having data-main-event attribute.
export function toggleEventRow(el, eventId){
    const mainCard = document.querySelector(`.event-card[data-event-id="${eventId}"]:not(.sub)`);
    if (!mainCard) return;

    // Sub-cards are rendered immediately after their parent main card
    let next = mainCard.nextElementSibling;
    const subCards = [];
    while (next && next.classList.contains('event-card') && next.classList.contains('sub')) {
        subCards.push(next);
        next = next.nextElementSibling;
    }

    if (subCards.length === 0) return;

    const isCurrentlyVisible = subCards[0].style.display !== 'none';
    subCards.forEach(card => {
        card.style.display = isCurrentlyVisible ? 'none' : '';
    });

    el.textContent = isCurrentlyVisible ? '▶' : '▼';
}
