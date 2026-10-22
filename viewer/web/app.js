const app = document.getElementById('app');
const crumbs = document.getElementById('crumbs');
const fresh = document.getElementById('fresh');
const drawer = document.getElementById('drawer');
const drawerTitle = document.getElementById('drawer-title');
const drawerBody = document.getElementById('drawer-body');
const toast = document.getElementById('toast');

let INDEX = null;

const STAGES = [
  ['observations', '观察'],
  ['aggregate', '汇总'],
  ['design', '出题'],
  ['leak', '泄露检查'],
  ['reference', '参考解'],
  ['gate', '模板 probe'],
  ['trainee', '试做'],
  ['hits', '缺陷归因'],
  ['route', '分流'],
];

const MODULES = [
  { id: 'observations', label: '观察', stage: 'observations', file: 'observations' },
  { id: 'attribution', label: '汇总与训练目标', stage: 'aggregate', file: 'attribution' },
  { id: 'design', label: '出题', stage: 'design', file: 'design' },
  { id: 'leak', label: '泄露检查', stage: 'leak', file: 'leak' },
  { id: 'reference', label: '参考解', stage: 'reference', file: 'validation', pointer: '/oracle' },
  { id: 'gate', label: '模板 probe', stage: 'gate', file: 'gate' },
  { id: 'trainee', label: '试做', stage: 'trainee', file: 'validation' },
  { id: 'hits', label: '缺陷归因', stage: 'hits', file: 'hits' },
  { id: 'route', label: '分流', stage: 'route', file: 'route' },
  { id: 'events', label: '事件与成本', stage: null, file: 'events', kind: 'text' },
  { id: 'synthesis', label: '运行元数据', stage: null, file: 'synthesis' },
  { id: 'raw', label: '原始产物', stage: null, file: null, kind: 'tree' },
];

const STATE_CLASS = {
  done: 'ok', accepted: 'ok', completed: 'ok', rejected: 'bad', failed: 'bad',
  error: 'bad', pending: 'warn', partial: 'warn', blocked: 'blocked',
  missing: 'muted',
};
const STAGE_ZH = Object.fromEntries(STAGES);
const KIND_ZH = { analysis: '分析', design: '出题', retry: '重试', unknown: '?' };

const enc = encodeURIComponent;
const dec = decodeURIComponent;

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid === undefined || kid === null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

async function getJSON(url) {
  const r = await fetch(url);
  const text = await r.text();
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${text.slice(0, 300)}`);
  return JSON.parse(text);
}

function apiJSON(root, path, pointer = '') {
  const q = new URLSearchParams({ root, path });
  if (pointer) q.set('pointer', pointer);
  return getJSON('/api/json?' + q.toString());
}

function apiText(root, path) {
  const q = new URLSearchParams({ root, path });
  return fetch('/api/text?' + q.toString()).then(async r => {
    const t = await r.text();
    if (!r.ok) throw new Error(`${r.status}: ${t.slice(0, 200)}`);
    return t;
  });
}

function apiTree(root, path, depth = 1) {
  const q = new URLSearchParams({ root, path, depth });
  return getJSON('/api/tree?' + q.toString());
}

function apiXlsx(root, path, sheet = '', mode = 'values') {
  const q = new URLSearchParams({ root, path, sheet, mode });
  return getJSON('/api/xlsx?' + q.toString());
}

function apiDiff(root, left, right) {
  const q = new URLSearchParams({ root, left, right });
  return getJSON('/api/xlsx/diff?' + q.toString());
}

function fmtBytes(n) {
  if (n === null || n === undefined) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function fmtPct(v) {
  if (typeof v !== 'number' || Number.isNaN(v)) return '—';
  return `${(v * 100).toFixed(1)}%`;
}

function fmtVal(v) {
  if (v === null || v === undefined || v === '') return '—';
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (Array.isArray(v)) return `[${v.length}]`;
  if (typeof v === 'object') return `{${Object.keys(v).length}}`;
  return String(v);
}

function badge(state, text) {
  const label = text ?? (state || '—');
  return h('span', { class: `badge ${STATE_CLASS[state] || 'muted'}` }, label);
}

function stageBadge(state) {
  const zh = { missing: '未产出', done: '完成', present: '有产物', pending: '等待',
               blocked: '阻塞', rejected: '拒收', accepted: '通过', failed: '失败',
               error: '错误', partial: '部分', completed: '完成' }[state] || state;
  return badge(state, zh || '—');
}

function copy(text, label = '已复制') {
  navigator.clipboard?.writeText(text).then(() => showToast(label)).catch(() => {});
}

let toastTimer = null;
function showToast(msg) {
  toast.textContent = msg;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 1400);
}

function go(hash) { location.hash = hash; }

function tagged(v) {
  if (typeof v !== 'string') return null;
  const m = v.match(/^(runs|batch):(.+)$/s);
  return m ? { root: m[1], path: m[2] } : null;
}

function openDrawer(title) {
  drawerTitle.textContent = title;
  drawerBody.replaceChildren();
  drawer.hidden = false;
}

function closeDrawer() { drawer.hidden = true; }

document.getElementById('drawer-close').onclick = closeDrawer;

function openJSON(root, path, pointer = '', title = '') {
  openDrawer(title || `${root}:${path}${pointer ? ' ' + pointer : ''}`);
  drawerBody.append(h('div', { class: 'loading' }, 'loading…'));
  apiJSON(root, path, pointer).then(res => {
    drawerBody.replaceChildren(
      h('div', { class: 'row', style: 'margin-bottom:10px' },
        h('code', {}, `${root}:${res.path}${pointer}`),
        h('button', { class: 'iconbtn', onclick: () => copy(`${root}:${res.path}${pointer}`) }, 'copy'),
        h('a', { class: 'iconbtn', href: `/api/file?root=${root}&path=${enc(res.path)}`, target: '_blank' }, 'download'),
        h('span', { class: 'spacer' }),
        h('span', { class: 'muted' }, `${fmtBytes(res.bytes)} · ${res.sha256?.slice(0, 12)}`)),
      h('pre', { class: 'json' }, JSON.stringify(res.value, null, 1)),
    );
  }).catch(err => drawerBody.replaceChildren(h('div', { class: 'error' }, String(err))));
}

function openText(root, path, title = '') {
  openDrawer(title || `${root}:${path}`);
  apiText(root, path).then(t => {
    drawerBody.replaceChildren(
      h('div', { class: 'row', style: 'margin-bottom:10px' },
        h('code', {}, `${root}:${path}`),
        h('button', { class: 'iconbtn', onclick: () => copy(`${root}:${path}`) }, 'copy'),
        h('a', { class: 'iconbtn', href: `/api/file?root=${root}&path=${enc(path)}`, target: '_blank' }, 'download')),
      h('pre', { class: 'text' }, t),
    );
  }).catch(err => drawerBody.replaceChildren(h('div', { class: 'error' }, String(err))));
}

function openRecord(obj, source, pointer, title) {
  openDrawer(title || 'record');
  drawerBody.replaceChildren(
    h('div', { class: 'row', style: 'margin-bottom:10px' },
      source ? h('button', { class: 'iconbtn', onclick: () => openJSON(source.root, source.path, pointer, title) }, '在源文件中定位') : null),
    h('pre', { class: 'json' }, JSON.stringify(obj, null, 1)),
  );
}

function fileButton(root, path, label) {
  const ext = (path.split('.').pop() || '').toLowerCase();
  if (ext === 'xlsx' || ext === 'xlsm') {
    return h('a', { class: 'iconbtn', href: `#/x/${root}/${enc(path)}` }, label || 'xlsx');
  }
  if (ext === 'json') {
    return h('button', { class: 'iconbtn', onclick: () => openJSON(root, path) }, label || 'json');
  }
  return h('button', { class: 'iconbtn', onclick: () => openText(root, path) }, label || 'view');
}

function taggedLink(v) {
  const t = tagged(v);
  if (!t) return null;
  const short = t.path.length > 46 ? '…' + t.path.slice(-45) : t.path;
  return h('span', { class: 'row', style: 'gap:4px' },
    fileButton(t.root, t.path, short),
    h('button', { class: 'iconbtn', onclick: () => copy(`${t.root}:${t.path}`) }, 'copy'));
}

function cell(v) {
  const link = taggedLink(v);
  if (link) return link;
  if (v && typeof v === 'object') {
    return h('span', { class: 'muted' }, Array.isArray(v) ? `[${v.length}]` : `{${Object.keys(v).length}}`);
  }
  return fmtVal(v);
}

function table(cols, rows, opts = {}) {
  const thead = h('thead', {}, h('tr', {}, cols.map(c => h('th', {}, c.label))));
  const tbody = h('tbody', {}, rows.map(r => h('tr', {
    class: opts.onRow ? 'clickable' : '',
    onclick: opts.onRow ? () => opts.onRow(r) : undefined,
  }, cols.map(c => h('td', { class: c.class || '' },
    c.render ? c.render(r) : cell(r[c.key]))))));
  return h('div', { class: 'table-wrap' }, h('table', {}, thead, tbody));
}

function kv(obj, source, basePointer = '') {
  const rows = [];
  for (const [k, v] of Object.entries(obj || {})) {
    const ptr = `${basePointer}/${k.replace(/~/g, '~0').replace(/\//g, '~1')}`;
    const isObj = v && typeof v === 'object';
    rows.push(h('div', {}, k));
    rows.push(h('div', {}, cell(v),
      h('span', { class: 'pv' },
        isObj ? h('button', { class: 'iconbtn', onclick: () => openJSON(source.root, source.path, ptr, k) }, '展开') : null,
        h('button', { class: 'iconbtn', onclick: () => openJSON(source.root, source.path, ptr, k) }, '{}'),
        h('span', { class: 'type' }, Array.isArray(v) ? 'array' : v === null ? 'null' : typeof v))));
  }
  return h('div', { class: 'kv' }, rows);
}

function objectTable(rows, source, basePointer, title, cols) {
  if (!rows || !rows.length) return h('div', { class: 'empty' }, '无记录');
  const keys = cols || [...new Set(rows.flatMap(r => Object.keys(r || {})))];
  return table(keys.map(k => ({ label: k, key: k })), rows, {
    onRow: r => {
      const i = rows.indexOf(r);
      openRecord(r, source, `${basePointer}/${i}`, title || 'record');
    },
  });
}

function panel(title, ...body) {
  return h('section', { class: 'panel' },
    h('div', { class: 'panel-head' }, h('h2', {}, title)), ...body);
}

function errorBox(err) { return h('div', { class: 'error' }, String(err)); }

function findTask(id) { return INDEX.tasks.find(t => t.id === id); }
function findRun(task, id) { return (task?.attempts || []).find(r => r.run_id === id); }
function currentRun(task) {
  return (task?.attempts || []).find(r => r.current) || (task?.attempts || [])[task.attempts.length - 1];
}
function passRate(run) { return run?.stages?.trainee?.pass_rate; }
function stageState(run, key) { return run?.stages?.[key]?.state || 'missing'; }

function stateOfModule(run, m) {
  if (m.stage) return run?.stages?.[m.stage]?.state || 'missing';
  if (m.kind === 'tree') return run?.episode_rel ? 'present' : 'missing';
  return run?.files?.[m.file] ? 'present' : 'missing';
}

function moduleMetrics(run, m) {
  const st = run?.stages || {};
  switch (m.id) {
    case 'observations': return [['状态', st.observations?.state], ['observation', st.observations?.n_observations], ['worker', st.observations?.n_workers]];
    case 'attribution': return [['claims', st.aggregate?.n_claims], ['训练目标', st.aggregate?.n_goals], ['复用观察', st.observations?.reused_from ? '是' : '否']];
    case 'design': return [['primary', st.design?.primary_goal_id], ['targets', st.design?.n_targets], ['preserves', st.design?.n_preserves]];
    case 'leak': return [['状态', st.leak?.state], ['verdicts', st.leak?.n_verdicts], ['模型', st.leak?.model]];
    case 'reference': return [['状态', st.reference?.state], ['reward', st.reference?.reward]];
    case 'gate': return [['状态', st.gate?.state], ['eligible', st.gate?.eligible], ['probes', st.gate?.n_probes]];
    case 'trainee': return [['状态', st.trainee?.status || st.trainee?.state], ['通过率', st.trainee?.pass_rate !== undefined ? fmtPct(st.trainee.pass_rate) : '—'], ['solved/completed', `${st.trainee?.solved ?? '—'}/${st.trainee?.completed ?? '—'}`]];
    case 'hits': return [['off_target', st.hits?.off_target], ['failed', st.hits?.failed], ['class hits', st.hits?.class_hits ? Object.entries(st.hits.class_hits).map(([k, v]) => `${k}:${v}`).join(' ') : '—']];
    case 'route': return [['route', st.route?.route], ['pass rate', st.route?.pass_rate !== undefined ? fmtPct(st.route.pass_rate) : '—'], ['band', st.route?.band ? st.route.band.join('–') : '—']];
    case 'events': return [['events', run?.episodes?.[0]?.metrics?.events?.n_events], ['耗时', run?.episodes?.[0]?.metrics?.synthesis?.elapsed_sec ? `${Math.round(run.episodes[0].metrics.synthesis.elapsed_sec)}s` : '—']];
    case 'synthesis': return [['delivered', run?.episodes?.[0]?.metrics?.synthesis?.n_delivered], ['budget', run?.episodes?.[0]?.metrics?.synthesis?.budget_mode]];
    case 'raw': return [['文件', run?.n_files], ['大小', fmtBytes(run?.size_bytes)]];
    default: return [];
  }
}

async function boot() {
  try {
    INDEX = await getJSON('/api/index');
    fresh.textContent = `index ${new Date(INDEX.built_at).toLocaleString()} · ${INDEX.batch.summary.n_runs} runs`;
  } catch (err) {
    app.replaceChildren(errorBox(`无法加载索引：${err}`));
    return;
  }
  window.addEventListener('hashchange', render);
  render();
}

function parts() {
  const raw = location.hash.replace(/^#/, '');
  return raw.split('/').filter(Boolean).map(s => { try { return dec(s); } catch { return s; } });
}

function setCrumbs(items) {
  crumbs.replaceChildren();
  items.forEach((it, i) => {
    if (i) crumbs.append(h('span', { class: 'sep' }, '/'));
    crumbs.append(it.href ? h('a', { href: it.href }, it.label) : h('span', {}, it.label));
  });
}

async function render() {
  const p = parts();
  app.replaceChildren(h('div', { class: 'loading' }, 'loading…'));
  try {
    if (!p.length || p[0] === 'b') return renderBatch();
    if (p[0] === 't') {
      const task = findTask(p[1]);
      if (!task) throw new Error(`unknown task ${p[1]}`);
      const runId = p[3];
      if (!runId) return renderTask(task);
      const run = findRun(task, runId);
      if (!run) throw new Error(`unknown run ${runId}`);
      if (p[4] === 'm') return renderModule(task, run, p[5]);
      return renderAttempt(task, run);
    }
    if (p[0] === 'dict') return renderDict();
    if (p[0] === 'tree') return renderTree(p[1], p[2] || '');
    if (p[0] === 'x') return renderXlsx(p[1], p[2] || '');
    throw new Error('unknown route');
  } catch (err) {
    app.replaceChildren(errorBox(err));
  }
}

/* ---------------------------------------------------------------- batch */

function renderBatch() {
  const s = INDEX.batch.summary;
  const batchHash = `#/b/${enc(INDEX.batch.id)}`;
  setCrumbs([{ label: INDEX.batch.id, href: batchHash }]);

  const attempts = s.n_attempts || 1;
  const funnel = h('div', { class: 'funnel' },
    STAGES.map(([key, label]) => {
      const n = s.funnel?.[key] ?? 0;
      const pct = Math.round(100 * n / attempts);
      return h('div', { class: 'funnel-row' },
        h('span', { class: 'muted' }, label),
        h('div', { class: 'funnel-bar' }, h('i', { class: pct >= 100 ? 'full' : '', style: `width:${pct}%` })),
        h('span', { class: 'mono', style: 'text-align:right' }, `${n} / ${attempts}`));
    }));

  const rows = INDEX.tasks.map(t => {
    const cur = currentRun(t);
    const st = cur?.stages || {};
    const hits = st.hits?.class_hits;
    return {
      task: t.id,
      run: cur,
      rate: st.trainee?.pass_rate,
      solved: st.trainee?.solved,
      completed: st.trainee?.completed,
      route: st.route?.route,
      leak: st.leak?.state,
      gate: st.gate?.state,
      hits,
      off: st.hits?.off_target,
      status: t.status?.stage,
    };
  });

  const cols = [
    { label: '原题', render: r => h('a', { href: `#/t/${enc(r.task)}` }, r.task) },
    { label: '当前尝试', render: r => r.run ? h('a', { href: `#/t/${enc(r.task)}/a/${enc(r.run.run_id)}` }, r.run.run_id) : h('span', { class: 'muted' }, '—') },
    { label: '通过率', render: r => r.completed ? `${r.solved}/${r.completed} · ${fmtPct(r.rate)}` : h('span', { class: 'muted' }, '—') },
    { label: '分流', render: r => r.route ? badge(r.route === 'rl' ? 'ok' : 'warn', r.route) : h('span', { class: 'muted' }, '—') },
    { label: '泄露', render: r => stageBadge(r.leak) },
    { label: '模板门', render: r => stageBadge(r.gate) },
    { label: '缺陷命中', render: r => r.hits ? Object.entries(r.hits).map(([k, v]) => `${k}:${v}`).join(' ') : h('span', { class: 'muted' }, '—') },
    { label: 'off_target', render: r => r.off === undefined ? '—' : (r.off ? badge('warn', 'yes') : badge('ok', 'no')) },
    { label: '尝试数', render: r => (INDEX.tasks.find(t => t.id === r.task)?.attempts || []).length },
    { label: '状态', render: r => r.status ? badge('present', r.status) : h('span', { class: 'muted' }, '—') },
  ];

  app.replaceChildren(
    h('div', { class: 'row' }, h('div', {},
      h('h1', {}, INDEX.batch.id),
      h('div', { class: 'sub' }, `只读审计视图 · 数据截至 ${INDEX.batch.status_updated ? new Date(INDEX.batch.status_updated * 1000).toLocaleString() : '—'}`)),
      h('span', { class: 'spacer' }),
      h('a', { class: 'chip', href: '#/dict' }, '字段字典')),
    h('div', { class: 'cards mt' },
      card('原题', s.n_tasks), card('尝试', s.n_attempts, `${s.n_runs} 个 run`),
      card('rl', s.routes?.rl || 0), card('sft', s.routes?.sft || 0),
      card('泄露通过', s.leak?.accepted || 0, `拒收 ${s.leak?.rejected || 0}`),
      card('模板门通过', s.gate?.accepted || 0)),
    panel('漏斗', funnel),
    panel('通过率分布', passStrip(INDEX.tasks)),
    panel('原题', table(cols, rows, { onRow: r => go(`#/t/${enc(r.task)}`) })),
    panel('全部尝试', attemptTable(INDEX.tasks.flatMap(t => t.attempts.map(a => ({ ...a, task: t.id }))))),
  );
}

function card(label, value, note) {
  return h('div', { class: 'card' },
    h('div', { class: 'label' }, label),
    h('div', { class: 'value' }, value ?? '—'),
    note ? h('div', { class: 'note' }, note) : null);
}

function passStrip(tasks) {
  const items = tasks.map(t => {
    const r = currentRun(t);
    const rate = passRate(r);
    return h('div', { class: 'row', style: 'gap:8px' },
      h('a', { href: `#/t/${enc(t.id)}`, style: 'width:92px' }, t.id.replace('task-', '')),
      h('div', { style: 'flex:1;height:12px;background:#edf0f4;border-radius:3px;overflow:hidden' },
        h('i', { style: `display:block;height:100%;width:${rate === undefined || rate === null ? 0 : rate * 100}%;background:${rate === undefined ? '#ccc' : (rate >= 0.75 || rate <= 0.125 ? '#d9a441' : '#5f9ea0')}` })),
      h('span', { class: 'mono', style: 'width:110px;text-align:right' },
        rate === undefined || rate === null ? '—' : `${(rate * 100).toFixed(1)}%`));
  });
  return h('div', { class: 'funnel' }, items,
    h('div', { class: 'sub' }, '绿色 = 在难度带内；橙色 = 太容易/太难'));
}

function attemptTable(attempts) {
  const cols = [
    { label: '原题', render: a => h('a', { href: `#/t/${enc(a.task)}` }, a.task) },
    { label: 'run', render: a => h('a', { href: `#/t/${enc(a.task)}/a/${enc(a.run_id)}` }, a.run_id) },
    { label: '类型', render: a => KIND_ZH[a.kind_label] || a.kind_label },
    { label: '最后阶段', render: a => a.last_stage ? badge('present', STAGE_ZH[a.last_stage] || a.last_stage) : badge('missing', '无') },
    { label: '阶段数', render: a => a.n_stages },
    { label: '通过率', render: a => passRate(a) !== undefined ? fmtPct(passRate(a)) : '—' },
    { label: '分流', render: a => a.stages?.route?.route || '—' },
    { label: '泄露', render: a => stageBadge(stageState(a, 'leak')) },
    { label: '大小', render: a => fmtBytes(a.size_bytes) },
    { label: '当前', render: a => a.current ? badge('ok', 'yes') : '' },
  ];
  return table(cols, attempts, { onRow: a => go(`#/t/${enc(a.task)}/a/${enc(a.run_id)}`) });
}

/* ----------------------------------------------------------------- task */

function renderTask(task) {
  const cur = currentRun(task);
  const batchHash = `#/b/${enc(INDEX.batch.id)}`;
  setCrumbs([{ label: INDEX.batch.id, href: batchHash },
              { label: task.id, href: `#/t/${enc(task.id)}` }]);

  const groups = { analysis: [], design: [], retry: [] };
  for (const a of task.attempts) (groups[a.kind_label] || (groups[a.kind_label] = [])).push(a);

  const tlRows = Object.entries(groups)
    .filter(([, items]) => items.length)
    .map(([kind, items]) => h('div', { class: 'row', style: 'margin:6px 0' },
      h('span', { class: 'muted', style: 'width:60px' }, KIND_ZH[kind] || kind),
      h('div', { class: 'chips' }, items.map(a => h('a', {
        class: `chip ${a.current ? 'current' : ''}`,
        href: `#/t/${enc(task.id)}/a/${enc(a.run_id)}`,
        title: `${a.last_stage || 'no stage'} · ${fmtBytes(a.size_bytes)}`,
      },
        a.run_id,
        a.stages?.trainee?.pass_rate !== undefined
          ? h('span', { class: 'muted' }, fmtPct(a.stages.trainee.pass_rate)) : null,
        a.current ? badge('ok', '当前') : null)))));

  const timeline = h('div', { class: 'panel' },
    h('div', { class: 'panel-head' }, h('h2', {}, '尝试时间线')),
    tlRows);

  const attrFile = cur?.files?.attribution ? `${cur.rel}/${cur.files.attribution}` : null;
  const attrPanel = panel('错误类库与训练目标',
    attrFile ? h('div', { class: 'loading' }, 'loading…') : h('div', { class: 'empty' }, '当前尝试没有汇总产物'));
  if (attrFile) {
    apiJSON('runs', attrFile).then(res => {
      const d = res.value || {};
      const claims = (d.claims || []).map(c => ({
        id: c.id, kind: c.kind, strength: c.evidence_strength,
        support: c.support_sample_count,
        statement: String(c.statement || ''),
      }));
      const goals = (d.training_goals || []).map(g => ({
        id: g.id, claim_ids: (g.claim_ids || []).join(','), ability: String(g.ability || ''),
      }));
      attrPanel.replaceChildren(
        h('div', { class: 'panel-head' }, h('h2', {}, '错误类库与训练目标'),
          h('span', { class: 'spacer' }),
          h('button', { class: 'iconbtn', onclick: () => openJSON('runs', attrFile, '', 'attribution.json') }, '原始 JSON')),
        claims.length ? table([
          { label: 'id', key: 'id' },
          { label: '类型', key: 'kind' },
          { label: '证据强度', render: c => badge(c.strength === 'replay_verified' ? 'ok' : 'warn', c.strength || '—') },
          { label: '支撑样本', key: 'support' },
          { label: '陈述', class: 'wrap', render: c => h('span', { title: c.statement }, c.statement.slice(0, 220) + (c.statement.length > 220 ? '…' : '')) },
        ], claims, { onRow: c => openRecord(c, { root: 'runs', path: attrFile }, `/claims/${claims.indexOf(c)}`, `claim ${c.id}`) })
          : h('div', { class: 'empty' }, '没有 claim'),
        h('h3', { class: 'mt' }, '训练目标'),
        goals.length ? table([
          { label: 'id', key: 'id' }, { label: 'claims', key: 'claim_ids' },
          { label: '能力', class: 'wrap', render: g => h('span', { title: g.ability }, g.ability.slice(0, 260) + (g.ability.length > 260 ? '…' : '')) },
        ], goals, { onRow: g => openRecord(g, { root: 'runs', path: attrFile }, `/training_goals/${goals.indexOf(g)}`, `goal ${g.id}`) })
          : h('div', { class: 'empty' }, '没有 training goal'),
      );
    }).catch(err => attrPanel.replaceChildren(errorBox(err)));
  }

  app.replaceChildren(
    h('div', { class: 'row' },
      h('div', {},
        h('h1', {}, task.id),
        h('div', { class: 'sub' }, `${task.attempts.length} 个尝试 · 当前 ${task.current_run || '—'}`)),
      h('span', { class: 'spacer' }),
      cur ? h('a', { class: 'chip', href: `#/t/${enc(task.id)}/a/${enc(cur.run_id)}` }, '打开当前尝试') : null),
    h('div', { class: 'cards mt' },
      card('通过率', passRate(cur) !== undefined ? fmtPct(passRate(cur)) : '—', cur?.stages?.trainee?.status || ''),
      card('分流', cur?.stages?.route?.route || '—'),
      card('泄露', cur?.stages?.leak?.state || '—'),
      card('模板门', cur?.stages?.gate?.state || '—'),
      card('off_target', cur?.stages?.hits?.off_target === undefined ? '—' : (cur.stages.hits.off_target ? 'yes' : 'no'))),
    timeline,
    panel('全部尝试', attemptTable(task.attempts.map(a => ({ ...a, task: task.id })))),
    attrPanel,
  );
}

/* -------------------------------------------------------------- attempt */

function renderAttempt(task, run) {
  setCrumbs([
    { label: INDEX.batch.id, href: `#/b/${enc(INDEX.batch.id)}` },
    { label: task.id, href: `#/t/${enc(task.id)}` },
    { label: run.run_id, href: `#/t/${enc(task.id)}/a/${enc(run.run_id)}` },
  ]);

  const rail = h('div', { class: 'stage-rail' }, STAGES.map(([key, label]) => {
    const st = run.stages?.[key]?.state || 'missing';
    return h('div', { class: 'stage' },
      h('a', { href: `#/t/${enc(task.id)}/a/${enc(run.run_id)}/m/${enc(key === 'aggregate' ? 'attribution' : key)}` }, label),
      h('div', { class: 'n' }, stageBadge(st)));
  }));

  const grid = h('div', { class: 'modules' }, MODULES.map(m => {
    const st = stateOfModule(run, m);
    const metrics = moduleMetrics(run, m);
    return h('div', { class: 'module' },
      h('header', {}, h('h3', {}, m.label), h('span', { class: 'spacer' }), stageBadge(st)),
      h('div', { class: 'metrics' }, metrics.flatMap(([k, v]) => [
        h('span', {}, k),
        h('span', { class: 'mono' }, fmtVal(v)),
      ])),
      h('div', { class: 'actions' },
        h('a', { class: 'chip', href: `#/t/${enc(task.id)}/a/${enc(run.run_id)}/m/${enc(m.id)}` }, '展开'),
        m.id === 'raw' ? null : fileForModule(run, m)),
    );
  }));

  app.replaceChildren(
    h('div', { class: 'row' },
      h('div', {},
        h('h1', {}, run.run_id),
        h('div', { class: 'sub' }, `${task.id} · ${KIND_ZH[run.kind_label] || run.kind_label} #${run.attempt_no ?? '—'} · ${fmtBytes(run.size_bytes)} · ${run.n_files} files`)),
      h('span', { class: 'spacer' }),
      h('button', { class: 'iconbtn', onclick: () => copy(run.run_id) }, 'copy run id')),
    run.warnings?.length ? h('div', { class: 'error mt' }, run.warnings.join(' · ')) : null,
    h('div', { class: 'cards mt' },
      card('通过率', passRate(run) !== undefined ? fmtPct(passRate(run)) : '—',
        run.stages?.trainee ? `${run.stages.trainee.solved ?? '—'}/${run.stages.trainee.completed ?? '—'}` : ''),
      card('分流', run.stages?.route?.route || '—'),
      card('泄露', run.stages?.leak?.state || '—'),
      card('模板门', run.stages?.gate?.state || '—'),
      card('缺陷', run.stages?.hits?.class_hits ? Object.entries(run.stages.hits.class_hits).map(([k, v]) => `${k}:${v}`).join(' ') : '—'),
      card('off_target', run.stages?.hits?.off_target === undefined ? '—' : (run.stages.hits.off_target ? 'yes' : 'no'))),
    panel('阶段', rail),
    grid,
    panel('原始目录', h('button', { class: 'chip', onclick: () => go(`#/tree/runs/${enc(run.rel)}`) }, run.rel)),
  );
}

function fileForModule(run, m) {
  if (m.id === 'raw') return null;
  const rel = run.files?.[m.file];
  if (!rel) return null;
  return fileButton('runs', `${run.rel}/${rel}`, '源文件');
}

/* --------------------------------------------------------------- module */

function moduleDef(id) { return MODULES.find(m => m.id === id) || { id, label: id }; }

async function renderModule(task, run, moduleId) {
  const m = moduleDef(moduleId);
  setCrumbs([
    { label: INDEX.batch.id, href: `#/b/${enc(INDEX.batch.id)}` },
    { label: task.id, href: `#/t/${enc(task.id)}` },
    { label: run.run_id, href: `#/t/${enc(task.id)}/a/${enc(run.run_id)}` },
    { label: m.label, href: `#/t/${enc(task.id)}/a/${enc(run.run_id)}/m/${enc(m.id)}` },
  ]);

  const head = h('div', { class: 'row' },
    h('div', {}, h('h1', {}, m.label), h('div', { class: 'sub' }, `${task.id} · ${run.run_id}`)),
    h('span', { class: 'spacer' }),
    h('a', { class: 'chip', href: `#/t/${enc(task.id)}/a/${enc(run.run_id)}` }, '返回尝试'));

  if (m.kind === 'tree') {
    app.replaceChildren(head, await treePanel('runs', `${run.rel}/${run.episode_rel || ''}`.replace(/\/$/, '')));
    return;
  }
  const rel = run.files?.[m.file] ? `${run.rel}/${run.files[m.file]}` : null;
  if (!rel) {
    app.replaceChildren(head, h('div', { class: 'empty mt' }, '这个尝试没有产出对应的文件。'));
    return;
  }
  try {
    if (m.kind === 'text') {
      const text = await apiText('runs', rel);
      app.replaceChildren(head,
        sourceBar('runs', rel),
        panel('事件', textPanel(text)),
        m.id === 'events' ? synthesisPanel(run) : null);
      return;
    }
    const res = await apiJSON('runs', rel, m.pointer || '');
    const data = res.value;
    const body = renderModuleBody(m, data, { root: 'runs', path: rel }, run);
    app.replaceChildren(head, sourceBar('runs', rel, m.pointer), ...body);
  } catch (err) {
    app.replaceChildren(head, errorBox(err));
  }
}

function sourceBar(root, path, pointer = '') {
  return h('div', { class: 'row mt' },
    h('code', {}, `${root}:${path}${pointer}`),
    h('button', { class: 'iconbtn', onclick: () => copy(`${root}:${path}`) }, 'copy'),
    h('button', { class: 'iconbtn', onclick: () => openJSON(root, path, pointer) }, '原始 JSON'),
    h('a', { class: 'iconbtn', href: `/api/file?root=${root}&path=${enc(path)}`, target: '_blank' }, 'download'));
}

function textPanel(text) {
  const lines = text.split('\n').filter(Boolean);
  const evs = [];
  for (const line of lines) {
    try { evs.push(JSON.parse(line)); } catch { evs.push({ raw: line }); }
  }
  return h('div', { class: 'timeline' }, evs.map(e => h('div', { class: 'ev' },
    h('span', { class: 't' }, e.time ? new Date(e.time * 1000).toLocaleTimeString() : '—'),
    h('span', { class: 'mono' }, e.event || 'event'),
    e.stage ? h('span', { class: 'muted' }, ` · ${e.stage}`) : null,
    e.harbor_rc !== undefined ? h('span', { class: 'muted' }, ` · rc=${e.harbor_rc}`) : null,
    e.commit_id ? h('span', { class: 'muted' }, ` · ${String(e.commit_id).slice(0, 10)}`) : null,
  )));
}

function renderModuleBody(m, data, source, run) {
  const out = [];
  switch (m.id) {
    case 'observations': {
      const workers = data?.workers || [];
      out.push(panel('observation index', kv({
        status: data?.status, schema_version: data?.schema_version,
        reuse_observations_from: data?.reuse_observations_from,
        n_observation_ids: (data?.observation_ids || []).length,
        n_workers: workers.length,
      }, source)));
      out.push(panel('workers', objectTable(workers, source, '/workers', 'worker', [
        'sample_id', 'status', 'agent_terminal_status', 'artifact_status', 'runtime',
      ])));
      break;
    }
    case 'attribution': {
      out.push(panel('汇总', kv({
        derivation: data?.derivation,
        n_claims: (data?.claims || []).length,
        n_goals: (data?.training_goals || []).length,
      }, source)));
      out.push(panel('claims', objectTable(data?.claims || [], source, '/claims', 'claim',
        ['id', 'kind', 'evidence_strength', 'support_sample_count', 'statement'])));
      out.push(panel('training goals', objectTable(data?.training_goals || [], source, '/training_goals', 'goal',
        ['id', 'claim_ids', 'ability', 'success_signal'])));
      break;
    }
    case 'design': {
      out.push(panel('设计决定', kv(data, source)));
      out.push(panel('targets', objectTable(data?.targets || [], source, '/targets', 'target')));
      out.push(panel('preserves', simpleList(data?.preserves)));
      out.push(panel('simplifies', simpleList(data?.simplifies)));
      break;
    }
    case 'leak': {
      out.push(panel('结果', kv({
        schema_version: data?.schema_version, model: data?.model,
        status: data?.status, attempts: data?.attempts, error: data?.error,
        usage: data?.usage,
      }, source)));
      out.push(panel('verdicts', objectTable(data?.verdicts || [], source, '/verdicts', 'verdict')));
      break;
    }
    case 'reference': {
      out.push(panel('oracle', kv(data || {}, source, '/oracle')));
      const cand = run.files?.dir_candidate || run.files?.candidate;
      if (cand) out.push(panel('候选', h('button', { class: 'chip', onclick: () => go(`#/tree/runs/${enc(cand)}`) }, cand)));
      break;
    }
    case 'gate': {
      out.push(panel('结果', kv({
        schema_version: data?.schema_version, origin: data?.origin,
        status: data?.status, eligible_for_training: data?.eligible_for_training,
        llm_review: data?.llm_review,
      }, source)));
      out.push(panel('probes', objectTable(data?.probes || [], source, '/probes', 'probe',
        ['id', 'expected_valid', 'actual_accept', 'execution_status', 'reason'])));
      out.push(panel('issues', objectTable(data?.issues || [], source, '/issues', 'issue')));
      break;
    }
    case 'trainee': {
      const attempts = data?.attempts || [];
      out.push(panel('试做汇总', kv({
        status: data?.status, requested_attempts: data?.requested_attempts,
        solved_attempts: data?.solved_attempts, completed_attempts: data?.completed_attempts,
        difficulty_status: data?.difficulty_status, quality_approved: data?.quality_approved,
      }, source)));
      out.push(panel('16 次试做', objectTable(attempts, source, '/attempts', 'attempt',
        ['id', 'kind', 'status', 'reward', 'error'])));
      const target = run.episodes?.[0]?.metrics?.target_xlsx;
      if (target) out.push(panel('单元格对比', xlsxDiffTool(run, target)));
      break;
    }
    case 'hits': {
      out.push(panel('命中汇总', kv({
        schema_version: data?.schema_version, threshold: data?.threshold,
        attempts: data?.attempts, failed: data?.failed,
        off_target: data?.off_target, categories: data?.categories,
        class_hits: data?.class_hits,
      }, source)));
      out.push(panel('逐条', objectTable(data?.rows || [], source, '/rows', 'row',
        ['id', 'reward', 'wrong_cells', 'total_cells', 'category', 'classes', 'probes'])));
      out.push(panel('probe 对一致性', kv(data?.probe_pair_agreement || {}, source, '/probe_pair_agreement')));
      break;
    }
    case 'route': {
      out.push(panel('路由', kv(data || {}, source)));
      out.push(panel('难度带', bandViz(data)));
      break;
    }
    case 'synthesis': {
      out.push(panel('运行元数据', kv(data || {}, source)));
      break;
    }
    default:
      out.push(panel('原始', kv(data || {}, source)));
  }
  return out;
}

function synthesisPanel(run) {
  const rel = run.files?.synthesis ? `${run.rel}/${run.files.synthesis}` : null;
  if (!rel) return null;
  const box = panel('运行元数据', h('div', { class: 'loading' }, 'loading…'));
  apiJSON('runs', rel).then(res => {
    box.replaceChildren(
      h('div', { class: 'panel-head' }, h('h2', {}, '运行元数据')),
      kv(res.value, { root: 'runs', path: rel }));
  }).catch(err => box.replaceChildren(errorBox(err)));
  return box;
}

function simpleList(v) {
  if (!Array.isArray(v) || !v.length) return h('div', { class: 'empty' }, '空');
  return h('ul', {}, v.map(x => h('li', {}, typeof x === 'object' ? JSON.stringify(x) : String(x))));
}

function bandViz(data) {
  const band = data?.band;
  const rate = data?.pass_rate;
  if (!Array.isArray(band) || typeof rate !== 'number') return h('div', { class: 'empty' }, '没有 band / pass_rate');
  const pct = v => `${Math.max(0, Math.min(100, v * 100))}%`;
  return h('div', {},
    h('div', { style: 'position:relative;height:18px;background:#edf0f4;border-radius:4px;overflow:hidden' },
      h('div', { style: `position:absolute;left:${pct(band[0])};width:${(band[1] - band[0]) * 100}%;top:0;bottom:0;background:#dff0e6` }),
      h('div', { style: `position:absolute;left:${pct(rate)};top:-2px;bottom:-2px;width:3px;background:#2f5d8a` })),
    h('div', { class: 'sub' }, `band ${band[0]}–${band[1]} · pass_rate ${fmtPct(rate)}`));
}

function xlsxDiffTool(run, target) {
  const final = run.episodes?.[0]?.metrics?.attempts?.find(a => a.final_xlsx)?.final_xlsx;
  const base = run.rel + '/' + (run.episode_rel || '');
  if (!final) {
    return h('div', { class: 'empty' }, `target: ${target} · 尚未找到 final.xlsx（试做尚未产出）`);
  }
  return h('div', { class: 'row' },
    h('code', {}, `target ↔ ${final}`),
    h('button', { class: 'chip', onclick: () => openDiff(run, target, final) }, '对比单元格'));
}

function openDiff(run, target, final) {
  const left = `${run.rel}/${run.episode_rel}/${target}`;
  const right = `${run.rel}/${run.episode_rel}/${final}`;
  openDrawer('单元格对比');
  drawerBody.replaceChildren(h('div', { class: 'loading' }, 'comparing…'));
  apiDiff('runs', left, right).then(d => {
    drawerBody.replaceChildren(
      h('div', { class: 'row' },
        h('code', {}, `changed ${d.changed_cells}`),
        h('span', { class: 'muted' }, JSON.stringify(d.per_sheet))),
      table([
        { label: 'sheet', key: 'sheet' },
        { label: 'cell', key: 'cell' },
        { label: 'target', render: r => h('span', { class: 'mono' }, fmtVal(r.left)) },
        { label: 'final', render: r => h('span', { class: 'mono' }, fmtVal(r.right)) },
      ], d.changes),
      d.truncated ? h('div', { class: 'sub' }, '结果已截断') : null);
  }).catch(err => drawerBody.replaceChildren(errorBox(err)));
}


/* ----------------------------------------------------------- dictionary */

const FIELD_DICT = [
  ['validation.json', 'status', 'string', 'pending / reference_completed / completed …'],
  ['validation.json', 'oracle.reward', 'number', '参考解在 verifier 下的 reward'],
  ['validation.json', 'attempts[].status', 'enum', 'pending / completed / failed'],
  ['validation.json', 'attempts[].reward', 'number', '该次试做的 reward（0/1）'],
  ['validation.json', 'solved_attempts / completed_attempts', 'int', '通过率 = solved / completed'],
  ['route.json', 'pass_rate', 'float', 'solved / completed'],
  ['route.json', 'band', '[float, float]', '难度带 [0.125, 0.75]'],
  ['route.json', 'route', 'enum', 'rl / sft'],
  ['defect-hits.json', 'class_hits', 'map', 'error class → 命中次数'],
  ['defect-hits.json', 'off_target', 'bool', '失败试做是否落在作者 probe 之外'],
  ['defect-hits.json', 'probe_pair_agreement', 'map', 'probe 对的一致率'],
  ['template-gate.json', 'probes[].expected_valid / actual_accept', 'bool', '确定性 probe 的期望与实得'],
  ['template-gate.json', 'eligible_for_training', 'bool', '模板门是否放行'],
  ['leak-check.json', 'verdicts[].verdict', 'enum', 'pass / leak …'],
  ['attribution.json', 'claims[].evidence_strength', 'enum', 'replay_verified 等'],
  ['attribution.json', 'training_goals[]', 'array', 'ability / rationale / success_signal'],
  ['observations/index.json', 'workers[].terminal', 'object', '每个 sample 的运行终态'],
  ['design.json', 'primary_goal_id / targets / preserves', 'mixed', '本轮出题的目标与约束'],
];

function renderDict() {
  setCrumbs([{ label: INDEX.batch.id, href: `#/b/${enc(INDEX.batch.id)}` },
             { label: '字段字典' }]);
  app.replaceChildren(
    h('h1', {}, '字段字典'),
    h('div', { class: 'sub' }, '常见字段的语义与来源；未知字段仍会在原始 JSON 中完整显示，不会被隐藏。'),
    panel('字段', table([
      { label: '来源文件', key: 'file' },
      { label: '字段', render: r => h('code', {}, r.field) },
      { label: '类型', key: 'type' },
      { label: '含义', key: 'note' },
    ], FIELD_DICT.map(([file, field, type, note]) => ({ file, field, type, note })))),
    panel('阶段', table([
      { label: '阶段', render: r => r[1] },
      { label: 'key', render: r => h('code', {}, r[0]) },
    ], STAGES.map(([k, l]) => [k, l]))),
    panel('状态值', h('div', { class: 'row' },
      ['done', 'accepted', 'pending', 'blocked', 'rejected', 'failed', 'missing'].map(s => stageBadge(s)))),
  );
}

/* ----------------------------------------------------------------- tree */

async function renderTree(root, path) {
  setCrumbs([{ label: INDEX.batch.id, href: `#/b/${enc(INDEX.batch.id)}` }, { label: 'tree', href: `#/tree/${root}` }]
    .concat(pathSegments(root, path)));
  const panelNode = await treePanel(root, path);
  app.replaceChildren(h('h1', {}, '文件树'), panelNode);
}

function pathSegments(root, path) {
  const out = [];
  const parts = path.split('/').filter(Boolean);
  let acc = '';
  for (const p of parts) {
    acc = acc ? `${acc}/${p}` : p;
    out.push({ label: p, href: `#/tree/${root}/${enc(acc)}` });
  }
  return out;
}

async function treePanel(root, path) {
  const data = await apiTree(root, path, 1);
  const rows = data.entries.map(e => h('div', { class: 'node' },
    e.type === 'dir' ? '📁' : '📄',
    e.type === 'dir'
      ? h('a', { href: `#/tree/${root}/${enc(path ? path + '/' + e.name : e.name)}` }, e.name + '/')
      : fileButton(root, path ? `${path}/${e.name}` : e.name, e.name),
    h('span', { class: 'sz' }, e.type === 'dir' ? '' : fmtBytes(e.bytes))));
  return h('section', { class: 'panel' },
    h('div', { class: 'panel-head' }, h('h2', {}, root + ':' + (path || '/')),
      h('span', { class: 'spacer' }),
      h('button', { class: 'iconbtn', onclick: () => copy(`${root}:${path}`) }, 'copy path')),
    h('div', { class: 'tree' }, rows.length ? rows : h('div', { class: 'empty' }, '空目录')));
}

/* ----------------------------------------------------------------- xlsx */

async function renderXlsx(root, path) {
  const name = path.split('/').pop();
  setCrumbs([{ label: INDEX.batch.id, href: `#/b/${enc(INDEX.batch.id)}` },
             { label: 'xlsx', href: `#/x/${root}/${enc(path)}` },
             { label: name }]);
  const box = h('div', { class: 'panel' }, h('div', { class: 'loading' }, 'loading workbook…'));
  app.replaceChildren(
    h('div', { class: 'row' }, h('div', {}, h('h1', {}, name), h('div', { class: 'sub mono' }, `${root}:${path}`)),
      h('span', { class: 'spacer' }),
      h('a', { class: 'chip', href: `/api/file?root=${root}&path=${enc(path)}`, target: '_blank' }, '下载')),
    box);
  try {
    const data = await apiXlsx(root, path);
    box.replaceChildren(xlsxBody(root, path, data));
  } catch (err) {
    box.replaceChildren(errorBox(err));
  }
}

function xlsxBody(root, path, data, mode = 'values') {
  const sheetBox = h('div', {});
  const tabs = h('div', { class: 'xlsx-tabs' });
  const toolbar = h('div', { class: 'toolbar' });

  const draw = async (sheet = data.sheet, m = mode) => {
    sheetBox.replaceChildren(h('div', { class: 'loading' }, 'loading sheet…'));
    try {
      const d = await apiXlsx(root, path, sheet, m);
      tabs.replaceChildren(...d.sheets.map(s => h('button', {
        class: s === d.sheet ? 'active' : '', onclick: () => draw(s, m),
      }, s)));
      sheetBox.replaceChildren(xlsxGrid(d));
    } catch (err) {
      sheetBox.replaceChildren(errorBox(err));
    }
  };

  toolbar.append(
    h('button', { onclick: () => draw(data.sheet, 'values') }, '值'),
    h('button', { onclick: () => draw(data.sheet, 'formulas') }, '公式'),
    h('span', { class: 'spacer' }),
    h('span', { class: 'muted' }, `${data.max_row} × ${data.max_column}${data.truncated ? ' (截断)' : ''}`),
  );

  const diffRow = h('div', { class: 'toolbar' },
    h('input', { type: 'text', id: 'other-xlsx', placeholder: '对比文件路径（root 内相对路径）' }),
    h('button', { onclick: () => runDiffFromInput(root, path) }, '对比'));
  draw();
  return h('div', {}, toolbar, tabs, sheetBox, h('hr'), h('h3', {}, '单元格 diff'), diffRow,
    h('div', { id: 'diff-result' }));
}

function xlsxGrid(d) {
  const cols = Math.max(0, ...d.rows.map(r => r.length));
  const header = h('tr', {}, h('th', { class: 'rownum' }, ''), Array.from({ length: cols }, (_, i) => h('th', {}, colLetter(i + 1))));
  const body = d.rows.map((row, ri) => h('tr', {},
    h('td', { class: 'rownum' }, ri + 1),
    Array.from({ length: cols }, (_, ci) => h('td', { title: String(row[ci] ?? '') }, fmtVal(row[ci])))));
  return h('div', { class: 'grid-wrap' }, h('table', { class: 'grid' }, h('thead', {}, header), h('tbody', {}, body)));
}

function colLetter(n) {
  let s = '';
  while (n > 0) { const m = (n - 1) % 26; s = String.fromCharCode(65 + m) + s; n = Math.floor((n - 1) / 26); }
  return s;
}

function runDiffFromInput(root, left) {
  const input = document.getElementById('other-xlsx');
  const out = document.getElementById('diff-result');
  if (!input || !out) return;
  const other = input.value.trim();
  if (!other) return;
  out.replaceChildren(h('div', { class: 'loading' }, 'comparing…'));
  apiDiff(root, left, other).then(d => {
    out.replaceChildren(
      h('div', { class: 'row' }, h('code', {}, `changed ${d.changed_cells}`), h('span', { class: 'muted' }, JSON.stringify(d.per_sheet))),
      table([
        { label: 'sheet', key: 'sheet' }, { label: 'cell', key: 'cell' },
        { label: 'left', render: r => h('span', { class: 'mono' }, fmtVal(r.left)) },
        { label: 'right', render: r => h('span', { class: 'mono' }, fmtVal(r.right)) },
      ], d.changes));
  }).catch(err => out.replaceChildren(errorBox(err)));
}

boot();
