// ─────────────────────────────────────────────────────────────────────────
//  explorer-analysis.js
//  Analysis tab: KL spectrum, correlation heatmap, MIG, and Phase-3
//  qualitative interactions (heatmap click, KL hover, MIG bar click).
// ─────────────────────────────────────────────────────────────────────────

// Top-level threshold constants (research-domain defaults from plan)
const MIG_GREEN = 0.4;
const MIG_AMBER = 0.2;
// DCI uses the same convention.
const DCI_GREEN = 0.4;
const DCI_AMBER = 0.2;
let _dciPollTimer = null;

// Single-dim traversal cache (keyed by `${dim}|${nSteps}|${range}`)
const _singleDimCache = new Map();

// ── KL spectrum ──────────────────────────────────────────────────────────
async function computeKL() {
  if (!window.modelLoaded) return;
  document.getElementById('kl-error').textContent = '';
  document.getElementById('kl-spinner').style.display = '';
  document.getElementById('btn-kl').disabled = true;
  try {
    const r    = await fetch('/api/kl_spectrum', {method: 'POST'});
    const data = await r.json();
    if (!r.ok) throw new Error(data.error);
    Plotly.newPlot('kl-plot', data.data, data.layout, {responsive: true, displayModeBar: false});
    window.globalKL = data.data[0].y;

    // Hook hover for thumbnail strip
    const plotEl = document.getElementById('kl-plot');
    plotEl.removeAllListeners?.('plotly_hover');
    plotEl.removeAllListeners?.('plotly_unhover');
    plotEl.on('plotly_hover',   evt => onKLBarHover(evt));
    plotEl.on('plotly_unhover', () => onKLBarUnhover());

    // If μ panel is currently rendered, re-render so colors pick up new globalKL
    if (window.currentMu) {
      renderMuBars(window.currentMu, window.currentLogvar, window.currentKLDims);
    }
    if (typeof refreshGeom1DDimList === 'function') refreshGeom1DDimList();
  } catch (e) {
    document.getElementById('kl-error').textContent = e.message;
  } finally {
    document.getElementById('kl-spinner').style.display = 'none';
    document.getElementById('btn-kl').disabled = false;
  }
}

// ── Correlation heatmap ──────────────────────────────────────────────────
async function computeCorr() {
  if (!window.modelLoaded) return;
  document.getElementById('corr-error').textContent = '';
  document.getElementById('corr-spinner').style.display = '';
  document.getElementById('btn-corr').disabled = true;
  try {
    const r    = await fetch('/api/correlation', {method: 'POST'});
    const data = await r.json();
    if (!r.ok) throw new Error(data.error);
    Plotly.newPlot('corr-plot', data.data, data.layout, {responsive: true, displayModeBar: false});

    // Hook click → mini traversal modal
    const plotEl = document.getElementById('corr-plot');
    plotEl.removeAllListeners?.('plotly_click');
    plotEl.on('plotly_click', evt => onHeatmapClick(evt));
  } catch (e) {
    document.getElementById('corr-error').textContent = e.message;
  } finally {
    document.getElementById('corr-spinner').style.display = 'none';
    document.getElementById('btn-corr').disabled = false;
  }
}

// ── MIG ──────────────────────────────────────────────────────────────────
async function startMIG() {
  if (!window.modelLoaded) return;
  document.getElementById('mig-error').textContent = '';
  document.getElementById('mig-result').style.display = 'none';
  document.getElementById('mig-spinner').style.display = '';
  document.getElementById('btn-mig').disabled = true;
  await fetch('/api/mig/start', {method: 'POST'});
  window.migPollTimer = setInterval(pollMIG, 2000);
}

async function pollMIG() {
  try {
    const r    = await fetch('/api/mig/status');
    const data = await r.json();
    if (data.status === 'done') {
      clearInterval(window.migPollTimer);
      document.getElementById('mig-spinner').style.display = 'none';
      document.getElementById('btn-mig').disabled = false;
      renderMIG(data.result);
    } else if (data.status === 'error') {
      clearInterval(window.migPollTimer);
      document.getElementById('mig-spinner').style.display = 'none';
      document.getElementById('btn-mig').disabled = false;
      document.getElementById('mig-error').textContent = data.error || 'Unknown error';
    }
  } catch (e) { /* retry */ }
}

function migBadge(val) {
  if (val >= MIG_GREEN) return '<span class="mig-badge mig-badge-green">strong</span>';
  if (val >= MIG_AMBER) return '<span class="mig-badge mig-badge-amber">moderate</span>';
  return '<span class="mig-badge mig-badge-red">poor</span>';
}

function renderMIG(result) {
  document.getElementById('mig-overall').textContent = result.score.toFixed(3);
  const overallBadge = document.getElementById('mig-overall-badge');
  if (overallBadge) overallBadge.innerHTML = migBadge(result.score);

  const container = document.getElementById('mig-per-factor');
  // Order: scale and orientation pinned to front (RQ-relevant), rest by score desc
  const RQ = ['scale', 'orientation'];
  const all = Object.entries(result.per_factor);
  const pinned = RQ.map(k => all.find(([n]) => n === k)).filter(Boolean);
  const rest   = all.filter(([n]) => !RQ.includes(n)).sort((a, b) => b[1] - a[1]);
  const ordered = [...pinned, ...rest];

  container.innerHTML = ordered.map(([name, val]) => {
    const pct  = Math.round(val * 100);
    const isRQ = RQ.includes(name);
    return `<div class="col-md-4 col-6">
      <div class="mig-row ${isRQ ? 'mig-row-hl' : ''}" data-factor="${name}">
        <div class="d-flex align-items-center justify-content-between">
          <span class="small text-muted">${name}${isRQ ? ' <small>(RQ)</small>' : ''}</span>
          ${migBadge(val)}
        </div>
        <div class="mig-bar"><div class="mig-fill" style="width:${Math.min(pct,100)}%"></div></div>
        <div class="small fw-semibold">${val.toFixed(3)}</div>
      </div>
    </div>`;
  }).join('');

  // Click handlers for conditional histogram
  container.querySelectorAll('.mig-row').forEach(el => {
    el.addEventListener('click', () => openCondHistModal(el.getAttribute('data-factor')));
    el.style.cursor = 'pointer';
  });

  document.getElementById('mig-result').style.display = '';
}

// ── DCI ──────────────────────────────────────────────────────────────────
async function startDCI() {
  if (!window.modelLoaded) return;
  document.getElementById('dci-error').textContent = '';
  document.getElementById('dci-result').style.display = 'none';
  document.getElementById('dci-spinner').style.display = '';
  document.getElementById('btn-dci').disabled = true;
  await fetch('/api/dci/start', {method: 'POST'});
  _dciPollTimer = setInterval(pollDCI, 2000);
}

async function pollDCI() {
  try {
    const r    = await fetch('/api/dci/status');
    const data = await r.json();
    if (data.status === 'done') {
      clearInterval(_dciPollTimer);
      document.getElementById('dci-spinner').style.display = 'none';
      document.getElementById('btn-dci').disabled = false;
      renderDCI(data.result);
    } else if (data.status === 'error') {
      clearInterval(_dciPollTimer);
      document.getElementById('dci-spinner').style.display = 'none';
      document.getElementById('btn-dci').disabled = false;
      document.getElementById('dci-error').textContent = data.error || 'Unknown error';
    }
  } catch (e) { /* retry next tick */ }
}

function dciBadge(v) {
  if (v >= DCI_GREEN) return '<span class="mig-badge mig-badge-green">strong</span>';
  if (v >= DCI_AMBER) return '<span class="mig-badge mig-badge-amber">moderate</span>';
  return '<span class="mig-badge mig-badge-red">poor</span>';
}

function renderDCI(result) {
  document.getElementById('dci-d-overall').textContent = result.D.toFixed(3);
  document.getElementById('dci-c-overall').textContent = result.C.toFixed(3);
  document.getElementById('dci-i-overall').textContent = result.I.toFixed(3);
  document.getElementById('dci-d-badge').innerHTML = dciBadge(result.D);
  document.getElementById('dci-c-badge').innerHTML = dciBadge(result.C);
  document.getElementById('dci-i-badge').innerHTML = dciBadge(result.I);

  // ----- Heatmap (latent × factor), rows sorted by KL desc -----
  const ldim = result.D_per_latent.length;
  const klRef = window.globalKL || new Array(ldim).fill(0);
  const order = klRef.map((k, i) => ({k, i})).sort((a, b) => b.k - a.k).map(o => o.i);
  const sortedImp = order.map(i => result.importance[i]);
  const ylabs = order.map(i => `z${i} (KL=${klRef[i].toFixed(2)})`);
  const data = [{
    type: 'heatmap',
    z: sortedImp,
    x: result.factor_names,
    y: ylabs,
    colorscale: 'Blues',
    zmin: 0,
    zmax: Math.max(0.01, Math.max(...sortedImp.flat())),
    text: sortedImp.map(row => row.map(v => v.toFixed(2))),
    texttemplate: '%{text}',
    showscale: true,
  }];
  const layout = {
    title: {text: 'Importance R(latent × factor)', font:{size:13}},
    xaxis: {title: 'Ground-truth factor', side: 'bottom'},
    yaxis: {title: 'Latent dim (↑ KL)', autorange: 'reversed'},
    margin: {t: 50, b: 60, l: 130, r: 20},
    height: Math.max(320, ldim * 28 + 120),
  };
  Plotly.newPlot('dci-heatmap', data, layout, {responsive: true, displayModeBar: false});

  // Click → reuse the existing single-dim traversal modal.
  const heatmapEl = document.getElementById('dci-heatmap');
  heatmapEl.removeAllListeners?.('plotly_click');
  heatmapEl.on('plotly_click', evt => onHeatmapClick(evt));

  // ----- Per-latent D bars (in KL order) -----
  const dContainer = document.getElementById('dci-d');
  dContainer.innerHTML = order.map(i => {
    const v   = result.D_per_latent[i];
    const pct = Math.round(v * 100);
    return `<div class="col-md-6">
      <div class="mig-row">
        <div class="d-flex align-items-center justify-content-between">
          <span class="small text-muted">z${i}</span>
          ${dciBadge(v)}
        </div>
        <div class="mig-bar"><div class="mig-fill" style="width:${Math.min(pct,100)}%"></div></div>
        <div class="small fw-semibold">${v.toFixed(3)}</div>
      </div>
    </div>`;
  }).join('');

  // ----- Per-factor C bars (RQ rows pinned to front, highlighted) -----
  const RQ = ['scale', 'orientation'];
  const reorderFactors = (vals) => {
    const all = result.factor_names.map((n, k) => [n, vals[k]]);
    const pinned = RQ.map(k => all.find(([n]) => n === k)).filter(Boolean);
    const rest   = all.filter(([n]) => !RQ.includes(n));
    return [...pinned, ...rest];
  };

  const cContainer = document.getElementById('dci-c');
  cContainer.innerHTML = reorderFactors(result.C_per_factor).map(([name, v]) => {
    const pct = Math.round(v * 100);
    const isRQ = RQ.includes(name);
    return `<div class="col-12">
      <div class="mig-row ${isRQ ? 'mig-row-hl' : ''}">
        <div class="d-flex align-items-center justify-content-between">
          <span class="small text-muted">${name}${isRQ ? ' <small>(RQ)</small>' : ''}</span>
          ${dciBadge(v)}
        </div>
        <div class="mig-bar"><div class="mig-fill" style="width:${Math.min(pct,100)}%"></div></div>
        <div class="small fw-semibold">${v.toFixed(3)}</div>
      </div>
    </div>`;
  }).join('');

  // ----- Per-factor I bars (R²) -----
  const iContainer = document.getElementById('dci-i');
  iContainer.innerHTML = reorderFactors(result.I_per_factor).map(([name, v]) => {
    const pct = Math.round(v * 100);
    const isRQ = RQ.includes(name);
    return `<div class="col-12">
      <div class="mig-row ${isRQ ? 'mig-row-hl' : ''}">
        <div class="d-flex align-items-center justify-content-between">
          <span class="small text-muted">${name}${isRQ ? ' <small>(RQ)</small>' : ''}</span>
          ${dciBadge(v)}
        </div>
        <div class="mig-bar"><div class="mig-fill" style="width:${Math.min(pct,100)}%"></div></div>
        <div class="small fw-semibold">${v.toFixed(3)}</div>
      </div>
    </div>`;
  }).join('');

  document.getElementById('dci-result').style.display = '';
}


// ─────────────────────────────────────────────────────────────────────────
// PHASE 3 — qualitative interactions
// ─────────────────────────────────────────────────────────────────────────

// ── Heatmap click → mini-traversal modal ─────────────────────────────────
async function onHeatmapClick(evt) {
  const pt = evt.points?.[0]; if (!pt) return;
  // y label is "z3 (KL=2.40)" — extract the digit
  const m = (typeof pt.y === 'string') ? pt.y.match(/^z(\d+)/) : null;
  const dim = m ? parseInt(m[1]) : null;
  if (dim === null) return;

  const factor = pt.x;
  const rho    = pt.z;

  if (!window.anchorMu) {
    alert("Set an anchor in the Explore tab first — the modal needs anchor μ/logvar to traverse.");
    return;
  }

  // Modal title
  document.getElementById('single-dim-title').innerHTML =
    `<b>z${dim}</b> &middot; factor: <b>${factor}</b> &middot; |ρ| = ${rho.toFixed(2)}`;
  document.getElementById('single-dim-strip').innerHTML =
    '<span class="text-muted small p-3">Decoding 9 frames …</span>';
  const modalEl = document.getElementById('single-dim-modal');
  bootstrap.Modal.getOrCreateInstance(modalEl).show();

  const data = await fetchSingleDimTraversal(dim, 9, 3.0);
  if (data?.error) {
    document.getElementById('single-dim-strip').innerHTML =
      `<div class="text-danger small p-3">${data.error}</div>`;
    return;
  }
  document.getElementById('single-dim-strip').innerHTML = data.images.map((src, k) => `
    <div class="frame">
      <img src="${src}" alt="">
      <span class="frame-label">${data.z_values[k].toFixed(2)}</span>
    </div>`).join('');
}

async function fetchSingleDimTraversal(dim, nSteps, rangeSigma) {
  const cacheKey = `${dim}|${nSteps}|${rangeSigma}`;
  if (_singleDimCache.has(cacheKey)) return _singleDimCache.get(cacheKey);

  const payload = {
    dim, n_steps: nSteps, range_sigma: rangeSigma,
    anchor_mu:     window.anchorMu,
    anchor_logvar: window.anchorLogvar,
  };
  try {
    const r = await fetch('/api/single_dim_traversal', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const d = await r.json();
    if (!r.ok) return {error: d.error || r.statusText};
    _singleDimCache.set(cacheKey, d);
    return d;
  } catch (e) {
    return {error: e.message};
  }
}

// ── KL bar hover → 3-thumbnail popover ───────────────────────────────────
let _klHoverPopover = null;
let _klHoverBarKey  = null;

async function onKLBarHover(evt) {
  if (!window.anchorMu) return;
  const pt = evt.points?.[0]; if (!pt) return;
  const m = (typeof pt.x === 'string') ? pt.x.match(/^z(\d+)/) : null;
  const dim = m ? parseInt(m[1]) : null;
  if (dim === null) return;

  const key = `kl-${dim}`;
  if (_klHoverBarKey === key && _klHoverPopover) return;  // already showing
  onKLBarUnhover();
  _klHoverBarKey = key;

  const data = await fetchSingleDimTraversal(dim, 3, 2.0);
  if (data?.error) return;

  // Anchor popover to the bar's bounding rect — find the trace bar element
  const target = document.elementFromPoint(evt.event.clientX, evt.event.clientY) || evt.event.target;
  if (!target) return;

  const html = `<div class="kl-thumb-popover">${data.images.map(src => `<img src="${src}">`).join('')}</div>`;
  _klHoverPopover = new bootstrap.Popover(target, {
    container: 'body',
    placement: 'top',
    html: true,
    trigger: 'manual',
    title: `z${dim} traversal ±2σ`,
    content: html,
  });
  _klHoverPopover.show();
}

function onKLBarUnhover() {
  if (_klHoverPopover) {
    _klHoverPopover.dispose();
    _klHoverPopover = null;
    _klHoverBarKey = null;
  }
}

// ── MIG factor bar click → conditional histogram modal ───────────────────
async function openCondHistModal(factor) {
  const titleEl = document.getElementById('cond-hist-title');
  const plotEl  = document.getElementById('cond-hist-plot');
  const noteEl  = document.getElementById('cond-hist-note');
  titleEl.textContent = `μ histograms grouped by ${factor}`;
  noteEl.innerHTML = '<span class="text-muted small">computing …</span>';
  plotEl.innerHTML = '';

  bootstrap.Modal.getOrCreateInstance(document.getElementById('cond-hist-modal')).show();

  try {
    const r = await fetch('/api/factor_conditional_histogram', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({factor, n_bins: 30}),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error);

    const traces = data.histograms.map(h => ({
      type: 'histogram',
      x: h.mu,
      name: `${factor}=${h.value}`,
      opacity: 0.55,
      nbinsx: 30,
    }));
    const layout = {
      barmode: 'overlay',
      title:   {text: `Conditional μ distribution — z${data.dim_used}`, font: {size: 13}},
      xaxis:   {title: `μ_z${data.dim_used}`},
      yaxis:   {title: 'count'},
      margin:  {t: 50, b: 50, l: 50, r: 20},
      height:  380,
      legend:  {orientation: 'h', y: -0.2},
    };
    Plotly.newPlot(plotEl, traces, layout, {responsive: true, displayModeBar: false});

    const methodNote = data.selection_method === 'mi'
      ? `<b>Top dim selected via mutual information</b> (orientation is cyclic — Spearman ρ underestimates).`
      : `Top dim selected via |Spearman ρ|.`;
    noteEl.innerHTML = `
      <div class="small text-muted">
        ${methodNote}<br>
        Cleanly separated humps = factor cleanly captured by z${data.dim_used}.
        Overlapping humps = entangled or weakly encoded.
      </div>`;
  } catch (e) {
    noteEl.innerHTML = `<div class="text-danger small">${e.message}</div>`;
  }
}

// ── Wire static buttons ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btn-kl').addEventListener('click',   computeKL);
  document.getElementById('btn-corr').addEventListener('click', computeCorr);
  document.getElementById('btn-mig').addEventListener('click',  startMIG);
  const dciBtn = document.getElementById('btn-dci');
  if (dciBtn) dciBtn.addEventListener('click', startDCI);
});
