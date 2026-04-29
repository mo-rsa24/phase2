// ─────────────────────────────────────────────────────────────────────────
//  explorer-geometric.js
//  Geometric KL panel:
//    1D — prior vs posterior PDF for a chosen latent dim (live)
//    2D — scatter of cached encodings, prior contour + posterior ellipse
// ─────────────────────────────────────────────────────────────────────────

let _geom1dDim = 0;
let _geom2dDimA = 0;
let _geom2dDimB = 1;
let _geom2dColorBy = 'scale';

let _cachedEncodings = null;       // {mu, factors, factor_names}
let _cachedReady     = false;
let _cachedFetchInProgress = false;

// Hover / click panel state
const _scatterSampleCache = new Map();   // cache_idx → {original, reconstruction, factors, ...}
let _hoverDebounceTimer  = null;
let _pinnedCacheIdx      = null;
const _SHAPE_NAMES = ["square", "ellipse", "heart"];

// Manual HSV-style cyclic colorscale (used when colouring by orientation).
// Plotly's named "Phase"/"HSV" availability in the browser bundle is patchy,
// so we ship the stops directly.
const CYCLIC_COLORSCALE = [
  [0.000, 'hsl(  0,80%,55%)'],
  [0.125, 'hsl( 45,80%,55%)'],
  [0.250, 'hsl( 90,80%,55%)'],
  [0.375, 'hsl(135,80%,55%)'],
  [0.500, 'hsl(180,80%,55%)'],
  [0.625, 'hsl(225,80%,55%)'],
  [0.750, 'hsl(270,80%,55%)'],
  [0.875, 'hsl(315,80%,55%)'],
  [1.000, 'hsl(360,80%,55%)'],
];

function gaussianPDF(x, mu, sigma) {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}
function klUnit(mu, sigma) {
  return 0.5 * (sigma * sigma + mu * mu - 1 - 2 * Math.log(Math.max(sigma, 1e-8)));
}

// ── Build "z3 (KL=4.22)" labels for the 1D dim selector ──────────────────
function refreshGeom1DDimList() {
  const sel = document.getElementById('geom-1d-dim');
  if (!sel) return;
  const ld = window.latentDim || 0;
  if (ld === 0) { sel.innerHTML = '<option>—</option>'; return; }

  // Use globalKL if available, otherwise use anchor's own KL or zeros
  let kls = window.globalKL;
  if (!kls && window.anchorMu && window.anchorLogvar) {
    kls = window.anchorMu.map((_, i) => {
      const lv = Math.min(Math.max(window.anchorLogvar[i], -10), 10);
      return -0.5 * (1 + lv - window.anchorMu[i] ** 2 - Math.exp(lv));
    });
  }
  if (!kls) kls = new Array(ld).fill(0);

  // Sort dims by KL desc and select top KL by default if not yet set
  const order = kls.map((k, i) => ({k, i})).sort((a, b) => b.k - a.k);

  const prevSel = sel.value;
  const lbl = (i) => (window.dimLabel ? window.dimLabel(i) : `z${i}`);
  sel.innerHTML = order.map(({k, i}) =>
    `<option value="${i}">${lbl(i)} (KL=${k.toFixed(2)})</option>`).join('');

  // Default to top-KL dim if no previous selection
  if (prevSel === '' || prevSel === '—' || prevSel === null) {
    sel.value = String(order[0].i);
    _geom1dDim = order[0].i;
  } else if (Number(prevSel) < ld) {
    sel.value = prevSel;
    _geom1dDim = Number(prevSel);
  } else {
    sel.value = String(order[0].i);
    _geom1dDim = order[0].i;
  }
}

// ── Render 1D plot (live; updates when slider for the chosen dim moves) ──
function refreshGeom1D() {
  const plotEl = document.getElementById('geom-1d-plot');
  if (!plotEl) return;
  if (!window.currentMu) {
    Plotly.purge('geom-1d-plot');
    plotEl.innerHTML = '<div class="text-muted small p-3">Encode an image to render.</div>';
    return;
  }
  const i  = _geom1dDim;
  const mu = window.currentMu[i];
  const lv = Math.min(Math.max(window.currentLogvar[i], -10), 10);
  const sigma = Math.exp(0.5 * lv);

  const N = 200;
  const xs = [], priorY = [], postY = [];
  const xMin = -4, xMax = 4;
  for (let k = 0; k < N; k++) {
    const x = xMin + (xMax - xMin) * (k / (N - 1));
    xs.push(x);
    priorY.push(gaussianPDF(x, 0, 1));
    postY.push(gaussianPDF(x, mu, Math.max(sigma, 0.01)));
  }
  const klVal = klUnit(mu, sigma);

  const traces = [
    {x: xs, y: priorY, type: 'scatter', mode: 'lines', name: 'prior 𝒩(0,1)',
     line: {color: '#9ca3af', width: 1.5}, fill: 'tozeroy', fillcolor: 'rgba(156,163,175,0.15)'},
    {x: xs, y: postY,  type: 'scatter', mode: 'lines', name: `posterior 𝒩(μ, σ²)`,
     line: {color: '#1d4ed8', width: 2},   fill: 'tozeroy', fillcolor: 'rgba(29,78,216,0.18)'},
  ];
  const lbl1d = window.dimLabel ? window.dimLabel(i) : `z${i}`;
  const layout = {
    title:  {text: `${lbl1d}: μ=${mu.toFixed(2)}, σ=${sigma.toFixed(2)}, KL=${klVal.toFixed(2)} nats`, font:{size:12}},
    xaxis:  {title: `z${i}`, range: [xMin, xMax]},
    yaxis:  {title: 'density', rangemode: 'tozero'},
    margin: {t: 40, b: 40, l: 50, r: 16},
    height: 240,
    showlegend: true,
    legend: {orientation: 'h', y: -0.25, font:{size:10}},
    shapes: [{
      type: 'line', x0: mu, x1: mu, yref: 'paper', y0: 0, y1: 1,
      line: {color: '#1d4ed8', width: 1, dash: 'dot'},
    }],
  };
  Plotly.react(plotEl, traces, layout, {responsive: true, displayModeBar: false});
}

function onGeom1DDimChange() {
  const sel = document.getElementById('geom-1d-dim');
  _geom1dDim = parseInt(sel.value);
  refreshGeom1D();
}

// Hook from explorer-core.js: when a latent slider moves, treat its value as μ
function onLatentSliderForGeom(dimIdx, val) {
  if (!window.currentMu || dimIdx !== _geom1dDim) return;
  // Use the slider's value as the live μ for visualisation; keep σ from logvar.
  window.currentMu[dimIdx] = val;
  refreshGeom1D();
}
window.onLatentSliderForGeom = onLatentSliderForGeom;

// ─────────────────────────────────────────────────────────────────────────
// 2D scatter
// ─────────────────────────────────────────────────────────────────────────

function invalidateCachedEncodings() {
  _cachedEncodings = null;
  _cachedReady = false;
  _cachedFetchInProgress = false;
}
window.invalidateCachedEncodings = invalidateCachedEncodings;

async function ensureCachedEncodings() {
  if (_cachedReady) return _cachedEncodings;
  if (_cachedFetchInProgress) return null;
  _cachedFetchInProgress = true;
  try {
    while (true) {
      const r = await fetch('/api/cached_encodings');
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || r.statusText);
      if (d.ready) {
        _cachedEncodings = d;
        _cachedReady = true;
        _cachedFetchInProgress = false;
        return d;
      }
      await new Promise(res => setTimeout(res, 1500));
    }
  } catch (e) {
    _cachedFetchInProgress = false;
    throw e;
  }
}

function refreshGeom2DDimList() {
  const ld = window.latentDim || 0;
  const lbl = (i) => (window.dimLabel ? window.dimLabel(i) : `z${i}`);
  ['geom-2d-dim-a', 'geom-2d-dim-b'].forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    sel.innerHTML = Array.from({length: ld}, (_, i) =>
      `<option value="${i}">${lbl(i)}</option>`).join('');
  });
  // Defaults: top-2 KL dims (distinct)
  let order = [0, 1];
  if (window.globalKL) {
    order = window.globalKL.map((k, i) => ({k, i})).sort((a, b) => b.k - a.k).map(o => o.i);
  }
  if (ld >= 2) {
    document.getElementById('geom-2d-dim-a').value = String(order[0]);
    document.getElementById('geom-2d-dim-b').value = String(order[1]);
    _geom2dDimA = order[0];
    _geom2dDimB = order[1];
  }
  // Show the "Orient ring" preset only for supervised runs where the
  // (sin, cos) pair literally lives at known dims.
  const ringBtn = document.getElementById('btn-orient-ring');
  if (ringBtn) {
    const ok = window.supervised
      && Array.isArray(window.zOrientIdx)
      && window.zOrientIdx.length === 2
      && window.zOrientIdx[0] < ld
      && window.zOrientIdx[1] < ld;
    ringBtn.style.display = ok ? '' : 'none';
  }
}

async function renderGeom2D() {
  const plotEl = document.getElementById('geom-2d-plot');
  if (!plotEl) return;
  plotEl.innerHTML = '<div class="text-muted small p-3">Loading cached encodings …</div>';

  let cache;
  try { cache = await ensureCachedEncodings(); }
  catch (e) {
    plotEl.innerHTML = `<div class="text-danger small p-3">${e.message}</div>`;
    return;
  }
  if (!cache) return;

  const a = _geom2dDimA, b = _geom2dDimB;
  const factorIdx = cache.factor_names.indexOf(_geom2dColorBy);
  const xs = cache.mu.map(row => row[a]);
  const ys = cache.mu.map(row => row[b]);
  const colorVals = (factorIdx >= 0)
    ? cache.factors.map(row => row[factorIdx])
    : null;

  // Cyclic colormap when colouring by the cyclic 'orientation' factor —
  // linear viridis would map orientation=0° and orientation=351° to the
  // most distant colours, masking the cyclic structure.
  const useCyclic = (_geom2dColorBy === 'orientation');
  const colorscale = useCyclic ? CYCLIC_COLORSCALE : 'Viridis';

  const cacheIdx = Array.from({length: xs.length}, (_, i) => i);

  // Scatter
  const scatter = {
    type: 'scattergl', mode: 'markers',
    x: xs, y: ys,
    customdata: cacheIdx,
    marker: {
      size: 5, opacity: 0.7,
      color: colorVals,
      colorscale,
      showscale: !!colorVals,
      colorbar: colorVals ? {
        title: _geom2dColorBy + (useCyclic ? ' (cyclic)' : ''),
        len: 0.7,
      } : null,
    },
    name: 'encoded samples',
    hovertemplate:
      `cache#%{customdata} · z${a}=%{x:.2f} · z${b}=%{y:.2f}` +
      (colorVals ? `<br>${_geom2dColorBy}=%{marker.color}` : '') +
      '<extra></extra>',
  };

  // Prior 95% contour: circle of radius 2.45
  const N = 100, theta = Array.from({length: N + 1}, (_, k) => 2 * Math.PI * k / N);
  const priorRing = {
    type: 'scatter', mode: 'lines',
    x: theta.map(t => 2.45 * Math.cos(t)),
    y: theta.map(t => 2.45 * Math.sin(t)),
    line: {color: '#9ca3af', dash: 'dot', width: 1.5},
    name: 'prior 95% (≈2.45σ)',
    hoverinfo: 'skip',
  };

  // Posterior ellipse for the current encoding
  const traces = [scatter, priorRing];
  if (window.currentMu && window.currentLogvar) {
    const muA = window.currentMu[a], muB = window.currentMu[b];
    const sigA = Math.exp(0.5 * Math.min(Math.max(window.currentLogvar[a], -10), 10));
    const sigB = Math.exp(0.5 * Math.min(Math.max(window.currentLogvar[b], -10), 10));
    const k = 2.0;  // 2σ
    traces.push({
      type: 'scatter', mode: 'lines',
      x: theta.map(t => muA + k * sigA * Math.cos(t)),
      y: theta.map(t => muB + k * sigB * Math.sin(t)),
      line: {color: '#1d4ed8', dash: 'dash', width: 1.6},
      name: 'current posterior 2σ',
      hoverinfo: 'skip',
    });
    traces.push({
      type: 'scatter', mode: 'markers',
      x: [muA], y: [muB],
      marker: {color: '#1d4ed8', size: 8, symbol: 'x'},
      name: 'current μ',
      hoverinfo: 'skip',
    });
  }
  if (window.anchorMu) {
    traces.push({
      type: 'scatter', mode: 'markers',
      x: [window.anchorMu[a]], y: [window.anchorMu[b]],
      marker: {color: '#dc2626', size: 9, symbol: 'star'},
      name: 'anchor μ',
      hoverinfo: 'skip',
    });
  }

  // Unit-circle reference: for supervised runs, z[Z_ORIENT_IDX] regresses to
  // (sin(kθ), cos(kθ)), so plotting (z_orient_sin vs z_orient_cos) should land
  // on the unit circle. Drawing it makes the target geometry obvious.
  const orient = Array.isArray(window.zOrientIdx) ? window.zOrientIdx : null;
  const onOrientPair = !!orient && (
    (a === orient[0] && b === orient[1]) || (a === orient[1] && b === orient[0])
  );
  if (window.supervised && onOrientPair) {
    traces.push({
      type: 'scatter', mode: 'lines',
      x: theta.map(t => Math.cos(t)),
      y: theta.map(t => Math.sin(t)),
      line: {color: '#16a34a', dash: 'dash', width: 1.5},
      name: 'orient target (unit circle)',
      hoverinfo: 'skip',
    });
  }

  const lbl = (i) => (window.dimLabel ? window.dimLabel(i) : `z${i}`);
  const layout = {
    title:  {text: `${lbl(a)} vs ${lbl(b)} (colored by ${_geom2dColorBy})`, font:{size:12}},
    xaxis:  {title: lbl(a), scaleanchor: 'y', scaleratio: 1, zeroline: true, zerolinecolor:'#cbd5e1'},
    yaxis:  {title: lbl(b), zeroline: true, zerolinecolor:'#cbd5e1'},
    margin: {t: 40, b: 40, l: 50, r: 16},
    height: 380,
    legend: {orientation: 'h', y: -0.18, font:{size:10}},
  };
  Plotly.newPlot(plotEl, traces, layout, {responsive: true, displayModeBar: false});

  // Hook hover + click for the original/recon side-panel.
  // Plotly attaches its event API to the <div> after newPlot completes.
  plotEl.removeAllListeners?.('plotly_hover');
  plotEl.removeAllListeners?.('plotly_click');
  plotEl.on('plotly_hover',   onScatterHover);
  plotEl.on('plotly_click',   onScatterClick);
}

function refreshGeom2DPosterior() {
  // Lightweight refresh — only updates the posterior ellipse markers if the
  // 2D plot is already drawn. If never drawn yet, do nothing (lazy).
  if (_cachedReady) renderGeom2D();
}
window.refreshGeom2DPosterior = refreshGeom2DPosterior;

function onGeom2DDimChange() {
  _geom2dDimA = parseInt(document.getElementById('geom-2d-dim-a').value);
  _geom2dDimB = parseInt(document.getElementById('geom-2d-dim-b').value);
  clearScatterSidePanel();
  renderGeom2D();
}
function onGeom2DColorChange() {
  _geom2dColorBy = document.getElementById('geom-2d-color').value;
  // Colour change keeps the same dots; side-panel reading stays valid.
  renderGeom2D();
}

// "Orient ring" preset — only meaningful for supervised runs. Sets dim x =
// z_orient_cos, dim y = z_orient_sin, colour = orientation, then renders.
// The unit-circle reference is added by renderGeom2D when it detects the
// orient pair is on screen.
function applyOrientRingPreset() {
  const orient = Array.isArray(window.zOrientIdx) ? window.zOrientIdx : null;
  if (!window.supervised || !orient || orient.length !== 2) return;
  const [iSin, iCos] = orient;
  const ld = window.latentDim || 0;
  if (iSin >= ld || iCos >= ld) return;

  const dimA = document.getElementById('geom-2d-dim-a');
  const dimB = document.getElementById('geom-2d-dim-b');
  const col  = document.getElementById('geom-2d-color');
  if (!dimA || !dimB) return;

  // x = cos, y = sin so the point at θ=0 sits at (1, 0) — matches the
  // mathematical unit-circle convention readers expect.
  dimA.value = String(iCos);
  dimB.value = String(iSin);
  _geom2dDimA = iCos;
  _geom2dDimB = iSin;
  if (col) {
    col.value = 'orientation';
    _geom2dColorBy = 'orientation';
  }
  clearScatterSidePanel();
  renderGeom2D();
}

// ─────────────────────────────────────────────────────────────────────────
// Hover / click side-panel (original + reconstruction for cached point)
// ─────────────────────────────────────────────────────────────────────────

function onScatterHover(evt) {
  if (_pinnedCacheIdx !== null) return;   // pinned → don't follow cursor
  const pt = evt.points?.[0]; if (!pt) return;
  const idx = (typeof pt.customdata === 'number') ? pt.customdata : null;
  if (idx === null) return;
  if (_hoverDebounceTimer) clearTimeout(_hoverDebounceTimer);
  _hoverDebounceTimer = setTimeout(() => {
    if (_pinnedCacheIdx === null) showCachedSample(idx, false);
  }, 100);
}

function onScatterClick(evt) {
  const pt = evt.points?.[0]; if (!pt) return;
  const idx = (typeof pt.customdata === 'number') ? pt.customdata : null;
  if (idx === null) return;
  // Toggle: clicking the already-pinned point unpins.
  if (_pinnedCacheIdx === idx) { unpinScatter(); return; }
  _pinnedCacheIdx = idx;
  showCachedSample(idx, true);
}

function unpinScatter() {
  _pinnedCacheIdx = null;
  document.getElementById('geom-2d-side').classList.remove('pinned');
  const btn = document.getElementById('geom-2d-unpin');
  if (btn) btn.style.display = 'none';
}

async function fetchCachedSample(cacheIdx) {
  if (_scatterSampleCache.has(cacheIdx)) return _scatterSampleCache.get(cacheIdx);
  try {
    const r = await fetch(`/api/cached_sample/${cacheIdx}`);
    const d = await r.json();
    if (!r.ok) return null;
    _scatterSampleCache.set(cacheIdx, d);
    return d;
  } catch (e) { return null; }
}

async function showCachedSample(cacheIdx, pinned) {
  const data = await fetchCachedSample(cacheIdx);
  if (!data) return;
  // Race guard: if another hover replaced the request mid-flight, skip.
  if (!pinned && _pinnedCacheIdx !== null) return;

  document.getElementById('geom-2d-side-empty').style.display = 'none';
  document.getElementById('geom-2d-side-content').style.display = '';
  document.getElementById('geom-2d-orig').src  = data.original;
  document.getElementById('geom-2d-recon').src = data.reconstruction;

  const f = data.factors;
  const RQ = ['scale', 'orientation'];
  const lines = [
    `<div class="small-mono text-muted">cache#${data.cache_idx} · img#${data.dataset_idx}</div>`,
    `<div><span class="factor-name">shape:</span> ${f.shape} (${_SHAPE_NAMES[f.shape]})</div>`,
    `<div class="${RQ.includes('scale') ? 'factor-rq' : ''}"><span class="factor-name">scale:</span> ${f.scale}</div>`,
    `<div class="${RQ.includes('orientation') ? 'factor-rq' : ''}"><span class="factor-name">orient:</span> ${f.orientation} (${Math.round(f.orientation*9)}°)</div>`,
    `<div><span class="factor-name">pos:</span> (${f.pos_x}, ${f.pos_y})</div>`,
  ];
  document.getElementById('geom-2d-factors').innerHTML = lines.join('');

  const side = document.getElementById('geom-2d-side');
  const btn  = document.getElementById('geom-2d-unpin');
  if (pinned) {
    side.classList.add('pinned');
    if (btn) btn.style.display = '';
  } else {
    side.classList.remove('pinned');
    if (btn) btn.style.display = 'none';
  }
}

// Invalidate side-panel when dim selection changes (the displayed point's
// position will no longer match the new projection — confusing if left up).
function clearScatterSidePanel() {
  unpinScatter();
  document.getElementById('geom-2d-side-empty').style.display = '';
  document.getElementById('geom-2d-side-content').style.display = 'none';
}

// ── Init ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // 1D
  const sel1 = document.getElementById('geom-1d-dim');
  if (sel1) {
    refreshGeom1DDimList();
    sel1.addEventListener('change', onGeom1DDimChange);
    refreshGeom1D();
  }

  // 2D
  const dimA = document.getElementById('geom-2d-dim-a');
  const dimB = document.getElementById('geom-2d-dim-b');
  const colorBy = document.getElementById('geom-2d-color');
  if (dimA && dimB) {
    refreshGeom2DDimList();
    dimA.addEventListener('change', onGeom2DDimChange);
    dimB.addEventListener('change', onGeom2DDimChange);
    if (colorBy) {
      colorBy.addEventListener('change', onGeom2DColorChange);
      _geom2dColorBy = colorBy.value;
    }
    document.getElementById('btn-load-2d').addEventListener('click', renderGeom2D);
    const ringBtn = document.getElementById('btn-orient-ring');
    if (ringBtn) ringBtn.addEventListener('click', applyOrientRingPreset);
  }

  // Unpin button on the side-panel
  const unpinBtn = document.getElementById('geom-2d-unpin');
  if (unpinBtn) unpinBtn.addEventListener('click', unpinScatter);

  // Clear side-panel when a new model is loaded (cache_idx semantics change).
  // Hook into the existing invalidator so we don't have to touch core.js.
  const origInv = window.invalidateCachedEncodings;
  window.invalidateCachedEncodings = function () {
    _scatterSampleCache.clear();
    clearScatterSidePanel();
    if (origInv) origInv();
  };

  // When the model loads, the dim list and 1D plot need to refresh too —
  // expose hooks so explorer-core.js can call them.
  window.refreshGeom1D = refreshGeom1D;
  window.refreshGeom1DDimList = refreshGeom1DDimList;
  window.refreshGeom2DDimList = refreshGeom2DDimList;
});
