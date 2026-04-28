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
  sel.innerHTML = order.map(({k, i}) =>
    `<option value="${i}">z${i} (KL=${k.toFixed(2)})</option>`).join('');

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
  const layout = {
    title:  {text: `z${i}: μ=${mu.toFixed(2)}, σ=${sigma.toFixed(2)}, KL=${klVal.toFixed(2)} nats`, font:{size:12}},
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
  ['geom-2d-dim-a', 'geom-2d-dim-b'].forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    sel.innerHTML = Array.from({length: ld}, (_, i) =>
      `<option value="${i}">z${i}</option>`).join('');
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

  // Scatter
  const scatter = {
    type: 'scattergl', mode: 'markers',
    x: xs, y: ys,
    marker: {
      size: 4, opacity: 0.6,
      color: colorVals,
      colorscale: 'Viridis',
      showscale: !!colorVals,
      colorbar: colorVals ? {title: _geom2dColorBy, len: 0.7} : null,
    },
    name: 'encoded samples',
    hoverinfo: 'skip',
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

  const layout = {
    title:  {text: `z${a} vs z${b} (colored by ${_geom2dColorBy})`, font:{size:12}},
    xaxis:  {title: `z${a}`, scaleanchor: 'y', scaleratio: 1, zeroline: true, zerolinecolor:'#cbd5e1'},
    yaxis:  {title: `z${b}`, zeroline: true, zerolinecolor:'#cbd5e1'},
    margin: {t: 40, b: 40, l: 50, r: 16},
    height: 380,
    legend: {orientation: 'h', y: -0.18, font:{size:10}},
  };
  Plotly.newPlot(plotEl, traces, layout, {responsive: true, displayModeBar: false});
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
  renderGeom2D();
}
function onGeom2DColorChange() {
  _geom2dColorBy = document.getElementById('geom-2d-color').value;
  renderGeom2D();
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
  }

  // When the model loads, the dim list and 1D plot need to refresh too —
  // expose hooks so explorer-core.js can call them.
  window.refreshGeom1D = refreshGeom1D;
  window.refreshGeom1DDimList = refreshGeom1DDimList;
  window.refreshGeom2DDimList = refreshGeom2DDimList;
});
