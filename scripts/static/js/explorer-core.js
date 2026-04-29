// ─────────────────────────────────────────────────────────────────────────
//  explorer-core.js
//  State + load + encode/decode + slider building (Explore tab) + grid
// ─────────────────────────────────────────────────────────────────────────

// ── Global state ─────────────────────────────────────────────────────────
window.modelLoaded   = window.__BOOT__?.modelLoaded ?? false;
window.latentDim     = window.__BOOT__?.latentDim   ?? 0;
window.supervised    = window.__BOOT__?.supervised  ?? false;
window.zScaleIdx     = window.__BOOT__?.zScaleIdx   ?? 0;
window.zOrientIdx    = window.__BOOT__?.zOrientIdx  ?? [1, 2];
window.anchorMu      = null;
window.anchorLogvar  = null;
window.anchorRecon   = null;
window.currentMu     = null;
window.currentLogvar = null;
window.currentKLDims = null;     // per-dim KL of current encoding
window.globalKL      = null;     // (latent_dim,) from /api/kl_spectrum
window.migPollTimer  = null;

const SHAPE_NAMES = ["square", "ellipse", "heart"];

// Semantic label for a latent dim. For supervised runs, z[Z_SCALE_IDX] is
// regressed to scale and z[Z_ORIENT_IDX] to (sin(kθ), cos(kθ)) on the unit
// circle — surface those names everywhere we'd otherwise show a bare "z3".
window.dimLabel = function (i) {
  if (!window.supervised) return `z${i}`;
  if (i === window.zScaleIdx) return `z${i} (scale)`;
  const [iSin, iCos] = window.zOrientIdx || [];
  if (i === iSin) return `z${i} (orient sin)`;
  if (i === iCos) return `z${i} (orient cos)`;
  return `z${i}`;
};

// ── Tiny helpers ─────────────────────────────────────────────────────────
function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function setStatus(text, state) {
  const dot  = document.getElementById('status-dot');
  const span = document.getElementById('status-text');
  dot.className = 'dot ' + (state === 'loaded' ? 'dot-green'
                          : state === 'loading' ? 'dot-spin' : 'dot-red');
  span.textContent = text;
}

function setImg(boxId, src) {
  const box = document.getElementById(boxId);
  box.innerHTML = `<img src="${src}" alt="">`;
}

// ── Load model ───────────────────────────────────────────────────────────
async function loadPreset(checkpoint, expId) {
  document.querySelectorAll('.exp-card').forEach(c => c.classList.remove('loaded'));
  await doLoad(checkpoint);
  if (window.modelLoaded) {
    document.getElementById(`exp-card-${expId}`).classList.add('loaded');
    const btn = document.getElementById(`seed-dd-${expId}`);
    const m   = checkpoint.match(/_seed(\d+)\//);
    if (btn && m) btn.textContent = `seed=${m[1]}`;
  }
}

async function loadSeed(expId, seed, checkpoint) {
  await loadPreset(checkpoint, expId);
}

// Card-level click delegate: ignore clicks that originated inside the seed
// dropdown (Bootstrap handles those itself), otherwise load the default seed.
function onExpCardClick(event, checkpoint, expId) {
  if (event.target.closest('.dropdown')) return;
  loadPreset(checkpoint, expId);
}

async function loadCustom() {
  const ckpt = document.getElementById('custom-ckpt').value.trim();
  if (!ckpt) return;
  await doLoad(ckpt);
}

async function doLoad(checkpoint) {
  setStatus('Loading …', 'loading');
  document.getElementById('load-error').textContent = '';
  try {
    const r    = await fetch('/api/load', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({checkpoint}),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || r.statusText);

    window.modelLoaded = true;
    window.latentDim   = data.latent_dim;
    window.supervised  = !!data.supervised;
    if (typeof data.z_scale_idx === 'number') window.zScaleIdx = data.z_scale_idx;
    if (Array.isArray(data.z_orient_idx))     window.zOrientIdx = data.z_orient_idx;
    document.getElementById('load-info').innerHTML =
      `<div class="text-success">&#10003; Loaded: <span class="font-monospace">${data.checkpoint}</span></div>
       <div>latent_dim = ${window.latentDim}</div>`;
    setStatus(`Loaded — z=${window.latentDim}`, 'loaded');

    // Refresh objective strip with the matched β / γ values
    if (typeof updateObjectiveStrip === 'function') {
      updateObjectiveStrip(data.beta, data.checkpoint, data.gamma);
    }
    // Refresh provenance bar with the new run's metadata.
    if (typeof updateProvenanceBar === 'function') {
      const isFactor = (data.checkpoint || '').includes('factorvae') ||
                       (data.checkpoint || '').includes('factor_vae');
      updateProvenanceBar({
        ckpt:      data.checkpoint,
        split:     data.split,
        corrA:     data.corr_factor_a,
        corrB:     data.corr_factor_b,
        corrDir:   data.corr_direction,
        beta:      data.beta,
        gamma:     data.gamma,
        latentDim: data.latent_dim,
        model:     isFactor ? 'FactorVAE' : 'VAE',
      });
    }

    buildLatentSliders();

    // Repopulate the geometric dim selectors with the new latent_dim and pick
    // up semantic labels / orient-ring visibility for the freshly loaded run.
    if (typeof window.refreshGeom1DDimList === 'function') window.refreshGeom1DDimList();
    if (typeof window.refreshGeom2DDimList === 'function') window.refreshGeom2DDimList();

    // Reset anchor state
    window.anchorMu = window.anchorLogvar = window.anchorRecon =
      window.currentMu = window.currentLogvar = null;
    document.getElementById('btn-anchor').disabled = true;
    document.getElementById('traversal-panel').style.display = 'none';
    document.getElementById('btn-reset-all').disabled = true;

    // Reset cached encodings flag for the 2D viz
    if (typeof invalidateCachedEncodings === 'function') invalidateCachedEncodings();

    onFactorChange();
  } catch (e) {
    document.getElementById('load-error').textContent = e.message;
    setStatus('Load failed', 'unloaded');
  }
}

// ── Factor sliders ───────────────────────────────────────────────────────
function updateFactorLabels() {
  const shape  = parseInt(document.getElementById('f-shape').value);
  const orient = parseInt(document.getElementById('f-orientation').value);
  document.getElementById('v-shape').textContent       = `${shape} (${SHAPE_NAMES[shape]})`;
  document.getElementById('v-scale').textContent       = document.getElementById('f-scale').value;
  document.getElementById('v-orientation').textContent = `${Math.round(orient * 9)}° (${orient})`;
  document.getElementById('v-pos_x').textContent       = document.getElementById('f-pos_x').value;
  document.getElementById('v-pos_y').textContent       = document.getElementById('f-pos_y').value;
}

const doEncode = debounce(async () => {
  if (!window.modelLoaded) return;
  updateFactorLabels();
  const payload = {
    color:       0,
    shape:       parseInt(document.getElementById('f-shape').value),
    scale:       parseInt(document.getElementById('f-scale').value),
    orientation: parseInt(document.getElementById('f-orientation').value),
    pos_x:       parseInt(document.getElementById('f-pos_x').value),
    pos_y:       parseInt(document.getElementById('f-pos_y').value),
  };
  try {
    const r    = await fetch('/api/encode_factors', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await r.json();
    if (!r.ok) { console.warn(data.error); return; }

    setImg('orig-box',  data.original);
    setImg('recon-box', data.reconstruction);
    setImg('diff-box',  data.difference);
    document.getElementById('encode-meta').textContent =
      `MSE: ${data.mse.toFixed(5)}  |  sample #${data.sample_index || ''}`;

    window.currentMu     = data.mu;
    window.currentLogvar = data.logvar;
    window.currentKLDims = data.kl_per_dim;
    // Enable the anchor button as soon as we have a valid encoding — don't
    // gate it on the μ-bar render, which can throw on first paint (Bootstrap
    // Tooltip init on freshly-created rows).
    document.getElementById('btn-anchor').disabled = false;
    try { renderMuBars(data.mu, data.logvar, data.kl_per_dim); }
    catch (e) { console.error('renderMuBars failed:', e); }

    if (typeof refreshGeom1D === 'function') refreshGeom1D();
    if (typeof refreshGeom2DPosterior === 'function') refreshGeom2DPosterior();
  } catch (e) { console.error(e); }
}, 120);

function onFactorChange() { doEncode(); }

// ── Posterior μ panel ────────────────────────────────────────────────────
function klFromMuLogvar(mu, lv) {
  const lvc = Math.min(Math.max(lv, -10), 10);
  return -0.5 * (1 + lvc - mu * mu - Math.exp(lvc));
}

function gaussianPdf(x, mu, sigma) {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

// 60×18px canvas: prior N(0,1) (grey) + posterior N(mu, sigma²) (blue)
function drawMiniGaussianPair(canvas, mu, sigma) {
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  const xMin = -3.5, xMax = 3.5;
  const N = 60;
  const xs = [], priorY = [], postY = [];
  let yMax = 0;
  for (let k = 0; k < N; k++) {
    const x = xMin + (xMax - xMin) * (k / (N - 1));
    xs.push(x);
    const p = gaussianPdf(x, 0, 1);
    const q = gaussianPdf(x, mu, Math.max(sigma, 0.05));
    priorY.push(p);
    postY.push(q);
    if (p > yMax) yMax = p;
    if (q > yMax) yMax = q;
  }
  const xToPix = x => ((x - xMin) / (xMax - xMin)) * W;
  const yToPix = y => H - (y / yMax) * (H - 2);

  // x-axis baseline
  ctx.strokeStyle = '#e5e7eb';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, H - 0.5);
  ctx.lineTo(W, H - 0.5);
  ctx.stroke();

  // prior — grey
  ctx.strokeStyle = '#9ca3af';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let k = 0; k < N; k++) {
    const px = xToPix(xs[k]);
    const py = yToPix(priorY[k]);
    if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // posterior — blue
  ctx.strokeStyle = '#2563eb';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  for (let k = 0; k < N; k++) {
    const px = xToPix(xs[k]);
    const py = yToPix(postY[k]);
    if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // μ marker
  ctx.strokeStyle = '#1d4ed8';
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 2]);
  ctx.beginPath();
  ctx.moveTo(xToPix(mu), 1);
  ctx.lineTo(xToPix(mu), H);
  ctx.stroke();
  ctx.setLineDash([]);
}

// μ-color via global KL spectrum (or per-encoding KL if global not yet ready)
function colorForKL(kl) {
  // 0 → light grey, ≥ 4 → deep blue
  const k = Math.max(0, Math.min(kl, 4)) / 4;
  const r = Math.round(225 - 196 * k);  // 225 → 29
  const g = Math.round(231 - 153 * k);  // 231 → 78
  const b = Math.round(255 - 39  * k);  // 255 → 216
  return `rgb(${r},${g},${b})`;
}

function renderMuBars(mu, logvar, klDims) {
  const container = document.getElementById('mu-bars');
  if (!mu || mu.length === 0) { container.innerHTML = '—'; return; }

  const maxAbs = Math.max(1, ...mu.map(Math.abs));
  // Sort by KL descending so high-KL dims top-most.
  const klRef = window.globalKL || klDims || mu.map(() => 0);
  const order = klRef.map((k, i) => ({k, i})).sort((a, b) => b.k - a.k);

  const showN = Math.min(15, mu.length);
  container.innerHTML = '';
  order.slice(0, showN).forEach(({k, i}) => {
    const v       = mu[i];
    const lv      = logvar ? logvar[i] : 0;
    const sigma   = Math.exp(0.5 * Math.min(Math.max(lv, -10), 10));
    const klLocal = klDims ? klDims[i] : klFromMuLogvar(v, lv);
    const klShow  = window.globalKL ? window.globalKL[i] : klLocal;
    const sign    = v >= 0 ? 'mu-sign-pos' : 'mu-sign-neg';
    const pct     = Math.abs(v / maxAbs * 50);
    const fillL   = v >= 0 ? 50 : (50 - pct);
    const barColor = colorForKL(klShow);

    const anchorMu = window.anchorMu ? window.anchorMu[i] : null;
    const dMu = (anchorMu !== null) ? (v - anchorMu) : null;
    const dMuStr = dMu !== null ? `Δ ${dMu >= 0 ? '+' : ''}${dMu.toFixed(2)}` : '';
    const anchorPct = (anchorMu !== null) ? ((anchorMu / maxAbs) * 50 + 50) : null;

    const row = document.createElement('div');
    row.className = 'mu-row';
    const muLbl = window.dimLabel ? window.dimLabel(i) : `z${i}`;
    row.innerHTML = `
      <span class="mu-label">${muLbl}</span>
      <canvas class="mu-spark" width="60" height="18"></canvas>
      <div class="mu-bar-wrap">
        <div class="mu-bar-fill" style="left:${fillL}%; width:${pct}%; background:${barColor};"></div>
        ${anchorPct !== null ? `<div class="mu-anchor-tick" style="left:calc(${anchorPct}% - 1px);"></div>` : ''}
      </div>
      <span class="mu-value ${sign}">${v >= 0 ? '+' : ''}${v.toFixed(2)}</span>
      <span class="mu-kl">KL=${klShow.toFixed(2)}</span>
      <span class="mu-delta">${dMuStr}</span>`;
    container.appendChild(row);

    drawMiniGaussianPair(row.querySelector('canvas'), v, sigma);

    // Tooltip (Bootstrap)
    const klBits = (klShow * 1.443).toFixed(2);
    const distSigma = sigma > 0 ? Math.abs(v / sigma).toFixed(1) : '∞';
    row.setAttribute('data-bs-toggle', 'tooltip');
    row.setAttribute('data-bs-placement', 'left');
    row.setAttribute('data-bs-html', 'true');
    row.setAttribute('title',
      `<b>z${i}</b>: μ=${v.toFixed(3)}, σ=${sigma.toFixed(3)}<br>` +
      `KL=${klShow.toFixed(2)} nats (≈ ${klBits} bits)<br>` +
      `posterior is ${distSigma}σ from prior centre`);
    new bootstrap.Tooltip(row);
  });
}

// ── Anchor ───────────────────────────────────────────────────────────────
function setAsAnchor() {
  if (!window.currentMu) return;
  window.anchorMu     = [...window.currentMu];
  window.anchorLogvar = [...window.currentLogvar];
  const reconImg = document.getElementById('recon-box').querySelector('img');
  window.anchorRecon = reconImg ? reconImg.src : null;

  document.getElementById('traversal-panel').style.display = '';
  if (window.anchorRecon) setImg('anchor-box', window.anchorRecon);

  populateLatentSlidersFromAnchor();
  decodeLatent();
  document.getElementById('anchor-hint').textContent =
    '✓ Anchor set. Drag latent sliders to traverse.';
  document.getElementById('btn-reset-all').disabled = false;

  // Re-render μ bars to include anchor tick
  if (window.currentMu) renderMuBars(window.currentMu, window.currentLogvar, window.currentKLDims);

  if (typeof refreshGeom2DPosterior === 'function') refreshGeom2DPosterior();
}

// ── Latent sliders ───────────────────────────────────────────────────────
function buildLatentSliders() {
  const wrap = document.getElementById('latent-sliders');
  wrap.innerHTML = '';
  for (let i = 0; i < window.latentDim; i++) {
    wrap.appendChild(buildLatentRow(i, 0, null, 0));
  }
}

function buildLatentRow(dimIdx, value, anchorVal, kl) {
  const row = document.createElement('div');
  row.className = 'slider-row';
  const klBadge = kl ? `<span class="kl-badge">KL=${kl.toFixed(2)}</span>` : '';
  const label = window.dimLabel ? window.dimLabel(dimIdx) : `z${dimIdx}`;
  row.innerHTML = `
    <span class="slider-lbl" id="lbl-z${dimIdx}">${label} ${klBadge}</span>
    <button class="btn-reset-z" data-dim="${dimIdx}" title="Reset z${dimIdx} to anchor μ"
            ${anchorVal === null ? 'disabled' : ''}>↺</button>
    <input type="range" id="z-slider-${dimIdx}" min="-4" max="4" step="0.05" value="${value.toFixed(3)}"
           oninput="onLatentChange(${dimIdx}, this.value)">
    <span class="slider-val" id="zval-${dimIdx}">${value.toFixed(2)}</span>
    <span class="delta-sigma" id="dsig-${dimIdx}">${anchorVal === null ? '—' : 'Δ 0.0σ'}</span>`;
  // Wire up the per-row reset
  row.querySelector('.btn-reset-z').addEventListener('click', () => resetSingleDim(dimIdx));
  return row;
}

function populateLatentSlidersFromAnchor() {
  if (!window.anchorMu) return;
  const kls = window.globalKL || window.anchorMu.map((_, i) =>
    klFromMuLogvar(window.anchorMu[i], window.anchorLogvar[i]));
  const order = kls.map((k, i) => ({k, i})).sort((a, b) => b.k - a.k);
  const wrap = document.getElementById('latent-sliders');
  wrap.innerHTML = '';
  order.forEach(({k, i}) => {
    wrap.appendChild(buildLatentRow(i, window.anchorMu[i], window.anchorMu[i], k));
  });
}

function onLatentChange(dimIdx, val) {
  const v = parseFloat(val);
  document.getElementById(`zval-${dimIdx}`).textContent = v.toFixed(2);

  // Δσ from anchor
  if (window.anchorMu && window.anchorLogvar) {
    const sigma = Math.exp(0.5 * Math.min(Math.max(window.anchorLogvar[dimIdx], -10), 10));
    const dSig  = sigma > 0 ? (v - window.anchorMu[dimIdx]) / sigma : 0;
    const el    = document.getElementById(`dsig-${dimIdx}`);
    if (el) {
      el.textContent = `Δ ${dSig >= 0 ? '+' : ''}${dSig.toFixed(1)}σ`;
      const a = Math.abs(dSig);
      el.className = 'delta-sigma ' + (a < 0.5 ? 'dsig-low' : a < 2 ? 'dsig-mid' : 'dsig-high');
    }
  }

  // Live update geometric 1D viz if it's currently showing this dim
  if (typeof onLatentSliderForGeom === 'function') onLatentSliderForGeom(dimIdx, v);

  doDecode();
}

function resetSingleDim(dimIdx) {
  if (!window.anchorMu) return;
  const sl = document.getElementById(`z-slider-${dimIdx}`);
  if (!sl) return;
  sl.value = window.anchorMu[dimIdx].toFixed(3);
  onLatentChange(dimIdx, sl.value);
}

function resetAllDims() {
  if (!window.anchorMu) return;
  for (let i = 0; i < window.latentDim; i++) {
    const sl = document.getElementById(`z-slider-${i}`);
    if (sl) {
      sl.value = window.anchorMu[i].toFixed(3);
      onLatentChange(i, sl.value);
    }
  }
}

const doDecode = debounce(async () => {
  if (!window.anchorMu) return;
  const z = window.anchorMu.map((_, i) => {
    const sl = document.getElementById(`z-slider-${i}`);
    return sl ? parseFloat(sl.value) : window.anchorMu[i];
  });
  try {
    const r    = await fetch('/api/decode', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({z}),
    });
    const data = await r.json();
    if (!r.ok) return;
    setImg('trav-box', data.reconstruction);
  } catch (e) { console.error(e); }
}, 80);

function decodeLatent() { doDecode(); }
function resetToAnchor() { resetAllDims(); }

// ── Traversal grid ───────────────────────────────────────────────────────
async function generateGrid() {
  if (!window.anchorMu) {
    document.getElementById('grid-error').textContent = 'Set an anchor in the Explore tab first.';
    return;
  }
  document.getElementById('grid-error').textContent = '';
  document.getElementById('grid-spinner').style.display = '';
  document.getElementById('grid-img').style.display = 'none';
  document.getElementById('grid-placeholder').style.display = 'none';

  const payload = {
    mu:               window.anchorMu,
    logvar:           window.anchorLogvar,
    n_steps:          parseInt(document.getElementById('g-steps').value),
    sigma_range:      parseFloat(document.getElementById('g-range').value),
    max_dims:         parseInt(document.getElementById('g-maxdims').value),
    use_anchor_sigma: document.getElementById('g-anchor-sigma').checked,
  };
  try {
    const r    = await fetch('/api/traversal_grid', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error);
    const img = document.getElementById('grid-img');
    img.src = data.grid;
    img.style.display = '';
  } catch (e) {
    document.getElementById('grid-error').textContent = e.message;
  } finally {
    document.getElementById('grid-spinner').style.display = 'none';
  }
}

// ── Init ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  updateFactorLabels();

  // Wire factor sliders
  ['f-shape','f-scale','f-orientation','f-pos_x','f-pos_y']
    .forEach(id => document.getElementById(id).addEventListener('input', onFactorChange));

  // Wire static buttons
  document.getElementById('btn-anchor').addEventListener('click', setAsAnchor);
  document.getElementById('btn-reset-trav').addEventListener('click', resetToAnchor);
  document.getElementById('btn-reset-all').addEventListener('click', resetAllDims);
  document.getElementById('btn-load-custom').addEventListener('click', loadCustom);
  document.getElementById('btn-grid').addEventListener('click', generateGrid);

  // Grid value labels
  document.getElementById('g-steps').addEventListener('input', e =>
    document.getElementById('v-steps').textContent = e.target.value);
  document.getElementById('g-range').addEventListener('input', e =>
    document.getElementById('v-range').textContent = e.target.value + 'σ');
  document.getElementById('g-maxdims').addEventListener('input', e =>
    document.getElementById('v-maxdims').textContent = e.target.value);

  if (window.modelLoaded) {
    setStatus(`Loaded — z=${window.latentDim}`, 'loaded');
    buildLatentSliders();
    onFactorChange();
  } else {
    setStatus('No model loaded', 'unloaded');
  }
});
