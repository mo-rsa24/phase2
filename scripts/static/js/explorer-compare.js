// ─────────────────────────────────────────────────────────────────────────
//  explorer-compare.js
//  Compare tab: pin up to 4 checkpoints, render summary metrics in a
//  table + per-checkpoint side-by-side KL spectrum / correlation heatmap.
// ─────────────────────────────────────────────────────────────────────────

const COMPARE_MAX = 4;
const _comparePins = [];          // array of metric payloads (≤ 4)
let _experiments   = null;        // EXPERIMENTS list from /api/experiments

const _MIG_GREEN_C = 0.4;
const _MIG_AMBER_C = 0.2;

function _scoreClass(v) {
  if (v >= _MIG_GREEN_C) return 'green';
  if (v >= _MIG_AMBER_C) return 'amber';
  return 'red';
}

function _splitClass(s) {
  if (s === 'iid')        return 'row-split-iid';
  if (s === 'correlated') return 'row-split-correlated';
  if (s === 'heldout')    return 'row-split-heldout';
  return '';
}

function _detectModelFromCkpt(ckpt) {
  return (ckpt || '').includes('factorvae') ? 'FactorVAE' : 'VAE';
}

function _shortLabel(m) {
  // From a metrics payload, build a compact label like "Exp 6 (corr·VAE z=10 β=1)".
  // We look up the experiment ID via _experiments by matching checkpoint.
  const ckpt = m.checkpoint;
  const exp  = (_experiments || []).find(e => ckpt.includes(e.checkpoint));
  if (exp) return `Exp ${exp.id}`;
  return ckpt.split('/').slice(-2).join('/');
}

// ── Picker population ────────────────────────────────────────────────────
async function loadExperiments() {
  if (_experiments) return _experiments;
  try {
    const r = await fetch('/api/experiments');
    const d = await r.json();
    _experiments = d.experiments;
    return _experiments;
  } catch (e) { console.error(e); return []; }
}

async function refreshComparePicker() {
  const exps = await loadExperiments();
  const sel  = document.getElementById('compare-picker');
  if (!sel) return;
  sel.innerHTML = '<option value="">Choose a sweep checkpoint to pin …</option>';

  // Group options by split with optgroup labels.
  const splits = {};
  exps.forEach(e => {
    const s = e.split || 'iid';
    if (!splits[s]) splits[s] = [];
    splits[s].push(e);
  });
  for (const s of Object.keys(splits)) {
    const og = document.createElement('optgroup');
    og.label = (s === 'iid' ? 'IID' :
                s === 'correlated' ? 'Correlated' :
                s === 'heldout' ? 'Held-out' : s);
    splits[s].forEach(e => {
      const opt = document.createElement('option');
      opt.value = e.checkpoint;
      opt.textContent = `${e.label}`;
      og.appendChild(opt);
    });
    sel.appendChild(og);
  }
}

// ── Pin / unpin ──────────────────────────────────────────────────────────
async function pinSelectedCheckpoint() {
  const sel  = document.getElementById('compare-picker');
  const ckpt = sel.value;
  if (!ckpt) return;
  if (_comparePins.find(p => p.checkpoint === ckpt)) {
    _setPinStatus('already pinned');
    return;
  }
  if (_comparePins.length >= COMPARE_MAX) {
    _setPinStatus(`max ${COMPARE_MAX} pins — remove one first`);
    return;
  }
  _setPinStatus(`computing metrics for ${ckpt} … (up to ~60 s)`);
  try {
    const r = await fetch('/api/compare/metrics', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({checkpoint: ckpt}),
    });
    const m = await r.json();
    if (!r.ok) throw new Error(m.error || r.statusText);
    _comparePins.push(m);
    _setPinStatus('');
    renderCompareTab();
  } catch (e) {
    _setPinStatus('error: ' + e.message);
  }
}

function unpinCheckpoint(ckpt) {
  const i = _comparePins.findIndex(p => p.checkpoint === ckpt);
  if (i >= 0) {
    _comparePins.splice(i, 1);
    renderCompareTab();
  }
}

function _setPinStatus(msg) {
  const el = document.getElementById('compare-pin-status');
  if (el) el.textContent = msg || '';
}

// ── Render: pinned cards + table + per-pin deep-dive ─────────────────────
function renderCompareTab() {
  renderPinnedCards();
  renderMetricsTable();
  renderDeepDive();
}

function renderPinnedCards() {
  const wrap = document.getElementById('compare-pinned');
  if (!wrap) return;
  if (_comparePins.length === 0) {
    wrap.innerHTML = '<span class="text-muted small">No checkpoints pinned yet.</span>';
    return;
  }
  wrap.innerHTML = _comparePins.map(m => {
    const split  = m.split || 'unknown';
    const model  = _detectModelFromCkpt(m.checkpoint);
    const hp     = (m.beta != null) ? `β=${m.beta}` :
                    (m.gamma != null) ? `γ=${m.gamma}` : '';
    return `<div class="compare-pin-card split-${split}">
      <div class="d-flex justify-content-between align-items-center">
        <span class="pin-label">${_shortLabel(m)}</span>
        <button class="pin-close" data-ckpt="${m.checkpoint}" title="unpin">✕</button>
      </div>
      <span class="pin-meta">${split} · ${model} · z=${m.latent_dim}${hp ? ' · ' + hp : ''}</span>
    </div>`;
  }).join('');
  wrap.querySelectorAll('.pin-close').forEach(btn => {
    btn.addEventListener('click', () => unpinCheckpoint(btn.getAttribute('data-ckpt')));
  });
}

function renderMetricsTable() {
  const wrap = document.getElementById('compare-table');
  if (!wrap) return;
  if (_comparePins.length === 0) {
    wrap.innerHTML = '<div class="text-muted small">Pin a checkpoint to populate.</div>';
    return;
  }
  // Header = factor names from the first pin's payload.
  const factors = _comparePins[0].factor_names || ['shape','scale','orientation','pos_x','pos_y'];
  const headerCells = [
    '<th>Run</th>',
    '<th>split</th>',
    '<th>model</th>',
    '<th class="numeric">active dims</th>',
    '<th class="numeric">MIG</th>',
    '<th class="numeric">DCI-D</th>',
    '<th class="numeric">DCI-C</th>',
    '<th class="numeric">DCI-I</th>',
    ...factors.map(f => `<th>${f}<br><small>top-z · |ρ|</small></th>`),
    '<th>flag</th>',
  ].join('');

  const rows = _comparePins.map(m => {
    const split = m.split || 'unknown';
    const model = _detectModelFromCkpt(m.checkpoint);
    const dci = m.dci || {};

    // Per-factor: top dim and |ρ|
    const factorCells = m.corr_top_dim_per_factor.map((dim, k) => {
      const rho = m.corr_max_per_factor[k];
      return `<td>z${dim} · ${rho.toFixed(2)}</td>`;
    }).join('');

    // Flag: scale and orientation share a top dim?
    const scale_idx  = factors.indexOf('scale');
    const orient_idx = factors.indexOf('orientation');
    let flag = '';
    if (scale_idx >= 0 && orient_idx >= 0 &&
        m.corr_top_dim_per_factor[scale_idx] === m.corr_top_dim_per_factor[orient_idx]) {
      flag = `<span class="warn-flag" title="scale & orientation share top latent dim — possible entanglement">⚠ entangled</span>`;
    }

    const migClass = !isNaN(m.mig) ? _scoreClass(m.mig) : '';
    const dClass   = (dci.D != null) ? _scoreClass(dci.D) : '';
    const cClass   = (dci.C != null) ? _scoreClass(dci.C) : '';
    const iClass   = (dci.I != null) ? _scoreClass(dci.I) : '';

    return `<tr class="${_splitClass(split)}">
      <td>${_shortLabel(m)}</td>
      <td>${split}</td>
      <td>${model}</td>
      <td class="numeric">${m.active_dims}/${m.latent_dim}</td>
      <td class="numeric ${migClass}">${!isNaN(m.mig) ? m.mig.toFixed(3) : '—'}</td>
      <td class="numeric ${dClass}">${dci.D != null ? dci.D.toFixed(3) : '—'}</td>
      <td class="numeric ${cClass}">${dci.C != null ? dci.C.toFixed(3) : '—'}</td>
      <td class="numeric ${iClass}">${dci.I != null ? dci.I.toFixed(3) : '—'}</td>
      ${factorCells}
      <td>${flag}</td>
    </tr>`;
  }).join('');

  wrap.innerHTML = `<table class="compare-table">
    <thead><tr>${headerCells}</tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function renderDeepDive() {
  const wrap = document.getElementById('compare-deepdive');
  if (!wrap) return;
  if (_comparePins.length === 0) {
    wrap.innerHTML = '<div class="col-12 text-muted small">Pin checkpoints to render side-by-side spectrums.</div>';
    return;
  }

  // Decide column width based on pin count.
  const cols = _comparePins.length === 1 ? 'col-12' :
               _comparePins.length === 2 ? 'col-md-6' :
               'col-md-6 col-xl-3';

  wrap.innerHTML = _comparePins.map((m, idx) => `
    <div class="${cols}">
      <div class="compare-deepdive-card">
        <div class="header">${_shortLabel(m)} <span class="text-muted small">— ${m.split}</span></div>
        <div id="compare-kl-${idx}" style="min-height:240px"></div>
      </div>
    </div>`).join('');

  // Render KL spectrum bar charts per pin.
  _comparePins.forEach((m, idx) => {
    const data = [{
      type: 'bar',
      x: m.kl_per_dim.map((_, i) => `z${i}`),
      y: m.kl_per_dim,
      marker: {color: m.kl_per_dim.map(v => v > 0.1 ? '#2563eb' : '#d1d5db')},
      text: m.kl_per_dim.map(v => v.toFixed(2)),
      textposition: 'outside',
    }];
    const layout = {
      title: {text: 'mean KL per dim', font:{size:11}},
      xaxis: {title: ''},
      yaxis: {title: 'nats', rangemode: 'tozero'},
      margin: {t: 30, b: 30, l: 40, r: 10},
      height: 240,
      shapes: [{type:'line', x0:-0.5, x1:m.kl_per_dim.length-0.5,
                y0:0.1, y1:0.1,
                line:{color:'#ef4444', dash:'dot', width:1}}],
    };
    Plotly.newPlot(`compare-kl-${idx}`, data, layout,
                   {responsive:true, displayModeBar:false});
  });
}

// ── Init ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  refreshComparePicker();
  const btn = document.getElementById('btn-compare-pin');
  if (btn) btn.addEventListener('click', pinSelectedCheckpoint);
});
