// ─────────────────────────────────────────────────────────────────────────
//  explorer-help.js
//  HELP popovers, KaTeX rendering, global Help-Mode toggle, cross-link pulse,
//  Model & Objective collapsible strip.
// ─────────────────────────────────────────────────────────────────────────

const _katexRenderedFor = new Set();   // popover dedupe

// Build popover HTML body for a given help slug
function buildPopoverContent(slug) {
  const entry = window.HELP?.[slug];
  if (!entry) return `<i>No help for "${slug}"</i>`;
  const formulaSlot = entry.formula
    ? `<div class="formula-slot" data-formula-slug="${slug}"></div>`
    : '';
  return `${entry.long || ''}${formulaSlot}`;
}

// Render KaTeX into any .formula-slot present in the body
function renderKatexIn(rootEl) {
  if (!rootEl || typeof katex === 'undefined') return;
  rootEl.querySelectorAll('.formula-slot').forEach(slot => {
    const slug = slot.getAttribute('data-formula-slug');
    if (!slug) return;
    const formula = window.HELP?.[slug]?.formula;
    if (!formula) return;
    try {
      katex.render(formula, slot, {throwOnError: false, displayMode: true});
    } catch (e) { console.warn('KaTeX failed for', slug, e); }
  });
}

// Pulse linked UI elements when a popover is shown
function pulseLinks(slug) {
  const links = window.HELP?.[slug]?.links || [];
  links.forEach(sel => {
    document.querySelectorAll(sel).forEach(el => {
      el.classList.remove('help-pulse');
      void el.offsetWidth;  // force reflow
      el.classList.add('help-pulse');
    });
  });
}

// ── Init: wire all .help-icon elements to popovers ───────────────────────
function initHelpIcons() {
  document.querySelectorAll('.help-icon[data-help-slug]').forEach(el => {
    const slug  = el.getAttribute('data-help-slug');
    const entry = window.HELP?.[slug];
    if (!entry) return;
    const pop = new bootstrap.Popover(el, {
      container: 'body',
      placement: 'auto',
      html: true,
      trigger: 'click',
      title: entry.title || slug,
      content: () => buildPopoverContent(slug),
    });
    el.addEventListener('shown.bs.popover', () => {
      const tip = bootstrap.Popover.getInstance(el)?.tip;
      if (tip) renderKatexIn(tip);
      pulseLinks(slug);
    });
  });

  // Click-outside to close (Bootstrap normally requires another click on the icon)
  document.addEventListener('click', (e) => {
    document.querySelectorAll('.help-icon').forEach(icon => {
      const pop = bootstrap.Popover.getInstance(icon);
      if (!pop) return;
      const tip = pop.tip;
      if (e.target === icon) return;
      if (tip && tip.contains(e.target)) return;
      pop.hide();
    });
  });
}

// ── Help-Mode toggle ─────────────────────────────────────────────────────
let _helpModeOn = false;
const _termTooltips = [];

function enableHelpMode() {
  if (_helpModeOn) return;
  _helpModeOn = true;
  document.querySelectorAll('span.term[data-help-slug]').forEach(el => {
    el.classList.add('term-active');
    const slug  = el.getAttribute('data-help-slug');
    const entry = window.HELP?.[slug];
    if (!entry) return;
    el.setAttribute('data-bs-toggle', 'tooltip');
    el.setAttribute('data-bs-placement', 'top');
    el.setAttribute('title', entry.short || entry.title || slug);
    _termTooltips.push(new bootstrap.Tooltip(el));
  });
}

function disableHelpMode() {
  if (!_helpModeOn) return;
  _helpModeOn = false;
  while (_termTooltips.length) {
    try { _termTooltips.pop().dispose(); } catch (e) {}
  }
  document.querySelectorAll('span.term').forEach(el => {
    el.classList.remove('term-active');
    el.removeAttribute('data-bs-toggle');
    el.removeAttribute('data-bs-placement');
    el.removeAttribute('title');
  });
}

// ── Objective strip — render KaTeX + populate β ──────────────────────────
function renderObjectiveStrip() {
  const formulaEl = document.getElementById('objective-formula');
  if (formulaEl && typeof katex !== 'undefined') {
    katex.render(
      window.HELP?.elbo?.formula ||
      'L_{\\beta\\text{-VAE}} = E_{q_\\phi(z|x)}[\\log p_\\theta(x|z)] - \\beta \\cdot D_{KL}(q_\\phi(z|x) \\| p(z))',
      formulaEl, {throwOnError: false, displayMode: true});
  }
  // Inline annotations also render KaTeX
  document.querySelectorAll('#objective-strip .katex-inline').forEach(el => {
    if (typeof katex !== 'undefined') {
      try {
        katex.render(el.getAttribute('data-formula') || el.textContent, el,
          {throwOnError: false, displayMode: false});
      } catch (e) {}
    }
  });
}

// ── Provenance bar ───────────────────────────────────────────────────────
function _splitDescription(split, a, b, dir) {
  if (split === 'iid')        return ['IID (random 70/15/15)', 'split-iid'];
  if (split === 'correlated') return [`correlated(${a}, ${b}, ${dir})`, 'split-correlated'];
  if (split === 'heldout')    return ['heldout pair', 'split-heldout'];
  return ['unknown', 'split-unknown'];
}

function updateProvenanceBar(info) {
  // info: {ckpt, split, corrA, corrB, corrDir, beta, gamma, latentDim, seed, model}
  const ckptEl  = document.getElementById('prov-ckpt');
  const splitEl = document.getElementById('prov-split');
  const modelEl = document.getElementById('prov-model');
  const hpEl    = document.getElementById('prov-hp');
  const seedEl  = document.getElementById('prov-seed');
  if (!ckptEl) return;

  ckptEl.textContent = info.ckpt || '—';

  const [splitText, splitClass] = _splitDescription(
    info.split, info.corrA, info.corrB, info.corrDir);
  splitEl.textContent = splitText;
  splitEl.className   = `prov-value ${splitClass}`;

  modelEl.textContent = info.model || '—';

  // Hyperparameter summary string
  const parts = [];
  if (info.latentDim) parts.push(`z=${info.latentDim}`);
  if (info.beta  !== null && info.beta  !== undefined) parts.push(`β=${info.beta}`);
  if (info.gamma !== null && info.gamma !== undefined) parts.push(`γ=${info.gamma}`);
  hpEl.textContent = parts.length ? parts.join(', ') : '—';

  // Seed: pulled from ckpt path if present (heuristic — no harm if missing).
  let seed = info.seed;
  if (!seed && info.ckpt) {
    const m = info.ckpt.match(/_seed(\d+)/);
    if (m) seed = m[1];
  }
  seedEl.textContent = seed || '—';
}
window.updateProvenanceBar = updateProvenanceBar;

function _detectModel(ckpt) {
  if (!ckpt) return null;
  if (ckpt.includes('factorvae') || ckpt.includes('factor_vae')) return 'FactorVAE';
  return 'VAE';
}

function updateObjectiveStrip(beta, ckpt, gamma) {
  const betaEl = document.getElementById('objective-beta');
  if (betaEl) {
    const parts = [];
    if (beta  !== null && beta  !== undefined) parts.push(`β = ${beta}`);
    if (gamma !== null && gamma !== undefined) parts.push(`γ = ${gamma}`);
    betaEl.textContent = parts.length ? parts.join(' · ') : 'β = ?';
  }
  const ckptEl = document.getElementById('objective-ckpt');
  if (ckptEl) ckptEl.textContent = ckpt || '';
}
window.updateObjectiveStrip = updateObjectiveStrip;

// ── KaTeX warm-up for any pre-rendered .katex-block / .formula-inline ────
function renderInlineFormulas() {
  if (typeof katex === 'undefined') return;
  document.querySelectorAll('.formula-inline[data-formula]').forEach(el => {
    try {
      katex.render(el.getAttribute('data-formula'), el,
        {throwOnError: false, displayMode: false});
    } catch (e) {}
  });
}

// ── Init ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initHelpIcons();
  renderObjectiveStrip();
  renderInlineFormulas();
  if (window.__BOOT__?.beta !== undefined || window.__BOOT__?.gamma !== undefined) {
    updateObjectiveStrip(
      window.__BOOT__.beta,
      window.__BOOT__.ckpt,
      window.__BOOT__.gamma,
    );
  }

  if (window.__BOOT__) {
    updateProvenanceBar({
      ckpt:      window.__BOOT__.ckpt,
      split:     window.__BOOT__.split,
      corrA:     window.__BOOT__.corrFactorA,
      corrB:     window.__BOOT__.corrFactorB,
      corrDir:   window.__BOOT__.corrDirection,
      beta:      window.__BOOT__.beta,
      gamma:     window.__BOOT__.gamma,
      latentDim: window.__BOOT__.latentDim,
      model:     _detectModel(window.__BOOT__.ckpt),
    });
  }

  const toggle = document.getElementById('help-mode-toggle');
  if (toggle) {
    toggle.addEventListener('change', () => {
      if (toggle.checked) enableHelpMode(); else disableHelpMode();
    });
  }
});
