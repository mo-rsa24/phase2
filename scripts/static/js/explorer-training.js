// ─────────────────────────────────────────────────────────────────────────
//  explorer-training.js
//  Train-new-seed modal · job tray · seed dropdown refresh
// ─────────────────────────────────────────────────────────────────────────

(function () {
  const POLL_MS = 4000;
  let pollTimer = null;
  let knownDoneJobs = new Set();   // job_ids we've already refreshed seeds for

  // ── Train modal ────────────────────────────────────────────────────────
  window.openTrainModal = function (expId, label) {
    document.getElementById('train-exp-id').value = expId;
    document.getElementById('train-modal-label').textContent = label || `Exp ${expId}`;
    document.getElementById('train-error').textContent = '';
    const modal = bootstrap.Modal.getOrCreateInstance(document.getElementById('train-modal'));
    modal.show();
  };

  async function submitTrainJob() {
    const expId   = parseInt(document.getElementById('train-exp-id').value, 10);
    const seed    = parseInt(document.getElementById('train-seed').value, 10);
    const epochsRaw = document.getElementById('train-epochs').value.trim();
    const gpu     = document.getElementById('train-gpu').value.trim();
    const runtime = document.getElementById('train-runtime').value.trim();
    const errEl   = document.getElementById('train-error');
    errEl.textContent = '';

    if (!Number.isFinite(expId) || !Number.isFinite(seed)) {
      errEl.textContent = 'exp_id and seed must be numbers.';
      return;
    }

    const body = { exp_id: expId, seed, gpu, runtime, node: runtime };
    if (epochsRaw) body.epochs = parseInt(epochsRaw, 10);

    try {
      const r = await fetch('/api/train/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || r.statusText);

      bootstrap.Modal.getInstance(document.getElementById('train-modal')).hide();
      showJobsTray();
      pollJobs(); // refresh now
    } catch (e) {
      errEl.textContent = String(e.message || e);
    }
  }

  // ── Seed dropdown refresh ──────────────────────────────────────────────
  // Re-fetches /api/seeds and replaces the <ul> contents for the given
  // exp_id with the freshly discovered seeds. Keeps the "+ Train new seed…"
  // affordance at the bottom.
  async function refreshSeedDropdowns(expIds /* optional */) {
    let r, data;
    try {
      r    = await fetch('/api/seeds');
      data = await r.json();
    } catch { return; }
    const map = data.seeds_by_exp || {};
    document.querySelectorAll('[id^="seed-dd-list-"]').forEach(ul => {
      const m = ul.id.match(/^seed-dd-list-(\d+)$/);
      if (!m) return;
      const expId = parseInt(m[1], 10);
      if (expIds && !expIds.includes(expId)) return;
      const seeds = map[expId] || [];
      const wrap = document.getElementById(`seed-dd-wrap-${expId}`);
      const label = wrap?.querySelector('.dropdown-toggle');
      const expLabelText = label?.dataset.label || `Exp ${expId}`;

      let html = '';
      if (seeds.length === 0) {
        html += '<li><span class="dropdown-item-text small text-muted">No checkpoints yet</span></li>';
        html += '<li><hr class="dropdown-divider"></li>';
      }
      for (const s of seeds) {
        html += `<li><a class="dropdown-item small-mono" href="#"
                    onclick="loadSeed(${expId}, ${s.seed}, '${s.checkpoint}'); return false;">
                    seed=${s.seed}</a></li>`;
      }
      if (seeds.length > 0) html += '<li><hr class="dropdown-divider"></li>';
      html += `<li><a class="dropdown-item small text-primary" href="#"
                  onclick="openTrainModal(${expId}, '${expLabelText.replace(/'/g, "\\'")}'); return false;">
                  + Train new seed…</a></li>`;
      ul.innerHTML = html;
    });
  }
  window.refreshSeedDropdowns = refreshSeedDropdowns;

  // ── Jobs tray ──────────────────────────────────────────────────────────
  function showJobsTray() {
    document.getElementById('training-jobs-tray').classList.remove('d-none');
    document.getElementById('btn-jobs-tray-open').classList.add('d-none');
    if (!pollTimer) pollTimer = setInterval(pollJobs, POLL_MS);
  }
  function hideJobsTray() {
    document.getElementById('training-jobs-tray').classList.add('d-none');
    document.getElementById('btn-jobs-tray-open').classList.remove('d-none');
  }

  function fmtDuration(startISO, endISO) {
    const s = new Date(startISO).getTime();
    const e = endISO ? new Date(endISO).getTime() : Date.now();
    const sec = Math.max(0, Math.round((e - s) / 1000));
    const m = Math.floor(sec / 60), ss = sec % 60;
    return m > 0 ? `${m}m${String(ss).padStart(2,'0')}s` : `${ss}s`;
  }

  function renderJobs(jobs) {
    const list = document.getElementById('training-jobs-list');
    document.getElementById('training-jobs-count').textContent = jobs.length;
    document.getElementById('training-jobs-fab-count').textContent = jobs.length;
    if (jobs.length === 0) {
      list.innerHTML = '<div class="text-muted small p-2">No jobs yet.</div>';
      return;
    }
    list.innerHTML = jobs.map(j => {
      const dur = fmtDuration(j.started, j.ended);
      const epochs = j.epochs ? `epochs=${j.epochs}` : 'epochs=cfg';
      const cancelBtn = j.status === 'running'
        ? `<button class="btn btn-sm btn-outline-danger py-0 px-1 small ms-1"
                   onclick="cancelJob('${j.job_id}')">stop</button>`
        : '';
      return `
        <div class="training-job-card status-${j.status}">
          <div class="d-flex justify-content-between align-items-center">
            <div class="job-title">exp ${j.exp_id} · seed ${j.seed} · ${j.status}</div>
            <div>
              <button class="btn btn-sm btn-outline-secondary py-0 px-1 small"
                      onclick="openJobLog('${j.job_id}')">log</button>
              ${cancelBtn}
            </div>
          </div>
          <div class="job-meta">
            ${epochs} · gpu=${j.gpu || '?'} · ${j.runtime} · ${dur}
            ${j.returncode != null ? ` · rc=${j.returncode}` : ''}
          </div>
        </div>`;
    }).join('');
  }

  async function pollJobs() {
    let r, data;
    try {
      r    = await fetch('/api/train/jobs');
      data = await r.json();
    } catch { return; }
    const jobs = data.jobs || [];
    renderJobs(jobs);

    // Refresh seed dropdowns for any newly-completed (rc=0) job we haven't seen.
    const newlyDone = jobs.filter(j =>
      j.status === 'done' && !knownDoneJobs.has(j.job_id)
    );
    if (newlyDone.length) {
      newlyDone.forEach(j => knownDoneJobs.add(j.job_id));
      const expIds = [...new Set(newlyDone.map(j => j.exp_id))];
      refreshSeedDropdowns(expIds);
    }

    // Stop polling once nothing is running.
    if (!jobs.some(j => j.status === 'running') && pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }

    // Auto-open the tray on first job.
    if (jobs.length > 0 &&
        document.getElementById('training-jobs-tray').classList.contains('d-none') &&
        document.getElementById('btn-jobs-tray-open').classList.contains('d-none')) {
      document.getElementById('btn-jobs-tray-open').classList.remove('d-none');
      document.getElementById('btn-jobs-tray-open').textContent = `Training jobs (${jobs.length})`;
    }
  }

  // ── Job log modal ──────────────────────────────────────────────────────
  window.openJobLog = async function (jobId) {
    const titleEl = document.getElementById('job-log-title');
    const preEl   = document.getElementById('job-log-pre');
    titleEl.textContent = jobId;
    preEl.textContent = '(loading…)';
    bootstrap.Modal.getOrCreateInstance(document.getElementById('job-log-modal')).show();
    try {
      const r = await fetch(`/api/train/job/${encodeURIComponent(jobId)}`);
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || r.statusText);
      preEl.textContent = data.log_tail || '(empty)';
      preEl.scrollTop = preEl.scrollHeight;
    } catch (e) {
      preEl.textContent = `Error: ${e.message || e}`;
    }
  };

  window.cancelJob = async function (jobId) {
    if (!confirm(`Stop job ${jobId}?`)) return;
    await fetch(`/api/train/cancel/${encodeURIComponent(jobId)}`, {method: 'POST'});
    pollJobs();
  };

  // ── Wire up listeners on DOM ready ─────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('btn-train-submit')?.addEventListener('click', submitTrainJob);
    document.getElementById('btn-jobs-tray-close')?.addEventListener('click', hideJobsTray);
    document.getElementById('btn-jobs-tray-open')?.addEventListener('click', showJobsTray);

    // Initial fetch — if any jobs from previous sessions are still in-memory
    // (e.g. server kept running), surface them; otherwise the FAB stays hidden
    // until the user starts one.
    pollJobs();
  });
})();
