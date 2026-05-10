// Top-level wiring: state, sidebar bindings, training flow, tab switching.
(() => {

  const LR_VALUES    = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1];
  const BATCH_VALUES = [8, 16, 32, 64, 128];

  const state = {
    datasets: {},
    dataset: null,           // {X, y, n_features, ...}
    sessionId: null,
    history: null,
    layers: null,
    training: false,
    uploadedData: null,      // overrides dataset selection if present
  };

  // Charts (created lazily)
  let lossChart = null;
  let accChart  = null;
  let dataChart = null;

  // ── DOM helpers ────────────────────────────────────────────────────────
  const $  = sel => document.querySelector(sel);
  const $$ = sel => Array.from(document.querySelectorAll(sel));

  // ── Range visual fill ──────────────────────────────────────────────────
  function paintRange(input) {
    const min = +input.min, max = +input.max, val = +input.value;
    const pct = ((val - min) / (max - min)) * 100;
    input.style.setProperty('--val', pct + '%');
  }

  // ── Sidebar bindings ───────────────────────────────────────────────────
  function bindSidebar() {
    const sliders = [
      ['cfg-samples',  'lbl-samples',  v => v],
      ['cfg-noise',    'lbl-noise',    v => (+v).toFixed(2)],
      ['cfg-split',    'lbl-split',    v => Math.round(v * 100) + '%'],
      ['cfg-layers',   'lbl-layers',   v => v],
      ['cfg-width',    'lbl-width',    v => v],
      ['cfg-lr',       'lbl-lr',       v => LR_VALUES[+v].toString()],
      ['cfg-epochs',   'lbl-epochs',   v => v],
      ['cfg-batch',    'lbl-batch',    v => BATCH_VALUES[+v].toString()],
      ['cfg-l2',       'lbl-l2',       v => (+v).toFixed(3)],
      ['cfg-dropout',  'lbl-dropout',  v => (+v).toFixed(2)],
      ['cfg-patience', 'lbl-patience', v => v],
      ['cfg-rbf',      'lbl-rbf',      v => v],
    ];
    sliders.forEach(([sid, lid, fmt]) => {
      const s = document.getElementById(sid);
      const l = document.getElementById(lid);
      if (!s) return;
      const update = () => { l.textContent = fmt(s.value); paintRange(s); refreshIfDataAffected(sid); };
      s.addEventListener('input', update);
      update();
    });

    // Model select toggles layer/width visibility
    $('#cfg-model').addEventListener('change', () => {
      const m = $('#cfg-model').value;
      const isMlp = m === 'mlp';
      const isRbf = m === 'rbf';
      $('#cfg-layers-row').classList.toggle('hidden', !isMlp);
      $('#cfg-width-row').classList.toggle('hidden', !isMlp);
      $('#cfg-widths-row').classList.toggle('hidden', !isMlp);
      $('#cfg-rbf-row').classList.toggle('hidden', !isRbf);
      previewArch();
    });
    $('#cfg-act').addEventListener('change', previewArch);
    $('#cfg-layers').addEventListener('input', previewArch);
    $('#cfg-width').addEventListener('input', previewArch);
    $('#cfg-widths').addEventListener('input', previewArch);
    $('#cfg-rbf').addEventListener('input', previewArch);

    // Regularisation method toggle
    $('#cfg-reg').addEventListener('change', () => {
      const m = $('#cfg-reg').value;
      $('#reg-l2-row').classList.toggle('hidden', m !== 'l2');
      $('#reg-dropout-row').classList.toggle('hidden', m !== 'dropout');
      $('#reg-es-row').classList.toggle('hidden', m !== 'early');
    });

    // Problem-type → loss default → output activation default
    $('#cfg-prob').addEventListener('change', () => {
      const p = $('#cfg-prob').value;
      const loss = $('#cfg-loss');
      if (p === 'binary')      loss.value = 'bce';
      else if (p === 'regression') loss.value = 'mse';
      else                     loss.value = 'cce';
      syncOutputActivation();
      loadDataset();
    });
    $('#cfg-loss').addEventListener('change', syncOutputActivation);

    $('#cfg-dataset').addEventListener('change', () => {
      state.uploadedData = null;
      // Auto-pick problem type from dataset metadata so multi-class datasets
      // route to CCE + softmax automatically.
      const ds = state.datasets[$('#cfg-dataset').value];
      if (ds && ds.type) {
        const probSel = $('#cfg-prob');
        probSel.value = ds.type;
        const lossSel = $('#cfg-loss');
        if (ds.type === 'binary')          lossSel.value = 'bce';
        else if (ds.type === 'multiclass') lossSel.value = 'cce';
        else if (ds.type === 'regression') lossSel.value = 'mse';
        syncOutputActivation();
      }
      loadDataset();
    });

    // CSV upload
    $('#cfg-upload').addEventListener('change', async e => {
      const file = e.target.files[0];
      if (!file) return;
      try {
        setStatus('uploading…', 'info');
        const probType = $('#cfg-prob').value;
        const data = await API.uploadCsv(file, probType);
        state.uploadedData = data;
        state.dataset = data;
        Charts.updateScatter(dataChart, data.X, data.y);
        setStatus(`loaded ${data.n_samples} × ${data.n_features} from ${file.name}`, 'ok');
        previewArch();
      } catch (err) {
        setStatus('upload failed: ' + err.message, 'err');
      }
    });

    $('#btn-train').addEventListener('click', startTraining);
  }

  function refreshIfDataAffected(sid) {
    if (state.uploadedData) return; // user-provided data; don't regenerate
    if (sid === 'cfg-samples' || sid === 'cfg-noise') {
      clearTimeout(refreshIfDataAffected._t);
      refreshIfDataAffected._t = setTimeout(loadDataset, 200);
    }
  }

  // ── Datasets ───────────────────────────────────────────────────────────
  async function loadDatasetList() {
    const list = await API.jsonGet('/api/datasets');
    const sel = $('#cfg-dataset');
    sel.innerHTML = list.map(d => `<option value="${d.id}">${d.label}</option>`).join('');
    list.forEach(d => state.datasets[d.id] = d);
    sel.value = 'circles';
  }

  async function loadDataset() {
    const ds = $('#cfg-dataset').value;
    const samples = +$('#cfg-samples').value;
    const noise   = +$('#cfg-noise').value;
    try {
      const data = await API.jsonPost('/api/dataset', {
        dataset: ds, samples, noise,
      });
      state.dataset = data;
      Charts.updateScatter(dataChart, data.X, data.y);
      previewArch();
    } catch (err) {
      setStatus('dataset error: ' + err.message, 'err');
    }
  }

  // ── Output activation auto-sync ────────────────────────────────────────
  // Loss → matching output activation. Visible to the user so they can see
  // softmax being selected for multi-class.
  function syncOutputActivation() {
    const loss = $('#cfg-loss').value;
    const map = { bce: 'sigmoid', cce: 'softmax', mse: 'linear' };
    const target = map[loss];
    if (target) $('#cfg-out-act').value = target;
  }

  // ── Architecture preview ───────────────────────────────────────────────
  function parseWidths() {
    const raw = ($('#cfg-widths').value || '').trim();
    if (!raw) return null;
    const parts = raw.split(/[,\s]+/).map(s => parseInt(s, 10)).filter(n => Number.isFinite(n) && n > 0);
    return parts.length ? parts : null;
  }

  function readArchitecture() {
    const inDim = state.dataset ? state.dataset.n_features : 2;
    let outDim;
    if ($('#cfg-prob').value === 'multiclass' && state.dataset && state.dataset.y[0]) {
      outDim = Array.isArray(state.dataset.y[0]) ? state.dataset.y[0].length : 1;
    } else {
      outDim = 1;
    }
    const model = $('#cfg-model').value;
    if (model === 'perceptron' || model === 'perceptron_act') {
      return [inDim, outDim];
    }
    if (model === 'rbf') {
      const k = +$('#cfg-rbf').value;
      return [inDim, k, outDim];
    }
    const widths = parseWidths();
    if (widths) return [inDim, ...widths, outDim];
    const n = +$('#cfg-layers').value;
    const w = +$('#cfg-width').value;
    return [inDim, ...Array(n).fill(w), outDim];
  }

  function previewArch() {
    const layers = readArchitecture();
    state.layers = layers;
    NetworkViz.renderArchitecture($('#network-viz'), layers);
    $('#arch-pill').textContent = NetworkViz.archDescription(layers);
  }

  // ── Status ─────────────────────────────────────────────────────────────
  function setStatus(text, kind = 'info') {
    const banner = $('#status-banner');
    banner.className = 'status-banner status-' + kind;
    banner.innerHTML = `<span>${text}</span>`;
    $('#status-text').textContent = text;
  }
  function setDot(kind) {
    const d = $('#status-dot');
    d.classList.remove('training', 'done');
    if (kind) d.classList.add(kind);
  }

  // ── Telemetry ──────────────────────────────────────────────────────────
  function setTelemetry(epoch, total, m) {
    $('#t-epoch').textContent = `${epoch}/${total}`;
    $('#t-tloss').textContent = m.train_loss != null ? m.train_loss.toFixed(4) : '—';
    $('#t-vloss').textContent = m.val_loss   != null ? m.val_loss.toFixed(4)   : '—';
    $('#t-tacc').textContent  = m.train_acc  != null ? (m.train_acc * 100).toFixed(1) + '%' : '—';
    $('#t-vacc').textContent  = m.val_acc    != null ? (m.val_acc * 100).toFixed(1) + '%' : '—';
    if (m.train_loss != null && m.val_loss != null) {
      const gap = m.val_loss - m.train_loss;
      const el  = $('#t-gap');
      el.textContent = gap.toFixed(4);
      el.classList.toggle('neg', gap > 0.05);
      el.classList.toggle('pos', gap <= 0.02);
    }
    const pct = epoch / total;
    const ring = $('#progress-ring');
    ring.style.strokeDashoffset = (97.4 * (1 - pct)).toFixed(2);
    $('#progress-text').textContent = Math.round(pct * 100) + '%';
  }
  function resetTelemetry() {
    ['t-epoch','t-tloss','t-vloss','t-tacc','t-vacc','t-gap'].forEach(id => $('#'+id).textContent = '—');
    $('#progress-ring').style.strokeDashoffset = 97.4;
    $('#progress-text').textContent = '0%';
  }

  // ── Training flow ──────────────────────────────────────────────────────
  function gatherTrainPayload() {
    const reg = $('#cfg-reg').value;
    const widths = parseWidths();
    return {
      X: state.dataset.X,
      y: state.dataset.y,
      test_split: +$('#cfg-split').value,
      model_type: $('#cfg-model').value,
      hidden_act: $('#cfg-act').value,
      output_act: $('#cfg-out-act').value,
      hidden_layers: +$('#cfg-layers').value,
      width: +$('#cfg-width').value,
      widths: widths || undefined,
      rbf_centers: +$('#cfg-rbf').value,
      lr: LR_VALUES[+$('#cfg-lr').value],
      epochs: +$('#cfg-epochs').value,
      batch_size: BATCH_VALUES[+$('#cfg-batch').value],
      loss: $('#cfg-loss').value,
      l2: reg === 'l2' ? +$('#cfg-l2').value : 0,
      dropout: reg === 'dropout' ? +$('#cfg-dropout').value : 0,
      early_stopping_patience: reg === 'early' ? +$('#cfg-patience').value : 0,
    };
  }

  async function startTraining() {
    if (state.training) return;
    if (!state.dataset) {
      setStatus('no dataset loaded', 'err');
      return;
    }

    state.training = true;
    state.history = { train_loss: [], val_loss: [], train_acc: [], val_acc: [] };
    state.sessionId = null;

    setDot('training');
    setStatus('training…', 'info');
    $('#btn-train').disabled = true;
    resetTelemetry();

    const payload = gatherTrainPayload();
    const liveHistory = state.history;

    try {
      await API.streamNdjson('/api/train', payload, msg => {
        if (msg.type === 'init') {
          state.layers = msg.layers;
          $('#arch-pill').textContent = NetworkViz.archDescription(msg.layers);
          NetworkViz.renderWeights($('#network-viz'), msg.layers, msg.weights);
        } else if (msg.type === 'epoch') {
          liveHistory.train_loss.push(msg.train_loss);
          liveHistory.val_loss.push(msg.val_loss);
          liveHistory.train_acc.push(msg.train_acc);
          liveHistory.val_acc.push(msg.val_acc);

          setTelemetry(msg.epoch, msg.total, msg);
          Charts.updateLoss(lossChart, liveHistory);
          if (Charts.updateAcc(accChart, liveHistory)) {
            $('#acc-empty').style.display = 'none';
          } else {
            $('#acc-empty').style.display = 'block';
          }
          if (msg.weights) {
            NetworkViz.renderWeights($('#network-viz'), state.layers, msg.weights);
            NetworkViz.pulse($('#network-viz'));
          }
        } else if (msg.type === 'done') {
          state.sessionId = msg.session_id;
          NetworkViz.renderWeights($('#network-viz'), state.layers, msg.weights);
          finishTraining();
        }
      });
    } catch (err) {
      setStatus('training failed: ' + err.message, 'err');
      setDot(null);
      $('#btn-train').disabled = false;
      state.training = false;
    }
  }

  async function finishTraining() {
    state.training = false;
    setDot('done');
    $('#btn-train').disabled = false;

    // Diagnose fit
    const h = state.history;
    const gap = h.val_loss[h.val_loss.length - 1] - h.train_loss[h.train_loss.length - 1];
    const finalTrain = h.train_loss[h.train_loss.length - 1];
    let diag = { kind: 'ok', text: `clean fit · gen. gap = ${gap.toFixed(3)}` };
    if (gap > 0.08) diag = { kind: 'warn', text: `possible overfitting · gen. gap = ${gap.toFixed(3)}` };
    else if (finalTrain > 0.45) diag = { kind: 'info', text: `high train loss · model may be underfitting` };
    setStatus('done · ' + diag.text, diag.kind === 'warn' ? 'warn' : 'ok');

    // Populate Results, Boundary tabs.
    try {
      const r = await API.jsonGet(`/api/results/${state.sessionId}`);
      renderResults(r);
    } catch (err) { setStatus('results fetch failed: ' + err.message, 'err'); }

    let boundaryPayload = null;
    try {
      const b = await API.jsonGet(`/api/boundary/${state.sessionId}?res=80`);
      boundaryPayload = b;
      renderBoundary(b);
    } catch (err) {
      $('#boundary-empty').textContent = err.message.includes('2D') ?
        'Decision boundary requires 2D input.' : 'Boundary unavailable.';
      $('#boundary-empty').classList.remove('hidden');
      $('#boundary-content').classList.add('hidden');
    }

    // 3D tab — probability surface (from boundary grid) + loss landscape.
    if (boundaryPayload) {
      $('#viz3d-empty').classList.add('hidden');
      $('#viz3d-content').classList.remove('hidden');
      Viz3D.renderSurface($('#viz3d-surface'), boundaryPayload);
      try {
        const land = await API.jsonGet(`/api/landscape/${state.sessionId}?res=21&span=0.6`);
        Viz3D.renderLandscape($('#viz3d-landscape'), land);
      } catch (err) {
        $('#viz3d-landscape').innerHTML =
          `<div class="empty">Loss landscape unavailable: ${err.message}</div>`;
      }
    } else {
      $('#viz3d-empty').classList.remove('hidden');
      $('#viz3d-content').classList.add('hidden');
    }
  }

  // ── Results renderer ───────────────────────────────────────────────────
  function renderResults(r) {
    $('#results-empty').classList.add('hidden');
    $('#results-content').classList.remove('hidden');

    const bars = $('#metrics-bars');
    bars.innerHTML = '';
    const tbl = $('#per-class-table');
    tbl.innerHTML = '';

    if (r.regression) {
      const m = r.regression;
      const items = [
        ['MSE',  m.mse,  Math.min(1, m.mse)],
        ['RMSE', m.rmse, Math.min(1, m.rmse)],
        ['R²',   m.r2,   Math.max(0, Math.min(1, m.r2))],
      ];
      items.forEach(([lbl, val, fill]) => {
        bars.insertAdjacentHTML('beforeend', `
          <div class="metric-bar">
            <span class="lbl">${lbl}</span>
            <div class="track"><div class="fill" style="width:${(fill*100).toFixed(0)}%"></div></div>
            <span class="val">${val.toFixed(4)}</span>
          </div>`);
      });
      $('#confusion').innerHTML = '<div class="muted small" style="padding:20px;text-align:center;">Not applicable for regression.</div>';
      $('#per-class-table').innerHTML = '';
      $('#diagnosis').className = 'diagnosis info';
      $('#diagnosis').textContent = `R² = ${m.r2.toFixed(3)}`;
      return;
    }

    // Classification
    const c = r.classification;
    const pc = c.per_class;
    const classes = Object.keys(pc);
    const avg = key => classes.reduce((s, k) => s + pc[k][key], 0) / classes.length;
    const items = [
      ['Accuracy',   c.accuracy],
      ['F1 score',   avg('f1')],
      ['Precision',  avg('precision')],
      ['Recall',     avg('recall')],
    ];
    items.forEach(([lbl, val]) => {
      bars.insertAdjacentHTML('beforeend', `
        <div class="metric-bar">
          <span class="lbl">${lbl}</span>
          <div class="track"><div class="fill" style="width:${(val*100).toFixed(1)}%"></div></div>
          <span class="val">${(val*100).toFixed(1)}%</span>
        </div>`);
    });

    // Confusion matrix
    const cm = c.confusion_matrix;
    const max = Math.max(...cm.flat()) || 1;
    let cmHtml = '';
    for (let i = 0; i < cm.length; i++) {
      cmHtml += '<div class="cm-row">';
      for (let j = 0; j < cm[i].length; j++) {
        const n = cm[i][j];
        const t = n / max;
        // mint -> deeper teal as t grows
        const bg = `rgba(125,249,199,${0.06 + t*0.55})`;
        const tc = t > 0.55 ? '#0a0d14' : '#e6ebf5';
        cmHtml += `<div class="cm-cell" style="background:${bg};color:${tc}">${n}</div>`;
      }
      cmHtml += '</div>';
    }
    cmHtml += '<div class="cm-axis"><span>predicted →</span></div>';
    $('#confusion').innerHTML = cmHtml;

    // Per-class table
    const head = '<thead><tr><th>class</th><th>prec</th><th>rec</th><th>f1</th><th>support</th></tr></thead>';
    const rows = classes.map(k => `
      <tr>
        <td>${k}</td>
        <td>${pc[k].precision.toFixed(3)}</td>
        <td>${pc[k].recall.toFixed(3)}</td>
        <td>${pc[k].f1.toFixed(3)}</td>
        <td>${pc[k].support}</td>
      </tr>`).join('');
    tbl.innerHTML = head + '<tbody>' + rows + '</tbody>';

    // Diagnosis
    const h = state.history;
    const gap = h.val_loss[h.val_loss.length - 1] - h.train_loss[h.train_loss.length - 1];
    const last = h.train_loss[h.train_loss.length - 1];
    let cls = 'ok', txt = `Good fit · gen. gap = ${gap.toFixed(3)}`;
    if (gap > 0.08) { cls = 'warn'; txt = `Possible overfitting · gap = ${gap.toFixed(3)}`; }
    else if (last > 0.45) { cls = 'info'; txt = `High train loss — try more layers or epochs`; }
    $('#diagnosis').className = 'diagnosis ' + cls;
    $('#diagnosis').textContent = txt;
  }

  // ── Boundary tab ───────────────────────────────────────────────────────
  function renderBoundary(b) {
    $('#boundary-empty').classList.add('hidden');
    $('#boundary-content').classList.remove('hidden');
    Boundary.render($('#boundary-train'), { ...b, train: b.train, test: null });
    Boundary.render($('#boundary-test'),  { ...b, train: null, test: b.test });
  }

  // ── Tab switching ──────────────────────────────────────────────────────
  function bindTabs() {
    $$('.tab').forEach(t => t.addEventListener('click', () => {
      $$('.tab').forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      const name = t.dataset.tab;
      $$('.tab-panel').forEach(p => p.classList.toggle('active', p.dataset.panel === name));

      // Re-render network viz on tab change in case viewport changed
      if (state.layers) {
        NetworkViz.renderArchitecture($('#network-viz'), state.layers);
      }

      // Plotly charts need a resize when their panel becomes visible.
      if (name === 'viz3d' && window.Plotly) {
        ['viz3d-surface', 'viz3d-landscape'].forEach(id => {
          const el = document.getElementById(id);
          if (el && el.children.length) Plotly.Plots.resize(el);
        });
      }
    }));
  }

  // ── Theory cards ───────────────────────────────────────────────────────
  const THEORY = [
    {
      title: 'Perceptron',
      eyebrow: 'origin',
      body: `<p>The <strong>perceptron</strong> (Rosenblatt, 1958) is a single weighted sum followed by an activation.
      The historical form uses a step function; replacing it with a differentiable activation enables gradient-based learning.
      A single perceptron can only separate linearly separable classes — the famous XOR limitation.</p>`,
      formula: 'ŷ = f(w · x + b)',
    },
    {
      title: 'Multi-Layer Perceptron',
      eyebrow: 'feed-forward stack',
      body: `<p>An <strong>MLP</strong> stacks affine maps and non-linear activations. With at least one hidden layer
      and a non-polynomial activation it is a <em>universal approximator</em> — given enough neurons it can fit any
      continuous function on a bounded domain. Depth lets the network compose features hierarchically; width gives
      each layer expressive capacity.</p>`,
      formula: 'h⁽ˡ⁾ = σ(W⁽ˡ⁾ h⁽ˡ⁻¹⁾ + b⁽ˡ⁾)',
    },
    {
      title: 'RBF network',
      eyebrow: 'distance-based',
      body: `<p>An <strong>RBF</strong> (Radial Basis Function) network places K Gaussian “bumps” at chosen centers
      and learns a linear combination of their responses. Each hidden unit answers <em>“how close is x to my center?”</em>.
      Local, fast to train, but scales poorly with input dimensionality.</p>`,
      formula: 'φₖ(x) = exp(−‖x − cₖ‖² / 2σ²)',
    },
    {
      title: 'Backpropagation',
      eyebrow: 'learning rule',
      body: `<p>Backpropagation applies the <strong>chain rule</strong> to compute gradients layer by layer,
      propagating error from the output backward. Each weight is updated as <code>w ← w − η · ∂L/∂w</code>.
      Mini-batch SGD averages the gradient over a small batch — a noisy but cheap estimator that often generalises better than full-batch.</p>`,
      formula: '∂L/∂Wⁱ  =  ∂L/∂ŷ · ∂ŷ/∂z⁽ⁱ⁺¹⁾ · … · ∂h⁽ⁱ⁾/∂z⁽ⁱ⁾ · x',
    },
    {
      title: 'Activations',
      eyebrow: 'nonlinearities',
      body: `<table class="theory-table">
        <thead><tr><th>name</th><th>range</th><th>typical use</th></tr></thead>
        <tbody>
          <tr><td>sigmoid</td><td>(0,1)</td><td>binary output</td></tr>
          <tr><td>tanh</td><td>(−1,1)</td><td>hidden — saturating</td></tr>
          <tr><td>relu</td><td>[0,∞)</td><td>hidden — default</td></tr>
          <tr><td>softmax</td><td>(0,1)</td><td>multi-class output</td></tr>
        </tbody>
      </table>`,
    },
    {
      title: 'Loss functions',
      eyebrow: 'training signals',
      body: `<p><strong>MSE</strong> · <code>Σ(y−ŷ)² / n</code> · regression.</p>
      <p><strong>Binary CE</strong> · <code>−[y·log(ŷ) + (1−y)·log(1−ŷ)]</code> · binary classification.</p>
      <p><strong>Categorical CE</strong> · <code>−Σ yᵢ · log(ŷᵢ)</code> · multi-class.</p>`,
    },
    {
      title: 'Regularisation',
      eyebrow: 'capacity control',
      body: `<table class="theory-table">
        <thead><tr><th>technique</th><th>mechanism</th></tr></thead>
        <tbody>
          <tr><td>L2</td><td>adds <code>λ Σ w²</code> — penalises large weights</td></tr>
          <tr><td>dropout</td><td>randomly zeros activations during training</td></tr>
          <tr><td>early stopping</td><td>halts when val loss stops improving</td></tr>
        </tbody>
      </table>`,
      formula: 'L_reg = L + λ · Σ w²',
    },
    {
      title: 'Underfitting',
      eyebrow: 'high bias',
      body: `<p>The model is <strong>too simple</strong> to capture the underlying structure. Symptom:
      <em>both</em> train and val loss stay high and refuse to come down. Fixes: add layers/neurons,
      switch to a richer activation (ReLU/tanh over linear), train longer, lower regularisation.</p>
      <p class="muted small">Try the <em>Perceptron</em> on the XOR or spirals dataset to see this live.</p>`,
    },
    {
      title: 'Overfitting',
      eyebrow: 'high variance',
      body: `<p>The model has <strong>memorised the training set</strong> but doesn’t generalise. Symptom:
      train loss keeps falling while val loss plateaus or rises — a widening generalisation gap. Fixes:
      L2, dropout, early stopping, more data, smaller architecture.</p>
      <p class="muted small">Run the <em>Overfitting vs regularisation</em> experiment to compare directly.</p>`,
    },
    {
      title: 'Bias / Variance',
      eyebrow: 'decomposition',
      body: `<p>Generalisation error decomposes into <strong>bias²</strong> (systematic error from a too-simple model),
      <strong>variance</strong> (sensitivity to the particular training sample), and irreducible <strong>noise</strong>.
      Increasing capacity trades bias for variance; regularisation pushes the other way.</p>`,
      formula: 'Error ≈ Bias² + Variance + Noise',
    },
  ];

  function renderTheory() {
    const grid = $('#theory-grid');
    grid.innerHTML = THEORY.map(t => `
      <div class="panel-card">
        <div class="card-head"><div>
          <div class="card-title">${t.title}</div>
          <div class="card-eyebrow">${t.eyebrow}</div>
        </div></div>
        <div class="theory-card-body">${t.body}</div>
        ${t.formula ? `<div class="formula">${t.formula}</div>` : ''}
      </div>`).join('');
  }

  // ── Experiments tab ────────────────────────────────────────────────────
  let currentExp = 'perceptron_vs_mlp';
  async function loadExperiments() {
    const list = await API.jsonGet('/api/experiments');
    const seg = $('#exp-seg');
    seg.innerHTML = list.map((e, i) =>
      `<button data-id="${e.id}" class="${i===0?'active':''}">${e.label}</button>`
    ).join('');
    seg.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
      seg.querySelectorAll('button').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      currentExp = b.dataset.id;
    }));
    $('#btn-exp').addEventListener('click', runExperiment);
  }

  async function runExperiment() {
    const out = $('#exp-results');
    out.innerHTML = `<div class="empty">running ${currentExp}…</div>`;
    const collected = [];
    let scatter = null;

    try {
      await API.streamNdjson('/api/experiment', { id: currentExp }, msg => {
        if (msg.type === 'progress') {
          out.innerHTML = `<div class="empty">trained <span class="mono">${msg.i}</span> · last: ${msg.name}</div>`;
        } else if (msg.type === 'done') {
          collected.push(...msg.results);
          scatter = msg.scatter;
        }
      });
    } catch (err) {
      out.innerHTML = `<div class="empty">experiment failed: ${err.message}</div>`;
      return;
    }

    if (!collected.length) return;

    // Render summary card + boundary cards
    const maxAcc = Math.max(...collected.map(r => r.acc));
    const maxLoss = Math.max(...collected.map(r => r.loss));
    const fmtAcc = v => (v == null) ? '—' : (v * 100).toFixed(1) + '%';
    const fmtLoss = v => (v == null) ? '—' : v.toFixed(3);
    const summary = `
      <div class="panel-card">
        <div class="card-head"><div>
          <div class="card-title">Comparison · ours vs scikit-learn</div>
          <div class="card-eyebrow">test accuracy · test loss · sklearn baseline (same architecture)</div>
        </div></div>
        <div class="exp-row exp-row-head muted small">
          <div>configuration</div>
          <div class="num">ours · acc</div>
          <div class="num dim">ours · loss</div>
          <div class="num">sk · acc</div>
          <div class="num dim">sk · loss</div>
        </div>
        ${collected.map(r => {
          const sk = r.sklearn || {};
          const skAcc = sk.acc != null ? sk.acc : null;
          const skLoss = sk.loss != null ? sk.loss : null;
          return `
          <div class="exp-row">
            <div>
              <div class="name">${r.name}</div>
              <div class="muted small mono">${r.layers.join(' · ')}</div>
            </div>
            <div class="num">${fmtAcc(r.acc)}</div>
            <div class="num dim">${fmtLoss(r.loss)}</div>
            <div class="num">${fmtAcc(skAcc)}</div>
            <div class="num dim">${fmtLoss(skLoss)}</div>
          </div>`;
        }).join('')}
      </div>`;

    const boundaryCards = `
      <div class="grid-3">
        ${collected.map((r, i) => `
          <div class="panel-card">
            <div class="card-head"><div>
              <div class="card-title">${r.name}</div>
              <div class="card-eyebrow">acc ${(r.acc*100).toFixed(1)}%</div>
            </div></div>
            <canvas id="exp-canvas-${i}" width="320" height="260"></canvas>
          </div>`).join('')}
      </div>`;

    out.innerHTML = summary + boundaryCards;

    // Paint each boundary
    collected.forEach((r, i) => {
      const c = document.getElementById('exp-canvas-' + i);
      if (!c) return;
      Boundary.render(c, {
        grid: r.boundary.grid,
        extent: r.boundary.extent,
        scatter,
      });
    });
  }

  // ── Init ──────────────────────────────────────────────────────────────
  async function init() {
    bindSidebar();
    bindTabs();
    renderTheory();

    // Create chart instances
    lossChart = Charts.makeLine(document.getElementById('chart-loss'));
    accChart  = Charts.makeLine(document.getElementById('chart-acc'));
    dataChart = Charts.makeScatter(document.getElementById('chart-data'));

    // Initial dataset list & preview
    await loadDatasetList();
    await loadDataset();

    // Architecture preview
    previewArch();

    // Experiments
    loadExperiments();

    // Reflow viz on resize
    window.addEventListener('resize', () => {
      if (state.layers) NetworkViz.renderArchitecture($('#network-viz'), state.layers);
    });

    setStatus('ready · configure on the left, then train', 'info');
  }

  document.addEventListener('DOMContentLoaded', init);
})();
