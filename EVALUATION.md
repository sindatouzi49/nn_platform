# Report — Neuralab (Neural Networks Educational Platform)

Project located at `C:\Users\sinda\Downloads\nn_platform`. Inspected source for `backend/`, `frontend/`, and the legacy `nn_platform/` Streamlit copy.

## 1. Project at a glance

- **Stack:** Flask + pure-NumPy ML engine on the backend; vanilla HTML/CSS/JS (Chart.js + Plotly via CDN) on the frontend. No PyTorch/TensorFlow/Keras — every line of math is hand-rolled. `requirements.txt` only pulls Flask, NumPy, Pandas, scikit-learn (the latter only for baselines in experiments).
- **Run:** `pip install -r requirements.txt` then `python run.py` → `http://127.0.0.1:5000`.
- **Layout mirrors the spec almost exactly** (`backend/{models,training,utils}` + `frontend/`). The legacy single-file `nn_platform/app.py` (Streamlit) is dead code kept for archive.
- A duplicate `PRESENTATION_REPORT.md` (~63 KB) and a concise `REPORT.md` already ship inside the project.

## 2. Coverage vs. the spec

| Spec requirement | Status | Where |
|---|---|---|
| **UI — load CSV, choose problem (binary/multi/regression), choose model** | Done | `frontend/index.html` sidebar; `backend/app.py:/api/upload` |
| **L1 — Historical perceptron** | Done | `backend/models/perceptron.py` (`historical=True` → linear out) |
| **L1 — Perceptron with activation** | Done | same file, `historical=False`, also wired as `perceptron_act` model option |
| **L2 — MLP forward** | Done | `backend/models/mlp.py:35` (`forward`) |
| **L2 — Configurable hidden layers / neurons / activation** | Done | sliders + free-form `widths` text override (`16,8,4`) → `_resolve_widths` in `app.py:134` |
| **L2 — sigmoid / ReLU / tanh / softmax** | Done | `backend/utils/activation.py` (all four + linear, with derivatives) |
| **L3 — Cost functions: MSE, BCE, CCE** | Done | `backend/utils/loss.py` |
| **L3 — Gradient descent + (bonus) mini-batch GD** | Done | `backend/training/optimizer.py` (mini-batch SGD as the default) |
| **L3 — Backpropagation** | Done | `backend/models/mlp.py:59` (`backward`), thin re-export in `training/backward.py` |
| **L4 — L2 regularisation** | Done | `mlp.py:83` `+ self.l2_lambda * self.W[i]` |
| **L4 — Dropout (advanced bonus)** | Done | `mlp.py:50` (inverted-dropout mask, applied per epoch when `training=True`) |
| **L4 — Early stopping** | Done | `optimizer.py:63` and inline copy in `app.py:293` (with best-weight rollback) |
| **L4 — Pedagogy: overfit/underfit/bias-variance** | Done | "Theory" tab + Experiment 3 (`regularisation`) wired in `app.py:430` |
| **L5 — Classification: accuracy, P/R, F1, confusion matrix** | Done | `backend/utils/metrics.py:5` (`classification_report_dict`) |
| **L5 — Regression: MSE, RMSE, R²** | Done | `metrics.py:34` (`regression_metrics`) |
| **L6 — Loss/Acc vs Epochs (live)** | Done | NDJSON streaming → Chart.js (`frontend/js/charts.js`, `app.js`) |
| **L6 — Decision boundary (2D)** | Done | `/api/boundary/<sid>` + canvas heatmap (`frontend/js/boundary.js`) |
| **L6 — Train vs Test comparison** | Done | side-by-side scatters in the "Decision boundary" tab |
| **Mandatory experiments** | Done | `EXPERIMENTS` dict in `app.py:413` covers Perceptron-vs-MLP, depth, regularisation |
| **Bonus — sklearn comparison** | Done | `_sklearn_baseline()` in `app.py:441` produces side-by-side acc/loss columns |
| **Bonus — Advanced interactive UI** | Done | live network topology with weight-pulse animation (`frontend/js/network.js`) |
| **Bonus — 3D visualisation** | Done | "3D" tab: probability surface + filter-normalised loss landscape (`backend/training/landscape.py`, `frontend/js/viz3d.js`) |
| **Extra — RBF network** | Done (beyond spec) | `backend/models/rbf.py` (Gaussian centers from data, σ from median pairwise distance, linear head trained) |

**Conclusion:** every required deliverable from L1–L6, all three mandatory experiments, and all three named bonuses are implemented. RBF is a free extra.

## 3. Architecture notes worth highlighting

- **Streaming training (NDJSON).** `POST /api/train` yields one JSON line per epoch (loss/acc + occasional weight snapshots) and a final `done` line carrying a `session_id`. Frontend uses a streaming reader (`frontend/js/api.js`) to update charts live without polling. Snapshots are subsampled to ~30/run to keep the wire light.
- **Model-agnostic training loop.** `MLP`, `Perceptron`, and `RBFNetwork` all expose the same `forward / backward / apply_gradients / predict / predict_classes` surface, so the optimizer works on any of them unchanged.
- **Session-keyed in-memory store.** Trained models live in `MODEL_STORE` keyed by `session_id` so the boundary, results, and landscape endpoints can run without the client re-uploading the network. Bounded to 12 entries (oldest evicted) — fine for demo, would not survive a restart.
- **Filter-normalised loss landscape.** `backend/training/landscape.py` implements Li et al. (2018) per-parameter normalisation so no single weight matrix dominates the random direction — a thoughtful detail for a teaching tool.
- **He init**, gradient-clean composite gradients (softmax+CCE and sigmoid+BCE collapsed to `(ŷ−y)/N` to avoid numerical issues): `mlp.py:64` and `loss.py:28`.

## 4. Mapped to the evaluation rubric

| Criterion (weight) | Assessment |
|---|---|
| **Correct implementation (30%)** | Strong. All four activations + derivatives, three losses + derivatives, working backprop, mini-batch SGD, L2, dropout, early stopping, full classification + regression metrics. Sklearn baselines act as a sanity check. |
| **Theoretical understanding (20%)** | The shipped `REPORT.md` covers perceptron limits, universal approximation, activations/losses, backprop chain rule, regularisation mechanics, bias-variance. The "Theory" UI tab surfaces the same material in-app. |
| **Experimentation (20%)** | Three mandatory experiments are one-click, with sklearn columns and per-config decision-boundary thumbnails. Numbers in `REPORT.md` look credible for the circles dataset. |
| **Platform quality (20%)** | Clean separation (models/training/utils), consistent model interface, streaming UI, dark "scientific instrument" theme, live network viz, 3D tab, no build step required. |
| **Innovation / bonus (10%)** | All three named bonuses delivered + RBF network + filter-normalised loss landscape — solid bonus coverage. |

## 5. Gaps / things to mention in a defense

- **Optimiser is plain SGD** — no momentum, no Adam. Acceptable for the 2D educational datasets; would matter on harder problems. `REPORT.md` flags this honestly.
- **RBF centers are non-trainable** (chosen from data + median-distance σ). Pedagogically clean; not state-of-the-art.
- **In-memory model store** — sessions vanish on server restart; no persistence layer.
- **No automated tests, no linter, no CI.** `CLAUDE.md` says so explicitly.
- **CSV upload is naive** — drops any row with a NaN, label-encodes object columns by sort order, normalises features but does not sanity-check that the last column is actually the label.
- **Two duplicate copies of the legacy Streamlit app** under `nn_platform/` and `nn_platform/nn_platform_new/`. Safe to delete; they're not on the run path.
- **`.pyc` caches checked in** under every `__pycache__/`. Cosmetic, but worth a `.gitignore`.

## 6. Recommendation

The codebase already meets the spec; what would lift the grade further:

1. Drop the legacy `nn_platform/` Streamlit copies — they confuse the layout and add nothing.
2. Add an Adam optimiser alongside SGD and expose it as a UI choice — cheap to implement, opens up another experiment ("SGD vs Adam convergence").
3. Add a small `tests/` folder (gradient check via finite differences for `MLP.backward`, sanity tests for losses/metrics). Even ten asserts would prove correctness defensibly.
4. Persist sessions to disk (pickle) so a demo survives a server restart — not load-bearing for grading, but a good polish point.
