# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An interactive educational web app for learning neural networks. Users configure network architectures, train them on built-in or uploaded datasets, and visualize decision boundaries, loss curves, and metrics — all backed by a pure-NumPy ML engine (no PyTorch/TensorFlow).

## Running the App

```bash
pip install -r requirements.txt
python run.py             # http://127.0.0.1:5000
python run.py --port 8000 # custom port
```

There is no build step, test suite, or linter configured.

## Architecture

Three layers, strictly separated.

**`backend/`** — pure ML engine, organised per the spec:

- `models/` — `perceptron.py`, `mlp.py`, `rbf.py`. Each model exposes the same `forward / backward / apply_gradients / predict / predict_classes` interface so the shared training loop is model-agnostic.
- `training/` — `forward.py`, `backward.py` (thin entry points), `optimizer.py` (mini-batch SGD with optional early stopping).
- `utils/` — `activation.py` (sigmoid/relu/tanh/softmax + derivatives), `loss.py` (MSE/BCE/CCE + derivatives), `metrics.py` (classification report + regression metrics), `datasets.py` (six built-in 2D problems).
- `nn_engine.py` — backwards-compatible aggregator that re-exports the public surface so old imports keep working.

**`backend/app.py`** — Flask API. Exposes `/api/*` routes, streams training as NDJSON, keeps trained models in an in-memory `MODEL_STORE` keyed by an opaque `session_id` so the client can request decision-boundary grids and reports without re-uploading the network. Serves the `frontend/` directory as static files.

**`frontend/`** — HTML/CSS/JS only, no build tools.
- `index.html` — single-page shell with sidebar config, hero (live network + telemetry), and five tabs.
- `css/styles.css` — design tokens + components ("scientific instrument" dark aesthetic).
- `js/api.js` — `fetch` wrappers including an NDJSON streaming reader.
- `js/charts.js` — Chart.js wrappers (line, scatter) tuned to the dark palette.
- `js/network.js` — animated SVG network viz; edges thickness/colour reflect weight magnitude/sign and pulse during training snapshots.
- `js/boundary.js` — canvas heatmap renderer for decision boundaries.
- `js/app.js` — top-level wiring: sidebar bindings, training flow, tab content, experiment runner.

## Key Data Flow

```
Sidebar config + dataset
    │
    ▼
POST /api/dataset → {X, y}
    │
    ▼ (Train button)
POST /api/train (NDJSON stream) ──► live charts + network pulse
    │
    │  final 'done' line carries session_id
    ▼
GET /api/results/<sid>  ─► classification report / regression metrics
GET /api/boundary/<sid> ─► 2D probability grid + extent
```

## API Reference

| Route | Method | Purpose |
|---|---|---|
| `/api/datasets` | GET | List built-in datasets. |
| `/api/dataset` | POST | Generate `{X, y}` for a built-in dataset. |
| `/api/upload` | POST | CSV upload, label-encoded + standardised. |
| `/api/train` | POST | NDJSON stream — one line per epoch + `init` and `done`. |
| `/api/boundary/<sid>` | GET | Decision-boundary grid + train/test scatter. |
| `/api/results/<sid>` | GET | Classification / regression metrics. |
| `/api/experiments` | GET | List comparative experiments. |
| `/api/experiment` | POST | NDJSON stream — runs a comparison. |

## Extension Points

- **New dataset:** add a generator function to `backend/utils/datasets.py` and register it in the `DATASETS` dict.
- **New activation/loss:** add the function and its derivative in `backend/utils/activation.py` or `loss.py`; the `MLP` class picks them up by name through the `ACTIVATIONS` / `LOSSES` registries.
- **New model:** add a class under `backend/models/` exposing `forward / backward / apply_gradients / predict / predict_classes`; wire it into `_build_model` in `backend/app.py` and add an `<option>` to `#cfg-model` in `index.html`.
- **New experiment:** add an entry to `EXPERIMENTS` in `backend/app.py`.
- **New tab:** add a `<button class="tab">` and `<section class="tab-panel">` in `index.html`; tab switching is name-based and fully data-driven.

## Notes

- The legacy Streamlit app under `nn_platform/` is no longer used. Treat `backend/` + `frontend/` as the canonical source.
- Models are kept in memory only; the store evicts the oldest entry after 12 trained sessions to bound memory.
- The NDJSON training stream snapshots weights ~30 times per run so the network viz has live frames without flooding the wire.
