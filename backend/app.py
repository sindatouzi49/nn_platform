"""Flask backend for the Neural Networks Platform.

Architecture:
- Frontend lives in ../frontend and is served as static files.
- API routes are under /api/*.
- Trained models are kept in-memory in MODEL_STORE keyed by an opaque session id
  so the client can request decision-boundary grids and predictions without
  re-uploading the network.
- /api/train streams NDJSON: one JSON object per line, one line per epoch,
  ending with a final `done` line carrying the session id.
"""
from __future__ import annotations

import io
import json
import time
import uuid
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

from . import nn_engine as nne

# ── App setup ────────────────────────────────────────────────────────────────

FRONTEND_DIR = (__file__.rsplit("backend", 1)[0]).rstrip("\\/").rstrip("/") + "/frontend"

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path="",
)

MODEL_STORE: dict[str, dict[str, Any]] = {}


# ── Static frontend ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ── Datasets ─────────────────────────────────────────────────────────────────

@app.route("/api/datasets")
def list_datasets():
    return jsonify([
        {"id": k, "label": v["label"], "type": v["type"]}
        for k, v in nne.DATASETS.items()
    ])


@app.route("/api/dataset", methods=["POST"])
def generate_dataset():
    cfg = request.get_json(force=True)
    name = cfg.get("dataset", "circles")
    n = int(cfg.get("samples", 300))
    noise = float(cfg.get("noise", 0.1))
    seed = cfg.get("seed")

    if name not in nne.DATASETS:
        return jsonify({"error": f"unknown dataset {name}"}), 400

    if seed is not None:
        np.random.seed(int(seed))

    fn = nne.DATASETS[name]["fn"]
    try:
        X, y = fn(n=n, noise=noise)
    except TypeError:
        X, y = fn(n=n)

    return jsonify({
        "X": X.tolist(),
        "y": y.tolist(),
        "type": nne.DATASETS[name]["type"],
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
    })


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    prob_type = request.form.get("prob_type", "binary")

    df = pd.read_csv(io.BytesIO(request.files["file"].read()))
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if df.shape[1] < 2:
        return jsonify({"error": "need at least 2 columns"}), 400

    feat_cols = df.columns[:-1]
    label_col = df.columns[-1]
    X_df = df[feat_cols].copy()
    y = df[label_col].values

    for col in X_df.columns:
        if X_df[col].dtype == object:
            codes = {v: i for i, v in enumerate(sorted(X_df[col].unique()))}
            X_df[col] = X_df[col].map(codes)

    X = X_df.values.astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)

    if prob_type == "regression":
        y = y.astype(np.float32)
        y = (y - y.min()) / (y.max() - y.min() + 1e-9)
        y = y.reshape(-1, 1).astype(np.float32)
    else:
        classes = np.unique(y)
        if prob_type == "binary" or len(classes) == 2:
            y = (y == classes[-1]).astype(np.float32).reshape(-1, 1)
        else:
            oh = np.zeros((len(y), len(classes)), dtype=np.float32)
            for i, c in enumerate(classes):
                oh[y == c, i] = 1
            y = oh

    return jsonify({
        "X": X.tolist(),
        "y": y.tolist(),
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "feature_names": [str(c) for c in feat_cols],
    })


# ── Training (streaming NDJSON) ──────────────────────────────────────────────

def _resolve_widths(cfg: dict) -> list[int]:
    """Return per-layer hidden widths.

    Accepts either an explicit `widths` array (e.g. [16, 8, 4]) or a string
    like "16,8,4". Falls back to `[width] * hidden_layers` for the simple
    "uniform layers" case the slider UI still supports.
    """
    raw = cfg.get("widths")
    if isinstance(raw, str):
        raw = [tok for tok in raw.replace(" ", "").split(",") if tok]
    if isinstance(raw, list) and raw:
        widths = [int(w) for w in raw if int(w) > 0]
        if widths:
            return widths
    n_hidden = int(cfg.get("hidden_layers", 2))
    width    = int(cfg.get("width", 8))
    return [width] * n_hidden


def _build_model(cfg: dict, in_dim: int, out_dim: int) -> tuple[Any, list[int]]:
    model_type   = cfg.get("model_type", "mlp")
    hidden_act   = cfg.get("hidden_act", "relu")
    loss_fn      = cfg.get("loss", "bce")
    l2_lambda    = float(cfg.get("l2", 0.0))
    dropout_rate = float(cfg.get("dropout", 0.0))
    seed         = cfg.get("seed")

    # Output activation: explicit choice from the UI wins; otherwise pick a
    # sensible default from the loss.
    out_act = cfg.get("output_act") or (
        "sigmoid" if loss_fn == "bce" else ("softmax" if loss_fn == "cce" else "linear")
    )

    if model_type == "perceptron":
        layers, act_used = [in_dim, out_dim], "linear"
    elif model_type == "perceptron_act":
        layers, act_used = [in_dim, out_dim], hidden_act
    elif model_type == "rbf":
        n_centers = int(cfg.get("rbf_centers", cfg.get("width", 20)))
        model = nne.RBFNetwork(
            in_dim=in_dim, n_centers=n_centers, out_dim=out_dim,
            output_act=out_act, loss=loss_fn, l2_lambda=l2_lambda,
            seed=int(seed) if seed is not None else None,
        )
        return model, [in_dim, n_centers, out_dim]
    else:  # mlp
        widths = _resolve_widths(cfg)
        layers, act_used = [in_dim] + widths + [out_dim], hidden_act

    model = nne.MLP(
        layers=layers, hidden_act=act_used, output_act=out_act, loss=loss_fn,
        l2_lambda=l2_lambda, dropout_rate=dropout_rate,
        seed=int(seed) if seed is not None else None,
    )
    return model, layers


def _split(X: np.ndarray, y: np.ndarray, test_split: float):
    n = len(X)
    n_test = max(10, int(n * test_split))
    idx = np.random.permutation(n)
    return (
        X[idx[n_test:]], y[idx[n_test:]],
        X[idx[:n_test]], y[idx[:n_test]],
    )


def _weight_snapshot(model: nne.MLP) -> list[list[list[float]]]:
    """Compact representation of weights for the network viz (rounded)."""
    return [np.round(W, 4).tolist() for W in model.W]


@app.route("/api/train", methods=["POST"])
def train_route():
    cfg = request.get_json(force=True)
    X = np.asarray(cfg["X"], dtype=np.float32)
    y = np.asarray(cfg["y"], dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    test_split = float(cfg.get("test_split", 0.2))
    epochs     = int(cfg.get("epochs", 200))
    lr         = float(cfg.get("lr", 0.01))
    batch_size = int(cfg.get("batch_size", 32))
    es_pat     = int(cfg.get("early_stopping_patience", 0))
    seed       = cfg.get("seed")
    if seed is not None:
        np.random.seed(int(seed))

    X_tr, y_tr, X_te, y_te = _split(X, y, test_split)
    model, layers = _build_model(cfg, X_tr.shape[1], y_tr.shape[1])

    # RBF needs centers chosen from the full training set, not a single batch.
    if isinstance(model, nne.RBFNetwork):
        model.fit_centers(X_tr)
        layers = list(model.layers)

    sid = uuid.uuid4().hex[:12]

    def generate():
        yield json.dumps({
            "type": "init",
            "session_id": sid,
            "layers": layers,
            "n_train": int(X_tr.shape[0]),
            "n_test": int(X_te.shape[0]),
            "weights": _weight_snapshot(model),
        }) + "\n"

        # Custom training loop so we can stream weight snapshots periodically.
        loss_fn, _ = nne.LOSSES[model.loss_name]
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val, patience_counter = np.inf, 0
        best_W = [w.copy() for w in model.W]
        best_b = [b.copy() for b in model.b]

        n = X_tr.shape[0]
        snapshot_every = max(1, epochs // 30)  # cap weight snapshots to ~30

        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            X_sh, y_sh = X_tr[idx], y_tr[idx]
            for start in range(0, n, batch_size):
                Xb = X_sh[start:start + batch_size]
                yb = y_sh[start:start + batch_size]
                model.forward(Xb, training=True)
                dW, db = model.backward(yb)
                model.apply_gradients(dW, db, lr)

            y_tr_pred  = model.predict(X_tr)
            y_val_pred = model.predict(X_te)
            t_loss = loss_fn(y_tr_pred, y_tr)
            v_loss = loss_fn(y_val_pred, y_te)

            if model.loss_name != "mse":
                if y_tr.shape[1] == 1:
                    t_acc = float(np.mean((y_tr_pred >= 0.5).astype(int) == y_tr))
                    v_acc = float(np.mean((y_val_pred >= 0.5).astype(int) == y_te))
                else:
                    t_acc = float(np.mean(np.argmax(y_tr_pred, 1) == np.argmax(y_tr, 1)))
                    v_acc = float(np.mean(np.argmax(y_val_pred, 1) == np.argmax(y_te, 1)))
            else:
                t_acc = v_acc = None

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["train_acc"].append(t_acc)
            history["val_acc"].append(v_acc)

            payload: dict[str, Any] = {
                "type": "epoch",
                "epoch": epoch, "total": epochs,
                "train_loss": t_loss, "val_loss": v_loss,
                "train_acc": t_acc, "val_acc": v_acc,
            }
            if epoch == 1 or epoch == epochs or epoch % snapshot_every == 0:
                payload["weights"] = _weight_snapshot(model)
            yield json.dumps(payload) + "\n"

            if es_pat > 0:
                if v_loss < best_val - 1e-4:
                    best_val = v_loss
                    patience_counter = 0
                    best_W = [w.copy() for w in model.W]
                    best_b = [b.copy() for b in model.b]
                else:
                    patience_counter += 1
                    if patience_counter >= es_pat:
                        model.W, model.b = best_W, best_b
                        history["stopped_early"] = epoch
                        break

        MODEL_STORE[sid] = {
            "model": model,
            "X_train": X_tr, "y_train": y_tr,
            "X_test":  X_te, "y_test":  y_te,
            "history": history,
            "config": cfg,
            "created": time.time(),
        }
        # Keep store from growing unboundedly.
        if len(MODEL_STORE) > 12:
            oldest = sorted(MODEL_STORE.items(), key=lambda kv: kv[1]["created"])[0][0]
            MODEL_STORE.pop(oldest, None)

        yield json.dumps({
            "type": "done",
            "session_id": sid,
            "history": history,
            "weights": _weight_snapshot(model),
        }) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


# ── Boundary, results, predictions ───────────────────────────────────────────

@app.route("/api/boundary/<sid>")
def boundary(sid: str):
    if sid not in MODEL_STORE:
        return jsonify({"error": "session expired"}), 404
    entry = MODEL_STORE[sid]
    model: nne.MLP = entry["model"]
    X_tr = entry["X_train"]
    X_te = entry["X_test"]
    if X_tr.shape[1] != 2:
        return jsonify({"error": "boundary requires 2D input"}), 400

    res = int(request.args.get("res", 80))
    res = max(20, min(160, res))

    X_all = np.vstack([X_tr, X_te])
    x0_min, x0_max = float(X_all[:, 0].min() - 0.4), float(X_all[:, 0].max() + 0.4)
    x1_min, x1_max = float(X_all[:, 1].min() - 0.4), float(X_all[:, 1].max() + 0.4)
    xx, yy = np.meshgrid(np.linspace(x0_min, x0_max, res),
                         np.linspace(x1_min, x1_max, res))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    Z = model.predict(grid)
    if Z.shape[1] == 1:
        Z = Z.reshape(res, res)
    else:
        Z = np.argmax(Z, axis=1).reshape(res, res).astype(float) / max(1, Z.shape[1] - 1)

    return jsonify({
        "grid": Z.tolist(),
        "extent": [x0_min, x0_max, x1_min, x1_max],
        "resolution": res,
        "train": {"X": X_tr.tolist(), "y": entry["y_train"].tolist()},
        "test":  {"X": X_te.tolist(), "y": entry["y_test"].tolist()},
    })


@app.route("/api/landscape/<sid>")
def landscape(sid: str):
    """3D loss surface around the trained weights."""
    if sid not in MODEL_STORE:
        return jsonify({"error": "session expired"}), 404
    entry = MODEL_STORE[sid]
    model = entry["model"]
    res  = int(request.args.get("res", 21))
    res  = max(11, min(31, res))
    span = float(request.args.get("span", 1.0))
    span = max(0.1, min(3.0, span))

    alphas, betas, Z = nne.loss_landscape(
        model, entry["X_train"], entry["y_train"], span=span, res=res, seed=0,
    )
    return jsonify({"alpha": alphas, "beta": betas, "loss": Z})


@app.route("/api/results/<sid>")
def results(sid: str):
    if sid not in MODEL_STORE:
        return jsonify({"error": "session expired"}), 404
    entry = MODEL_STORE[sid]
    model: nne.MLP = entry["model"]
    X_te = entry["X_test"]
    y_te = entry["y_test"]

    out: dict[str, Any] = {
        "loss_name": model.loss_name,
        "history":   entry["history"],
        "layers":    model.layers,
    }

    if model.loss_name == "mse":
        y_pred = model.predict(X_te)
        out["regression"] = nne.regression_metrics(y_te, y_pred)
    else:
        y_pred_cls = model.predict_classes(X_te)
        n_cls = y_te.shape[1] if y_te.ndim > 1 and y_te.shape[1] > 1 else 2
        y_true_cls = y_te.ravel() if y_te.shape[1] == 1 else np.argmax(y_te, 1)
        out["classification"] = nne.classification_report_dict(
            y_true_cls, y_pred_cls, n_classes=max(n_cls, 2))
    return jsonify(out)


# ── Experiments ──────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "perceptron_vs_mlp": {
        "label": "Perceptron vs MLP",
        "configs": [
            {"name": "Perceptron",   "layers": [2, 1],         "act": "linear"},
            {"name": "Perceptron+σ", "layers": [2, 1],         "act": "sigmoid"},
            {"name": "MLP 1 hidden", "layers": [2, 8, 1],      "act": "relu"},
            {"name": "MLP 2 hidden", "layers": [2, 16, 16, 1], "act": "relu"},
        ],
    },
    "depth": {
        "label": "Effect of depth",
        "configs": [
            {"name": f"{d} hidden", "layers": [2] + [8] * d + [1], "act": "relu"}
            for d in range(1, 6)
        ],
    },
    "regularisation": {
        "label": "Overfitting vs regularisation",
        "configs": [
            {"name": "No reg",       "layers": [2, 32, 32, 1], "act": "relu", "l2": 0.0,   "dropout": 0.0},
            {"name": "L2 λ=0.001",   "layers": [2, 32, 32, 1], "act": "relu", "l2": 0.001, "dropout": 0.0},
            {"name": "Dropout 30%",  "layers": [2, 32, 32, 1], "act": "relu", "l2": 0.0,   "dropout": 0.3},
        ],
    },
}


def _sklearn_baseline(layers: list[int], act: str, X_tr, y_tr, X_te, y_te) -> dict | None:
    """Train a comparable scikit-learn model on the same split.

    Maps our config to the closest sklearn estimator:
      - [in, out] with linear act     -> Perceptron
      - [in, out] with sigmoid/relu/tanh -> LogisticRegression
      - [in, ..., out]                -> MLPClassifier (same hidden sizes)
    Returns {"acc": float, "loss": float} or None if comparison fails.
    """
    try:
        y_tr_flat = y_tr.ravel().astype(int)
        y_te_flat = y_te.ravel().astype(int)
        if len(layers) == 2:
            if act == "linear":
                from sklearn.linear_model import Perceptron as SkPerceptron
                clf = SkPerceptron(max_iter=200, tol=1e-4, random_state=0)
            else:
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(max_iter=200, random_state=0)
        else:
            from sklearn.neural_network import MLPClassifier
            sk_act = {"sigmoid": "logistic", "tanh": "tanh", "relu": "relu"}.get(act, "relu")
            clf = MLPClassifier(
                hidden_layer_sizes=tuple(layers[1:-1]),
                activation=sk_act, solver="sgd",
                learning_rate_init=0.01, max_iter=120,
                batch_size=32, random_state=0,
            )
        clf.fit(X_tr, y_tr_flat)
        acc = float(clf.score(X_te, y_te_flat))
        # Approximate held-out BCE: use predict_proba where available, else hinge-ish.
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(X_te)[:, 1]
            p = np.clip(p, 1e-9, 1 - 1e-9)
            loss = float(-np.mean(y_te_flat * np.log(p) + (1 - y_te_flat) * np.log(1 - p)))
        else:
            preds = clf.predict(X_te)
            loss = float(np.mean(preds != y_te_flat))
        return {"acc": acc, "loss": loss}
    except Exception as e:
        return {"acc": None, "loss": None, "error": str(e)}


@app.route("/api/experiments")
def list_experiments():
    return jsonify([
        {"id": k, "label": v["label"], "n_configs": len(v["configs"])}
        for k, v in EXPERIMENTS.items()
    ])


@app.route("/api/experiment", methods=["POST"])
def run_experiment():
    body = request.get_json(force=True)
    exp_id = body.get("id", "perceptron_vs_mlp")
    if exp_id not in EXPERIMENTS:
        return jsonify({"error": "unknown experiment"}), 400

    np.random.seed(7)
    X, y = nne.make_circles(n=300, noise=0.1)
    n = len(X); idx = np.random.permutation(n); sp = int(n * 0.8)
    X_tr, y_tr = X[idx[:sp]], y[idx[:sp]]
    X_te, y_te = X[idx[sp:]], y[idx[sp:]]

    def generate():
        yield json.dumps({"type": "start", "n_configs": len(EXPERIMENTS[exp_id]["configs"])}) + "\n"
        results = []
        for i, c in enumerate(EXPERIMENTS[exp_id]["configs"]):
            m = nne.MLP(
                layers=c["layers"], hidden_act=c["act"], output_act="sigmoid",
                loss="bce", l2_lambda=c.get("l2", 0.0), dropout_rate=c.get("dropout", 0.0),
            )
            nne.train(m, X_tr, y_tr, X_te, y_te, lr=0.01, epochs=120, batch_size=32)
            acc = float(np.mean(m.predict_classes(X_te) == y_te.ravel()))
            lv  = nne.bce_loss(m.predict(X_te), y_te)

            # Boundary grid for this trained model, for the cards.
            res = 60
            x0_min, x0_max = float(X[:, 0].min() - 0.4), float(X[:, 0].max() + 0.4)
            x1_min, x1_max = float(X[:, 1].min() - 0.4), float(X[:, 1].max() + 0.4)
            xx, yy = np.meshgrid(np.linspace(x0_min, x0_max, res),
                                 np.linspace(x1_min, x1_max, res))
            grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
            Z = m.predict(grid).reshape(res, res)

            sk = _sklearn_baseline(c["layers"], c["act"], X_tr, y_tr, X_te, y_te)

            results.append({
                "name": c["name"], "acc": acc, "loss": lv,
                "layers": c["layers"],
                "sklearn": sk,
                "boundary": {
                    "grid": Z.tolist(),
                    "extent": [x0_min, x0_max, x1_min, x1_max],
                    "resolution": res,
                },
            })
            yield json.dumps({"type": "progress", "i": i + 1, "name": c["name"]}) + "\n"

        yield json.dumps({
            "type": "done",
            "results": results,
            "scatter": {"X": X_te.tolist(), "y": y_te.tolist()},
        }) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
