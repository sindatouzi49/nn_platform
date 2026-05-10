import numpy as np

# ── Activations ──────────────────────────────────────────────────────────────

def sigmoid(z):    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def relu(z):       return np.maximum(0, z)
def tanh_(z):      return np.tanh(z)
def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
def linear(z):     return z

def d_sigmoid(z):  s = sigmoid(z); return s * (1 - s)
def d_relu(z):     return (z > 0).astype(float)
def d_tanh(z):     return 1 - np.tanh(z) ** 2
def d_linear(z):   return np.ones_like(z)

ACTIVATIONS = {
    "relu":    (relu,    d_relu),
    "sigmoid": (sigmoid, d_sigmoid),
    "tanh":    (tanh_,   d_tanh),
    "linear":  (linear,  d_linear),
}

# ── Loss functions ────────────────────────────────────────────────────────────

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def bce_loss(y_pred, y_true):
    p = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

def cce_loss(y_pred, y_true):
    p = np.clip(y_pred, 1e-9, 1.0)
    return -np.mean(np.sum(y_true * np.log(p), axis=1))

def d_mse(y_pred, y_true):      return 2 * (y_pred - y_true) / y_true.shape[0]
def d_bce(y_pred, y_true):
    p = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return (-(y_true / p) + (1 - y_true) / (1 - p)) / y_true.shape[0]
def d_cce_softmax(y_pred, y_true): return (y_pred - y_true) / y_true.shape[0]

LOSSES = {
    "mse":  (mse_loss,  d_mse),
    "bce":  (bce_loss,  d_bce),
    "cce":  (cce_loss,  d_cce_softmax),
}

# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP:
    """
    General multi-layer perceptron.
    layers      : list of ints, e.g. [2, 8, 8, 1]
    hidden_act  : 'relu' | 'sigmoid' | 'tanh'
    output_act  : 'sigmoid' | 'softmax' | 'linear'
    loss        : 'mse' | 'bce' | 'cce'
    l2_lambda   : L2 regularisation coefficient
    dropout_rate: fraction of neurons to zero during training (0 = disabled)
    """

    def __init__(self, layers, hidden_act="relu", output_act="sigmoid",
                 loss="bce", l2_lambda=0.0, dropout_rate=0.0):
        self.layers       = layers
        self.hidden_act   = hidden_act
        self.output_act   = output_act
        self.loss_name    = loss
        self.l2_lambda    = l2_lambda
        self.dropout_rate = dropout_rate
        self._init_weights()

    def _init_weights(self):
        self.W, self.b = [], []
        for i in range(len(self.layers) - 1):
            fan_in = self.layers[i]
            scale  = np.sqrt(2.0 / fan_in)
            self.W.append(np.random.randn(self.layers[i], self.layers[i + 1]) * scale)
            self.b.append(np.zeros((1, self.layers[i + 1])))

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, X, training=False):
        h_fn, _ = ACTIVATIONS[self.hidden_act]
        o_fn, _ = ACTIVATIONS[self.output_act] if self.output_act != "softmax" else (softmax, None)

        self._cache = {"A": [X], "Z": [], "masks": []}
        A = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A @ W + b
            self._cache["Z"].append(Z)
            is_last = (i == len(self.W) - 1)
            if is_last:
                A = softmax(Z) if self.output_act == "softmax" else o_fn(Z)
                self._cache["masks"].append(None)
            else:
                A = h_fn(Z)
                if training and self.dropout_rate > 0:
                    mask = (np.random.rand(*A.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                    A   *= mask
                    self._cache["masks"].append(mask)
                else:
                    self._cache["masks"].append(None)
            self._cache["A"].append(A)
        return A

    # ── Backward pass ─────────────────────────────────────────────────────────

    def backward(self, y_true):
        _, d_loss = LOSSES[self.loss_name]
        _, d_h    = ACTIVATIONS[self.hidden_act]

        y_pred = self._cache["A"][-1]
        # output delta
        if self.output_act == "softmax" or self.loss_name in ("bce",):
            dA = d_loss(y_pred, y_true)
        else:
            _, d_o = ACTIVATIONS[self.output_act]
            dA     = d_loss(y_pred, y_true) * d_o(self._cache["Z"][-1])

        dW_list, db_list = [], []
        for i in reversed(range(len(self.W))):
            A_prev = self._cache["A"][i]
            Z      = self._cache["Z"][i]
            mask   = self._cache["masks"][i]

            if i == len(self.W) - 1:
                dZ = dA  # already correct for output
            else:
                dZ = dA * d_h(Z)
                if mask is not None:
                    dZ *= mask

            dW = A_prev.T @ dZ + self.l2_lambda * self.W[i]
            db = dZ.sum(axis=0, keepdims=True)
            dW_list.insert(0, dW)
            db_list.insert(0, db)

            if i > 0:
                dA = dZ @ self.W[i].T

        return dW_list, db_list

    def apply_gradients(self, dW_list, db_list, lr):
        for i in range(len(self.W)):
            self.W[i] -= lr * dW_list[i]
            self.b[i]  -= lr * db_list[i]

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X):
        return self.forward(X, training=False)

    def predict_classes(self, X):
        p = self.predict(X)
        if p.shape[1] == 1:
            return (p >= 0.5).astype(int).ravel()
        return np.argmax(p, axis=1)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, X_train, y_train, X_val, y_val,
          lr=0.01, epochs=200, batch_size=32,
          early_stopping_patience=0,
          progress_callback=None):
    """
    Returns history dict with keys: train_loss, val_loss, train_acc, val_acc.
    progress_callback(epoch, total, metrics_dict) called every epoch.
    """
    loss_fn, _ = LOSSES[model.loss_name]
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, patience_counter = np.inf, 0
    best_W = [w.copy() for w in model.W]
    best_b = [b.copy() for b in model.b]

    n = X_train.shape[0]

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(n)
        X_sh, y_sh = X_train[idx], y_train[idx]

        # mini-batch SGD
        for start in range(0, n, batch_size):
            Xb = X_sh[start:start + batch_size]
            yb = y_sh[start:start + batch_size]
            model.forward(Xb, training=True)
            dW, db = model.backward(yb)
            model.apply_gradients(dW, db, lr)

        # metrics
        y_tr_pred  = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        t_loss = loss_fn(y_tr_pred, y_train)
        v_loss = loss_fn(y_val_pred, y_val)

        # accuracy (skip for regression)
        if model.loss_name != "mse":
            if y_train.shape[1] == 1:
                t_acc = np.mean((y_tr_pred >= 0.5).astype(int) == y_train)
                v_acc = np.mean((y_val_pred >= 0.5).astype(int) == y_val)
            else:
                t_acc = np.mean(np.argmax(y_tr_pred, 1) == np.argmax(y_train, 1))
                v_acc = np.mean(np.argmax(y_val_pred, 1) == np.argmax(y_val, 1))
        else:
            t_acc = v_acc = None

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        if progress_callback:
            progress_callback(epoch, epochs, {"t_loss": t_loss, "v_loss": v_loss,
                                               "t_acc": t_acc, "v_acc": v_acc})

        # early stopping
        if early_stopping_patience > 0:
            if v_loss < best_val - 1e-4:
                best_val = v_loss
                patience_counter = 0
                best_W = [w.copy() for w in model.W]
                best_b = [b.copy() for b in model.b]
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    model.W, model.b = best_W, best_b
                    break

    return history


# ── Metrics ───────────────────────────────────────────────────────────────────

def classification_report_dict(y_true, y_pred_classes, n_classes=2):
    """Returns dict with accuracy, per-class precision/recall/f1, confusion matrix."""
    y_t = y_true.ravel() if y_true.ndim > 1 else y_true
    acc = np.mean(y_t == y_pred_classes)
    cm  = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_t, y_pred_classes):
        cm[int(t)][int(p)] += 1

    per_class = {}
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec   = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1     = 2 * prec * recall / (prec + recall + 1e-9)
        per_class[c] = {"precision": prec, "recall": recall, "f1": f1, "support": int(cm[c].sum())}

    return {"accuracy": acc, "per_class": per_class, "confusion_matrix": cm}

def regression_metrics(y_true, y_pred):
    mse  = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2   = 1 - ss_res / (ss_tot + 1e-9)
    return {"mse": mse, "rmse": rmse, "r2": r2}


# ── Data generators ───────────────────────────────────────────────────────────

def make_circles(n=300, noise=0.1):
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii  = np.where(np.random.rand(n) < 0.5, 0.35, 0.75)
    X = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    X += np.random.randn(n, 2) * noise
    y = (radii < 0.55).astype(int).reshape(-1, 1)
    return X.astype(np.float32), y.astype(np.float32)

def make_xor(n=300, noise=0.1):
    X = np.random.uniform(-1, 1, (n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int).reshape(-1, 1)
    X += np.random.randn(n, 2) * noise
    return X.astype(np.float32), y.astype(np.float32)

def make_moons(n=300, noise=0.15):
    h = n // 2
    t1 = np.linspace(0, np.pi, h)
    t2 = np.linspace(0, np.pi, n - h)
    X1 = np.column_stack([np.cos(t1), np.sin(t1)])
    X2 = np.column_stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5])
    X  = np.vstack([X1, X2]) + np.random.randn(n, 2) * noise
    y  = np.array([0]*h + [1]*(n-h), dtype=np.float32).reshape(-1, 1)
    mu, std = X.mean(0), X.std(0) + 1e-9
    return ((X - mu) / std).astype(np.float32), y

def make_linear(n=300, noise=0.1):
    X = np.random.uniform(-1, 1, (n, 2)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] + np.random.randn(n) * noise) > 0).astype(np.float32).reshape(-1, 1)
    return X, y

def make_sine(n=300, noise=0.05):
    x = np.random.uniform(-np.pi, np.pi, n)
    y = np.sin(x) + np.random.randn(n) * noise
    X = (x / np.pi).reshape(-1, 1).astype(np.float32)
    y = ((y + 1) / 2).reshape(-1, 1).astype(np.float32)
    return X, y

DATASETS = {
    "Concentric circles": make_circles,
    "XOR pattern":        make_xor,
    "Half moons":         make_moons,
    "Linearly separable": make_linear,
    "Sine wave (regression)": make_sine,
}
