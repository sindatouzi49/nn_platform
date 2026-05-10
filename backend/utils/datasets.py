"""Synthetic dataset generators used by the platform."""
import numpy as np


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
    y  = np.array([0] * h + [1] * (n - h), dtype=np.float32).reshape(-1, 1)
    mu, std = X.mean(0), X.std(0) + 1e-9
    return ((X - mu) / std).astype(np.float32), y


def make_spiral(n=300, noise=0.08):
    """Two intertwined spirals — classic 'hard' nonlinear classification."""
    h = n // 2
    t = np.linspace(0.2, 3.5, h)
    X1 = np.column_stack([t * np.cos(2 * t), t * np.sin(2 * t)])
    X2 = np.column_stack([t * np.cos(2 * t + np.pi), t * np.sin(2 * t + np.pi)])
    X  = np.vstack([X1, X2]) / 3.5 + np.random.randn(n, 2) * noise
    y  = np.array([0] * h + [1] * (n - h), dtype=np.float32).reshape(-1, 1)
    return X.astype(np.float32), y


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


def make_blobs3(n=300, noise=0.18):
    """Three Gaussian clusters arranged in a triangle. One-hot labels."""
    centers = np.array([[0.0, 1.0], [-1.0, -0.6], [1.0, -0.6]], dtype=np.float32)
    counts  = [n // 3, n // 3, n - 2 * (n // 3)]
    X_parts, y_parts = [], []
    for k, (c, cnt) in enumerate(zip(centers, counts)):
        X_parts.append(np.random.randn(cnt, 2).astype(np.float32) * noise + c)
        y_parts.append(np.full(cnt, k, dtype=int))
    X = np.vstack(X_parts)
    y_idx = np.concatenate(y_parts)
    y = np.zeros((n, 3), dtype=np.float32)
    y[np.arange(n), y_idx] = 1.0
    perm = np.random.permutation(n)
    return X[perm], y[perm]


def make_quadrants(n=400, noise=0.12):
    """Four-class problem — one cluster per quadrant. Forces softmax."""
    centers = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float32) * 0.7
    per     = n // 4
    X_parts, y_parts = [], []
    for k, c in enumerate(centers):
        cnt = per if k < 3 else n - 3 * per
        X_parts.append(np.random.randn(cnt, 2).astype(np.float32) * noise + c)
        y_parts.append(np.full(cnt, k, dtype=int))
    X = np.vstack(X_parts)
    y_idx = np.concatenate(y_parts)
    y = np.zeros((n, 4), dtype=np.float32)
    y[np.arange(n), y_idx] = 1.0
    perm = np.random.permutation(n)
    return X[perm], y[perm]


DATASETS = {
    "circles":   {"label": "Concentric circles", "fn": make_circles,   "type": "binary"},
    "xor":       {"label": "XOR pattern",        "fn": make_xor,       "type": "binary"},
    "moons":     {"label": "Half moons",         "fn": make_moons,     "type": "binary"},
    "spiral":    {"label": "Two spirals",        "fn": make_spiral,    "type": "binary"},
    "linear":    {"label": "Linearly separable", "fn": make_linear,    "type": "binary"},
    "blobs3":    {"label": "3 blobs (multi)",    "fn": make_blobs3,    "type": "multiclass"},
    "quadrants": {"label": "4 quadrants (multi)","fn": make_quadrants, "type": "multiclass"},
    "sine":      {"label": "Sine wave",          "fn": make_sine,      "type": "regression"},
}
