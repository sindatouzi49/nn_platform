"""Radial Basis Function network.

Two-layer model:
    1. RBF hidden layer with K Gaussian units. Centers are picked from the
       training set; bandwidth σ is set from the median pairwise distance.
       These parameters are fixed (non-trainable) for educational simplicity.
    2. Linear output layer trained by gradient descent against the chosen loss.

Exposes the same `forward / backward / apply_gradients / predict` interface
as MLP so the shared training loop in `training.optimizer` can drive it.
"""
import numpy as np

from ..utils.activation import softmax
from ..utils.loss import LOSSES


class RBFNetwork:
    def __init__(self, in_dim, n_centers=20, out_dim=1, output_act="sigmoid",
                 loss="bce", l2_lambda=0.0, seed=None):
        self.in_dim     = in_dim
        self.n_centers  = n_centers
        self.out_dim    = out_dim
        self.output_act = output_act
        self.loss_name  = loss
        self.l2_lambda  = l2_lambda
        self.dropout_rate = 0.0  # not supported on RBF — kept for API symmetry
        self.layers     = [in_dim, n_centers, out_dim]

        if seed is not None:
            np.random.seed(seed)

        # Centers and bandwidth are placeholders until fit_centers() is called.
        self.centers = None
        self.sigma   = 1.0

        # Trainable head: φ(x) → ŷ
        scale = np.sqrt(2.0 / max(1, n_centers))
        self.W = [np.random.randn(n_centers, out_dim).astype(np.float32) * scale]
        self.b = [np.zeros((1, out_dim), dtype=np.float32)]

    def fit_centers(self, X):
        """Pick centers from X and set σ from the median pairwise distance."""
        n = X.shape[0]
        k = min(self.n_centers, n)
        idx = np.random.choice(n, size=k, replace=False)
        self.centers = X[idx].astype(np.float32)

        # Median pairwise distance between centers — scale-aware bandwidth.
        if k >= 2:
            diffs = self.centers[:, None, :] - self.centers[None, :, :]
            dists = np.sqrt((diffs ** 2).sum(-1))
            iu = np.triu_indices(k, k=1)
            med = float(np.median(dists[iu])) if iu[0].size else 1.0
            self.sigma = max(med, 1e-3)
        else:
            self.sigma = 1.0

        # If we asked for more centers than samples, shrink to what we have.
        if k != self.n_centers:
            self.n_centers = k
            self.layers    = [self.in_dim, k, self.out_dim]
            scale = np.sqrt(2.0 / max(1, k))
            self.W = [np.random.randn(k, self.out_dim).astype(np.float32) * scale]
            self.b = [np.zeros((1, self.out_dim), dtype=np.float32)]

    def _phi(self, X):
        # ||x - c||² for every (sample, center) pair.
        diffs = X[:, None, :] - self.centers[None, :, :]
        d2    = (diffs ** 2).sum(-1)
        return np.exp(-d2 / (2 * self.sigma ** 2 + 1e-9))

    def forward(self, X, training=False):
        if self.centers is None:
            self.fit_centers(X)
        phi = self._phi(X)
        Z = phi @ self.W[0] + self.b[0]
        if self.output_act == "softmax":
            A = softmax(Z)
        elif self.output_act == "sigmoid":
            A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        else:  # linear
            A = Z
        self._cache = {"phi": phi, "Z": Z, "A": A}
        return A

    def backward(self, y_true):
        _, d_loss = LOSSES[self.loss_name]
        y_pred = self._cache["A"]
        phi    = self._cache["phi"]

        # bce-with-sigmoid and cce-with-softmax both reduce to (ŷ - y)/N.
        if self.output_act == "softmax" or self.loss_name == "bce":
            dZ = d_loss(y_pred, y_true)
        elif self.output_act == "sigmoid":
            s  = y_pred
            dZ = d_loss(y_pred, y_true) * s * (1 - s)
        else:  # linear output
            dZ = d_loss(y_pred, y_true)

        dW = phi.T @ dZ + self.l2_lambda * self.W[0]
        db = dZ.sum(axis=0, keepdims=True)
        return [dW], [db]

    def apply_gradients(self, dW_list, db_list, lr):
        self.W[0] -= lr * dW_list[0]
        self.b[0] -= lr * db_list[0]

    def predict(self, X):
        return self.forward(X, training=False)

    def predict_classes(self, X):
        p = self.predict(X)
        if p.shape[1] == 1:
            return (p >= 0.5).astype(int).ravel()
        return np.argmax(p, axis=1)
