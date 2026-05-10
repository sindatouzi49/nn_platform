"""Multi-Layer Perceptron with He initialisation, dropout, and L2.

Forward and backward live here as methods because the training loop needs the
intermediate cache. Free functions in `training/forward.py` and
`training/backward.py` re-expose them for code-organisation purposes (the spec
asks for those modules explicitly).
"""
import numpy as np

from ..utils.activation import ACTIVATIONS
from ..utils.loss import LOSSES


class MLP:
    def __init__(self, layers, hidden_act="relu", output_act="sigmoid",
                 loss="bce", l2_lambda=0.0, dropout_rate=0.0, seed=None):
        self.layers       = layers
        self.hidden_act   = hidden_act
        self.output_act   = output_act
        self.loss_name    = loss
        self.l2_lambda    = l2_lambda
        self.dropout_rate = dropout_rate
        if seed is not None:
            np.random.seed(seed)
        self._init_weights()

    def _init_weights(self):
        self.W, self.b = [], []
        for i in range(len(self.layers) - 1):
            fan_in = self.layers[i]
            scale  = np.sqrt(2.0 / fan_in)
            self.W.append(np.random.randn(self.layers[i], self.layers[i + 1]) * scale)
            self.b.append(np.zeros((1, self.layers[i + 1])))

    def forward(self, X, training=False):
        h_fn, _ = ACTIVATIONS[self.hidden_act]
        o_fn, _ = ACTIVATIONS[self.output_act]

        self._cache = {"A": [X], "Z": [], "masks": []}
        A = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A @ W + b
            self._cache["Z"].append(Z)
            is_last = (i == len(self.W) - 1)
            if is_last:
                A = o_fn(Z)
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

    def backward(self, y_true):
        _, d_loss = LOSSES[self.loss_name]
        _, d_h    = ACTIVATIONS[self.hidden_act]

        y_pred = self._cache["A"][-1]
        if self.output_act == "softmax" or self.loss_name == "bce":
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
                dZ = dA
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
            self.b[i] -= lr * db_list[i]

    def predict(self, X):
        return self.forward(X, training=False)

    def predict_classes(self, X):
        p = self.predict(X)
        if p.shape[1] == 1:
            return (p >= 0.5).astype(int).ravel()
        return np.argmax(p, axis=1)
