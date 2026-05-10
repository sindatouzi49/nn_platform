"""Activation functions and their derivatives."""
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def relu(z):
    return np.maximum(0, z)


def tanh_(z):
    return np.tanh(z)


def linear(z):
    return z


def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)


def d_relu(z):
    return (z > 0).astype(float)


def d_tanh(z):
    return 1 - np.tanh(z) ** 2


def d_linear(z):
    return np.ones_like(z)


def d_softmax(z):
    """Pseudo-derivative for softmax.

    The true Jacobian is non-diagonal, but in practice softmax is only used as
    the output activation paired with categorical cross-entropy — and the
    composite gradient simplifies to (ŷ − y) / N, computed directly in
    `MLP.backward`. We expose ones here so the registry stays consistent and
    nothing breaks if someone wires softmax in unusually.
    """
    return np.ones_like(z)


ACTIVATIONS = {
    "relu":    (relu,    d_relu),
    "sigmoid": (sigmoid, d_sigmoid),
    "tanh":    (tanh_,   d_tanh),
    "linear":  (linear,  d_linear),
    "softmax": (softmax, d_softmax),
}
