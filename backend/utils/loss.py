"""Loss functions and their derivatives."""
import numpy as np


def mse_loss(y_pred, y_true):
    return float(np.mean((y_pred - y_true) ** 2))


def bce_loss(y_pred, y_true):
    p = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def cce_loss(y_pred, y_true):
    p = np.clip(y_pred, 1e-9, 1.0)
    return float(-np.mean(np.sum(y_true * np.log(p), axis=1)))


def d_mse(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.shape[0]


def d_bce(y_pred, y_true):
    p = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return (-(y_true / p) + (1 - y_true) / (1 - p)) / y_true.shape[0]


def d_cce_softmax(y_pred, y_true):
    return (y_pred - y_true) / y_true.shape[0]


LOSSES = {
    "mse": (mse_loss, d_mse),
    "bce": (bce_loss, d_bce),
    "cce": (cce_loss, d_cce_softmax),
}
