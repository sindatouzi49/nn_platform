"""Backwards-compatible aggregator.

Historically this module held every piece of math in the project. The code is
now split into `models/`, `training/`, and `utils/` per the spec. This file
re-exports the public surface so existing callers (notably `app.py` and any
notebooks) keep working.
"""
from .models import MLP, Perceptron, RBFNetwork
from .training import train, forward, backward, gradient_descent_step, loss_landscape
from .utils.activation import (
    ACTIVATIONS, sigmoid, relu, tanh_, linear, softmax,
    d_sigmoid, d_relu, d_tanh, d_linear, d_softmax,
)
from .utils.loss import (
    LOSSES, mse_loss, bce_loss, cce_loss,
    d_mse, d_bce, d_cce_softmax,
)
from .utils.metrics import classification_report_dict, regression_metrics
from .utils.datasets import (
    DATASETS,
    make_circles, make_xor, make_moons, make_spiral, make_linear, make_sine,
    make_blobs3, make_quadrants,
)

__all__ = [
    # models
    "MLP", "Perceptron", "RBFNetwork",
    # training
    "train", "forward", "backward", "gradient_descent_step", "loss_landscape",
    # activations
    "ACTIVATIONS", "sigmoid", "relu", "tanh_", "linear", "softmax",
    "d_sigmoid", "d_relu", "d_tanh", "d_linear", "d_softmax",
    # losses
    "LOSSES", "mse_loss", "bce_loss", "cce_loss",
    "d_mse", "d_bce", "d_cce_softmax",
    # metrics
    "classification_report_dict", "regression_metrics",
    # datasets
    "DATASETS",
    "make_circles", "make_xor", "make_moons", "make_spiral", "make_linear", "make_sine",
    "make_blobs3", "make_quadrants",
]
