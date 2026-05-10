"""Stateless helpers: activations, losses, metrics, dataset generators."""
from .activation import ACTIVATIONS, sigmoid, relu, tanh_, linear, softmax
from .loss import LOSSES, mse_loss, bce_loss, cce_loss
from .metrics import classification_report_dict, regression_metrics
from .datasets import (
    DATASETS,
    make_circles, make_xor, make_moons, make_spiral, make_linear, make_sine,
    make_blobs3, make_quadrants,
)
