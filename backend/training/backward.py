"""Backpropagation entry point.

Models own the chain-rule logic so they can reuse the forward cache. This
module exposes the same call as a free function for organisational symmetry.
"""


def backward(model, y_true):
    """Compute (dW_list, db_list) for `model` given the most recent forward."""
    return model.backward(y_true)
