"""Forward propagation entry point.

Models own the actual forward math (because it shares state with backward).
This module exists for organisational symmetry with the recommended layout.
"""


def forward(model, X, training=False):
    """Run forward propagation on `model` and return the output activations."""
    return model.forward(X, training=training)
