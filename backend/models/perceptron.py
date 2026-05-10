"""Perceptron — historical (linear) and modern (with activation).

The historical Rosenblatt perceptron is a single-layer linear classifier with
a step output. The modern variant replaces the step with a differentiable
activation so that gradient descent works. Both forms are special cases of the
MLP with no hidden layers, so we reuse MLP rather than duplicating the math.
"""
from .mlp import MLP


class Perceptron(MLP):
    """Single-layer perceptron — kept as a named class for clarity.

    `historical=True` forces a linear unit (closest to the original step
    perceptron once predictions are thresholded); otherwise `activation` is
    used as the (output) activation.
    """

    def __init__(self, in_dim, out_dim=1, historical=True, activation="sigmoid",
                 loss="bce", seed=None):
        out_act = "linear" if historical else activation
        super().__init__(
            layers=[in_dim, out_dim],
            hidden_act="linear",
            output_act=out_act,
            loss=loss,
            seed=seed,
        )
        self.historical = historical
