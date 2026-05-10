"""Training utilities — forward, backward, the gradient-descent loop, and a
loss-landscape sampler for the 3D visualisation."""
from .forward import forward
from .backward import backward
from .optimizer import train, gradient_descent_step
from .landscape import loss_landscape
