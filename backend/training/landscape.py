"""Loss-landscape sampler.

Picks two random orthogonal directions in flattened weight space and evaluates
the loss on a grid of perturbations around the trained weights. Useful for
visualising the bowl/saddle shape near the solution.
"""
import numpy as np

from ..utils.loss import LOSSES


def _filter_normalised_direction(rng, params):
    """Make a random direction whose per-parameter norm matches that parameter's norm.

    This is the "filter normalisation" trick from Li et al. (2018) — stops one
    huge weight matrix from dominating the perturbation and keeps the surface
    well-conditioned across architectures.
    """
    out = []
    for P in params:
        d = rng.randn(*P.shape).astype(np.float32)
        scale = (np.linalg.norm(P) + 1e-9) / (np.linalg.norm(d) + 1e-9)
        out.append((d * scale).ravel())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def loss_landscape(model, X, y, span=1.0, res=21, seed=0):
    """Return (alphas, betas, Z) where Z[i, j] is loss at theta + αᵢ d₁ + βⱼ d₂."""
    loss_fn, _ = LOSSES[model.loss_name]

    params = list(model.W) + list(model.b)
    parts = [P.ravel() for P in params]
    theta = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)

    rng = np.random.RandomState(seed)
    d1 = _filter_normalised_direction(rng, params)
    d2 = _filter_normalised_direction(rng, params)
    # Gram-Schmidt for orthogonality, preserving d2's norm.
    if d1.size and float(d1 @ d1) > 0:
        d2_norm = float(np.linalg.norm(d2))
        d2 -= (d2 @ d1) / (d1 @ d1) * d1
        if np.linalg.norm(d2) > 1e-9:
            d2 *= d2_norm / np.linalg.norm(d2)

    alphas = np.linspace(-span, span, res).astype(np.float32)
    betas  = np.linspace(-span, span, res).astype(np.float32)

    saved_W = [W.copy() for W in model.W]
    saved_b = [b.copy() for b in model.b]

    shapes_W = [W.shape for W in model.W]
    sizes_W  = [W.size  for W in model.W]
    shapes_b = [b.shape for b in model.b]
    sizes_b  = [b.size  for b in model.b]

    Z = np.zeros((res, res), dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for i, a in enumerate(alphas):
            for j, b_ in enumerate(betas):
                new = theta + a * d1 + b_ * d2
                off = 0
                for k, (sh, sz) in enumerate(zip(shapes_W, sizes_W)):
                    model.W[k] = new[off:off + sz].reshape(sh).astype(np.float32)
                    off += sz
                for k, (sh, sz) in enumerate(zip(shapes_b, sizes_b)):
                    model.b[k] = new[off:off + sz].reshape(sh).astype(np.float32)
                    off += sz
                y_pred = model.predict(X)
                v = loss_fn(y_pred, y)
                Z[i, j] = v if np.isfinite(v) else np.nan

    # Restore.
    model.W = saved_W
    model.b = saved_b

    # Cap blow-ups so the surface is plottable. Replace nan/inf with a
    # finite ceiling derived from the rest of the grid.
    finite = Z[np.isfinite(Z)]
    cap = float(np.nanmax(finite)) * 1.5 if finite.size else 1.0
    Z = np.where(np.isfinite(Z), Z, cap)

    return alphas.tolist(), betas.tolist(), Z.tolist()
