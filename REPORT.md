# Neural Networks — Educational Platform

> *"You are not just implementing neural networks — you are building a platform that teaches how they work."*

This report accompanies **Neuralab**, a single-page web application for learning, experimenting with, and visualising neural networks. The whole engine is implemented from scratch in NumPy — no PyTorch, no TensorFlow, no scikit-learn for the models themselves — so every line of math is inspectable.

---

## 1. Theoretical foundation

### 1.1 The perceptron and its limits

A perceptron computes `ŷ = f(w · x + b)`. Rosenblatt’s 1958 form used a step activation; modern variants use a differentiable activation so gradient descent applies. A single perceptron only separates **linearly separable** classes — XOR is its canonical failure mode.

### 1.2 Multi-layer perceptron (MLP)

Stacking `h⁽ˡ⁾ = σ(W⁽ˡ⁾ h⁽ˡ⁻¹⁾ + b⁽ˡ⁾)` gives a feed-forward network that, with at least one hidden layer and a non-polynomial activation, is a *universal approximator* (Hornik et al., 1989). Two design knobs:

- **Depth** — composes hierarchical features.
- **Width** — increases per-layer capacity.

### 1.3 RBF networks

Radial Basis Function networks place K Gaussian centers `cₖ` and compute `φₖ(x) = exp(−‖x − cₖ‖² / 2σ²)`. The hidden layer is *fixed* (centers chosen from data, σ from the median pairwise distance); only a linear output head is trained. This makes RBF fast to train and easy to reason about, but it scales poorly with input dimensionality.

### 1.4 Activations

| name | range | typical use |
|---|---|---|
| sigmoid | (0,1) | binary output |
| tanh | (−1,1) | hidden, saturating |
| ReLU | [0,∞) | hidden, default |
| softmax | (0,1) | multi-class output |

### 1.5 Loss functions

| problem | loss | formula |
|---|---|---|
| regression | MSE | `Σ(y−ŷ)² / n` |
| binary classification | BCE | `−[y·log(ŷ) + (1−y)·log(1−ŷ)]` |
| multi-class | CCE | `−Σ yᵢ · log(ŷᵢ)` |

### 1.6 Backpropagation

Backprop applies the chain rule layer by layer:

```
∂L/∂Wⁱ = ∂L/∂ŷ · ∂ŷ/∂z⁽ⁱ⁺¹⁾ · … · ∂h⁽ⁱ⁾/∂z⁽ⁱ⁾ · x
```

Each weight is updated `w ← w − η · ∂L/∂w`. Mini-batch SGD averages the gradient over a small batch — a noisy but cheap estimator that often generalises better than full-batch GD.

### 1.7 Regularisation

| technique | mechanism |
|---|---|
| L2 | adds `λ Σ w²` to the loss; penalises large weights |
| Dropout | randomly zeros activations during training |
| Early stopping | halts training when validation loss stops improving |

### 1.8 Bias / variance

`Error ≈ Bias² + Variance + Noise`. Increasing capacity trades bias for variance; regularisation pushes the other way. Two failure modes the platform highlights:

- **Underfitting** — both train and val loss stay high.
- **Overfitting** — train loss falls while val loss plateaus or rises.

---

## 2. Implementation

Project layout follows the spec’s recommended split:

```
backend/
  models/
    perceptron.py      # historical + activation variants
    mlp.py             # forward, backward, dropout, L2
    rbf.py             # Gaussian RBF network
  training/
    forward.py         # forward propagation entry point
    backward.py        # backpropagation entry point
    optimizer.py       # mini-batch SGD + early stopping
  utils/
    activation.py      # sigmoid, tanh, ReLU, softmax + derivatives
    loss.py            # MSE, BCE, CCE + derivatives
    metrics.py         # accuracy, precision, recall, F1, MSE/RMSE/R²
    datasets.py        # circles, XOR, moons, spirals, linear, sine
  app.py               # Flask API + NDJSON streaming
  nn_engine.py         # backwards-compat aggregator (re-exports)
frontend/
  index.html           # SPA shell with five tabs
  css/styles.css       # dark "scientific instrument" palette
  js/                  # app, charts, network viz, decision boundary
```

The backend uses Flask + NumPy + pandas (CSV parsing only). The frontend is plain HTML/CSS/JS with Chart.js — no build tooling.

### 2.1 Training as a stream

Training is exposed as `POST /api/train` returning **NDJSON**: one JSON line per epoch (loss/accuracy values + occasional weight snapshots), terminated by a `done` line carrying a `session_id`. The frontend reads the stream and updates charts and the network visualisation in real time. Weight snapshots are sub-sampled to ~30 per training run to keep the wire light.

### 2.2 Per-layer architectures

The MLP can be configured either by uniform sliders (`hidden_layers`, `width`) or via a per-layer override (e.g. `16,8,4`). The frontend passes both; the backend prefers the explicit `widths` array when provided.

### 2.3 RBF integration

`RBFNetwork` exposes the same `forward / backward / apply_gradients / predict` interface as the MLP, so the shared training loop works on it unchanged. Centers are picked from the training set, σ from the median pairwise distance.

---

## 3. Experimental results

All experiments train on the **concentric circles** dataset (n=300, noise=0.1, 80/20 split) for 120 epochs at lr=0.01.

### Experiment 1 — Perceptron vs MLP

| model | layers | test accuracy |
|---|---|---|
| Perceptron (linear) | [2, 1] | ≈ 50–55% |
| Perceptron + sigmoid | [2, 1] | ≈ 55–60% |
| MLP, 1 hidden (8) | [2, 8, 1] | ≈ 92–96% |
| MLP, 2 hidden (16,16) | [2, 16, 16, 1] | ≈ 96–99% |

**Reading.** A linear perceptron cannot separate concentric circles because no straight line splits them. Even adding sigmoid does not change the decision surface’s linearity. A single hidden layer of 8 ReLU units is already enough for >90% accuracy. Adding a second hidden layer of 16 units gives a smoother boundary and a small accuracy gain.

### Experiment 2 — Effect of depth

| hidden layers | test accuracy |
|---|---|
| 1 | ≈ 92–95% |
| 2 | ≈ 95–98% |
| 3 | ≈ 96–98% |
| 4 | ≈ 96–98% |
| 5 | ≈ 95–98% |

**Reading.** Returns plateau quickly on this 2D problem. Beyond two hidden layers we mostly trade extra parameters for a marginally smoother boundary; on a small dataset, deeper models also become harder to optimise without careful initialisation.

### Experiment 3 — Overfitting vs regularisation

Architecture fixed at [2, 32, 32, 1] — intentionally over-capacity for the dataset.

| configuration | train loss | val loss | gen. gap |
|---|---|---|---|
| No regularisation | very low | rises late | wide |
| L2 (λ = 0.001) | low | low | small |
| Dropout 30% | moderate | low | small |

**Reading.** Without regularisation, the unrestricted network drives training loss near zero while validation loss starts climbing — classic overfitting. L2 keeps weights small and recovers most of the test accuracy. Dropout achieves a similar effect through a different mechanism: it forces the network to make redundant features, so no single neuron carries too much weight.

---

## 4. Critical analysis

### What works well

- **Pure NumPy engine.** Every operation is visible, debuggable, and pedagogically transparent.
- **Streaming training.** NDJSON lets the UI animate per-epoch progress without polling.
- **Live network visualisation.** Edge thickness/colour reflects weight magnitude/sign and pulses on snapshot — students can *see* learning happen.
- **Three mandatory experiments** are wired in and one click away.

### Limitations

- **Optimiser is plain SGD.** No momentum, no Adam. For deep networks this matters; for the educational 2D problems it doesn’t.
- **RBF centers are fixed.** A real RBF would optimise centers and σ jointly (e.g. via k-means + gradient descent). Our version trades that for clarity and stability.
- **No GPU / no batching backends.** Everything runs on CPU NumPy. Acceptable for ≤ a few thousand samples.
- **In-memory model store.** Trained sessions are lost on server restart. Fine for a single-user demo; would need persistence in a multi-user context.

### Possible extensions

- Add **Adam** alongside SGD for a fairer optimiser comparison.
- Add **k-means** initialisation for RBF centers and a learnable σ.

---

## 4b. Implemented bonuses

### Comparison with scikit-learn

Each row of every experiment now ships with a sklearn baseline trained on the same train/test split, mapped to the closest official estimator:

| our config | sklearn estimator |
|---|---|
| `[in, out]` linear | `linear_model.Perceptron` |
| `[in, out]` + activation | `linear_model.LogisticRegression` |
| `[in, …, out]` MLP | `neural_network.MLPClassifier` (same hidden sizes, same activation) |

The Experiments tab shows four columns side-by-side: ours · acc, ours · loss, sk · acc, sk · loss. This validates that our pure-NumPy engine reaches comparable accuracy to a battle-tested implementation, and exposes places where it doesn’t (small differences in optimiser, init, and stopping criteria).

### 3D visualisations (Plotly)

A new **3D** tab appears after training and renders two surfaces:

- **3D probability surface** — the model’s `P(y = 1)` evaluated on a grid over the input plane, plotted as a height field with the training points overlaid in 3D. Makes the *shape* of the decision surface immediately legible (sigmoidal ramp for a perceptron, multi-bump landscape for an RBF, valley-and-ridge for a deep MLP).

- **3D loss landscape** — two random directions are drawn in flattened weight space and *filter-normalised* per parameter (Li et al., 2018) so no single weight matrix dominates. Loss is then evaluated on a 21×21 grid of perturbations around the trained weights. The trained solution sits at α = β = 0; how steep the bowl is around it is a proxy for the sharpness of the minimum.

Both surfaces use Plotly via CDN — no build step required.

---

## 5. How to run

```bash
pip install -r requirements.txt
python run.py             # http://127.0.0.1:5000
python run.py --port 8000 # custom port
```

Configure on the left, click **Train**, watch the loss/accuracy curves live, then explore the *Results*, *Decision boundary*, *Theory*, and *Experiments* tabs.
