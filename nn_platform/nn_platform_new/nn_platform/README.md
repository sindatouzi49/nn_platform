# 🧠 Neural Networks Educational Platform

An interactive Streamlit app to train, visualise, and understand neural networks — entirely from scratch (pure NumPy, no PyTorch/TensorFlow).

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Features

| Tab | What you get |
|-----|-------------|
| **Configure** (sidebar) | Dataset, architecture, hyperparameters, regularisation |
| **Training** | Live loss & accuracy curves updating every 10 epochs |
| **Results** | Accuracy, F1, precision, recall, confusion matrix, train/test gap |
| **Decision boundary** | 2D contour plot, train vs test side-by-side |
| **Theory** | Expandable explanations of every concept with formulas |
| **Experiments** | One-click: Perceptron vs MLP, effect of depth, overfitting vs regularisation |

## Models implemented

- **Perceptron (historical)** — step/linear activation, no gradient
- **Perceptron + activation** — single layer with sigmoid/ReLU/tanh
- **MLP** — configurable depth (1–5 hidden layers), width (2–32 neurons/layer)

## What's built from scratch (pure NumPy)

- Forward propagation
- Backpropagation (chain rule)
- Mini-batch gradient descent
- Activations: ReLU, Sigmoid, Tanh, Softmax, Linear
- Losses: MSE, Binary Cross-Entropy, Categorical Cross-Entropy
- Regularisation: L2, Dropout, Early stopping
- Metrics: Accuracy, Precision, Recall, F1, Confusion matrix, MSE/RMSE/R²

## CSV upload format

Last column = label. Example:

```
x1,x2,label
0.1,0.5,0
0.9,0.2,1
...
```

## Project structure

```
nn_platform/
├── app.py          # Streamlit UI
├── nn_engine.py    # All ML logic (pure NumPy)
├── requirements.txt
└── README.md
```
