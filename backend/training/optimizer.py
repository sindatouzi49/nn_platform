"""Gradient-descent training loop with mini-batches and early stopping."""
import numpy as np

from ..utils.loss import LOSSES


def gradient_descent_step(model, Xb, yb, lr):
    """One mini-batch update."""
    model.forward(Xb, training=True)
    dW, db = model.backward(yb)
    model.apply_gradients(dW, db, lr)


def train(model, X_train, y_train, X_val, y_val,
          lr=0.01, epochs=200, batch_size=32,
          early_stopping_patience=0,
          progress_callback=None):
    """Mini-batch SGD. Returns the per-epoch training history."""
    loss_fn, _ = LOSSES[model.loss_name]
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = np.inf
    patience_counter = 0
    best_W = [w.copy() for w in model.W]
    best_b = [b.copy() for b in model.b]

    n = X_train.shape[0]

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(n)
        X_sh, y_sh = X_train[idx], y_train[idx]

        for start in range(0, n, batch_size):
            Xb = X_sh[start:start + batch_size]
            yb = y_sh[start:start + batch_size]
            gradient_descent_step(model, Xb, yb, lr)

        y_tr_pred  = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        t_loss = loss_fn(y_tr_pred, y_train)
        v_loss = loss_fn(y_val_pred, y_val)

        if model.loss_name != "mse":
            if y_train.shape[1] == 1:
                t_acc = float(np.mean((y_tr_pred >= 0.5).astype(int) == y_train))
                v_acc = float(np.mean((y_val_pred >= 0.5).astype(int) == y_val))
            else:
                t_acc = float(np.mean(np.argmax(y_tr_pred, 1) == np.argmax(y_train, 1)))
                v_acc = float(np.mean(np.argmax(y_val_pred, 1) == np.argmax(y_val, 1)))
        else:
            t_acc = v_acc = None

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        if progress_callback:
            progress_callback(epoch, epochs, {
                "t_loss": t_loss, "v_loss": v_loss,
                "t_acc": t_acc, "v_acc": v_acc,
            })

        if early_stopping_patience > 0:
            if v_loss < best_val - 1e-4:
                best_val = v_loss
                patience_counter = 0
                best_W = [w.copy() for w in model.W]
                best_b = [b.copy() for b in model.b]
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    model.W, model.b = best_W, best_b
                    history["stopped_early"] = epoch
                    break

    return history
