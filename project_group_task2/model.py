"""
model.py — COS40007 Group Task 2 (Week 7 Studio)
Obesity Level Prediction — GitHub Actions ML Pipeline Demo

Pipeline:
  1. Generate synthetic regression data
  2. Train a neural network (MLP) with TensorFlow/Keras
  3. Evaluate on held-out test set
  4. Save model_results.png  (training curves + scatter + residuals)
  5. Save metrics.txt        (all evaluation metrics)

Both output files are picked up by the upload-artifact step in train.yml.
"""

import sys
import os

# ── Non-interactive backend BEFORE any other matplotlib import ────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  COS40007 | Group Task 2 | GitHub Actions ML Pipeline")
print("=" * 60)

# ── TensorFlow import with version info ───────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"  Python     : {sys.version.split()[0]}")
print(f"  TensorFlow : {tf.__version__}")
print(f"  NumPy      : {np.__version__}")
print("=" * 60)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Generate synthetic dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Generating synthetic dataset...")

N_SAMPLES  = 600
N_FEATURES = 10

X = np.random.randn(N_SAMPLES, N_FEATURES).astype(np.float32)

# Non-linear target: weighted sum + interaction + quadratic + noise
weights = np.array([3.0, -1.5, 0.8, 2.0, -0.5, 1.2, -2.2, 0.6, 1.8, -1.0])
y = (
    X @ weights
    + 0.6 * X[:, 0] ** 2
    - 0.4 * X[:, 1] * X[:, 2]
    + 0.3 * np.sin(X[:, 3])
    + np.random.randn(N_SAMPLES) * 0.7
).astype(np.float32)

# Stratified-style split: 70 / 15 / 15
n_train = int(0.70 * N_SAMPLES)   # 420
n_val   = int(0.15 * N_SAMPLES)   # 90
# test   = remaining 90

X_train, y_train = X[:n_train],              y[:n_train]
X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_test,  y_test  = X[n_train+n_val:],        y[n_train+n_val:]

print(f"  Train : {X_train.shape[0]} samples | "
      f"Val : {X_val.shape[0]} | Test : {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Build model
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Building neural network...")

model = keras.Sequential(
    [
        layers.Input(shape=(N_FEATURES,), name="input"),
        layers.Dense(128, activation="relu", name="hidden_1"),
        layers.Dense(64,  activation="relu", name="hidden_2"),
        layers.Dense(32,  activation="relu", name="hidden_3"),
        layers.Dense(1,                      name="output"),
    ],
    name="obesity_regression_net",
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.003),
    loss="mse",
    metrics=["mae"],
)

model.summary()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Train
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Training...")

EPOCHS     = 15
BATCH_SIZE = 32

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1,
)

epochs_run = len(history.history["loss"])
print(f"\n  Training stopped at epoch {epochs_run}/{EPOCHS}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Evaluate
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Evaluating on test set...")

test_mse, test_mae = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test, verbose=0).flatten()

ss_res  = float(np.sum((y_test - y_pred) ** 2))
ss_tot  = float(np.sum((y_test - np.mean(y_test)) ** 2))
r2      = 1.0 - ss_res / ss_tot
rmse    = float(np.sqrt(test_mse))
test_mae = float(test_mae)
test_mse = float(test_mse)

best_val_loss  = float(min(history.history["val_loss"]))
final_val_loss = float(history.history["val_loss"][-1])
final_trn_loss = float(history.history["loss"][-1])

print(f"  MSE   : {test_mse:.4f}")
print(f"  RMSE  : {rmse:.4f}")
print(f"  MAE   : {test_mae:.4f}")
print(f"  R²    : {r2:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5a — Save metrics.txt
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Saving artefacts...")

metrics_content = f"""\
============================================================
  COS40007 Group Task 2 — Pipeline Metrics
  Obesity Level Prediction System (Neural Network Demo)
============================================================
  Model name     : {model.name}
  Architecture   : {N_FEATURES} → 128 → 64 → 32 → 1 (ReLU)
  Dataset        : Synthetic regression (N={N_SAMPLES})
  Features       : {N_FEATURES}
  Epochs trained : {epochs_run} / {EPOCHS} (EarlyStopping)
  Batch size     : {BATCH_SIZE}
  Split          : 70 / 15 / 15  (train / val / test)
------------------------------------------------------------
  TEST SET RESULTS
  MSE            : {test_mse:.4f}
  RMSE           : {rmse:.4f}
  MAE            : {test_mae:.4f}
  R²             : {r2:.4f}
------------------------------------------------------------
  TRAINING SUMMARY
  Final train loss    : {final_trn_loss:.4f}
  Final val loss      : {final_val_loss:.4f}
  Best val loss (ESt) : {best_val_loss:.4f}
============================================================
"""

with open("metrics.txt", "w") as f:
    f.write(metrics_content)

print("  ✓ metrics.txt saved")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5b — Save model_results.png
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "COS40007 Group Task 2 — ML Pipeline Results\n"
    "Obesity Level Prediction System  |  Neural Network Regression",
    fontsize=13, fontweight="bold", y=1.02,
)

# ── Plot 1: Loss curves ───────────────────────────────────────────────────────
ax = axes[0]
epochs_x = range(1, epochs_run + 1)
ax.plot(epochs_x, history.history["loss"],     color="#2E75B6", lw=2, label="Train Loss")
ax.plot(epochs_x, history.history["val_loss"], color="#E05C2A", lw=2, ls="--", label="Val Loss")
ax.axvline(
    x=epochs_run - early_stop.patience if early_stop.stopped_epoch > 0 else epochs_run,
    color="grey", ls=":", lw=1, label="Best epoch"
)
ax.set_title("Training & Validation Loss (MSE)", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Plot 2: Predicted vs Actual ───────────────────────────────────────────────
ax = axes[1]
ax.scatter(y_test, y_pred, alpha=0.6, color="#2E75B6",
           edgecolors="white", lw=0.4, s=45, label="Predictions")
lo = min(float(y_test.min()), float(y_pred.min()))
hi = max(float(y_test.max()), float(y_pred.max()))
ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
ax.set_title("Predicted vs Actual (Test Set)", fontweight="bold")
ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.93, f"R² = {r2:.3f}", transform=ax.transAxes,
        fontsize=11, fontweight="bold", color="#1F497D",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#D5E8F0", alpha=0.85))

# ── Plot 3: Residuals ─────────────────────────────────────────────────────────
ax = axes[2]
residuals = y_test - y_pred
ax.scatter(y_pred, residuals, alpha=0.6, color="#5B9D5B",
           edgecolors="white", lw=0.4, s=45)
ax.axhline(0, color="red", ls="--", lw=1.5, label="Zero residual")
ax.set_title("Residual Plot (Test Set)", fontweight="bold")
ax.set_xlabel("Predicted Values")
ax.set_ylabel("Residuals  (Actual − Predicted)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.93, f"RMSE = {rmse:.3f}", transform=ax.transAxes,
        fontsize=11, fontweight="bold", color="#1F497D",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#D5E8F0", alpha=0.85))

plt.tight_layout()
plt.savefig("model_results.png", dpi=150, bbox_inches="tight")
plt.close()

print("  ✓ model_results.png saved")

# ── Final verification ────────────────────────────────────────────────────────
assert os.path.exists("metrics.txt"),       "ERROR: metrics.txt not found!"
assert os.path.exists("model_results.png"), "ERROR: model_results.png not found!"

print("\n  ✓ Both artefacts verified on disk.")
print("  ✓ Pipeline complete — ready for artifact upload.")
print("=" * 60)
