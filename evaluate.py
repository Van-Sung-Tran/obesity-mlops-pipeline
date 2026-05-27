import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
from datetime import datetime
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)

# ── Load hyperparameters ───────────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

TEST_SIZE    = params["data"]["test_size"]
BATCH_SIZE   = params["model"]["batch_size"]
EPOCHS       = params["model"]["epochs"]
METRICS_PATH = params["evaluate"]["metrics_path"]

artifacts_dir = "artifacts"

# ── Load test data saved by model.py ──────────────────────────────────────────
# model.py saves the exact split it trained on so evaluate.py
# uses the identical partition — no risk of data leakage from re-splitting.
for path in ["artifacts/X_test_cnn.npy", "artifacts/y_test.npy",
             "artifacts/training_history.json", "models/model.keras",
             "artifacts/feature_columns.json"]:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found — run model.py first")
        sys.exit(1)

print("Loading test data and model...")
X_test_cnn = np.load("artifacts/X_test_cnn.npy")
y_test     = np.load("artifacts/y_test.npy")
print(f"X_test_cnn: {X_test_cnn.shape}  y_test: {y_test.shape}")

with open("artifacts/training_history.json") as f:
    history_dict = json.load(f)

with open("artifacts/feature_columns.json") as f:
    feat_meta = json.load(f)

label_classes = feat_meta["label_classes"]
print(f"Classes: {label_classes}")

# ── Load model ─────────────────────────────────────────────────────────────────
model = tf.keras.models.load_model("models/model.keras")
print("Model loaded successfully")

# ── Evaluate ───────────────────────────────────────────────────────────────────
score = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"\nTest Loss     : {score[0]:.4f}")
print(f"Test Accuracy : {score[1]:.4f}")

# ── Predictions ────────────────────────────────────────────────────────────────
y_prob = model.predict(X_test_cnn, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

# ── Sklearn metrics ────────────────────────────────────────────────────────────
accuracy   = float(accuracy_score(y_test, y_pred))
f1_macro   = float(f1_score(y_test, y_pred, average="macro"))
f1_weighted= float(f1_score(y_test, y_pred, average="weighted"))
cm         = confusion_matrix(y_test, y_pred)
report     = classification_report(
    y_test, y_pred, target_names=label_classes, output_dict=True
)

print(f"\nAccuracy   : {accuracy:.4f}")
print(f"F1 macro   : {f1_macro:.4f}")
print(f"F1 weighted: {f1_weighted:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_classes))

# ── Plot: confusion matrix ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("COS40007 Group Task 3 — Model Evaluation\n"
             "Obesity Level Prediction", fontweight="bold", fontsize=13)

short = [c.replace("_", "\n") for c in label_classes]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=short, yticklabels=short,
            ax=axes[0], linewidths=0.5)
axes[0].set_title(f"Confusion Matrix  |  Accuracy: {accuracy:.3f}",
                  fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Per-class F1
class_f1 = [report[c]["f1-score"] for c in label_classes if c in report]
colors   = ["#2E75B6" if v >= 0.8 else "#E05C2A" for v in class_f1]
bars     = axes[1].barh(short, class_f1, color=colors, edgecolor="white")
axes[1].axvline(f1_macro, color="grey", ls="--", lw=1.5,
                label=f"Macro F1 = {f1_macro:.3f}")
axes[1].set_title("Per-Class F1 Score", fontweight="bold")
axes[1].set_xlabel("F1 Score")
axes[1].set_xlim(0, 1)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="x")
for bar, val in zip(bars, class_f1):
    axes[1].text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{artifacts_dir}/predictions_vs_actual.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved: predictions_vs_actual.png")

# ── Plot: residuals (class prediction errors) ──────────────────────────────────
residuals = y_test.astype(int) - y_pred.astype(int)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(residuals, bins=range(min(residuals)-1, max(residuals)+2),
             edgecolor='black', alpha=0.7, color="#2E75B6")
axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('Prediction Error (true class - predicted class)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Classification Error Distribution')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_pred, residuals, alpha=0.4, color="#2E75B6", s=15)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Class')
axes[1].set_ylabel('Error (true - predicted)')
axes[1].set_title('Error vs Predicted Class')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residuals_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{artifacts_dir}/residuals_analysis.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("Saved: residuals_analysis.png")

# ── metrics.txt ───────────────────────────────────────────────────────────────
with open('metrics.txt', 'w') as f:
    f.write("=" * 50 + "\n")
    f.write("MODEL PERFORMANCE METRICS\n")
    f.write("COS40007 Group Task 3 — Obesity Level Prediction\n")
    f.write("=" * 50 + "\n")
    f.write(f"Accuracy   : {accuracy:.4f}\n")
    f.write(f"F1 macro   : {f1_macro:.4f}\n")
    f.write(f"F1 weighted: {f1_weighted:.4f}\n")
    f.write("=" * 50 + "\n\n")
    f.write("TRAINING HISTORY\n")
    f.write("=" * 50 + "\n")
    f.write(f"Final Train Loss    : {history_dict['loss'][-1]:.4f}\n")
    f.write(f"Final Val Loss      : {history_dict['val_loss'][-1]:.4f}\n")
    f.write(f"Final Train Accuracy: {history_dict['accuracy'][-1]:.4f}\n")
    f.write(f"Final Val Accuracy  : {history_dict['val_accuracy'][-1]:.4f}\n")
print("Saved: metrics.txt")

# ── metrics.json — read by DVC ─────────────────────────────────────────────────
metrics_out = {
    "timestamp":  datetime.now().isoformat(),
    "model_type": "1D CNN Classification",
    "test_size":  TEST_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs":     EPOCHS,
    "final_epoch": len(history_dict['loss']),
    "metrics": {
        "accuracy":    float(accuracy),
        "f1_macro":    float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "test_loss":   float(score[0]),
    },
    "training_history": {
        "final_train_loss":     float(history_dict['loss'][-1]),
        "final_val_loss":       float(history_dict['val_loss'][-1]),
        "final_train_accuracy": float(history_dict['accuracy'][-1]),
        "final_val_accuracy":   float(history_dict['val_accuracy'][-1]),
    }
}
with open(METRICS_PATH, 'w', encoding='utf-8') as f:
    json.dump(metrics_out, f, indent=4)
print(f"Saved: {METRICS_PATH}")

print("\nevaluate.py complete")
