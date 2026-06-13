import os
import sys
import io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import json
from datetime import datetime
from io import StringIO
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from dvclive import Live

# ── Load hyperparameters ───────────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

EPOCHS        = params["model"]["epochs"]
BATCH_SIZE    = params["model"]["batch_size"]
LEARNING_RATE = params["model"]["learning_rate"]
SEED          = params["data"]["random_seed"]
TEST_SIZE     = params["data"]["test_size"]
CNN_FILTERS   = params["model"]["cnn_filters"]
KERNEL_SIZE   = params["model"]["kernel_size"]
POOL_SIZE     = params["model"]["pool_size"]
DENSE_1       = params["model"]["dense_units_1"]
DENSE_2       = params["model"]["dense_units_2"]
DENSE_3       = params["model"]["dense_units_3"]
DROPOUT_1     = params["model"]["dropout_1"]
DROPOUT_2     = params["model"]["dropout_2"]
NUM_CLASSES   = params["model"]["num_classes"]
ES_PATIENCE   = params["callbacks"]["early_stopping_patience"]
LR_PATIENCE   = params["callbacks"]["reduce_lr_patience"]
LR_FACTOR     = params["callbacks"]["reduce_lr_factor"]
LR_MIN        = params["callbacks"]["reduce_lr_min_lr"]

# ── Directories ────────────────────────────────────────────────────────────────
artifacts_dir = "artifacts"
os.makedirs(artifacts_dir, exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 50)
print("STARTING 1D CNN CLASSIFICATION MODEL — TRAINING")
print("Obesity Level Prediction — COS40007 Group Task 3")
print("=" * 50)

# ── Load data ──────────────────────────────────────────────────────────────────
for path in ["train/train.csv", "test/test.csv"]:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found!")
        print("CWD:", os.getcwd())
        print("Files:", os.listdir('.'))
        sys.exit(1)

print("\nLoading data...")
data  = pd.read_csv("train/train.csv")
dtest = pd.read_csv("test/test.csv")
print(f"Train shape: {data.shape}  Test shape: {dtest.shape}")

# ── Missing values report ──────────────────────────────────────────────────────
print(f"Missing — train: {data.isnull().any().sum()}  "
      f"test: {dtest.isnull().any().sum()}")

train_test_data = [data, dtest]
for dataset in train_test_data:
    num_vars = [v for v in dataset.columns if dataset[v].dtype != 'O']
    print(f"Numerical variables: {len(num_vars)}")

# ── Drop constant columns ──────────────────────────────────────────────────────
suspiciousData = [col for col in data.columns
                  if data[col].nunique() == 1]
if suspiciousData:
    print(f"Dropping {len(suspiciousData)} constant columns")
    for dataset in train_test_data:
        dataset.drop(suspiciousData, axis=1, inplace=True)
else:
    print("No constant columns found")

# ── Encode target column ───────────────────────────────────────────────────────
TARGET_COL = "NObeyesdad"
le_target  = LabelEncoder()
data[TARGET_COL]  = le_target.fit_transform(data[TARGET_COL].astype(str))
dtest[TARGET_COL] = le_target.transform(dtest[TARGET_COL].astype(str))
print(f"Target classes: {list(le_target.classes_)}")

# Save label encoder classes
label_classes = list(le_target.classes_)

# ── Encode categorical variables (frequency encoding — matches tutor approach) ─
cat_vars = [v for v in data.columns
            if data[v].dtype == 'O' and v != TARGET_COL]
print(f"Categorical variables: {len(cat_vars)}")

if cat_vars:
    for var in cat_vars:
        freq = data[var].value_counts().to_dict()
        data[f"{var}_freq"]  = data[var].map(freq)
        dtest[f"{var}_freq"] = dtest[var].map(freq).fillna(0)
    data  = data.drop(cat_vars, axis=1)
    dtest = dtest.drop(cat_vars, axis=1)
    print("Categorical variables frequency-encoded")

# ── Features and target ────────────────────────────────────────────────────────
X = data.drop(TARGET_COL, axis=1).apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean()).fillna(0).values
y = data[TARGET_COL].values

print(f"X: {X.shape}  y: {y.shape}")

# Save feature column names so evaluate.py can align test data exactly
feature_columns = list(data.drop(TARGET_COL, axis=1).columns)
with open("artifacts/feature_columns.json", "w", encoding='utf-8') as f:
    json.dump({
        "feature_columns": feature_columns,
        "label_classes":   label_classes,
        "num_classes":     NUM_CLASSES,
    }, f, indent=4)
print(f"Saved: artifacts/feature_columns.json ({len(feature_columns)} features)")

# ── StandardScaler ─────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler
import joblib
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "artifacts/scaler.pkl")
print("Saved: artifacts/scaler.pkl")

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test_arr, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)
print(f"X_train: {X_train.shape}  X_test: {X_test_arr.shape}")

# ── Reshape for 1D CNN: (samples, features, 1) ────────────────────────────────
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn  = X_test_arr.reshape(X_test_arr.shape[0], X_test_arr.shape[1], 1)
print(f"CNN shapes — train: {X_train_cnn.shape}  test: {X_test_cnn.shape}")

# ── Save split data for evaluate.py ───────────────────────────────────────────
np.save("artifacts/X_test_cnn.npy", X_test_cnn)
np.save("artifacts/y_test.npy",     y_test)
print("Saved: artifacts/X_test_cnn.npy  artifacts/y_test.npy")

# ── Build 1D CNN model ─────────────────────────────────────────────────────────
tf.random.set_seed(SEED)

model = Sequential([
    Conv1D(CNN_FILTERS[0], kernel_size=KERNEL_SIZE, activation='relu',
           input_shape=(X_train_cnn.shape[1], 1), padding='same'),
    MaxPooling1D(pool_size=POOL_SIZE),
    Conv1D(CNN_FILTERS[1], kernel_size=KERNEL_SIZE, activation='relu',
           padding='same'),
    MaxPooling1D(pool_size=POOL_SIZE),
    Conv1D(CNN_FILTERS[2], kernel_size=KERNEL_SIZE, activation='relu',
           padding='same'),
    MaxPooling1D(pool_size=POOL_SIZE),
    Flatten(),
    Dense(DENSE_1, activation='relu'),
    Dropout(DROPOUT_1),
    Dense(DENSE_2, activation='relu'),
    Dropout(DROPOUT_2),
    Dense(DENSE_3, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax'),
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=['accuracy'],
)

model.summary()

# Capture summary to string
stream = StringIO()
model.summary(print_fn=lambda x: stream.write(x + '\n'))
summary_str = stream.getvalue()
with open('model_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_str)
print("Saved: model_summary.txt")

# ── Callbacks ──────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=ES_PATIENCE,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=LR_FACTOR,
                      patience=LR_PATIENCE, min_lr=LR_MIN, verbose=1),
]

# ── Train with dvclive ─────────────────────────────────────────────────────────
print("\nTraining model...")
with Live(dir="dvclive", report="html") as live:
    live.log_param("epochs",       EPOCHS)
    live.log_param("batch_size",   BATCH_SIZE)
    live.log_param("lr",           LEARNING_RATE)
    live.log_param("cnn_filters",  str(CNN_FILTERS))
    live.log_param("num_classes",  NUM_CLASSES)

    history = model.fit(
        X_train_cnn, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test_cnn, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    for i in range(len(history.history['loss'])):
        live.log_metric("train_loss",     history.history['loss'][i])
        live.log_metric("val_loss",       history.history['val_loss'][i])
        live.log_metric("train_accuracy", history.history['accuracy'][i])
        live.log_metric("val_accuracy",   history.history['val_accuracy'][i])
        live.next_step()

print("Training completed!")

# ── Save model ─────────────────────────────────────────────────────────────────
model.save("models/model.keras")
print("Saved: models/model.keras")

# ── Training history plots ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(history.history['loss'],     label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['accuracy'],     label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1].set_title('Model Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{artifacts_dir}/model_results.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: model_results.png")

# ── Save training history for evaluate.py ──────────────────────────────────────
history_dict = {
    "loss":         [float(v) for v in history.history['loss']],
    "val_loss":     [float(v) for v in history.history['val_loss']],
    "accuracy":     [float(v) for v in history.history['accuracy']],
    "val_accuracy": [float(v) for v in history.history['val_accuracy']],
}
with open("artifacts/training_history.json", "w", encoding='utf-8') as f:
    json.dump(history_dict, f, indent=4)
print("Saved: artifacts/training_history.json")

# ── Save data info ─────────────────────────────────────────────────────────────
data_info = {
    "train_samples":             int(X_train.shape[0]),
    "test_samples":              int(X_test_arr.shape[0]),
    "features_count":            int(X.shape[1]),
    "categorical_vars_original": len(cat_vars),
    "constant_features_dropped": len(suspiciousData),
    "num_classes":               NUM_CLASSES,
    "target_classes":            label_classes,
}
with open('data_info.json', 'w', encoding='utf-8') as f:
    json.dump(data_info, f, indent=4)
print("Saved: data_info.json")


# ── SECOND MODEL: Decision Tree & Random Forest ────────────────────────────────
print("\n" + "=" * 50)
print("SECOND MODEL — Decision Tree & Random Forest")
print("Obesity Level Prediction — COS40007 Group Task 3")
print("=" * 50)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# Flatten CNN arrays back to 2D for sklearn
X_train_flat = X_train_cnn.reshape(X_train_cnn.shape[0], -1)
X_test_flat  = X_test_cnn.reshape(X_test_cnn.shape[0], -1)

sklearn_results = []

for sk_name, clf in [
    ("Decision Tree", DecisionTreeClassifier(
        max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=SEED
    )),
    ("Random Forest", RandomForestClassifier(
        n_estimators=100, max_depth=15,
        min_samples_split=5, random_state=SEED, n_jobs=-1
    )),
]:
    print(f"\nTraining {sk_name}...")
    clf.fit(X_train_flat, y_train)
    y_pred = clf.predict(X_test_flat)
    acc    = float(accuracy_score(y_test, y_pred))
    f1_mac = float(f1_score(y_test, y_pred, average="macro"))
    f1_wt  = float(f1_score(y_test, y_pred, average="weighted"))
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 macro : {f1_mac:.4f}")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))
    fname = sk_name.lower().replace(" ", "_")
    joblib.dump(clf, f"models/model_{fname}.pkl")
    print(f"Saved: models/model_{fname}.pkl")
    sklearn_results.append({
        "model_name": sk_name, "accuracy": acc,
        "f1_macro": f1_mac, "f1_weighted": f1_wt,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    })

# Save sklearn metrics
with open("artifacts/sklearn_metrics.json", "w") as f:
    json.dump({"models": sklearn_results}, f, indent=4)
print("Saved: artifacts/sklearn_metrics.json")

# Model comparison plot
cnn_pred = model.predict(X_test_cnn, verbose=0).argmax(axis=1)
cnn_acc  = float(accuracy_score(y_test, cnn_pred))
cnn_f1   = float(f1_score(y_test, cnn_pred, average="macro"))
all_models = [{"model_name": "1D CNN", "accuracy": cnn_acc, "f1_macro": cnn_f1}] + sklearn_results
names    = [r["model_name"] for r in all_models]
acc_vals = [r["accuracy"]   for r in all_models]
f1_vals  = [r["f1_macro"]   for r in all_models]
x = np.arange(len(names)); width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("COS40007 Group Task 3 — Model Comparison\n1D CNN vs Decision Tree vs Random Forest",
             fontweight="bold", fontsize=13)
bars1 = ax.bar(x - width/2, acc_vals, width, label="Accuracy",  color="#2E75B6", edgecolor="white")
bars2 = ax.bar(x + width/2, f1_vals,  width, label="F1 Macro",  color="#E05C2A", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel("Score"); ax.set_ylim(0, 1.0); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.savefig(f"{artifacts_dir}/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: model_comparison.png")

# Save comparison summary
best = max(all_models, key=lambda x: x["f1_macro"])
with open("model_comparison.txt", "w") as f:
    f.write("=" * 55 + "\n")
    f.write("  COS40007 Group Task 3 — Model Comparison\n")
    f.write("=" * 55 + "\n\n")
    for r in all_models:
        f.write(f"  {r['model_name']:<20} Accuracy: {r['accuracy']:.4f}  F1 macro: {r['f1_macro']:.4f}\n")
    f.write(f"\n  Best model: {best['model_name']} (F1 macro: {best['f1_macro']:.4f})\n")
    f.write("=" * 55 + "\n")
print("Saved: model_comparison.txt")

print("\nmodel.py complete — all models trained successfully!")
