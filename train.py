"""
train.py
--------
Loads the Iris dataset, performs an 80/20 train-test split, trains a
RandomForestClassifier, prints evaluation metrics, saves the model to
disk with joblib, and writes the split data to data/ as CSV files.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
MODEL_PATH   = "model.joblib"
DATA_DIR     = "data"

# ── Load dataset ──────────────────────────────────────────────────────────────
print("=" * 55)
print("  Iris Flower Classification — Training")
print("=" * 55)

iris    = load_iris()
X       = iris.data           # shape (150, 4)
y       = iris.target         # 0 = setosa, 1 = versicolor, 2 = virginica
names   = list(iris.target_names)
columns = list(iris.feature_names)

print(f"\n📦 Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"   Classes: {names}")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y,            # preserve class proportions
)

print(f"\n✂️  Split: {len(X_train)} train  |  {len(X_test)} test  (80/20, stratified)")

# ── Save CSVs ─────────────────────────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)

def to_df(X_arr, y_arr):
    df = pd.DataFrame(X_arr, columns=columns)
    df["species"] = [names[i] for i in y_arr]
    return df

to_df(X_train, y_train).to_csv(f"{DATA_DIR}/train.csv", index=False)
to_df(X_test,  y_test ).to_csv(f"{DATA_DIR}/test.csv",  index=False)
print(f"💾 CSV splits written to {DATA_DIR}/train.csv and {DATA_DIR}/test.csv")

# ── Train model ───────────────────────────────────────────────────────────────
clf = RandomForestClassifier(
    n_estimators = 100,
    max_depth    = None,
    random_state = RANDOM_STATE,
)
clf.fit(X_train, y_train)
print("\n🌲 RandomForestClassifier trained (100 trees)")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred   = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 Test accuracy: {accuracy * 100:.2f}%\n")
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=names))

print("Confusion matrix (rows = actual, cols = predicted):")
cm = confusion_matrix(y_test, y_pred)
header = f"{'':14}" + "  ".join(f"{n:>10}" for n in names)
print(header)
for i, row in enumerate(cm):
    print(f"{names[i]:14}" + "  ".join(f"{v:>10}" for v in row))

# ── Feature importance ────────────────────────────────────────────────────────
print("\nFeature importances:")
importances = clf.feature_importances_
for feat, imp in sorted(zip(columns, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {feat:<30} {imp:.4f}  {bar}")

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(clf, MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")
print("=" * 55)
