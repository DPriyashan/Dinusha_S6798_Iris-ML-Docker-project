"""
predict.py
----------
Loads the trained RandomForestClassifier from disk and runs predictions
on a set of sample inputs, printing the predicted class and per-species
confidence scores.

Can also be called as a module:
    from predict import predict_sample
    result = predict_sample([5.1, 3.5, 1.4, 0.2])
"""

import sys
import joblib
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "model.joblib"
CLASS_NAMES  = ["setosa", "versicolor", "virginica"]
FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

# ── Sample inputs ─────────────────────────────────────────────────────────────
# Each tuple: (description, [sepal_length, sepal_width, petal_length, petal_width])
SAMPLES = [
    ("Classic setosa",     [5.1, 3.5, 1.4, 0.2]),
    ("Classic versicolor", [6.0, 2.9, 4.5, 1.5]),
    ("Classic virginica",  [6.7, 3.0, 5.2, 2.3]),
    ("Borderline sample",  [5.9, 3.0, 4.8, 1.8]),
    ("Custom input",       [4.8, 3.1, 1.6, 0.2]),
]

# ── Helper ────────────────────────────────────────────────────────────────────

def predict_sample(features: list, model=None) -> dict:
    """
    Predict the Iris species for a single feature vector.

    Parameters
    ----------
    features : list of 4 floats
        [sepal_length, sepal_width, petal_length, petal_width]
    model : sklearn estimator, optional
        If None, loads from MODEL_PATH.

    Returns
    -------
    dict with keys:
        predicted_class (str), confidence (float), probabilities (dict)
    """
    if model is None:
        model = joblib.load(MODEL_PATH)

    X     = np.array(features).reshape(1, -1)
    cls   = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    return {
        "predicted_class" : CLASS_NAMES[cls],
        "confidence"      : float(probs[cls]),
        "probabilities"   : {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    }


def print_result(label: str, features: list, result: dict) -> None:
    """Pretty-print a single prediction result."""
    print(f"\n  Sample : {label}")
    print(f"  Input  : {dict(zip(FEATURE_NAMES, features))}")
    print(f"  ▶ Predicted : {result['predicted_class'].upper()}"
          f"  (confidence {result['confidence'] * 100:.1f}%)")
    print("  Probabilities:")
    for species, prob in result["probabilities"].items():
        bar  = "█" * int(prob * 30)
        dash = "░" * (30 - len(bar))
        print(f"    {species:<12} {prob * 100:5.1f}%  {bar}{dash}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Iris Flower Classification — Prediction")
    print("=" * 55)

    try:
        model = joblib.load(MODEL_PATH)
        print(f"\n✅ Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"\n❌ Model file '{MODEL_PATH}' not found.")
        print("   Run train.py first to generate the model.")
        sys.exit(1)

    print(f"\nRunning predictions on {len(SAMPLES)} samples...\n")
    print("-" * 55)

    for label, features in SAMPLES:
        result = predict_sample(features, model=model)
        print_result(label, features, result)
        print("-" * 55)

    print("\n✅ All predictions complete.")
    print("=" * 55)
