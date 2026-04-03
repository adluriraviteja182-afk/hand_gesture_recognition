"""
Model Training Script
======================
Trains a gesture classifier on landmark features extracted during data collection.
Supports Random Forest and MLP (Multi-Layer Perceptron) classifiers.
Saves the best model as a .pkl file for use in the recognizer.

Usage:
    python train_model.py
    python train_model.py --model mlp --epochs 100
"""

import json
import pickle
import numpy as np
import os
from pathlib import Path
from collections import Counter


def load_dataset(dataset_path="data/raw/dataset.json"):
    """Load the collected gesture dataset."""
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}")
        print("[HINT] Run collect_data.py first to gather training samples.")
        return None, None, None

    with open(dataset_path) as f:
        data = json.load(f)

    X = np.array(data["X"], dtype=np.float32)
    y = np.array(data["y"], dtype=np.int32)
    gesture_map = data.get("gesture_map", {})

    print(f"\n[INFO] Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    counts = Counter(y)
    inv = {v: k for k, v in gesture_map.items()}
    for label, count in sorted(counts.items()):
        print(f"  Class {label} ({inv.get(label, '?')}): {count} samples")

    return X, y, gesture_map


def train_random_forest(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    print("\n[INFO] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_mlp(X_train, y_train):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    print("\n[INFO] Training MLP Neural Network...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=True,
    )
    model.fit(X_scaled, y_train)

    # Wrap scaler with model so it's applied automatically
    class ScaledMLP:
        def __init__(self, scaler, model):
            self.scaler = scaler
            self.model = model

        def predict(self, X):
            return self.model.predict(self.scaler.transform(X))

        def predict_proba(self, X):
            return self.model.predict_proba(self.scaler.transform(X))

    return ScaledMLP(scaler, model)


def evaluate(model, X_test, y_test, gesture_map):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    inv = {v: k for k, v in gesture_map.items()}
    target_names = [inv.get(i, f"class_{i}") for i in sorted(set(y_test))]

    print(f"\n{'='*50}")
    print(f"  Test Accuracy: {acc*100:.2f}%")
    print(f"{'='*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return acc


def train(model_type="random_forest", dataset_path="data/raw/dataset.json",
          output_path="models/gesture_model.pkl"):
    """Full training pipeline."""
    from sklearn.model_selection import train_test_split

    # Load data
    X, y, gesture_map = load_dataset(dataset_path)
    if X is None:
        return

    if len(X) < 50:
        print(f"[WARNING] Only {len(X)} samples. Consider collecting more data (200+ per gesture).")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO] Train: {len(X_train)}, Test: {len(X_test)}")

    # Train
    if model_type == "mlp":
        model = train_mlp(X_train, y_train)
    else:
        model = train_random_forest(X_train, y_train)

    # Evaluate
    acc = evaluate(model, X_test, y_test, gesture_map)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\n✅ Model saved to: {output_path}")
    print(f"   Test Accuracy: {acc*100:.2f}%")
    print(f"\n   Run recognizer with:")
    print(f"   python src/gesture_recognizer.py --model {output_path}")

    return model, acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train gesture recognition model")
    parser.add_argument("--model", choices=["random_forest", "mlp"], default="random_forest")
    parser.add_argument("--dataset", default="data/raw/dataset.json")
    parser.add_argument("--output", default="models/gesture_model.pkl")
    args = parser.parse_args()

    train(model_type=args.model, dataset_path=args.dataset, output_path=args.output)
