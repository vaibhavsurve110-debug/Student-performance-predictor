"""
train.py — Train and compare 5 ML models on the UCI Student Performance dataset.
Saves the best model (Random Forest) and scaler/encoders for inference.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import load_and_preprocess, FEATURE_NAMES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2,
        random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric="logloss"
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10, min_samples_split=5, random_state=42
    ),
    "SVM": SVC(
        kernel="rbf", C=1.0, gamma="scale",
        probability=True, random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
}


def train_all():
    print("=" * 60)
    print("  Student Performance Prediction — Model Training")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, scaler, feature_names, encoders = \
        load_and_preprocess(apply_smote=True)

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")
    print(f"Features:     {len(feature_names)}")
    print(f"Classes:      0=Fail, 1=Pass, 2=Excellent\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    best_model = None
    best_score = 0.0
    best_name = ""

    for name, model in MODELS.items():
        print(f"Training: {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring="accuracy", n_jobs=-1)

        result = {
            "name": name,
            "accuracy": round(float(acc) * 100, 1),
            "precision": round(float(prec) * 100, 1),
            "recall": round(float(rec) * 100, 1),
            "f1_score": round(float(f1) * 100, 1),
            "cv_mean": round(float(cv_scores.mean()) * 100, 1),
            "cv_std": round(float(cv_scores.std()) * 100, 2),
        }
        results.append(result)
        print(f"  Accuracy: {result['accuracy']}%  |  F1: {result['f1_score']}%  "
              f"| CV: {result['cv_mean']} ± {result['cv_std']}%")

        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

    print(f"\n Best Model: {best_name} ({best_score*100:.1f}% accuracy)")

    # Feature importance (Random Forest / GBM / DT)
    feature_importances = []
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        ranked = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1], reverse=True
        )
        feature_importances = [
            {"feature": name, "importance": round(float(imp), 4), "rank": i + 1}
            for i, (name, imp) in enumerate(ranked)
        ]
        print("\n Top 10 Features:")
        for fi in feature_importances[:10]:
            bar = "#" * int(fi["importance"] * 100)
            print(f"  {fi['rank']:2}. {fi['feature']:15} {fi['importance']:.4f} {bar}")

    # Save artifacts
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

    # Add notes for each model
    notes_map = {
        "XGBoost": "Advanced ensemble — best for M.E. project",
        "Random Forest": "Strong baseline ensemble",
        "Gradient Boosting": "Slower to train, highly accurate",
        "Decision Tree": "Interpretable, prone to overfit",
        "SVM": "Good with scaled features",
        "Logistic Regression": "Baseline; fast training",
    }
    for r in results:
        r["notes"] = notes_map.get(r["name"], "")

    # Save metrics and feature importances as JSON
    with open(os.path.join(MODELS_DIR, "model_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(MODELS_DIR, "feature_importances.json"), "w") as f:
        json.dump(feature_importances, f, indent=2)

    print(f"\nAll models saved to: {MODELS_DIR}")
    print("Training complete!")
    return results, feature_importances


if __name__ == "__main__":
    train_all()
