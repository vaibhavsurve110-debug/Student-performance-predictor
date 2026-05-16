"""
model.py — Load trained model artifacts and run inference.
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime
from typing import Optional

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "models")

CLASS_LABELS = {0: "Fail", 1: "Pass", 2: "Excellent"}
CLASS_COLORS = {0: "#ef4444", 1: "#f59e0b", 2: "#22c55e"}

_model = None
_scaler = None
_encoders = None
_feature_names = None
_metrics = None
_feature_importances = None


def _load_artifacts():
    global _model, _scaler, _encoders, _feature_names, _metrics, _feature_importances
    if _model is None:
        model_path = os.path.join(MODELS_DIR, "best_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model not found. Please run 'python ml/train.py' first."
            )
        _model = joblib.load(model_path)
        _scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        _encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
        _feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

        metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                _metrics = json.load(f)

        fi_path = os.path.join(MODELS_DIR, "feature_importances.json")
        if os.path.exists(fi_path):
            with open(fi_path) as f:
                _feature_importances = json.load(f)


def predict(student_dict: dict) -> dict:
    """Run inference on a single student input dictionary."""
    _load_artifacts()

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ml.preprocess import preprocess_single

    X = preprocess_single(student_dict, _scaler, _encoders, _feature_names)

    predicted_class = int(_model.predict(X)[0])
    proba = _model.predict_proba(X)[0]
    confidence = float(np.max(proba))

    probabilities = {
        CLASS_LABELS[i]: round(float(p) * 100, 1)
        for i, p in enumerate(proba)
        if i in CLASS_LABELS
    }

    # Top influencing features for this prediction
    top_features = []
    if _feature_importances:
        top_features = _feature_importances[:10]

    # Calculate SHAP values for the specific prediction
    shap_explanation = {}
    try:
        import shap
        explainer = shap.TreeExplainer(_model)
        # Convert X to dense if necessary
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        shap_values = explainer.shap_values(X_dense)
        expected_value = explainer.expected_value

        if isinstance(shap_values, list):
            class_shap_values = shap_values[predicted_class][0]
            if isinstance(expected_value, list) or isinstance(expected_value, np.ndarray):
                base_value = float(expected_value[predicted_class])
            else:
                base_value = float(expected_value)
        else:
            if len(shap_values.shape) == 3:
                class_shap_values = shap_values[0, :, predicted_class]
                base_value = float(expected_value[predicted_class])
            else:
                class_shap_values = shap_values[0]
                base_value = float(expected_value)

        feature_shap = []
        for i, feature_name in enumerate(_feature_names):
            feature_shap.append({
                "feature": feature_name,
                "value": float(class_shap_values[i])
            })
        feature_shap.sort(key=lambda x: abs(x["value"]), reverse=True)
        shap_explanation = {
            "base_value": base_value,
            "features": feature_shap[:10]
        }
    except Exception as e:
        shap_explanation = {"error": str(e)}

    return {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_LABELS[predicted_class],
        "confidence": round(confidence * 100, 1),
        "probabilities": probabilities,
        "top_features": top_features,
        "shap_explanation": shap_explanation,
        "timestamp": datetime.now().isoformat(),
    }


def get_model_metrics() -> list:
    _load_artifacts()
    return _metrics or []


def get_feature_importances() -> list:
    _load_artifacts()
    return _feature_importances or []


def is_model_ready() -> bool:
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    return os.path.exists(model_path)
