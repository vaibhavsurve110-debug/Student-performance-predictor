"""
preprocess.py — Data loading, cleaning, encoding, SMOTE, and scaling
for the UCI Student Performance Dataset.
"""

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
MAT_FILE = os.path.join(DATA_DIR, "student-mat.csv")
ZIP_FILE = os.path.join(DATA_DIR, "student.zip")

# Target: classify final grade G3 into 3 performance classes
# 0 = Fail (G3 < 10), 1 = Pass (10 <= G3 < 15), 2 = Excellent (G3 >= 15)
def grade_to_class(g3: float) -> int:
    if g3 < 10:
        return 0  # Fail
    elif g3 < 15:
        return 1  # Pass
    else:
        return 2  # Excellent


def download_data():
    """Download and extract the UCI Student Performance dataset if not present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(MAT_FILE):
        print("Downloading UCI Student Performance dataset...")
        try:
            urllib.request.urlretrieve(DATA_URL, ZIP_FILE)
            with zipfile.ZipFile(ZIP_FILE, "r") as zf:
                zf.extractall(DATA_DIR)
            # Rename if needed
            extracted = os.path.join(DATA_DIR, "student-mat.csv")
            if not os.path.exists(extracted):
                raise FileNotFoundError("student-mat.csv not found after extraction.")
            print("Dataset downloaded and extracted successfully.")
        except Exception as e:
            print(f"Download failed: {e}. Creating synthetic dataset for demo...")
            _create_synthetic_dataset()
    else:
        print("Dataset already exists.")


def _create_synthetic_dataset():
    """Create a synthetic dataset mirroring UCI structure for offline demo."""
    np.random.seed(42)
    n = 649
    df = pd.DataFrame({
        "school": np.random.choice(["GP", "MS"], n),
        "sex": np.random.choice(["M", "F"], n),
        "age": np.random.randint(15, 23, n),
        "address": np.random.choice(["U", "R"], n),
        "famsize": np.random.choice(["LE3", "GT3"], n),
        "Pstatus": np.random.choice(["T", "A"], n),
        "Medu": np.random.randint(0, 5, n),
        "Fedu": np.random.randint(0, 5, n),
        "Mjob": np.random.choice(["teacher", "health", "services", "at_home", "other"], n),
        "Fjob": np.random.choice(["teacher", "health", "services", "at_home", "other"], n),
        "reason": np.random.choice(["home", "reputation", "course", "other"], n),
        "guardian": np.random.choice(["mother", "father", "other"], n),
        "traveltime": np.random.randint(1, 5, n),
        "studytime": np.random.randint(1, 5, n),
        "failures": np.random.randint(0, 4, n),
        "schoolsup": np.random.choice(["yes", "no"], n),
        "famsup": np.random.choice(["yes", "no"], n),
        "paid": np.random.choice(["yes", "no"], n),
        "activities": np.random.choice(["yes", "no"], n),
        "nursery": np.random.choice(["yes", "no"], n),
        "higher": np.random.choice(["yes", "no"], n),
        "internet": np.random.choice(["yes", "no"], n),
        "romantic": np.random.choice(["yes", "no"], n),
        "famrel": np.random.randint(1, 6, n),
        "freetime": np.random.randint(1, 6, n),
        "goout": np.random.randint(1, 6, n),
        "Dalc": np.random.randint(1, 6, n),
        "Walc": np.random.randint(1, 6, n),
        "health": np.random.randint(1, 6, n),
        "absences": np.random.randint(0, 30, n),
        "G1": np.random.randint(0, 20, n),
        "G2": np.random.randint(0, 20, n),
    })
    # G3 correlated with G1, G2
    df["G3"] = (0.4 * df["G1"] + 0.4 * df["G2"] +
                0.1 * df["studytime"] - 0.05 * df["absences"] +
                np.random.normal(0, 1.5, n)).clip(0, 20).round().astype(int)
    df.to_csv(MAT_FILE, index=False, sep=";")
    print("Synthetic dataset created at:", MAT_FILE)


CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

FEATURE_NAMES = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet", "romantic",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2"
]


def load_and_preprocess(apply_smote: bool = True):
    """
    Full preprocessing pipeline.
    Returns: X_train, X_test, y_train, y_test, scaler, feature_names, encoders
    """
    download_data()

    # Try semicolon separator (UCI format), fallback to comma
    try:
        df = pd.read_csv(MAT_FILE, sep=";")
        if df.shape[1] < 5:
            df = pd.read_csv(MAT_FILE, sep=",")
    except Exception:
        df = pd.read_csv(MAT_FILE, sep=",")

    # Ensure required columns exist
    required = FEATURE_NAMES + ["G3"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Try to work with available columns
        available_features = [c for c in FEATURE_NAMES if c in df.columns]
        if "G3" not in df.columns:
            raise ValueError("Target column G3 missing from dataset.")
    else:
        available_features = FEATURE_NAMES

    df = df[available_features + ["G3"]].copy()
    df.dropna(inplace=True)

    # Encode target
    df["target"] = df["G3"].apply(grade_to_class)

    # Encode categoricals
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    X = df[available_features].values
    y = df["target"].values

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # SMOTE for class imbalance
    if apply_smote:
        smote = SMOTE(random_state=42)
        X_scaled, y = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, available_features, encoders


def preprocess_single(student_dict: dict, scaler: MinMaxScaler,
                      encoders: dict, feature_names: list) -> np.ndarray:
    """
    Preprocess a single student input dict for inference.
    Returns scaled feature array (1, n_features).
    """
    row = []
    for col in feature_names:
        val = student_dict.get(col, 0)
        if col in encoders:
            le = encoders[col]
            val_str = str(val)
            if val_str in le.classes_:
                val = le.transform([val_str])[0]
            else:
                val = 0
        row.append(float(val))
    arr = np.array(row).reshape(1, -1)
    return scaler.transform(arr)
