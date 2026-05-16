# 🎓 Student Performance Predictor

> **M.E. Dissertation Project** — Pooja V. Mali, AJINKYA DY PATIL School of Engineering  
> AI-powered academic outcome prediction using Machine Learning on the UCI Student Performance Dataset.

---

## 🏆 Project Highlights

| Metric | Value |
|---|---|
| **Best Model** | Gradient Boosting — **93.1% accuracy**, 93.2% F1-Score |
| **Dataset** | UCI Student Performance (Math), 649 records, 32 features |
| **Target** | 3-class: `Fail (G3<10)` · `Pass (10≤G3<15)` · `Excellent (G3≥15)` |
| **Preprocessing** | MinMaxScaler + SMOTE oversampling + Label Encoding |
| **Evaluation** | 5-fold Stratified Cross-Validation |
| **Top Features** | G2, absences, G1, traveltime, age |

---

## 📁 Project Structure

```
Student Performance Predictor/
├── backend/
│   ├── main.py            # FastAPI REST API (5 endpoints)
│   ├── model.py           # Model loading & inference
│   ├── schemas.py         # Pydantic request/response models
│   ├── local_storage.py   # JSON-based prediction history
│   └── __init__.py
├── ml/
│   ├── train.py           # Training pipeline (6 models, 5-fold CV)
│   ├── preprocess.py      # Data cleaning, encoding, SMOTE, scaling
│   └── models/            # Saved trained model artifacts (.pkl + .json)
│       ├── best_model.pkl
│       ├── scaler.pkl
│       ├── encoders.pkl
│       ├── feature_names.pkl
│       ├── model_metrics.json
│       └── feature_importances.json
├── frontend/
│   └── app.py             # Streamlit dashboard (5 pages)
├── data/
│   └── student-mat.csv    # UCI dataset (auto-downloaded if missing)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (already pre-trained — skip if `ml/models/` exists)

```bash
python ml/train.py
```

Expected output:
```
Best Model: Gradient Boosting (93.1% accuracy)
All models saved to: ml/models/
Training complete!
```

---

## 🚀 Running the Application

### Step 1 — Start the FastAPI Backend

```bash
# From the project root directory
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at:
- **Docs (Swagger)**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Step 2 — Start the Streamlit Frontend

Open a **new terminal**, then:

```bash
streamlit run frontend/app.py
```

Dashboard will open at: **http://localhost:8501**

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API + model readiness check |
| `POST` | `/predict` | Predict student performance |
| `GET` | `/models` | Model comparison metrics (all 6 models) |
| `GET` | `/features` | Feature importance rankings |
| `GET` | `/history` | Recent prediction history |
| `POST` | `/history/clear` | Clear prediction history |

### Example Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "G1": 14, "G2": 15, "studytime": 3, "absences": 2,
    "failures": 0, "Medu": 3, "Fedu": 2, "higher": "yes",
    "school": "GP", "sex": "F", "age": 17, "address": "U",
    "famsize": "GT3", "Pstatus": "T", "Mjob": "teacher",
    "Fjob": "other", "reason": "course", "guardian": "mother",
    "traveltime": 1, "schoolsup": "no", "famsup": "yes",
    "paid": "no", "activities": "yes", "nursery": "yes",
    "internet": "yes", "romantic": "no", "famrel": 4,
    "freetime": 3, "goout": 2, "Dalc": 1, "Walc": 1, "health": 4
  }'
```

---

## 📊 ML Models Compared

| Model | Accuracy | F1-Score | CV Mean |
|---|---|---|---|
| **Gradient Boosting** ⭐ | **93.1%** | **93.2%** | 92.0% |
| Random Forest | 91.4% | 91.4% | 92.2% |
| Decision Tree | 91.4% | 91.4% | 89.6% |
| Logistic Regression | 87.1% | 86.6% | 82.6% |
| SVM | 86.2% | 86.0% | 80.9% |

---

## 🔑 Top Predictive Features

1. **G2** (2nd period grade) — 85.8% importance
2. **Absences** — early warning signal
3. **G1** (1st period grade) — consistent predictor
4. **Travel time** — proxy for lifestyle factors
5. **Age** — correlates with maturity and focus

---

## 🖥️ Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Home | Project overview, key stats, performance classes |
| 🔮 Predict | Input student data → get real-time AI prediction |
| 📊 Model Comparison | Grouped bar chart of all model metrics |
| 📈 Feature Importance | Horizontal bar chart of top predictors |
| 📋 History | View & clear all past predictions |

---

## 📚 Dataset Reference

**UCI Machine Learning Repository — Student Performance Data Set**  
P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School Student Performance." 2008.  
URL: https://archive.ics.uci.edu/ml/datasets/student+performance

---

## 👩‍💻 Author

**Pooja V. Mali**  
AJINKYA DY PATIL School of Engineering  
M.E. Dissertation Project — 2026
