# End-to-End Credit Card Fraud Detection Pipeline

An end-to-end Machine Learning pipeline designed for highly imbalanced transactional fraud detection (~0.58% positive class). Built using LightGBM, custom behavioral feature engineering, and threshold-tuning strategies optimized for Precision-Recall trade-offs.

---

## 📌 Key Highlights & Results

- **Class Imbalance Handling**: Addressed severe imbalance using `scale_pos_weight` and threshold optimization over the Precision-Recall curve (avoiding naive accuracy metrics).
- **Leakage-Free Feature Engineering**: Extracted temporal deltas (`time_diff`) and historical spending ratios (`amt_to_mean_ratio`) strictly isolated across time splits.
- **Evaluation on Unseen Test Data**:
  - **PR-AUC**: `0.9314`
  - **ROC-AUC**: `0.9985`
  - **F1-Score**: `0.8663` (Precision: `92.61%`, Recall: `81.37%`)
  - **Optimal Decision Threshold**: `0.9788`

---

## 🏗️ Project Architecture
```text
├── data/
│   ├── raw/             # Raw transaction CSVs (git-ignored)
│   └── processed/       # Engineered features & preprocessed splits
├── models/              # Serialized model artifacts and tuned threshold
├── notebooks/           # Exploratory Data Analysis & initial experiments
├── src/
│   ├── data_preprocessing.py # Cleaning, parsing, and base geospatial engineering
│   ├── features.py           # Temporal and user-behavioral feature generators
│   ├── train.py              # End-to-end training and evaluation pipeline
│   └── inference.py          # Production-ready inference module
├── config.yaml          # Pipeline hyperparameter & path configurations
├── requirements.txt     # Environment dependencies
└── README.md

---

## ⚙️ Feature Engineering Details

1. **Geospatial Distance**: Calculates geodesic distance between cardholder residence and transaction merchant coordinates via the Haversine formula.
2. **Temporal Signals**: Extraction of cyclical/calendar elements (`hour`, `day_of_week`, cardholder `age`).
3. **Cardholder Velocity (`time_diff`)**: Inter-transaction delta per card ID to capture burst frequency fraud patterns.
4. **Normalized Expenditure (`amt_to_mean_ratio`)**: Transaction amount scaled by the historical average spend for that specific cardholder.

---

## 🚀 Quickstart

### 1. Environment Setup
bash
# Clone the repository
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection

# Install dependencies
pip install -r requirements.txt

### 2. Training the Model
Ensure your raw data is placed at `data/raw/train.csv`, then run:
bash
python src/train.py
This saves the trained model artifact to `models/lgbm_model.pkl` and threshold configuration to `models/threshold.json`.

### 3. Inference
python
from src.inference import FraudDetector
import pandas as pd

detector = FraudDetector()
sample_df = pd.read_csv("data/raw/test.csv").head(10)
predictions = detector.predict(sample_df)
print(predictions)

---

## 📊 Metrics Overview

| Metric | Baseline | Feature Engineered + Tuned |
| :--- | :--- | :--- |
| **ROC-AUC** | 0.9975 | **0.9985** |
| **PR-AUC** | 0.9072 | **0.9314** |
| **Precision** | 88.77% | **92.61%** |
| **Recall** | 80.59% | **81.37%** |
| **F1-Score** | 0.8448 | **0.8663** |

---

## 🛠️ Tech Stack
- **Languages & Frameworks**: Python, Pandas, NumPy, Scikit-Learn, LightGBM
- **Workflow & Config**: YAML, Joblib
EOF
