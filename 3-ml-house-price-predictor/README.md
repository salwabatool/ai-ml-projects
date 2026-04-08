# 🏠 ML House Price Predictor

A complete **machine learning regression pipeline** using the California Housing dataset. Compares 7 models end-to-end with EDA, feature engineering, and rich visualisations.

## Features
- ✅ Exploratory Data Analysis (distributions, correlation heatmap)
- ✅ Feature Engineering (ratio features, polynomial terms)
- ✅ Scikit-learn preprocessing pipeline (imputation + scaling)
- ✅ 7 models compared: Linear, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost
- ✅ 5-fold cross-validation + test evaluation
- ✅ Feature importance plots, Actual vs Predicted charts

## Pipeline

```
Raw Data (California Housing)
        │
        ▼
   EDA & Visualisation
        │
        ▼
  Feature Engineering
  (ratio, polynomial features)
        │
        ▼
  Preprocessing Pipeline
  (SimpleImputer → StandardScaler)
        │
        ▼
  Model Training & CV (7 models)
        │
        ▼
  Evaluation + Visualisation
        │
        ▼
  Best Model saved → best_model.pkl
```

## Quick Start

```bash
pip install -r requirements.txt
python predictor.py

# Outputs:
#   feature_distributions.png
#   correlation_heatmap.png
#   model_comparison.png
#   actual_vs_predicted.png
#   feature_importance.png
#   best_model.pkl
```

## Results (typical)

| Model | Test R² | MAE |
|-------|---------|-----|
| XGBoost | 0.841 | 0.301 |
| Gradient Boosting | 0.835 | 0.310 |
| Random Forest | 0.812 | 0.325 |
| Ridge | 0.598 | 0.530 |
| Lasso | 0.597 | 0.532 |

## Dataset
**California Housing** — 20,640 samples, 8 features, auto-loaded via `sklearn.datasets`.

## 📁 Project Structure
```
3-ml-house-price-predictor/
├── predictor.py      # Full ML pipeline
├── requirements.txt
└── README.md
```

## 📄 License
MIT
