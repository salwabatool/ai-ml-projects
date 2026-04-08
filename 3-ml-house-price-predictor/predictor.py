"""
Machine Learning — House Price Predictor
Full scikit-learn pipeline: EDA → preprocessing → model comparison → evaluation
Models: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────
def exploratory_analysis(df: pd.DataFrame):
    print("\n── EDA ───────────────────────────────────────")
    print(df.describe().round(3).to_string())
    print(f"\nMissing values:\n{df.isnull().sum()}")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flat
    for i, col in enumerate(df.columns):
        axes[i].hist(df[col], bins=40, edgecolor="k", alpha=0.7, color="#4C72B0")
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel(""); axes[i].set_ylabel("Count")
    plt.suptitle("Feature Distributions", fontsize=14, y=1.01)
    plt.tight_layout(); plt.savefig("feature_distributions.png", dpi=120)
    plt.close(); print("  → feature_distributions.png")

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap"); plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=120); plt.close()
    print("  → correlation_heatmap.png")


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RoomsPerHouse"]   = df["AveRooms"]   / df["AveOccup"].clip(lower=1)
    df["BedroomsPerRoom"] = df["AveBedrms"]  / df["AveRooms"].clip(lower=1)
    df["PopPerHouse"]     = df["Population"] / df["HouseAge"].clip(lower=1)
    df["IncomeSq"]        = df["MedInc"] ** 2
    print(f"Features after engineering: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# 4. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
def make_preprocessor(feature_cols: list):
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]), feature_cols),
        ]
    )


# ─────────────────────────────────────────────
# 5. MODELS
# ─────────────────────────────────────────────
def get_models():
    models = {
        "LinearRegression":    LinearRegression(),
        "Ridge":               Ridge(alpha=1.0),
        "Lasso":               Lasso(alpha=0.01),
        "ElasticNet":          ElasticNet(alpha=0.01, l1_ratio=0.5),
        "RandomForest":        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoosting":    GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=200, learning_rate=0.1,
                                         random_state=42, verbosity=0, n_jobs=-1)
    return models


# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te) -> dict:
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2", n_jobs=-1)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return {
        "Model":   name,
        "CV_R2":   round(cv_scores.mean(), 4),
        "CV_std":  round(cv_scores.std(),  4),
        "Test_R2": round(r2_score(y_te, y_pred), 4),
        "MAE":     round(mean_absolute_error(y_te, y_pred), 4),
        "RMSE":    round(np.sqrt(mean_squared_error(y_te, y_pred)), 4),
    }


def compare_models(models, X_tr, y_tr, X_te, y_te) -> pd.DataFrame:
    print("\n── Training & Evaluating Models ─────────────")
    results = []
    for name, m in models.items():
        print(f"  {name} …", end=" ", flush=True)
        res = evaluate_model(name, m, X_tr, y_tr, X_te, y_te)
        results.append(res)
        print(f"R²={res['Test_R2']}  MAE={res['MAE']}")
    return pd.DataFrame(results).sort_values("Test_R2", ascending=False)


# ─────────────────────────────────────────────
# 7. VISUALISE RESULTS
# ─────────────────────────────────────────────
def plot_results(results_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(results_df["Model"], results_df["Test_R2"],
                 color="#4C72B0", edgecolor="k", alpha=0.8)
    axes[0].set_xlabel("R² Score"); axes[0].set_title("Model Comparison — R²")
    axes[0].axvline(0.8, color="red", ls="--", label="R²=0.8")
    axes[0].legend()

    axes[1].barh(results_df["Model"], results_df["MAE"],
                 color="#DD8452", edgecolor="k", alpha=0.8)
    axes[1].set_xlabel("MAE"); axes[1].set_title("Model Comparison — MAE")

    plt.tight_layout(); plt.savefig("model_comparison.png", dpi=120); plt.close()
    print("  → model_comparison.png")


def plot_actual_vs_predicted(best_model, X_te, y_te, model_name):
    y_pred = best_model.predict(X_te)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_te, y_pred, alpha=0.3, s=10, color="#4C72B0")
    lim = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
    plt.plot(lim, lim, "r--", lw=2, label="Perfect Prediction")
    plt.xlabel("Actual Price"); plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted — {model_name}")
    plt.legend(); plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=120); plt.close()
    print("  → actual_vs_predicted.png")


def plot_feature_importance(best_model, feature_names, model_name):
    if not hasattr(best_model, "feature_importances_"):
        return
    imp = pd.Series(best_model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=True)
    plt.figure(figsize=(8, 6))
    imp.plot(kind="barh", color="#4C72B0", edgecolor="k", alpha=0.8)
    plt.title(f"Feature Importance — {model_name}")
    plt.xlabel("Importance"); plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120); plt.close()
    print("  → feature_importance.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ML HOUSE PRICE PREDICTOR")
    print("=" * 60)

    df = load_data()
    exploratory_analysis(df)
    df = engineer_features(df)

    target = "MedHouseVal"
    features = [c for c in df.columns if c != target]

    X = df[features].values
    y = df[target].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = make_preprocessor(list(range(len(features))))
    X_tr = preprocessor.fit_transform(X_tr)
    X_te = preprocessor.transform(X_te)

    all_models = get_models()
    results_df = compare_models(all_models, X_tr, y_tr, X_te, y_te)

    print("\n── Results Table ─────────────────────────────")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model = all_models[best_name]
    best_model.fit(X_tr, y_tr)

    print(f"\n  🏆 Best model: {best_name}")
    plot_results(results_df)
    plot_actual_vs_predicted(best_model, X_te, y_te, best_name)
    plot_feature_importance(best_model, features, best_name)

    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")
    print("\n  Saved best_model.pkl + preprocessor.pkl")
    print("\nDone! ✓")


if __name__ == "__main__":
    main()
