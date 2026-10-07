"""
House Price Prediction - training and evaluation
------------------------------------------------
Load data -> explore -> preprocess -> train models -> compare against a
baseline -> cross-validate -> tune -> explain -> save model + metrics.

Dataset columns:
Id, Area, Bedrooms, Bathrooms, Floors, YearBuilt, Location, Condition, Garage, Price

Install requirements:
    python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib

Run:
    python house_price_pred.py

Outputs:
    house_price_model.joblib   trained pipeline
    model_metrics.json         evaluation numbers (used by the web app)
    model_comparison.csv       model comparison table
    *.png                      EDA and evaluation plots
"""

import json
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = "House Price Prediction Dataset.csv"
MODEL_PATH = "house_price_model.joblib"
METRICS_PATH = "model_metrics.json"
RANDOM_STATE = 42
CURRENT_YEAR = datetime.now().year


# ----------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("\nFirst rows:\n", df.head())
    print("\nInfo:")
    df.info()
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nSummary statistics:\n", df.describe())
    return df


# ----------------------------------------------------------------------------
# 2. Exploratory data analysis (saves plots as PNG files)
# ----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Price"], bins=40, kde=True)
    plt.title("Price Distribution")
    plt.tight_layout()
    plt.savefig("eda_price_distribution.png", dpi=150)
    plt.close()

    num_cols = df.select_dtypes(include=np.number).drop(columns=["Id"], errors="ignore")
    plt.figure(figsize=(8, 6))
    sns.heatmap(num_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("eda_correlation.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="Area", y="Price", hue="Location", alpha=0.6)
    plt.title("Price vs Area by Location")
    plt.tight_layout()
    plt.savefig("eda_price_vs_area.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, ["Location", "Condition", "Garage"]):
        sns.boxplot(data=df, x=col, y="Price", ax=ax)
        ax.set_title(f"Price by {col}")
    plt.tight_layout()
    plt.savefig("eda_price_by_category.png", dpi=150)
    plt.close()

    # How strongly does each feature relate to price?
    print("\nCorrelation of numeric features with Price:")
    print(num_cols.corr()["Price"].drop("Price").round(3).to_string())
    for col in ["Location", "Condition", "Garage"]:
        print(f"\nAverage price by {col}:")
        print(df.groupby(col)["Price"].mean().round(0).to_string())

    print("\nEDA plots saved as eda_*.png")


# ----------------------------------------------------------------------------
# 3. Feature engineering + preprocessing
# ----------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["HouseAge"] = CURRENT_YEAR - df["YearBuilt"]
    df["TotalRooms"] = df["Bedrooms"] + df["Bathrooms"]
    df["AreaPerBedroom"] = df["Area"] / df["Bedrooms"].replace(0, np.nan)
    df["AreaPerBedroom"] = df["AreaPerBedroom"].fillna(df["Area"])
    return df


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


# ----------------------------------------------------------------------------
# 4. Evaluation helper
# ----------------------------------------------------------------------------
def evaluate(name, model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name:<24} RMSE: {rmse:>10,.0f}  MAE: {mae:>10,.0f}  R2: {r2:>7.4f}")
    return {"model": name, "RMSE": rmse, "MAE": mae, "R2": r2}


# ----------------------------------------------------------------------------
# 5. Main pipeline
# ----------------------------------------------------------------------------
def main():
    df = load_data(DATA_PATH)

    df = df.drop_duplicates()
    df = df.dropna(subset=["Price"])
    run_eda(df)

    df = add_features(df)

    target = "Price"
    drop_cols = [target, "Id", "YearBuilt"]  # YearBuilt replaced by HouseAge
    X = df.drop(columns=drop_cols)
    y = df[target]

    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    print("\nNumeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Baseline (predict mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    # Train, cross-validate, and compare every model with the baseline
    print("\n--- Model comparison (5-fold CV on train set + hold-out test set) ---")
    results, fitted = [], {}
    for name, estimator in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        cv = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=kfold,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
        )
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        res = evaluate(name, pipe, X_test, y_test)
        res["CV_R2_mean"] = cv["test_r2"].mean()
        res["CV_R2_std"] = cv["test_r2"].std()
        res["CV_MAE"] = -cv["test_mae"].mean()
        results.append(res)

    baseline_r2 = next(r["R2"] for r in results if r["model"].startswith("Baseline"))

    # Tune Gradient Boosting
    print("\n--- Hyperparameter tuning (Gradient Boosting) ---")
    gb_pipe = Pipeline(
        [
            ("prep", preprocessor),
            ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]
    )
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
    }
    grid = GridSearchCV(gb_pipe, param_grid, cv=kfold, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best params:", grid.best_params_)
    tuned = grid.best_estimator_
    tuned_res = evaluate("Tuned Gradient Boosting", tuned, X_test, y_test)
    tuned_res["CV_R2_mean"] = grid.best_score_
    tuned_res["CV_R2_std"] = np.nan
    tuned_res["CV_MAE"] = np.nan
    results.append(tuned_res)
    fitted["Tuned Gradient Boosting"] = tuned

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    print("\n", results_df.round(4).to_string(index=False))
    results_df.to_csv("model_comparison.csv", index=False)

    # Choose the best real model (the baseline is only a yardstick)
    real = {n: m for n, m in fitted.items() if not n.startswith("Baseline")}
    best_name = max(real, key=lambda n: r2_score(y_test, real[n].predict(X_test)))
    final_model = real[best_name]
    preds = final_model.predict(X_test)
    best = next(r for r in results if r["model"] == best_name)
    print(f"\nBest model: {best_name}")

    # Honest verdict
    print("\n--- Verdict ---")
    print(f"Baseline R2 (always predict the average price): {baseline_r2:.4f}")
    print(f"Best model test R2: {best['R2']:.4f}   CV R2: {best['CV_R2_mean']:.4f}")
    if best["R2"] < 0.1:
        print(
            "The model is no better than guessing the average price. The features in\n"
            "this dataset show almost no relationship with Price, so any prediction\n"
            "from it is unreliable. Say this in your report as a data limitation."
        )
    elif best["R2"] > baseline_r2 + 0.1:
        print("The model clearly beats the baseline, so it has learned real signal.")

    # Actual vs predicted + residual plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(y_test, preds, alpha=0.5)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    axes[0].plot(lims, lims, "r--")
    axes[0].set_xlabel("Actual Price")
    axes[0].set_ylabel("Predicted Price")
    axes[0].set_title(f"Actual vs Predicted ({best_name})")
    axes[1].hist(y_test - preds, bins=40)
    axes[1].set_xlabel("Error (actual - predicted)")
    axes[1].set_title("Residuals")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    plt.close()

    # Permutation importance on the test set (works for any model)
    perm = permutation_importance(
        final_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    print("\nPermutation importance (drop in R2 when a feature is shuffled):")
    print(imp.round(4).to_string())
    plt.figure(figsize=(8, 5))
    sns.barplot(x=imp.values, y=imp.index)
    plt.title("Permutation Feature Importance")
    plt.xlabel("Drop in R2 when shuffled")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

    # Save model and metrics
    joblib.dump(final_model, MODEL_PATH)
    metrics = {
        "best_model": best_name,
        "n_rows": int(len(df)),
        "test_r2": float(best["R2"]),
        "test_mae": float(best["MAE"]),
        "test_rmse": float(best["RMSE"]),
        "cv_r2_mean": float(best["CV_R2_mean"]),
        "baseline_r2": float(baseline_r2),
        "price_mean": float(y.mean()),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nModel saved to {MODEL_PATH}, metrics saved to {METRICS_PATH}")


# ----------------------------------------------------------------------------
# 6. Reusable prediction function
# ----------------------------------------------------------------------------
def predict_price(house: pd.DataFrame, model_path: str = MODEL_PATH) -> float:
    """
    `house` must contain: Area, Bedrooms, Bathrooms, Floors, YearBuilt,
    Location, Condition, Garage
    """
    model = joblib.load(model_path)
    house = add_features(house).drop(columns=["YearBuilt"])
    return float(model.predict(house)[0])


if __name__ == "__main__":
    main()