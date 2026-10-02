"""Train, compare, persist, and predict with regression models for Bike Sales in Europe.

Target:
  - Profit: The net profit generated from the order.

Features (from src.data_processing):
  - Numeric: Customer_Age, Order_Quantity, Unit_Cost, Unit_Price, Month_Number, Year
  - Categorical: Customer_Gender, Country, Product_Category, Sub_Category
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_processing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    get_features_target,
    load_clean_enriched,
)

MODEL_PATH = "models/model.joblib"

CANDIDATE_MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    #"Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

# Month names to 1-indexed integers
MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

FEATURE_ALIASES = {
    "customer_age": "Customer_Age",
    "age": "Customer_Age",
    "customerage": "Customer_Age",
    "customer_gender": "Customer_Gender",
    "gender": "Customer_Gender",
    "sex": "Customer_Gender",
    "customergender": "Customer_Gender",
    "country": "Country",
    "product_category": "Product_Category",
    "category": "Product_Category",
    "productcategory": "Product_Category",
    "sub_category": "Sub_Category",
    "subcategory": "Sub_Category",
    "sub_cat": "Sub_Category",
    "order_quantity": "Order_Quantity",
    "quantity": "Order_Quantity",
    "orderquantity": "Order_Quantity",
    "unit_cost": "Unit_Cost",
    "cost": "Unit_Cost",
    "unitcost": "Unit_Cost",
    "unit_price": "Unit_Price",
    "price": "Unit_Price",
    "unitprice": "Unit_Price",
    "month_number": "Month_Number",
    "month": "Month_Number",
    "monthnumber": "Month_Number",
    "year": "Year",
}


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer that one-hot encodes categoricals and passes numeric features."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )


def build_pipeline(estimator) -> Pipeline:
    """Combine preprocessing and the given regression estimator into a Pipeline."""
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    models: Optional[Dict[str, Any]] = None,
    sample_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], str, pd.DataFrame, pd.Series]:
    """Train candidate regression models on the training split and evaluate on the test split.

    Args:
        X: Feature DataFrame containing NUMERIC_FEATURES and CATEGORICAL_FEATURES.
        y: Target Series (Profit).
        test_size: Proportion of dataset for test split (default 0.2).
        random_state: Random state for reproducible train/test split.
        models: Optional mapping of model names to estimators. Defaults to CANDIDATE_MODELS.
        sample_size: Optional maximum rows to train on for fast evaluation.

    Returns:
        results_df: DataFrame sorted by R2 descending with columns: model, R2, RMSE, MAE.
        fitted: Dict mapping model names to fitted Pipeline objects.
        best_name: Name of the top-performing model.
        X_test: Test feature set.
        y_test: Test target values.
    """
    if models is None:
        models = CANDIDATE_MODELS

    # Select only required model features if extra columns exist
    required_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if all(col in X.columns for col in required_features):
        X = X[required_features]

    if sample_size is not None and len(X) > sample_size:
        X, _, y, _ = train_test_split(X, y, train_size=sample_size, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results = []
    fitted = {}
    for name, estimator in models.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        results.append({"model": name, "R2": r2, "RMSE": rmse, "MAE": mae})
        fitted[name] = pipeline

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    best_name = str(results_df.iloc[0]["model"])
    return results_df, fitted, best_name, X_test, y_test


def save_model(pipeline: Pipeline, path: str = MODEL_PATH) -> str:
    """Save the fitted model pipeline to disk, creating parent folders if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: str = MODEL_PATH) -> Pipeline:
    """Load a persisted model pipeline from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)


def predict_single(
    pipeline: Pipeline,
    customer_age: Optional[Union[int, Dict[str, Any], pd.DataFrame, pd.Series]] = None,
    customer_gender: Optional[Union[str, int, float]] = None,
    country: Optional[Union[str, int, float]] = None,
    product_category: Optional[str] = None,
    sub_category: Optional[str] = None,
    order_quantity: Optional[int] = None,
    unit_cost: Optional[float] = None,
    unit_price: Optional[float] = None,
    month_number: Optional[Union[int, str]] = None,
    year: Optional[int] = None,
    **kwargs: Any,
) -> float:
    """Predict profit for a single bike sales order.

    Accepts arguments via:
      1. Named keyword or positional arguments:
         customer_age, customer_gender, country, product_category, sub_category,
         order_quantity, unit_cost, unit_price, month_number, year
      2. A dictionary, pandas Series, or single-row DataFrame passed as the 2nd argument.
      3. Flexible alias keyword arguments (e.g. age=..., gender=..., category=...).

    Returns:
        Predicted profit as a float.
    """
    raw_data: Dict[str, Any] = {}

    # Case 1: dictionary, DataFrame, or Series passed as customer_age
    if isinstance(customer_age, (dict, pd.Series)):
        raw_data = dict(customer_age)
    elif isinstance(customer_age, pd.DataFrame):
        if len(customer_age) > 0:
            raw_data = customer_age.iloc[0].to_dict()
        else:
            raise ValueError("Input DataFrame is empty.")
    else:
        # Case 2: Positional arguments check
        # Check if caller passed in numeric-features-first order:
        # (age, quantity, unit_cost, unit_price, month_number, year, gender, country, category, sub_category)
        if isinstance(customer_gender, (int, float)) and not isinstance(customer_gender, bool):
            raw_data["Customer_Age"] = customer_age
            raw_data["Order_Quantity"] = customer_gender
            raw_data["Unit_Cost"] = country
            raw_data["Unit_Price"] = product_category
            raw_data["Month_Number"] = sub_category
            raw_data["Year"] = order_quantity
            raw_data["Customer_Gender"] = unit_cost
            raw_data["Country"] = unit_price
            raw_data["Product_Category"] = month_number
            raw_data["Sub_Category"] = year
        else:
            # Standard order: (age, gender, country, category, sub_category, quantity, unit_cost, unit_price, month, year)
            if customer_age is not None:
                raw_data["Customer_Age"] = customer_age
            if customer_gender is not None:
                raw_data["Customer_Gender"] = customer_gender
            if country is not None:
                raw_data["Country"] = country
            if product_category is not None:
                raw_data["Product_Category"] = product_category
            if sub_category is not None:
                raw_data["Sub_Category"] = sub_category
            if order_quantity is not None:
                raw_data["Order_Quantity"] = order_quantity
            if unit_cost is not None:
                raw_data["Unit_Cost"] = unit_cost
            if unit_price is not None:
                raw_data["Unit_Price"] = unit_price
            if month_number is not None:
                raw_data["Month_Number"] = month_number
            if year is not None:
                raw_data["Year"] = year

    # Merge additional kwargs
    raw_data.update(kwargs)

    # Normalize feature names
    normalized_data: Dict[str, Any] = {}
    for k, v in raw_data.items():
        canonical = FEATURE_ALIASES.get(str(k).lower(), k)
        normalized_data[canonical] = v

    # Validate required columns
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [f for f in all_features if f not in normalized_data or normalized_data[f] is None]
    if missing:
        raise ValueError(
            f"Missing required feature(s) for prediction: {missing}. "
            f"Required features are: {all_features}"
        )

    # Handle month name conversion if string provided
    m_val = normalized_data["Month_Number"]
    if isinstance(m_val, str):
        cleaned_m = m_val.strip().lower()
        if cleaned_m in MONTH_MAP:
            normalized_data["Month_Number"] = MONTH_MAP[cleaned_m]
        else:
            try:
                normalized_data["Month_Number"] = int(cleaned_m)
            except ValueError:
                raise ValueError(f"Unknown month string: '{m_val}'")

    # Standardize gender formatting (e.g. 'm' -> 'M', 'female' -> 'F')
    g_val = str(normalized_data["Customer_Gender"]).strip()
    if g_val.lower() in ("m", "male"):
        normalized_data["Customer_Gender"] = "M"
    elif g_val.lower() in ("f", "female"):
        normalized_data["Customer_Gender"] = "F"

    # Build 1-row DataFrame matching the exact schema
    row = pd.DataFrame([{
        "Customer_Age": int(normalized_data["Customer_Age"]),
        "Order_Quantity": int(normalized_data["Order_Quantity"]),
        "Unit_Cost": float(normalized_data["Unit_Cost"]),
        "Unit_Price": float(normalized_data["Unit_Price"]),
        "Month_Number": int(normalized_data["Month_Number"]),
        "Year": int(normalized_data["Year"]),
        "Customer_Gender": str(normalized_data["Customer_Gender"]),
        "Country": str(normalized_data["Country"]),
        "Product_Category": str(normalized_data["Product_Category"]),
        "Sub_Category": str(normalized_data["Sub_Category"]),
    }])

    return float(pipeline.predict(row)[0])


def predict_batch(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Predict profit for multiple records in a DataFrame.

    Automatically extracts the required numeric and categorical feature columns.
    """
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required features: {missing}")
    return pipeline.predict(df[required])


def train_best_model(path: str = "data/Sales.csv", save: bool = True) -> Tuple[Pipeline, str, float]:
    """Convenience helper to load data, train candidate models, and optionally save the best model."""
    clean_df = load_clean_enriched(path)
    X, y = get_features_target(clean_df)
    results_df, fitted, best_name, _, _ = train_and_evaluate(X, y)
    best_pipeline = fitted[best_name]
    best_r2 = float(results_df.iloc[0]["R2"])
    if save:
        save_model(best_pipeline)
    return best_pipeline, best_name, best_r2


# def main():
#     """Run model comparison and save the best pipeline."""
#     print("=== Bike Sales Profit Model Training ===")
#     print("Loading and preparing data from data/Sales.csv...")
#     clean_df = load_clean_enriched()
#     X, y = get_features_target(clean_df)
#     print(f"Dataset ready: {X.shape[0]} rows, {X.shape[1]} features.")

#     print("\nTraining and evaluating candidate models (Linear Regression, Random Forest, Gradient Boosting)...")
#     results_df, fitted, best_name, X_test, y_test = train_and_evaluate(X, y)
#     print("\nModel Comparison Results:")
#     print(results_df.to_string(index=False))

#     best_pipeline = fitted[best_name]
#     print(f"\nBest Model: {best_name}")
#     save_path = save_model(best_pipeline)
#     print(f"Saved model to: {save_path}")

#     print("\nVerifying sample single prediction:")
#     sample_profit = predict_single(
#         best_pipeline,
#         customer_age=19,
#         customer_gender="M",
#         country="Canada",
#         product_category="Accessories",
#         sub_category="Bike Racks",
#         order_quantity=8,
#         unit_cost=45,
#         unit_price=120,
#         month_number=11,
#         year=2013,
#     )
#     print(f"Predicted Profit: ${sample_profit:,.2f} (Actual CSV value was $590.00)")


# if __name__ == "__main__":
#     main()
