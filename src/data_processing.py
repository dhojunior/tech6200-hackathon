"""Loading, cleaning and feature preparation for the Bike Sales in Europe dataset."""

from pathlib import Path

import pandas as pd

# The path is built from this file's location, so it works from any working folder
DATA_PATH = str(Path(__file__).resolve().parent.parent / "data" / "Sales.csv")

# Columns used by the model. Cost and Revenue are NOT used,
# because Profit = Revenue - Cost (they would leak the answer).
NUMERIC_FEATURES = ["Customer_Age", "Order_Quantity", "Unit_Cost", "Unit_Price", "Month_Number", "Year"]
CATEGORICAL_FEATURES = ["Customer_Gender", "Country", "Product_Category", "Sub_Category"]
TARGET = "Profit"

# Columns that must exist in the CSV file
REQUIRED_COLUMNS = [
    "Date", "Year", "Customer_Age", "Customer_Gender", "Country", "Product_Category",
    "Sub_Category", "Order_Quantity", "Unit_Cost", "Unit_Price", "Profit", "Cost", "Revenue",
]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV file and check that the needed columns exist."""
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"The file is missing these columns: {missing}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and missing values, and convert Date to a real date."""
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add new columns used by the model and by the charts."""
    df = df.copy()
    df["Month_Number"] = df["Date"].dt.month
    df["Profit_Margin"] = (df["Profit"] / df["Revenue"] * 100).round(2)  # charts only
    return df


def prepare_data(path: str = DATA_PATH):
    """Run the whole preparation. Returns the clean data and a quality report."""
    raw = load_data(path)
    clean = add_features(clean_data(raw))
    report = {
        "rows_raw": len(raw),
        "rows_clean": len(clean),
        "duplicates_removed": int(raw.duplicated().sum()),
        "missing_values": int(raw.isna().sum().sum()),
        "columns": clean.shape[1],
    }
    return clean, report


def load_clean_enriched(path: str = DATA_PATH) -> pd.DataFrame:
    """Same as prepare_data, but returns only the clean data."""
    return prepare_data(path)[0]


def get_features_target(df: pd.DataFrame):
    """Return X (inputs) and y (what we want to predict)."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y


def get_form_options(df: pd.DataFrame) -> dict:
    """Values for the GUI prediction form (drop-down lists and number limits)."""
    return {
        "genders": sorted(df["Customer_Gender"].unique()),
        "countries": sorted(df["Country"].unique()),
        "categories": sorted(df["Product_Category"].unique()),
        # each category has its own sub-categories
        "sub_categories": {
            category: sorted(group["Sub_Category"].unique())
            for category, group in df.groupby("Product_Category")
        },
        "age_range": (int(df["Customer_Age"].min()), int(df["Customer_Age"].max())),
        "quantity_range": (int(df["Order_Quantity"].min()), int(df["Order_Quantity"].max())),
        "years": sorted(int(y) for y in df["Year"].unique()),
    }
