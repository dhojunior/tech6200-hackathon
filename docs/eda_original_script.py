# STEP 1: Import libraries and set file locations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# CSV file is in the Mac Downloads folder
DATA_PATH = Path.home() / "Downloads" / "Sales.csv"

# EDA charts and summary tables will be saved beside this script
OUTPUT_DIR = Path(__file__).resolve().parent / "eda_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.style.use("ggplot")
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda value: f"{value:,.2f}")


# STEP 2: Load the data

if not DATA_PATH.exists():
    print(f"Could not find the data file: {DATA_PATH}")
    print("\nCSV files found in Downloads:")
    for file in (Path.home() / "Downloads").glob("*.csv"):
        print(f" - {file.name}")
    sys.exit("\nCheck that your file is named sale.csv.")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("=" * 70)
print("BIKE SALES DATA: EXPLORATORY DATA ANALYSIS")
print("=" * 70)
print(f"\nFile loaded: {DATA_PATH}")
print(f"Dataset size: {df.shape[0]:,} rows and {df.shape[1]} columns")


# STEP 3: Inspect the data structure

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head().to_string(index=False))

print("\nData types and non-missing values:")
df.info()

print("\nSummary statistics for numeric columns:")
print(df.describe().T.round(2).to_string())


# STEP 4: Check data quality

print("\n" + "=" * 70)
print("DATA QUALITY CHECKS")
print("=" * 70)

# Check missing data
missing_report = pd.DataFrame({
    "Missing_Count": df.isna().sum(),
    "Missing_Percent": (df.isna().mean() * 100).round(2)
})

print("\nMissing values by column:")
print(missing_report.sort_values(
    "Missing_Count", ascending=False
).to_string())

# Check exact duplicate rows
duplicate_count = int(df.duplicated().sum())
print(f"\nExact duplicate rows: {duplicate_count:,}")

# Convert the date column into a date format
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    print(f"Invalid or missing dates: {df['Date'].isna().sum():,}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# Convert expected number columns to numeric
numeric_columns = [
    "Day", "Year", "Customer_Age", "Order_Quantity",
    "Unit_Cost", "Unit_Price", "Profit", "Cost", "Revenue"
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

# Report the number of different values in key category columns
category_columns = [
    "Customer_Gender", "Country", "Product_Category",
    "Sub_Category", "Product"
]

print("\nNumber of distinct values in categorical columns:")
for column in category_columns:
    if column in df.columns:
        print(f"{column}: {df[column].nunique(dropna=True)}")


# STEP 5: Check the financial calculations

print("\n" + "=" * 70)
print("FINANCIAL CONSISTENCY CHECK")
print("=" * 70)

required_for_check = {"Order_Quantity", "Unit_Cost", "Unit_Price"}

if required_for_check.issubset(df.columns):
    df["Revenue_Calculated"] = df["Unit_Price"] * df["Order_Quantity"]
    df["Cost_Calculated"] = df["Unit_Cost"] * df["Order_Quantity"]
    df["Profit_Calculated"] = df["Revenue_Calculated"] - df["Cost_Calculated"]

    checks = [
        ("Revenue", "Revenue_Calculated"),
        ("Cost", "Cost_Calculated"),
        ("Profit", "Profit_Calculated")
    ]

    for reported_column, calculated_column in checks:
        if reported_column in df.columns:
            valid_rows = (
                df[reported_column].notna()
                & df[calculated_column].notna()
            )

            if valid_rows.any():
                matches = np.isclose(
                    df.loc[valid_rows, reported_column],
                    df.loc[valid_rows, calculated_column]
                )

                match_percent = matches.mean() * 100
                mismatch_count = int((~matches).sum())

                print(
                    f"{reported_column}: "
                    f"{match_percent:.2f}% match; "
                    f"{mismatch_count:,} rows differ"
                )
else:
    print("Could not check formulas: required price, cost, or quantity column is missing.")


# STEP 6: Explore numeric variable distributions

print("\n" + "=" * 70)
print("NUMERIC VARIABLE DISTRIBUTIONS")
print("=" * 70)

distribution_columns = [
    column for column in [
        "Customer_Age", "Order_Quantity", "Unit_Cost",
        "Unit_Price", "Revenue", "Profit"
    ]
    if column in df.columns
]

if distribution_columns:
    print("\nMedian and mean values:")
    summary = df[distribution_columns].agg(["mean", "median", "min", "max"]).T
    print(summary.round(2).to_string())

    number_of_columns = 2
    number_of_rows = int(np.ceil(len(distribution_columns) / number_of_columns))

    fig, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(13, 4 * number_of_rows)
    )
    axes = np.array(axes).reshape(-1)

    for axis, column in zip(axes, distribution_columns):
        values = df[column].dropna()
        axis.hist(values, bins=30, color="steelblue", edgecolor="white")
        axis.set_title(f"Distribution of {column}")
        axis.set_xlabel(column)
        axis.set_ylabel("Number of records")

    for axis in axes[len(distribution_columns):]:
        axis.remove()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_numeric_distributions.png", dpi=300)
    plt.show()


# STEP 7: Explore sales and profit over time

print("\n" + "=" * 70)
print("TIME-BASED ANALYSIS")
print("=" * 70)

if {"Date", "Revenue", "Profit", "Order_Quantity"}.issubset(df.columns):
    dated_df = df.dropna(subset=["Date"]).copy()

    # Yearly summary
    yearly_summary = dated_df.groupby(
        dated_df["Date"].dt.year
    ).agg(
        Records=("Date", "size"),
        Units_Sold=("Order_Quantity", "sum"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    )

    print("\nSales and profit by year:")
    print(yearly_summary.round(2).to_string())

    yearly_summary[["Revenue", "Profit"]].plot(
        kind="bar",
        figsize=(10, 5),
        color=["steelblue", "seagreen"]
    )
    plt.title("Revenue and Profit by Year")
    plt.xlabel("Year")
    plt.ylabel("Dataset currency units")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_yearly_revenue_profit.png", dpi=300)
    plt.show()

    # Monthly summary across the dataset
    dated_df["Month_Number"] = dated_df["Date"].dt.month

    monthly_summary = dated_df.groupby("Month_Number").agg(
        Units_Sold=("Order_Quantity", "sum"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    )

    print("\nSales and profit by calendar month:")
    print(monthly_summary.round(2).to_string())

    monthly_summary[["Revenue", "Profit"]].plot(
        figsize=(11, 5),
        marker="o",
        color=["steelblue", "seagreen"]
    )
    plt.title("Revenue and Profit by Calendar Month")
    plt.xlabel("Month number")
    plt.ylabel("Dataset currency units")
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_monthly_revenue_profit.png", dpi=300)
    plt.show()

    # Check how many records are present in each year/month
    dated_df["Year_Month"] = dated_df["Date"].dt.to_period("M").astype(str)
    monthly_coverage = dated_df.groupby("Year_Month").size()

    print("\nMonths with the fewest records:")
    print(monthly_coverage.sort_values().head(12).to_string())


# STEP 8: Compare performance by country

print("\n" + "=" * 70)
print("COUNTRY ANALYSIS")
print("=" * 70)

if {"Country", "Revenue", "Profit", "Order_Quantity"}.issubset(df.columns):
    country_summary = df.groupby("Country").agg(
        Records=("Country", "size"),
        Units_Sold=("Order_Quantity", "sum"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    ).sort_values("Revenue", ascending=False)

    print("\nSales and profit by country:")
    print(country_summary.round(2).to_string())

    country_summary[["Revenue", "Profit"]].plot(
        kind="bar",
        figsize=(12, 6),
        color=["steelblue", "seagreen"]
    )
    plt.title("Revenue and Profit by Country")
    plt.xlabel("Country")
    plt.ylabel("Dataset currency units")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_country_performance.png", dpi=300)
    plt.show()

# STEP 9: Compare performance by product

print("\n" + "=" * 70)
print("PRODUCT ANALYSIS")
print("=" * 70)

if {"Product_Category", "Revenue", "Profit", "Order_Quantity"}.issubset(df.columns):
    category_summary = df.groupby("Product_Category").agg(
        Units_Sold=("Order_Quantity", "sum"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    ).sort_values("Revenue", ascending=False)

    category_summary["Profit_Margin_Percent"] = (
        category_summary["Profit"]
        / category_summary["Revenue"].replace(0, np.nan)
        * 100
    )

    print("\nSales and profit by product category:")
    print(category_summary.round(2).to_string())

    category_summary[["Revenue", "Profit"]].plot(
        kind="bar",
        figsize=(9, 5),
        color=["steelblue", "seagreen"]
    )
    plt.title("Revenue and Profit by Product Category")
    plt.xlabel("Product category")
    plt.ylabel("Dataset currency units")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_product_categories.png", dpi=300)
    plt.show()

if {"Sub_Category", "Revenue", "Profit", "Order_Quantity"}.issubset(df.columns):
    subcategory_summary = df.groupby("Sub_Category").agg(
        Units_Sold=("Order_Quantity", "sum"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    ).sort_values("Revenue", ascending=False)

    print("\nTop 10 subcategories by revenue:")
    print(subcategory_summary.head(10).round(2).to_string())

    top_subcategories = subcategory_summary.head(10).sort_values("Revenue")

    top_subcategories["Revenue"].plot(
        kind="barh",
        figsize=(9, 6),
        color="steelblue"
    )
    plt.title("Top 10 Subcategories by Revenue")
    plt.xlabel("Dataset currency units")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_top_subcategories.png", dpi=300)
    plt.show()

# STEP 10: Explore customer age and gender

print("\n" + "=" * 70)
print("CUSTOMER ANALYSIS")
print("=" * 70)

# Create age groups if the dataset does not already contain them
if "Customer_Age" in df.columns and "Age_Group" not in df.columns:
    age_bins = [0, 24, 34, 64, np.inf]
    age_labels = ["Under 25", "25–34", "35–64", "65+"]

    df["Age_Group"] = pd.cut(
        df["Customer_Age"],
        bins=age_bins,
        labels=age_labels,
        include_lowest=True
    )

if {"Age_Group", "Revenue", "Profit", "Order_Quantity"}.issubset(df.columns):
    age_summary = df.groupby("Age_Group", observed=False).agg(
        Records=("Age_Group", "size"),
        Units_Sold=("Order_Quantity", "sum"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    ).sort_values("Revenue", ascending=False)

    print("\nSales and profit by age group:")
    print(age_summary.round(2).to_string())

    age_summary[["Revenue", "Profit"]].plot(
        kind="bar",
        figsize=(9, 5),
        color=["steelblue", "seagreen"]
    )
    plt.title("Revenue and Profit by Age Group")
    plt.xlabel("Age group")
    plt.ylabel("Dataset currency units")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_age_group_performance.png", dpi=300)
    plt.show()

if {"Customer_Gender", "Revenue", "Profit", "Order_Quantity"}.issubset(df.columns):
    gender_summary = df.groupby("Customer_Gender").agg(
        Records=("Customer_Gender", "size"),
        Units_Sold=("Order_Quantity", "sum"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum")
    )

    print("\nSales and profit by recorded customer gender:")
    print(gender_summary.round(2).to_string())


# STEP 11: Explore correlations

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

correlation_columns = [
    column for column in [
        "Customer_Age", "Order_Quantity", "Unit_Cost",
        "Unit_Price", "Profit", "Cost", "Revenue", "Year"
    ]
    if column in df.columns
]

if len(correlation_columns) >= 2:
    correlation = df[correlation_columns].corr(method="spearman")

    print("\nSpearman correlation matrix:")
    print(correlation.round(2).to_string())

    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)

    axis.set_xticks(range(len(correlation.columns)))
    axis.set_xticklabels(correlation.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(correlation.index)))
    axis.set_yticklabels(correlation.index)

    for row in range(len(correlation.index)):
        for column in range(len(correlation.columns)):
            axis.text(
                column,
                row,
                f"{correlation.iloc[row, column]:.2f}",
                ha="center",
                va="center"
            )

    axis.set_title("Spearman Correlations Between Numeric Variables")
    fig.colorbar(image, ax=axis, label="Correlation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_correlation_matrix.png", dpi=300)
    plt.show()


# STEP 12: Save summary tables and finish

if "Country" in df.columns and "Revenue" in df.columns:
    country_summary.to_csv(OUTPUT_DIR / "country_summary.csv")

if "Product_Category" in df.columns and "Revenue" in df.columns:
    category_summary.to_csv(OUTPUT_DIR / "category_summary.csv")

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)
print(f"Charts and summary tables are saved in:\n{OUTPUT_DIR}")
print(
    "\nInterpretation note: describe revenue and profit as dataset currency units "
    "unless the currency is confirmed. Correlation shows association, not cause."
)