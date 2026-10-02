"""Exploratory data analysis (EDA) for the Bike Sales in Europe dataset.

Every chart function takes the clean dataframe and returns a matplotlib Figure,
so the same chart can be used in the GUI or saved as an image.
Every summary function returns a pandas table.

To run the full EDA in the terminal (from the project root):
    python -m src.eda
"""

import numpy as np
import pandas as pd
from matplotlib import style
from matplotlib.figure import Figure

style.use("ggplot")

FIGSIZE = (8, 4.6)  # good size for the GUI window
BLUE, GREEN = "steelblue", "seagreen"
AGE_ORDER = ["Youth (<25)", "Young Adults (25-34)", "Adults (35-64)", "Seniors (64+)"]
CURRENCY_LABEL = "Dataset currency units"  # the currency is not confirmed in the dataset


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def summary_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Records, units sold, revenue, profit and margin grouped by one column."""
    table = df.groupby(column, observed=True).agg(
        Records=(column, "size"),
        Units_Sold=("Order_Quantity", "sum"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum"),
    )
    table["Profit_Margin_Percent"] = table["Profit"] / table["Revenue"].replace(0, np.nan) * 100
    return table.sort_values("Revenue", ascending=False)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, min and max of the main numeric columns."""
    columns = [c for c in ["Customer_Age", "Order_Quantity", "Unit_Cost", "Unit_Price", "Revenue", "Profit"]
               if c in df.columns]
    return df[columns].agg(["mean", "median", "min", "max"]).T


def financial_check(df: pd.DataFrame) -> pd.DataFrame:
    """Compare Revenue, Cost and Profit with the values calculated from price, cost and quantity."""
    revenue = df["Unit_Price"] * df["Order_Quantity"]
    cost = df["Unit_Cost"] * df["Order_Quantity"]
    calculated = {"Revenue": revenue, "Cost": cost, "Profit": revenue - cost}
    rows = []
    for name, values in calculated.items():
        matches = np.isclose(df[name], values)
        rows.append({"Column": name, "Match_Percent": matches.mean() * 100,
                     "Rows_That_Differ": int((~matches).sum())})
    return pd.DataFrame(rows).set_index("Column")


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between the numeric columns."""
    columns = [c for c in ["Customer_Age", "Order_Quantity", "Unit_Cost", "Unit_Price",
                           "Profit", "Cost", "Revenue", "Year"] if c in df.columns]
    return df[columns].corr(method="spearman")


def _age_groups(df: pd.DataFrame) -> pd.Series:
    """Use the Age_Group column, or create one if the data does not have it."""
    if "Age_Group" in df.columns:
        return df["Age_Group"]
    return pd.cut(df["Customer_Age"], bins=[0, 24, 34, 64, np.inf], labels=AGE_ORDER, include_lowest=True)


# ---------------------------------------------------------------------------
# Charts (each one returns a matplotlib Figure)
# ---------------------------------------------------------------------------

def _revenue_profit_bars(table: pd.DataFrame, title: str, xlabel: str, rotation: int = 0) -> Figure:
    """Helper: bar chart with Revenue and Profit side by side."""
    fig = Figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    table[["Revenue", "Profit"]].plot(kind="bar", ax=ax, color=[BLUE, GREEN])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(CURRENCY_LABEL)
    ax.tick_params(axis="x", rotation=rotation)
    fig.tight_layout()
    return fig


def fig_numeric_distributions(df: pd.DataFrame) -> Figure:
    """Histograms of the main numeric columns."""
    columns = list(numeric_summary(df).index)
    rows = int(np.ceil(len(columns) / 2))
    fig = Figure(figsize=(10, 3 * rows))
    for position, column in enumerate(columns, start=1):
        ax = fig.add_subplot(rows, 2, position)
        ax.hist(df[column].dropna(), bins=30, color=BLUE, edgecolor="white")
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Number of records")
    fig.tight_layout()
    return fig


def fig_age_distribution(df: pd.DataFrame) -> Figure:
    """How old are the customers?"""
    fig = Figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    ax.hist(df["Customer_Age"], bins=30, color=BLUE, edgecolor="white")
    ax.set_title("Customer age distribution")
    ax.set_xlabel("Customer age")
    ax.set_ylabel("Number of records")
    fig.tight_layout()
    return fig


def fig_sales_over_time(df: pd.DataFrame) -> Figure:
    """Revenue and profit by year."""
    yearly = df.groupby("Year")[["Revenue", "Profit"]].sum()
    return _revenue_profit_bars(yearly, "Revenue and Profit by Year", "Year")


def fig_monthly_pattern(df: pd.DataFrame) -> Figure:
    """Revenue and profit by calendar month (all years together)."""
    monthly = df.groupby(df["Date"].dt.month)[["Revenue", "Profit"]].sum()
    fig = Figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    monthly.plot(ax=ax, marker="o", color=[BLUE, GREEN])
    ax.set_title("Revenue and Profit by Calendar Month")
    ax.set_xlabel("Month number")
    ax.set_ylabel(CURRENCY_LABEL)
    ax.set_xticks(range(1, 13))
    fig.tight_layout()
    return fig


def fig_revenue_by_country(df: pd.DataFrame) -> Figure:
    """Revenue and profit for each country."""
    return _revenue_profit_bars(summary_by(df, "Country"), "Revenue and Profit by Country", "Country", rotation=40)


def fig_profit_by_category(df: pd.DataFrame) -> Figure:
    """Revenue and profit for each product category."""
    return _revenue_profit_bars(summary_by(df, "Product_Category"),
                                "Revenue and Profit by Product Category", "Product category")


def fig_top_subcategories(df: pd.DataFrame, top: int = 10) -> Figure:
    """The sub-categories with the highest revenue."""
    best = summary_by(df, "Sub_Category").head(top).sort_values("Revenue")
    fig = Figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    best["Revenue"].plot(kind="barh", ax=ax, color=BLUE)
    ax.set_title(f"Top {top} Sub-categories by Revenue")
    ax.set_xlabel(CURRENCY_LABEL)
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def fig_age_group_performance(df: pd.DataFrame) -> Figure:
    """Revenue and profit for each customer age group."""
    table = summary_by(df.assign(Age_Group=_age_groups(df)), "Age_Group")
    table = table.reindex([g for g in AGE_ORDER if g in table.index])
    return _revenue_profit_bars(table, "Revenue and Profit by Age Group", "Age group", rotation=20)


def fig_correlation(df: pd.DataFrame) -> Figure:
    """Heatmap of the Spearman correlations."""
    corr = correlation_table(df)
    fig = Figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111)
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.grid(False)
    for row in range(len(corr.index)):
        for col in range(len(corr.columns)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Spearman Correlations Between Numeric Variables")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    return fig


# Name shown in the GUI list -> chart function
CHARTS = {
    "Revenue and profit by year": fig_sales_over_time,
    "Revenue and profit by month": fig_monthly_pattern,
    "Revenue and profit by country": fig_revenue_by_country,
    "Revenue and profit by category": fig_profit_by_category,
    "Top 10 sub-categories": fig_top_subcategories,
    "Revenue and profit by age group": fig_age_group_performance,
    "Customer age distribution": fig_age_distribution,
    "Numeric distributions": fig_numeric_distributions,
    "Correlation heatmap": fig_correlation,
}


# ---------------------------------------------------------------------------
# Run the full EDA in the terminal: python -m src.eda
# ---------------------------------------------------------------------------

def run_full_eda(output_dir: str = "eda_outputs") -> None:
    from pathlib import Path

    from src.data_processing import prepare_data

    df, report = prepare_data()
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    print("=" * 70)
    print("BIKE SALES DATA: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    print(f"Rows: {report['rows_raw']:,} raw -> {report['rows_clean']:,} clean "
          f"({report['duplicates_removed']:,} duplicates removed, {report['missing_values']} missing values)")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    print("\nFinancial consistency check:")
    print(financial_check(df).round(2).to_string())
    print("\nNumeric summary:")
    print(numeric_summary(df).round(2).to_string())
    for column in ["Country", "Product_Category", "Customer_Gender"]:
        print(f"\nSales and profit by {column}:")
        print(summary_by(df, column).round(2).to_string())

    for number, (name, function) in enumerate(CHARTS.items(), start=1):
        file = out / f"{number:02d}_{function.__name__}.png"
        function(df).savefig(file, dpi=150)
    summary_by(df, "Country").to_csv(out / "country_summary.csv")
    summary_by(df, "Product_Category").to_csv(out / "category_summary.csv")

    print("\n" + "=" * 70)
    print(f"EDA complete. Charts and tables saved in: {out.resolve()}")
    print("Note: revenue and profit are in dataset currency units (the currency is not confirmed). "
          "Correlation shows association, not cause.")


if __name__ == "__main__":
    run_full_eda()
