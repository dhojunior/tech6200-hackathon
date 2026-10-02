"""Exploratory data analysis (EDA) for the Bike Sales in Europe dataset.

Every chart function takes the clean dataframe and returns a Plotly figure,
so it can be shown in the Streamlit app with st.plotly_chart().
Every summary function returns a pandas table.

To run the full EDA in the terminal (from the project root):
    python -m src.eda
"""

import numpy as np
import pandas as pd
import plotly.express as px

PALETTE = px.colors.qualitative.Set2
TEMPLATE = "plotly_white"
AGE_ORDER = ["Youth (<25)", "Young Adults (25-34)", "Adults (35-64)", "Seniors (64+)"]
SAMPLE_SIZE = 5000  # scatter charts show a random sample, so they stay fast
FULL_YEARS_START = 2013  # 2011 and 2012 have about 10 times fewer records than the other years


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def summary_by(df: pd.DataFrame, column) -> pd.DataFrame:
    """Records, units sold, revenue, profit and margin grouped by one column (or a list of columns)."""
    table = df.groupby(column, observed=True).agg(
        Records=("Profit", "size"),
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


def _sample(df: pd.DataFrame) -> pd.DataFrame:
    return df.sample(min(SAMPLE_SIZE, len(df)), random_state=42)


def _revenue_profit_bars(table: pd.DataFrame, x_name: str, title: str):
    """Helper: grouped bars with Revenue and Profit side by side."""
    data = table.reset_index().melt(id_vars=x_name, value_vars=["Revenue", "Profit"],
                                    var_name="Measure", value_name="Amount")
    fig = px.bar(data, x=x_name, y="Amount", color="Measure", barmode="group", title=title,
                 color_discrete_sequence=PALETTE, template=TEMPLATE)
    fig.update_layout(yaxis_title="Dataset currency units")
    return fig


# ---------------------------------------------------------------------------
# Charts (each one returns a Plotly figure)
# ---------------------------------------------------------------------------

def fig_profit_distribution(df: pd.DataFrame):
    """How much profit does a typical order make? (shown up to the 99th percentile)"""
    limit = df["Profit"].quantile(0.99)
    fig = px.histogram(df[df["Profit"] <= limit], x="Profit", nbins=50, marginal="box",
                       title="Distribution of Profit per Order (up to the 99th percentile)",
                       color_discrete_sequence=PALETTE, template=TEMPLATE)
    fig.update_layout(yaxis_title="Number of orders")
    return fig


def fig_category_split(df: pd.DataFrame):
    """Share of the total revenue by product category."""
    table = summary_by(df, "Product_Category").reset_index()
    return px.pie(table, names="Product_Category", values="Revenue", hole=0.4,
                  title="Share of Revenue by Product Category",
                  color_discrete_sequence=PALETTE, template=TEMPLATE)


def fig_profit_by_category(df: pd.DataFrame):
    """Revenue and profit for each product category."""
    return _revenue_profit_bars(summary_by(df, "Product_Category"), "Product_Category",
                                "Revenue and Profit by Product Category")


def fig_profit_by_country(df: pd.DataFrame):
    """Revenue and profit for each country."""
    return _revenue_profit_bars(summary_by(df, "Country"), "Country", "Revenue and Profit by Country")


def fig_profit_vs_price(df: pd.DataFrame):
    """Does a higher unit price mean a higher profit? (random sample of orders)"""
    return px.scatter(_sample(df), x="Unit_Price", y="Profit", color="Product_Category", opacity=0.6,
                      hover_data=["Sub_Category", "Order_Quantity"],
                      title=f"Profit vs Unit Price (sample of {SAMPLE_SIZE:,} orders)",
                      color_discrete_sequence=PALETTE, template=TEMPLATE)


def fig_profit_vs_age(df: pd.DataFrame):
    """Does the customer age change the profit? (random sample of orders)"""
    return px.scatter(_sample(df), x="Customer_Age", y="Profit", color="Product_Category", opacity=0.6,
                      hover_data=["Sub_Category", "Order_Quantity"],
                      title=f"Profit vs Customer Age (sample of {SAMPLE_SIZE:,} orders)",
                      color_discrete_sequence=PALETTE, template=TEMPLATE)


def fig_profit_by_age_group(df: pd.DataFrame):
    """Revenue and profit for each customer age group."""
    table = summary_by(df.assign(Age_Group=_age_groups(df)), "Age_Group")
    table = table.reindex([g for g in AGE_ORDER if g in table.index])
    return _revenue_profit_bars(table, "Age_Group", "Revenue and Profit by Age Group")


def fig_correlation_heatmap(df: pd.DataFrame):
    """Spearman correlation between the numeric columns."""
    return px.imshow(correlation_table(df), text_auto=".2f", color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, title="Spearman Correlations Between Numeric Variables",
                     template=TEMPLATE)


def fig_monthly_trend(df: pd.DataFrame):
    """Average profit per calendar month.

    Only the years with full records are used (2011 and 2012 have far fewer rows),
    and the sum of each year-month is averaged, so the months can be compared.
    """
    recent = df[df["Year"] >= FULL_YEARS_START]
    per_year_month = recent.groupby(["Year", "Month_Number"])["Profit"].sum()
    monthly = per_year_month.groupby("Month_Number").mean().reset_index(name="Average_Profit")
    fig = px.line(monthly, x="Month_Number", y="Average_Profit", markers=True,
                  title=f"Average Monthly Profit ({FULL_YEARS_START} onwards)",
                  color_discrete_sequence=PALETTE, template=TEMPLATE)
    fig.update_layout(xaxis=dict(tickmode="linear", dtick=1, title="Month"), yaxis_title="Average profit")
    return fig


def fig_margin_by_sub_category(df: pd.DataFrame):
    """Profit margin of each sub-category (profit as a % of revenue)."""
    table = summary_by(df, ["Product_Category", "Sub_Category"]).reset_index()
    table = table.sort_values("Profit_Margin_Percent")
    fig = px.bar(table, x="Profit_Margin_Percent", y="Sub_Category", color="Product_Category",
                 orientation="h", hover_data=["Revenue", "Profit"],
                 title="Profit Margin by Sub-category (%)",
                 color_discrete_sequence=PALETTE, template=TEMPLATE)
    fig.update_layout(xaxis_title="Profit margin (%)", yaxis_title="", height=560)
    return fig


# Name -> chart function (used to save every chart in run_full_eda)
CHARTS = {
    "profit_distribution": fig_profit_distribution,
    "category_split": fig_category_split,
    "profit_by_category": fig_profit_by_category,
    "profit_by_country": fig_profit_by_country,
    "profit_vs_price": fig_profit_vs_price,
    "profit_vs_age": fig_profit_vs_age,
    "profit_by_age_group": fig_profit_by_age_group,
    "correlation_heatmap": fig_correlation_heatmap,
    "monthly_trend": fig_monthly_trend,
    "margin_by_sub_category": fig_margin_by_sub_category,
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
        function(df).write_html(out / f"{number:02d}_{name}.html")
    summary_by(df, "Country").to_csv(out / "country_summary.csv")
    summary_by(df, "Product_Category").to_csv(out / "category_summary.csv")

    print("\n" + "=" * 70)
    print(f"EDA complete. Charts (HTML, open in the browser) and tables saved in: {out.resolve()}")
    print("Note: revenue and profit are in dataset currency units (the currency is not confirmed). "
          "Correlation shows association, not cause.")


if __name__ == "__main__":
    run_full_eda()
