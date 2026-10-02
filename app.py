import pandas as pd
import plotly.express as px
import streamlit as st

from src import eda
from src.data_processing import get_features_target, get_form_options, prepare_data
from src.model import predict_single, train_and_evaluate

st.set_page_config(page_title="Bike Sales Profit Predictor", page_icon="🚲", layout="wide")


@st.cache_data
def get_data():
    return prepare_data()


@st.cache_resource
def get_trained_models(df: pd.DataFrame):
    X, y = get_features_target(df)
    return train_and_evaluate(X, y)


df, report = get_data()
options = get_form_options(df)
with st.spinner("Training models on ~112k orders (first run only)..."):
    results_df, fitted_models, best_name, X_test, y_test = get_trained_models(df)
best_pipeline = fitted_models[best_name]

st.title("🚲 Bike Sales in Europe — Profit Predictor")
st.caption("Kaggle: Bike Sales in Europe — data prep, EDA, ML modeling and live prediction in one app.")

tab_data, tab_eda, tab_model, tab_predict = st.tabs(
    ["📋 Data", "📊 Exploratory Analysis", "Model Evaluation", "🔮 Predict"]
)

with tab_data:
    st.subheader("Raw / Cleaned Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders", f"{df.shape[0]:,}")
    c2.metric("Total Revenue", f"${df['Revenue'].sum():,.0f}")
    c3.metric("Total Profit", f"${df['Profit'].sum():,.0f}")
    c4.metric("Avg. Profit Margin", f"{df['Profit'].sum() / df['Revenue'].sum() * 100:.1f}%")

    st.dataframe(df.head(1000), width="stretch", height=300)
    st.caption("Showing the first 1,000 rows of the cleaned dataset.")

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include="number"), width="stretch")

    st.subheader("Data Quality")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Raw Rows", f"{report['rows_raw']:,}")
    q2.metric("Duplicates Removed", f"{report['duplicates_removed']:,}")
    q3.metric("Missing Values", report["missing_values"])
    q4.metric("Clean Rows", f"{report['rows_clean']:,}")
    st.write(
        "Duplicate rows and missing values were removed, `Date` was parsed and `Month_Number` / "
        "`Profit_Margin` were added during cleaning (`src/data_processing.py`). "
        "`Cost` and `Revenue` are excluded from the model because `Profit = Revenue - Cost`."
    )

with tab_eda:
    st.subheader("Exploratory Data Analysis")
    row1c1, row1c2 = st.columns(2)
    row1c1.plotly_chart(eda.fig_profit_distribution(df), width="stretch")
    row1c2.plotly_chart(eda.fig_category_split(df), width="stretch")

    row2c1, row2c2 = st.columns(2)
    row2c1.plotly_chart(eda.fig_profit_by_category(df), width="stretch")
    row2c2.plotly_chart(eda.fig_profit_by_country(df), width="stretch")

    row3c1, row3c2 = st.columns(2)
    row3c1.plotly_chart(eda.fig_profit_vs_price(df), width="stretch")
    row3c2.plotly_chart(eda.fig_profit_vs_age(df), width="stretch")

    row4c1, row4c2 = st.columns(2)
    row4c1.plotly_chart(eda.fig_profit_by_age_group(df), width="stretch")
    row4c2.plotly_chart(eda.fig_correlation_heatmap(df), width="stretch")

    st.plotly_chart(eda.fig_monthly_trend(df), width="stretch")
    st.plotly_chart(eda.fig_margin_by_sub_category(df), width="stretch")

with tab_model:
    st.subheader("Model Comparison")
    st.dataframe(
        results_df.style.format({"R2": "{:.3f}", "RMSE": "${:,.0f}", "MAE": "${:,.0f}"}),
        width="stretch",
    )
    st.success(f"Best model: **{best_name}** (R² = {results_df.iloc[0]['R2']:.3f})")

    metric_fig = px.bar(
        results_df, x="model", y="R2", color="model", title="R² Score by Model",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(metric_fig, width="stretch")

    st.subheader("Predicted vs Actual (Best Model, Test Set)")
    preds = best_pipeline.predict(X_test)
    comp_df = pd.DataFrame({"Actual": y_test.values, "Predicted": preds}).sample(
        min(5000, len(y_test)), random_state=42
    )
    scatter = px.scatter(
        comp_df, x="Actual", y="Predicted", opacity=0.6,
        title=f"{best_name}: Predicted vs Actual Profit (5,000 test orders)",
    )
    max_val = float(comp_df.max().max())
    scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(dash="dash", color="gray"))
    st.plotly_chart(scatter, width="stretch")

with tab_predict:
    st.subheader("Estimate Order Profit")
    st.write(f"Prediction powered by the best-performing model: **{best_name}**")

    # Category sits outside the form so the sub-category list refreshes when it changes
    category = st.selectbox("Product Category", options["categories"])

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        sub_category = c1.selectbox("Sub-Category", options["sub_categories"][category])
        country = c2.selectbox("Country", options["countries"])
        gender = c3.selectbox("Customer Gender", options["genders"])

        c4, c5, c6 = st.columns(3)
        age = c4.slider("Customer Age", *options["age_range"], 35)
        quantity = c5.slider("Order Quantity", *options["quantity_range"], 10)
        month = c6.slider("Month", 1, 12, 6)

        c7, c8, c9 = st.columns(3)
        unit_cost = c7.number_input("Unit Cost ($)", min_value=0.0, value=50.0, step=1.0)
        unit_price = c8.number_input("Unit Price ($)", min_value=0.0, value=100.0, step=1.0)
        year = c9.selectbox("Year", options["years"], index=len(options["years"]) - 1)

        submitted = st.form_submit_button("Predict Profit")

    if submitted:
        prediction = predict_single(
            best_pipeline, age, gender, country, category, sub_category,
            quantity, unit_cost, unit_price, month, year,
        )
        st.metric("Estimated Order Profit", f"${prediction:,.2f}")
        st.caption(f"Dataset average is ${df['Profit'].mean():,.2f} per order for comparison.")
