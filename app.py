"""Streamlit app: Data, Charts, Model and Predict tabs.

Run from the project root:
    streamlit run app.py
"""

import streamlit as st

from src import eda
from src.data_processing import get_features_target, get_form_options, prepare_data
from src.model import predict_single, train_and_evaluate

st.set_page_config(page_title="Bike Sales Profit Predictor", layout="wide")


@st.cache_data
def get_data():
    """Load and clean the data once."""
    return prepare_data()


@st.cache_resource
def get_models(_df):
    """Train the models once (this can take a minute the first time)."""
    X, y = get_features_target(_df)
    return train_and_evaluate(X, y)


df, report = get_data()
options = get_form_options(df)

st.title("Bike Sales in Europe - Profit Predictor")
st.caption("Data preparation, exploratory analysis, machine learning and live prediction in one app.")

with st.spinner("Training the models. This only happens the first time..."):
    results_df, fitted_models, best_name, X_test, y_test = get_models(df)
best_model = fitted_models[best_name]

tab_data, tab_charts, tab_model, tab_predict = st.tabs(["Data", "Charts", "Model", "Predict"])

# ---- Data tab
with tab_data:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Raw rows", f"{report['rows_raw']:,}")
    c2.metric("Clean rows", f"{report['rows_clean']:,}")
    c3.metric("Duplicates removed", f"{report['duplicates_removed']:,}")
    c4.metric("Missing values", report["missing_values"])
    c5.metric("Average profit", f"${df['Profit'].mean():,.0f}")
    st.subheader("Clean data (first 200 rows)")
    st.dataframe(df.head(200), width="stretch", height=320)
    st.subheader("Summary statistics")
    st.dataframe(eda.numeric_summary(df).round(2), width="stretch")

# ---- Charts tab
with tab_charts:
    name = st.selectbox("Choose a chart", list(eda.CHARTS))
    st.pyplot(eda.CHARTS[name](df))
    with st.expander("Summary table by country"):
        st.dataframe(eda.summary_by(df, "Country").round(2), width="stretch")

# ---- Model tab
with tab_model:
    st.subheader("Model comparison (test set)")
    st.dataframe(
        results_df.style.format({"R2": "{:.3f}", "RMSE": "${:,.0f}", "MAE": "${:,.0f}"}),
        width="stretch",
    )
    st.success(f"Best model: {best_name} (R2 = {results_df.iloc[0]['R2']:.3f})")

# ---- Predict tab
with tab_predict:
    st.subheader("Estimate the profit of an order")
    left, right = st.columns(2)
    age = left.slider("Customer age", *options["age_range"], value=35)
    gender = right.selectbox("Gender", options["genders"])
    country = left.selectbox("Country", options["countries"])
    category = right.selectbox("Product category", options["categories"])
    sub_category = left.selectbox("Sub-category", options["sub_categories"][category])
    quantity = right.slider("Order quantity", *options["quantity_range"], value=5)
    unit_cost = left.number_input("Unit cost", min_value=1, value=100)
    unit_price = right.number_input("Unit price", min_value=1, value=170)
    month = left.slider("Month", 1, 12, 6)
    year = right.selectbox("Year", options["years"], index=len(options["years"]) - 1)

    if st.button("Predict profit"):
        estimate = predict_single(
            best_model,
            Customer_Age=age, Order_Quantity=quantity, Unit_Cost=float(unit_cost), Unit_Price=float(unit_price),
            Month_Number=month, Year=year, Customer_Gender=gender, Country=country,
            Product_Category=category, Sub_Category=sub_category,
        )
        st.metric("Estimated profit", f"${estimate:,.2f}")
        st.caption(f"Average profit in the data: ${df['Profit'].mean():,.2f}  |  model: {best_name}")
