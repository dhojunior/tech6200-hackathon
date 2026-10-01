"""Simple checks for src/data_processing.py. Run from the project root: python -m tests.test_data_processing"""

from src.data_processing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    get_features_target,
    get_form_options,
    prepare_data,
)

df, report = prepare_data()
X, y = get_features_target(df)
options = get_form_options(df)

# 1. Duplicates were removed and nothing is missing
assert df.duplicated().sum() == 0, "There are still duplicate rows"
assert df.isna().sum().sum() == 0, "There are still missing values"
print(f"OK 1: clean data has {report['rows_clean']} rows ({report['duplicates_removed']} duplicates removed)")

# 2. New columns exist
assert {"Month_Number", "Profit_Margin"} <= set(df.columns), "New columns are missing"
print("OK 2: new columns were created")

# 3. Model inputs do not include Cost or Revenue (they would leak Profit)
assert "Cost" not in X.columns and "Revenue" not in X.columns, "Leakage: Cost or Revenue is in X"
assert list(X.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
assert len(X) == len(y)
print(f"OK 3: X has {X.shape[1]} columns and the target is {y.name}")

# 4. Form options are ready for the GUI
assert "Bikes" in options["sub_categories"], "Sub-categories are missing"
print(f"OK 4: form options ready ({len(options['countries'])} countries)")

print("All checks passed.")
