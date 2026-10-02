"""Entry point of the Bike Sales app.

Steps: prepare the data -> train the models -> open the GUI.
Run from the project root:
    python main.py            (opens the window)
    python main.py --no-gui   (runs everything and prints a summary, good for testing)
"""

import argparse

from src.data_processing import get_features_target, get_form_options, prepare_data
from src.model import train_and_evaluate


def build_app_data() -> dict:
    """Run data preparation and model training.
    Returns everything the GUI needs."""
    df, report = prepare_data()
    X, y = get_features_target(df)
    results_df, fitted_models, best_name, X_test, y_test = train_and_evaluate(X, y)
    return {
        "df": df,                      # clean data
        "report": report,              # data quality numbers (Data tab)
        "options": get_form_options(df),  # drop-down values (Predict tab)
        "results_df": results_df,      # model comparison table (Model tab)
        "fitted_models": fitted_models,
        "best_name": best_name,
        "X_test": X_test,
        "y_test": y_test,
    }


def print_summary(data: dict) -> None:
    """Show the main results in the terminal."""
    report = data["report"]
    print(f"Rows: {report['rows_raw']} raw -> {report['rows_clean']} clean "
          f"({report['duplicates_removed']} duplicates removed)")
    print("\nModel comparison:")
    print(data["results_df"].to_string(index=False))
    print(f"\nBest model: {data['best_name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bike Sales in Europe - profit predictor")
    parser.add_argument("--no-gui", action="store_true", help="run without opening the window")
    args = parser.parse_args()

    try:
        data = build_app_data()
    except FileNotFoundError:
        print("Could not find data/Sales.csv. Please run this file from the project root folder.")
        return

    if args.no_gui:
        print_summary(data)
        return

    from src.gui import App  # imported here so --no-gui works even without a display

    app = App(**data)
    app.mainloop()


if __name__ == "__main__":
    main()
