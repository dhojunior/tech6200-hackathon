"""Entry point of the Bike Sales app.

Run from the project root:
    python main.py                (opens the Streamlit app in the browser)
    python main.py --ui tkinter   (opens the Tkinter window instead)
    python main.py --no-gui       (runs data + models and prints a summary, good for testing)
"""

import argparse
import subprocess
import sys

from src.data_processing import get_features_target, get_form_options, prepare_data
from src.model import train_and_evaluate


def build_app_data() -> dict:
    """Run data preparation and model training. Returns everything a GUI needs."""
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
    parser.add_argument("--ui", choices=["streamlit", "tkinter"], default="streamlit",
                        help="which interface to open (default: streamlit)")
    parser.add_argument("--no-gui", action="store_true", help="run without opening any interface")
    args = parser.parse_args()

    if args.no_gui:
        print_summary(build_app_data())
        return

    if args.ui == "streamlit":
        # Streamlit loads the data and trains the models by itself (see app.py)
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=False)
        return

    from src.gui import App  # Tkinter window (only needed for --ui tkinter)

    App(**build_app_data()).mainloop()


if __name__ == "__main__":
    main()
