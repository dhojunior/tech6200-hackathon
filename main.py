"""Entry point of the Bike Sales app.

Just run this file (in PyCharm: right click main.py > Run, or in a terminal: python main.py).
The first time, it creates a local environment (.venv) and installs the packages by itself.

    python main.py            (opens the Streamlit app in the browser)
    python main.py --no-gui   (runs data preparation + models and prints a summary, good for testing)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_FILE = ROOT / "app.py"
REQUIREMENTS = ROOT / "requirements.txt"
VENV_DIR = ROOT / ".venv"


def build_app_data() -> dict:
    """Run data preparation and model training. Returns the main results."""
    from src.data_processing import get_features_target, get_form_options, prepare_data
    from src.model import train_and_evaluate

    df, report = prepare_data()
    X, y = get_features_target(df)
    results_df, fitted_models, best_name, X_test, y_test = train_and_evaluate(X, y)
    return {
        "df": df,                         # clean data
        "report": report,                 # data quality numbers
        "options": get_form_options(df),  # values for the prediction form
        "results_df": results_df,         # model comparison table
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


def venv_python() -> Path:
    """Path of the Python inside the local .venv folder."""
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_environment() -> Path:
    """Create the .venv and install the packages if they are not there yet."""
    python = venv_python()
    if not python.exists():
        print("First run: creating the local environment (.venv)...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])

    packages_ok = subprocess.run(
        [str(python), "-c", "import streamlit, plotly, sklearn, pandas, joblib"],
        capture_output=True,
    ).returncode == 0
    if not packages_ok:
        print("Installing the packages. This can take a few minutes, please wait...")
        subprocess.check_call([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    return python


def main() -> None:
    parser = argparse.ArgumentParser(description="Bike Sales in Europe - profit predictor")
    parser.add_argument("--no-gui", action="store_true", help="run without opening the app")
    args = parser.parse_args()

    if args.no_gui:
        print_summary(build_app_data())
        return

    try:
        python = ensure_environment()
    except subprocess.CalledProcessError:
        print("Could not set up the environment automatically. Please check your internet "
              "connection and that Python 3.11 or newer is installed, then run this file again.")
        return

    # Streamlit loads the data and trains the models by itself (see app.py)
    env = dict(os.environ, STREAMLIT_BROWSER_GATHER_USAGE_STATS="false")  # no first-run email prompt
    subprocess.run([str(python), "-m", "streamlit", "run", str(APP_FILE)], check=False, env=env)


if __name__ == "__main__":
    main()
