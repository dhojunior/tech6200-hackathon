# Bike Sales in Europe - Profit Predictor

TECH6200 Advanced Programming, Assessment 3 (Hackathon), Group 3.

A Streamlit app that cleans the Bike Sales in Europe data, shows exploratory charts, compares
machine learning models and predicts the profit of a new order.

## What you need

- Python 3.11 or newer (tested with Python 3.12 and 3.13)
- An internet connection to install the packages the first time

## How to run

Just run `main.py`. You do not need to type any commands.

- **PyCharm:** open the folder, right click `main.py` and choose **Run**.
- **Any computer with Python:** double-click `main.py`, or open it in your editor (IDLE, VS Code, Spyder) and press Run.

The first time, `main.py` does everything by itself: it creates a local environment (`.venv`),
installs the packages from `requirements.txt` and starts the Streamlit app. This can take a few
minutes, so please wait. The app opens in the browser (normally at http://localhost:8501).

When the page opens, the models are trained once (about 2 to 3 minutes). After that, the result
is kept in memory and the app is fast. The next runs skip the installation and start much sooner.

To stop the app, close the terminal window or press Ctrl+C.

## Other commands

| Command | What it does |
|---|---|
| `python main.py --no-gui` | Trains the models and prints the comparison table in the terminal |
| `python -m src.eda` | Runs the full EDA and saves the charts (HTML) and tables in `eda_outputs/` |
| `python -m tests.test_data_processing` | Runs simple checks on the data preparation |

## Project structure

```
app.py                  Streamlit interface (Data, Exploratory Analysis, Model Evaluation, Predict)
main.py                 Entry point (opens the app or prints a model summary)
requirements.txt        Packages needed
data/Sales.csv          Bike Sales in Europe dataset
src/data_processing.py  Loading, cleaning and features
src/eda.py              Exploratory charts and summary tables
src/model.py            Training, comparison and prediction of the models
tests/                  Simple checks for the data preparation
docs/log.md             Contribution log
```

## Notes

- The target is **Profit**. `Cost` and `Revenue` are not used as model inputs, because
  Profit = Revenue - Cost and they would give away the answer.
- The data preparation removes 1,000 duplicate rows (113,036 rows become 112,036).
- Revenue and profit are shown in dataset currency units because the currency is not confirmed.
- Tested with pandas 3.0, scikit-learn 1.9, Streamlit 1.64 and Plotly 7.1.
