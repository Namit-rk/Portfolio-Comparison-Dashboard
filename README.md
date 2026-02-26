# Portfolio Comparison Dashboard

A Python-based quantitative research dashboard for comparing multiple portfolio construction strategies on the same stock universe.

This project:
- Downloads and updates historical market data.
- Validates and cleans data into a modeling-ready return matrix.
- Builds multiple portfolio strategies (equal weight, max Sharpe, minimum variance, risk parity, momentum, and optional user-defined portfolios).
- Visualizes performance, risk, allocation, efficient frontier, Monte Carlo outcomes, and out-of-sample backtests in an interactive Dash app.

---

## Features

- **End-to-end pipeline**: ingestion → validation → feature engineering → analytics app.
- **Interactive controls**:
  - Select assets.
  - Select date ranges.
  - Add custom user portfolios with manual weights.
- **Portfolio analytics**:
  - Cumulative returns.
  - Drawdowns.
  - Strategy metric comparison (return, volatility, Sharpe, max drawdown, VaR, CVaR).
  - Correlation heatmap and rolling volatility.
  - Efficient frontier with random portfolio cloud.
  - Monthly return heatmaps.
  - Monte Carlo simulation paths.
  - Out-of-sample backtesting metrics.

---

## Tech Stack

- **Backend / analytics**: Python, NumPy, pandas, SciPy
- **Visualization**: Plotly
- **Web app**: Dash + dash-bootstrap-components
- **Data source**: yfinance
- **Storage**: CSV (raw), Parquet (processed)
- **Testing**: pytest

---

## Repository Structure

```text
Portfolio-Comparison-Dashboard/
├── scr/
│   ├── main.py                   # Orchestrates data pipeline + starts app
│   ├── dash_app.py               # Dash layout and callbacks
│   ├── config.py                 # Tickers, date range, risk-free assumptions
│   ├── data_ingestion.py         # Downloads/updates raw market data (CSV)
│   ├── data_validation.py        # Data quality checks
│   ├── data_cleaning.py          # Merge, clean, and create returns parquet
│   ├── portfolio_making.py       # MarketData/Portfolio objects and metrics
│   ├── optimization_methods.py   # Optimization strategy implementations
│   ├── efficient_frontier.py     # Frontier computation and plotting helpers
│   └── logger.py                 # Logging setup
├── tests/                        # Unit and pipeline tests
├── data/
│   ├── raw/                      # Per-ticker CSVs (created at runtime)
│   └── processed/                # returns.parquet (created at runtime)
├── requirements.txt
├── Procfile                      # Gunicorn entrypoint
└── README.md
```

> Note: the source package is named `scr` in this repository.

---

## Getting Started

### 1) Clone the repository

```bash
git clone <your-repo-url>
cd Portfolio-Comparison-Dashboard
```

### 2) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Edit `scr/config.py` to customize:
- `STOCKS`: portfolio universe ticker list.
- `START_DATE`, `END_DATE`: historical window.
- `RISK_FREE_RATE`: Sharpe ratio assumption.
- `NUM_TRADING_DAYS`: annualization constant.

Example tickers currently include NSE symbols.

---

## Running the Project

### Option A: Run full pipeline + app (recommended first run)

```bash
python -m scr.main
```

This will:
1. Download/update raw data for all configured tickers.
2. Run validation checks.
3. Build cleaned returns dataset (`data/processed/returns.parquet`).
4. Launch the Dash app in debug mode.

### Option B: Run app directly

```bash
python -m scr.dash_app
```

Use this after you already have `data/processed/returns.parquet` generated.

---

## Running Tests

```bash
pytest -q
```

Tests cover:
- Data ingestion formatting and update behavior.
- Validation checks (duplicates, missing values, abnormal returns).
- Feature engineering pipeline and parquet output.

---

## Deployment

A `Procfile` is included for WSGI deployment with Gunicorn:

```bash
web: gunicorn scr.dash_app:server
```

Typical deployment flow:
1. Ensure dependencies are installed.
2. Ensure `data/processed/returns.parquet` exists (run pipeline first).
3. Start app with Gunicorn using the Procfile command.

---

## Data and Logging

- Raw downloaded files are stored under `data/raw/`.
- Processed returns are stored under `data/processed/returns.parquet`.
- Runtime logs are written to `logs/workflow.log`.

---

## Troubleshooting

- **`returns.parquet` missing**: run `python -m scr.main` once to build processed data.
- **No/partial ticker data**: verify ticker symbols and market suffixes in `scr/config.py`.
- **App callbacks error after changing assets**: ensure custom portfolio weights sum to approximately 1.

---

## Future Improvements (Suggested)

- Add rebalance frequency support for strategies.
- Add transaction costs/slippage assumptions for backtests.
- Add persistent storage for user-created portfolios.
- Add CI workflow for automated tests and linting.
