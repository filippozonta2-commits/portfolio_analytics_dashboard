# Portfolio Analytics Dashboard

An interactive Streamlit application for building, analyzing, optimizing, and simulating multi-asset portfolios.

## Features

- Portfolio performance and risk metrics
- Historical and parametric VaR/CVaR
- Drawdown and rolling analytics
- Benchmark comparison, alpha, beta, and capture ratios
- Correlation and covariance analysis
- Efficient frontier and constrained portfolio optimization
- Bootstrap and distribution-based Monte Carlo simulation
- Company fundamentals for supported securities
- CSV exports for analysis outputs

## Project structure

```text
portfolio-analytics-dashboard/
├── app.py
├── requirements.txt
├── .streamlit/config.toml
└── src/
    ├── analytics.py
    ├── benchmark.py
    ├── charts.py
    ├── data.py
    ├── fundamentals.py
    ├── metrics.py
    ├── optimization.py
    ├── sidebar.py
    ├── simulation.py
    └── utils.py
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data sources

Market and fundamental data are downloaded from Yahoo Finance. Treasury yields are retrieved from FRED, with a configurable fallback rate when the service is unavailable.

## Notes

This project is intended for educational and analytical purposes only. It is not investment advice, and model outputs should not be treated as forecasts or trading recommendations.
