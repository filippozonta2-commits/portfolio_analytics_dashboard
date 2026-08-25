# PortfolioLab

**Professional portfolio analytics, risk, optimization and scenario simulation in Streamlit.**

Built by **Filippo Zonta, MSc**.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?logo=streamlit&logoColor=white)
![CI](https://github.com/filippozonta2-commits/portfolio_analytics_dashboard/actions/workflows/quality.yml/badge.svg)

## What it does

- Executive portfolio overview with benchmark-relative performance
- Annualized return, volatility, Sharpe, Sortino and Calmar ratios
- Historical and parametric VaR/CVaR, drawdown and rolling analytics
- Benchmark alpha, beta, active returns and capture ratios
- Correlation and covariance analysis
- Long-only efficient frontier and constrained portfolio optimization
- Current-versus-optimized allocation comparison
- Monte Carlo, GBM and historical-bootstrap simulations
- Company and fund fundamentals with resilient data fallbacks
- CSV exports and a complete in-app methodology reference

## Application structure

```text
portfolio_analytics_dashboard/
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── .streamlit/config.toml
├── .github/workflows/quality.yml
├── tests/
└── src/
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Quality checks

```bash
pip install -r requirements-dev.txt
python -m compileall -q app.py src
pytest -q
```

GitHub Actions runs these checks after every push and pull request to `main`.

## Methodology

Returns are computed from adjusted daily closing prices and annualized using a configurable trading-day convention. Portfolio optimization uses historical mean returns and the sample covariance matrix. The displayed efficient frontier contains only the upper efficient branch beginning at the global minimum-variance portfolio.

Historical VaR/CVaR use empirical quantiles. Parametric VaR assumes normally distributed returns. Simulation outputs are scenario estimates based on historical inputs and do not model transaction costs, taxes, liquidity or market impact.

## Data sources

- Yahoo Finance: prices and available security fundamentals
- FRED: US Treasury yields used as the risk-free rate

External data availability, accuracy and update frequency depend on those providers.

## Disclaimer

PortfolioLab is an educational analytics project. Its outputs are historical and model-dependent and do not constitute investment advice, a recommendation or a guarantee of future performance.
