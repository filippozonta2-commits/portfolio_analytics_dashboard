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

## Risk metrics and interpretation

- **Sharpe ratio:** arithmetic mean daily return multiplied by the configured trading days, minus the annual risk-free rate, divided by sample daily volatility multiplied by the square root of trading days. A zero-volatility series yields an undefined ratio.
- **Rolling Sharpe ratio:** uses the same arithmetic risk-free-rate convention as the headline Sharpe: daily excess return is daily portfolio return minus annual risk-free rate divided by the configured trading days. Each rolling window uses sample standard deviation (`ddof=1`), so a full-window rolling Sharpe agrees with the headline Sharpe on the identical observations. Rolling annualized *return* is separately compounded and must not be substituted into this arithmetic Sharpe formula.
- **Drawdown / maximum drawdown:** compounded wealth starts at **1.0 before the first return**. The initial investment is part of the high-water mark, so a loss on day one is included. Maximum drawdown is reported as a negative fraction.
- **Historical VaR and CVaR:** one-period loss measures from the empirical lower tail, shown as **nonnegative loss magnitudes**. CVaR is the average loss among observations at or below the empirical quantile; this finite-sample convention may differ from interpolated expected shortfall.
- **Parametric VaR:** one-period Gaussian approximation based on sample mean and sample standard deviation; not a guarantee or a robust estimate of extreme tails.
- **Benchmark alpha / beta:** computed on the **intersection of available return dates**. Jensen alpha uses annualized arithmetic returns and the selected annual risk-free rate. The displayed benchmark-relative results should not be compared with unaligned date windows.
- **Optimization:** expected returns and covariance are estimated from the selected historical sample. Optimized allocations are **in-sample illustrations**, not out-of-sample trading performance or demonstrated alpha.
- **Costs and realism:** historical returns and simulations exclude transaction costs, financing/borrow fees, taxes, slippage and market impact. The selected index benchmark may not be directly investable. Yahoo's adjusted prices and data availability may be revised.

### Reproducible synthetic risk checks

The repository includes `tests/test_risk_regressions.py`, which tests first-day drawdowns, recovery paths, date-aligned benchmark alpha, positive-loss VaR/CVaR conventions, Sharpe annualization and insufficient-data behavior. These tests use synthetic returns and do not require Yahoo Finance or FRED access.

Run `python -m pytest -q` after installing `requirements-dev.txt`. Passing synthetic tests does not validate market-data correctness, out-of-sample model performance or live trading assumptions.

## Data sources

- Yahoo Finance: prices and available security fundamentals
- FRED: US Treasury yields used as the risk-free rate

External data availability, accuracy and update frequency depend on those providers.

## Disclaimer

PortfolioLab is an educational analytics project. Its outputs are historical and model-dependent and do not constitute investment advice, a recommendation or a guarantee of future performance.
