import numpy as np
import pandas as pd
import pytest

from app import runSimulation
from src.analytics import portfolioReturns, portfolioVolatility
from src.data import getDividendSummary
from src.fundamentals import (
    dividendMetrics,
    formatFundamentals,
    multipleFundamentalsSummary
)
from src.sidebar import normalizeWeights, parseTickers, validateCustomWeightTotal, validateTickers
from src.simulation import (
    efficientFrontier,
    minimumVariancePortfolio,
    optimizationWeights,
    randomPortfolios
)


def test_ticker_parsing_is_clean_and_unlimited():
    assert parseTickers('aapl, MSFT; aapl, brk.b') == ['AAPL', 'MSFT', 'BRK.B']
    assert len(validateTickers([f'T{i}' for i in range(100)])) == 100


def test_london_gold_ticker_is_mapped_to_yahoo_symbol():
    assert parseTickers('SGLN') == ['SGLN.L']
    assert parseTickers('AAPL, SGLN, SGLN.L') == ['AAPL', 'SGLN.L']


def test_weights_normalize_and_reject_overallocation():
    assert np.allclose(normalizeWeights(np.array([2.0, 3.0])), [0.4, 0.6])
    with pytest.raises(ValueError, match='115.00%'):
        validateCustomWeightTotal(np.array([0.60, 0.55]))


def test_optimizer_residual_weights_are_cleaned():
    result = {'weights': np.array([0.70, 3e-17, 0.30])}
    weights = optimizationWeights(result, ['A', 'B', 'C'])
    assert weights['B'] == 0
    assert np.isclose(weights.sum(), 1.0)


def test_yahoo_dividend_yields_are_normalized_before_formatting():
    metrics = dividendMetrics(
        'AAPL',
        info={
            'dividendYield': 0.35,
            'trailingAnnualDividendYield': 0.35,
            'dividendRate': 1.08,
            'trailingAnnualDividendRate': 1.08,
            'currentPrice': 309.63,
            'previousClose': 309.63,
            'fiveYearAvgDividendYield': 0.55
        }
    )
    frame = pd.DataFrame({'AAPL': metrics})
    formatted = formatFundamentals(frame)

    assert formatted.loc['Dividend Yield', 'AAPL'] == '0.35%'
    assert formatted.loc['Trailing Annual Dividend Yield', 'AAPL'] == '0.35%'
    assert formatted.loc['Five Year Average Dividend Yield', 'AAPL'] == '0.55%'


@pytest.mark.parametrize('raw_yield', [35.0, 0.35, 0.0035])
def test_yahoo_dividend_yield_scales_produce_same_result(raw_yield):
    metrics = dividendMetrics(
        'AAPL',
        info={
            'dividendYield': raw_yield,
            'trailingAnnualDividendYield': raw_yield
        }
    )
    formatted = formatFundamentals(pd.DataFrame({'AAPL': metrics}))

    assert formatted.loc['Dividend Yield', 'AAPL'] == '0.35%'
    assert formatted.loc['Trailing Annual Dividend Yield', 'AAPL'] == '0.35%'


def test_scaled_dividend_rate_cannot_create_a_35_percent_yield():
    metrics = dividendMetrics(
        'AAPL',
        info={
            'dividendRate': 108.0,
            'trailingAnnualDividendRate': 108.0,
            'currentPrice': 309.63,
            'previousClose': 309.63
        }
    )
    formatted = formatFundamentals(pd.DataFrame({'AAPL': metrics}))

    assert formatted.loc['Dividend Yield', 'AAPL'] == '0.35%'
    assert formatted.loc['Trailing Annual Dividend Yield', 'AAPL'] == '0.35%'


def test_efficient_frontier_is_upper_and_monotonic():
    means = pd.Series([0.00025, 0.00055, 0.00090], index=['A', 'B', 'C'])
    covariance = pd.DataFrame(
        [[0.00010, 0.00002, 0.00001], [0.00002, 0.00018, 0.00004], [0.00001, 0.00004, 0.00035]],
        index=means.index,
        columns=means.index
    )
    frontier = efficientFrontier(means, covariance, points=40)
    minimumVariance = minimumVariancePortfolio(means, covariance)
    assert frontier['Expected Return'].is_monotonic_increasing
    assert frontier['Volatility'].is_monotonic_increasing
    assert frontier['Expected Return'].min() >= minimumVariance['expectedReturn'] - 1e-6



def test_short_positions_are_supported_only_when_enabled():
    dates = pd.date_range('2026-01-02', periods=2, freq='B')
    returns = pd.DataFrame(
        {'A': [0.01, 0.02], 'B': [-0.01, 0.01]},
        index=dates
    )
    weights = pd.Series({'A': 1.20, 'B': -0.20})

    with pytest.raises(ValueError, match='negative'):
        portfolioReturns(returns, weights)

    result = portfolioReturns(
        returns,
        weights,
        allowShortSelling=True
    )
    assert len(result) == len(returns)
    assert np.isfinite(result).all()


def test_rebalancing_frequency_changes_portfolio_path():
    dates = pd.to_datetime(['2026-01-30', '2026-02-02'])
    returns = pd.DataFrame(
        {'A': [0.10, 0.10], 'B': [0.0, 0.0]},
        index=dates
    )
    weights = pd.Series({'A': 0.50, 'B': 0.50})

    buyAndHold = portfolioReturns(
        returns, weights, rebalanceFrequency='None'
    )
    monthly = portfolioReturns(
        returns, weights, rebalanceFrequency='Monthly'
    )

    assert np.isclose(buyAndHold.iloc[0], 0.05)
    assert buyAndHold.iloc[1] > 0.05
    assert np.allclose(monthly.values, [0.05, 0.05])


def test_portfolio_volatility_annualized_flag_is_not_reversed():
    covariance = np.array([[0.0001]])
    daily = portfolioVolatility(
        covariance, [1.0], annualized=False
    )
    annual = portfolioVolatility(
        covariance, [1.0], annualized=True
    )

    assert np.isclose(daily, 0.01)
    assert np.isclose(annual, 0.01 * np.sqrt(252))


def test_random_portfolios_respect_weight_constraints():
    means = pd.Series([0.0002, 0.0004, 0.0006])
    covariance = np.diag([0.0001, 0.0002, 0.0003])
    samples = randomPortfolios(
        means,
        covariance,
        portfolios=500,
        minimumWeight=-0.20,
        maximumWeight=0.70,
        randomSeed=7
    )
    weights = np.vstack(samples['Weights'])

    assert np.allclose(weights.sum(axis=1), 1.0)
    assert weights.min() >= -0.20 - 1e-10
    assert weights.max() <= 0.70 + 1e-10


def test_gbm_uses_exponential_log_return_compounding():
    historicalReturns = pd.Series(
        [0.01, 0.01, 0.01],
        index=pd.date_range('2026-01-01', periods=3)
    )
    paths = runSimulation(
        historicalReturns,
        method='GBM',
        simulations=2,
        horizonDays=3,
        initialValue=100.0,
        randomSeed=42
    )

    expected = 100.0 * 1.01 ** np.arange(1, 4)
    assert np.allclose(paths.iloc[:, 0].values, expected)
    assert np.allclose(paths.iloc[:, 1].values, expected)


def test_unavailable_dividend_data_is_not_reported_as_zero(monkeypatch):
    class BrokenTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, **kwargs):
            raise RuntimeError('temporary failure')

    monkeypatch.setattr('src.data.yf.Ticker', BrokenTicker)
    getDividendSummary.clear()
    summary = getDividendSummary(
        ['AAPL'],
        pd.Timestamp('2025-01-01'),
        pd.Timestamp('2026-01-01')
    )

    assert pd.isna(summary.loc['AAPL', 'Dividend Yield (TTM)'])
    assert summary.loc['AAPL', 'Data Status'] == 'Unavailable'


def test_fundamental_failures_are_exposed(monkeypatch):
    def fail(ticker):
        raise ValueError('provider unavailable')

    monkeypatch.setattr(
        'src.fundamentals.fundamentalsSummary',
        fail
    )
    result = multipleFundamentalsSummary(['AAPL'])

    assert 'AAPL' in result.attrs['errors']
