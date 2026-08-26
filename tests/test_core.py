import numpy as np
import pandas as pd
import pytest

from src.fundamentals import dividendMetrics, formatFundamentals
from src.sidebar import normalizeWeights, parseTickers, validateCustomWeightTotal, validateTickers
from src.simulation import efficientFrontier, minimumVariancePortfolio, optimizationWeights


def test_ticker_parsing_is_clean_and_unlimited():
    assert parseTickers('aapl, MSFT; aapl, brk.b') == ['AAPL', 'MSFT', 'BRK.B']
    assert len(validateTickers([f'T{i}' for i in range(100)])) == 100


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
