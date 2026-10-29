"""Regression checks for portfolio risk and benchmark statistics.

All fixtures are synthetic and do not require external market-data access.
"""
import numpy as np
import pandas as pd
import pytest

from src.analytics import (
    alpha,
    drawdownSeries,
    historicalCVaR,
    historicalVaR,
    maxDrawdown,
    parametricVaR,
    sharpeRatio,
)


def test_drawdown_includes_initial_investment_baseline():
    returns = pd.Series([-0.20, 0.10], index=pd.date_range('2026-01-01', periods=2))
    # Wealth: 1.00 -> 0.80 -> 0.88, so the peak is still the initial 1.00.
    assert np.allclose(drawdownSeries(returns).to_numpy(), [-0.20, -0.12])
    assert maxDrawdown(returns) == pytest.approx(-0.20)


def test_drawdown_tracks_new_highs_and_subsequent_losses():
    returns = pd.Series([0.10, -0.10, 0.05])
    assert np.allclose(drawdownSeries(returns), [0.0, -0.10, -0.055])


def test_alpha_uses_only_common_portfolio_and_benchmark_dates():
    portfolio = pd.Series(
        [0.20, 0.20, 0.01, 0.02],
        index=pd.date_range('2026-01-01', periods=4),
    )
    benchmark = pd.Series(
        [0.005, 0.015, 0.10, 0.20],
        index=pd.date_range('2026-01-03', periods=4),
    )
    # On the common dates beta=1, mean excess daily return=0.005.
    assert alpha(portfolio, benchmark, tradingDays=252) == pytest.approx(1.26)


def test_alpha_rejects_insufficient_overlap():
    portfolio = pd.Series([0.01, 0.02], index=pd.to_datetime(['2026-01-01', '2026-01-02']))
    benchmark = pd.Series([0.01, 0.02], index=pd.to_datetime(['2026-01-02', '2026-01-03']))
    with pytest.raises(ValueError, match='aligned'):
        alpha(portfolio, benchmark)


def test_historical_var_and_cvar_have_positive_loss_convention():
    returns = pd.Series([-0.10, -0.02, 0.01, 0.02])
    assert historicalVaR(returns, confidenceLevel=0.95) == pytest.approx(0.088)
    assert historicalCVaR(returns, confidenceLevel=0.95) == pytest.approx(0.10)


def test_sharpe_uses_arithmetic_annualized_excess_return():
    returns = pd.Series([0.01, 0.03, -0.01, 0.02])
    rf = 0.04
    expected = (returns.mean() * 252 - rf) / (returns.std(ddof=1) * np.sqrt(252))
    assert sharpeRatio(returns, riskFreeRate=rf) == pytest.approx(expected)


@pytest.mark.parametrize('function', [drawdownSeries, historicalCVaR])
def test_all_missing_returns_raise_clear_error(function):
    with pytest.raises(ValueError, match='valid observations'):
        function(pd.Series([np.nan, np.nan]))


def test_parametric_var_requires_sample_variance():
    with pytest.raises(ValueError, match='two valid observations'):
        parametricVaR(pd.Series([0.01]))
