from __future__ import annotations

import pandas as pd

from src.analytics import (
    alpha,
    annualizedReturn,
    annualizedVolatility,
    beta,
    calmarRatio,
    historicalCVaR,
    historicalVaR,
    maxDrawdown,
    parametricVaR,
    sharpeRatio,
    sortinoRatio
)


TRADING_DAYS = 252


def portfolioSummary(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series | None = None,
    riskFreeRate: float = 0,
    confidenceLevel: float = 0.95,
    tradingDays: int = TRADING_DAYS
) -> pd.Series:
    '''Return a complete portfolio performance summary.'''
    if portfolioReturns.empty:
        raise ValueError(
            'portfolioReturns cannot be empty.'
        )

    summary = {
        'Annual Return': annualizedReturn(
            portfolioReturns,
            tradingDays=tradingDays
        ),
        'Annual Volatility': annualizedVolatility(
            portfolioReturns,
            tradingDays=tradingDays
        ),
        'Sharpe Ratio': sharpeRatio(
            portfolioReturns,
            riskFreeRate=riskFreeRate,
            tradingDays=tradingDays
        ),
        'Sortino Ratio': sortinoRatio(
            portfolioReturns,
            riskFreeRate=riskFreeRate,
            tradingDays=tradingDays
        ),
        'Calmar Ratio': calmarRatio(
            portfolioReturns,
            tradingDays=tradingDays
        ),
        'Maximum Drawdown': maxDrawdown(
            portfolioReturns
        ),
        f'Historical VaR {confidenceLevel:.0%}': historicalVaR(
            portfolioReturns,
            confidenceLevel=confidenceLevel
        ),
        f'Historical CVaR {confidenceLevel:.0%}': historicalCVaR(
            portfolioReturns,
            confidenceLevel=confidenceLevel
        ),
        f'Parametric VaR {confidenceLevel:.0%}': parametricVaR(
            portfolioReturns,
            confidenceLevel=confidenceLevel
        )
    }

    if benchmarkReturns is not None:
        summary['Beta'] = beta(
            portfolioReturns,
            benchmarkReturns
        )

        summary['Alpha'] = alpha(
            portfolioReturns,
            benchmarkReturns,
            riskFreeRate=riskFreeRate,
            tradingDays=tradingDays
        )

    return pd.Series(
        summary,
        name='Portfolio'
    )