from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.data import computeReturns, getData


DEFAULT_BENCHMARK = 'SPY'


def validateBenchmarkTicker(
    benchmarkTicker: str
) -> str:
    '''Validate and standardize a benchmark ticker.'''
    benchmarkTicker = str(
        benchmarkTicker
    ).strip().upper()

    if not benchmarkTicker:
        raise ValueError(
            'benchmarkTicker cannot be empty.'
        )

    return benchmarkTicker


def downloadBenchmarkPrices(
    benchmarkTicker: str,
    startDate,
    endDate
) -> pd.Series:
    '''Download benchmark prices for the selected date range.'''
    benchmarkTicker = validateBenchmarkTicker(
        benchmarkTicker
    )

    prices = getData(
        [benchmarkTicker],
        startDate,
        endDate
    )

    benchmarkPrices = prices[
        benchmarkTicker
    ].copy()

    benchmarkPrices.name = benchmarkTicker

    return benchmarkPrices


def benchmarkReturns(
    benchmarkPrices: pd.Series,
    returnType: str = 'simple'
) -> pd.Series:
    '''Compute benchmark returns from benchmark prices.'''
    if benchmarkPrices.empty:
        raise ValueError(
            'benchmarkPrices cannot be empty.'
        )

    returns = computeReturns(
        benchmarkPrices,
        returnType=returnType
    )

    returns.name = (
        benchmarkPrices.name
        if benchmarkPrices.name
        else 'Benchmark'
    )

    return returns


def alignPortfolioBenchmark(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series
) -> pd.DataFrame:
    '''Align portfolio and benchmark returns on common dates.'''
    if portfolioReturns.empty:
        raise ValueError(
            'portfolioReturns cannot be empty.'
        )

    if benchmarkReturns.empty:
        raise ValueError(
            'benchmarkReturns cannot be empty.'
        )

    alignedReturns = pd.concat(
        [
            portfolioReturns.rename('Portfolio'),
            benchmarkReturns.rename('Benchmark')
        ],
        axis=1
    ).dropna()

    if alignedReturns.empty:
        raise ValueError(
            'Portfolio and benchmark have no common observations.'
        )

    return alignedReturns


def activeReturns(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series
) -> pd.Series:
    '''Compute portfolio returns in excess of the benchmark.'''
    alignedReturns = alignPortfolioBenchmark(
        portfolioReturns,
        benchmarkReturns
    )

    activeReturnSeries = (
        alignedReturns['Portfolio']
        - alignedReturns['Benchmark']
    )

    activeReturnSeries.name = 'Active Return'

    return activeReturnSeries


def annualizedActiveReturn(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series,
    tradingDays: int = 252
) -> float:
    '''Compute annualized active return.'''
    if tradingDays <= 0:
        raise ValueError(
            'tradingDays must be positive.'
        )

    returns = activeReturns(
        portfolioReturns,
        benchmarkReturns
    )

    return float(
        returns.mean() * tradingDays
    )


def trackingError(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series,
    tradingDays: int = 252
) -> float:
    '''Compute annualized tracking error.'''
    if tradingDays <= 0:
        raise ValueError(
            'tradingDays must be positive.'
        )

    returns = activeReturns(
        portfolioReturns,
        benchmarkReturns
    )

    if len(returns) < 2:
        raise ValueError(
            'At least two aligned observations are required.'
        )

    return float(
        returns.std(ddof=1)
        * np.sqrt(tradingDays)
    )


def informationRatio(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series,
    tradingDays: int = 252
) -> float:
    '''Compute the annualized information ratio.'''
    activeReturn = annualizedActiveReturn(
        portfolioReturns,
        benchmarkReturns,
        tradingDays=tradingDays
    )

    error = trackingError(
        portfolioReturns,
        benchmarkReturns,
        tradingDays=tradingDays
    )

    if np.isclose(error, 0):
        return np.nan

    return activeReturn / error


def relativePerformance(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series
) -> pd.DataFrame:
    '''Compute cumulative portfolio and benchmark performance.'''
    alignedReturns = alignPortfolioBenchmark(
        portfolioReturns,
        benchmarkReturns
    )

    cumulativePerformance = (
        1 + alignedReturns
    ).cumprod() - 1

    cumulativePerformance[
        'Relative Performance'
    ] = (
        cumulativePerformance['Portfolio']
        - cumulativePerformance['Benchmark']
    )

    return cumulativePerformance


def upCaptureRatio(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series
) -> float:
    '''Compute the portfolio upside capture ratio.'''
    alignedReturns = alignPortfolioBenchmark(
        portfolioReturns,
        benchmarkReturns
    )

    positivePeriods = alignedReturns[
        alignedReturns['Benchmark'] > 0
    ]

    if positivePeriods.empty:
        return np.nan

    benchmarkReturn = positivePeriods[
        'Benchmark'
    ].mean()

    if np.isclose(benchmarkReturn, 0):
        return np.nan

    return float(
        positivePeriods['Portfolio'].mean()
        / benchmarkReturn
    )


def downCaptureRatio(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series
) -> float:
    '''Compute the portfolio downside capture ratio.'''
    alignedReturns = alignPortfolioBenchmark(
        portfolioReturns,
        benchmarkReturns
    )

    negativePeriods = alignedReturns[
        alignedReturns['Benchmark'] < 0
    ]

    if negativePeriods.empty:
        return np.nan

    benchmarkReturn = negativePeriods[
        'Benchmark'
    ].mean()

    if np.isclose(benchmarkReturn, 0):
        return np.nan

    return float(
        negativePeriods['Portfolio'].mean()
        / benchmarkReturn
    )


def captureRatio(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series
) -> float:
    '''Compute the ratio between upside and downside capture.'''
    upsideCapture = upCaptureRatio(
        portfolioReturns,
        benchmarkReturns
    )

    downsideCapture = downCaptureRatio(
        portfolioReturns,
        benchmarkReturns
    )

    if np.isnan(upsideCapture):
        return np.nan

    if np.isnan(downsideCapture):
        return np.nan

    if np.isclose(downsideCapture, 0):
        return np.nan

    return upsideCapture / downsideCapture


def benchmarkCorrelation(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series
) -> float:
    '''Compute return correlation with the benchmark.'''
    alignedReturns = alignPortfolioBenchmark(
        portfolioReturns,
        benchmarkReturns
    )

    return float(
        alignedReturns['Portfolio'].corr(
            alignedReturns['Benchmark']
        )
    )


def benchmarkSummary(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series,
    tradingDays: int = 252
) -> pd.Series:
    '''Return the main benchmark-relative metrics.'''
    summary = {
        'Active Return': annualizedActiveReturn(
            portfolioReturns,
            benchmarkReturns,
            tradingDays=tradingDays
        ),
        'Tracking Error': trackingError(
            portfolioReturns,
            benchmarkReturns,
            tradingDays=tradingDays
        ),
        'Information Ratio': informationRatio(
            portfolioReturns,
            benchmarkReturns,
            tradingDays=tradingDays
        ),
        'Correlation': benchmarkCorrelation(
            portfolioReturns,
            benchmarkReturns
        ),
        'Up Capture Ratio': upCaptureRatio(
            portfolioReturns,
            benchmarkReturns
        ),
        'Down Capture Ratio': downCaptureRatio(
            portfolioReturns,
            benchmarkReturns
        ),
        'Capture Ratio': captureRatio(
            portfolioReturns,
            benchmarkReturns
        )
    }

    return pd.Series(
        summary,
        name='Benchmark Analysis'
    )


def downloadMultipleBenchmarks(
    benchmarkTickers: Iterable[str],
    startDate,
    endDate
) -> pd.DataFrame:
    '''Download prices for multiple benchmark tickers.'''
    benchmarkTickers = [
        validateBenchmarkTicker(ticker)
        for ticker in benchmarkTickers
    ]

    if not benchmarkTickers:
        raise ValueError(
            'Enter at least one benchmark ticker.'
        )

    return getData(
        benchmarkTickers,
        startDate,
        endDate
    )