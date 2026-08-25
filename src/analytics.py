from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def validateWeights(
    weights: np.ndarray | pd.Series | list[float],
    tolerance: float = 1e-6
) -> np.ndarray:
    '''Validate and standardize portfolio weights.'''
    weights = np.asarray(weights, dtype=float)

    if weights.ndim != 1:
        raise ValueError('weights must be one-dimensional.')

    if weights.size == 0:
        raise ValueError('weights cannot be empty.')

    if not np.isfinite(weights).all():
        raise ValueError('weights must contain finite values.')

    if np.any(weights < 0):
        raise ValueError('weights cannot contain negative values.')

    if not np.isclose(weights.sum(), 1, atol=tolerance):
        raise ValueError('weights must sum to one.')

    return weights


def computeWeights(
    investedAmounts: pd.Series | dict | list[float]
) -> pd.Series:
    '''Compute portfolio weights from invested amounts.'''
    if isinstance(investedAmounts, dict):
        investedAmounts = pd.Series(
            investedAmounts,
            dtype=float
        )
    elif not isinstance(investedAmounts, pd.Series):
        investedAmounts = pd.Series(
            investedAmounts,
            dtype=float
        )
    else:
        investedAmounts = investedAmounts.astype(float)

    if investedAmounts.empty:
        raise ValueError('investedAmounts cannot be empty.')

    if not np.isfinite(investedAmounts).all():
        raise ValueError(
            'investedAmounts must contain finite values.'
        )

    if np.any(investedAmounts < 0):
        raise ValueError(
            'investedAmounts cannot contain negative values.'
        )

    totalInvestment = investedAmounts.sum()

    if totalInvestment <= 0:
        raise ValueError(
            'Total invested amount must be positive.'
        )

    weights = investedAmounts / totalInvestment
    weights.name = 'Weight'

    return weights


def alignWeights(
    weights: pd.Series,
    assetNames: list[str] | pd.Index
) -> pd.Series:
    '''Align portfolio weights with the selected asset order.'''
    if not isinstance(weights, pd.Series):
        raise TypeError('weights must be a pandas Series.')

    assetNames = list(assetNames)
    missingAssets = [
        asset
        for asset in assetNames
        if asset not in weights.index
    ]

    if missingAssets:
        missing = ', '.join(missingAssets)

        raise ValueError(
            f'Missing weights for: {missing}'
        )

    alignedWeights = weights.reindex(assetNames).astype(float)

    validateWeights(alignedWeights.values)

    return alignedWeights


def portfolioReturns(
    returns: pd.DataFrame,
    weights: pd.Series | np.ndarray | list[float]
) -> pd.Series:
    '''Compute the historical daily portfolio return series.'''
    if returns.empty:
        raise ValueError('returns cannot be empty.')

    if isinstance(weights, pd.Series):
        weights = alignWeights(
            weights,
            returns.columns
        ).values
    else:
        weights = validateWeights(weights)

        if len(weights) != returns.shape[1]:
            raise ValueError(
                'weights length must match the number of assets.'
            )

    portfolioReturnSeries = returns.dot(weights)
    portfolioReturnSeries.name = 'Portfolio Return'

    return portfolioReturnSeries


def cumulativeReturns(
    returns: pd.Series | pd.DataFrame,
    returnType: Literal['simple', 'log'] = 'simple'
) -> pd.Series | pd.DataFrame:
    '''Compute cumulative performance from periodic returns.'''
    if returns.empty:
        raise ValueError('returns cannot be empty.')

    if returnType == 'simple':
        cumulative = (1 + returns).cumprod() - 1
    elif returnType == 'log':
        cumulative = np.exp(returns.cumsum()) - 1
    else:
        raise ValueError(
            'returnType must be either simple or log.'
        )

    return cumulative


def annualizedReturn(
    returns: pd.Series,
    tradingDays: int = TRADING_DAYS,
    method: Literal['arithmetic', 'geometric'] = 'geometric'
) -> float:
    '''Compute the annualized return of a return series.'''
    if returns.empty:
        raise ValueError('returns cannot be empty.')

    if tradingDays <= 0:
        raise ValueError('tradingDays must be positive.')

    returns = returns.dropna().astype(float)

    if returns.empty:
        raise ValueError(
            'returns must contain valid observations.'
        )

    if method == 'arithmetic':
        return float(
            returns.mean() * tradingDays
        )

    if method == 'geometric':
        growth = float(
            (1 + returns).prod()
        )

        if growth <= 0:
            raise ValueError(
                'Geometric annualization requires positive cumulative growth.'
            )

        years = len(returns) / tradingDays

        return growth ** (1 / years) - 1

    raise ValueError(
        'method must be either arithmetic or geometric.'
    )


def annualizedVolatility(
    returns: pd.Series,
    tradingDays: int = TRADING_DAYS
) -> float:
    '''Compute annualized portfolio volatility.'''
    if returns.empty:
        raise ValueError('returns cannot be empty.')

    if tradingDays <= 0:
        raise ValueError('tradingDays must be positive.')

    returns = returns.dropna().astype(float)

    if len(returns) < 2:
        raise ValueError(
            'At least two observations are required.'
        )

    return float(
        returns.std(ddof=1) * np.sqrt(tradingDays)
    )


def portfolioExpectedReturn(
    meanReturns: pd.Series | np.ndarray,
    weights: pd.Series | np.ndarray | list[float],
    tradingDays: int = TRADING_DAYS
) -> float:
    '''Compute annualized expected portfolio return.'''
    if tradingDays <= 0:
        raise ValueError('tradingDays must be positive.')

    meanReturns = np.asarray(
        meanReturns,
        dtype=float
    )

    if isinstance(weights, pd.Series):
        weights = weights.values

    weights = validateWeights(weights)

    if len(meanReturns) != len(weights):
        raise ValueError(
            'meanReturns and weights must have equal length.'
        )

    return float(
        np.dot(meanReturns, weights) * tradingDays
    )


def portfolioVolatility(
    covarianceMatrix: pd.DataFrame | np.ndarray,
    weights: pd.Series | np.ndarray | list[float],
    tradingDays: int = TRADING_DAYS,
    annualized: bool = False
) -> float:
    '''Compute portfolio volatility from weights and covariance.'''
    if tradingDays <= 0:
        raise ValueError('tradingDays must be positive.')

    covarianceMatrix = np.asarray(
        covarianceMatrix,
        dtype=float
    )

    if isinstance(weights, pd.Series):
        weights = weights.values

    weights = validateWeights(weights)

    if covarianceMatrix.ndim != 2:
        raise ValueError(
            'covarianceMatrix must be two-dimensional.'
        )

    if covarianceMatrix.shape[0] != covarianceMatrix.shape[1]:
        raise ValueError(
            'covarianceMatrix must be square.'
        )

    if covarianceMatrix.shape[0] != len(weights):
        raise ValueError(
            'covarianceMatrix dimensions must match weights.'
        )

    variance = float(
        weights.T @ covarianceMatrix @ weights
    )

    if variance < 0 and not np.isclose(variance, 0):
        raise ValueError(
            'Portfolio variance cannot be negative.'
        )

    volatility = np.sqrt(max(variance, 0))

    if not annualized:
        volatility *= np.sqrt(tradingDays)

    return float(volatility)

def sharpeRatio(
    returns: pd.Series,
    riskFreeRate: float = 0,
    tradingDays: int = TRADING_DAYS
) -> float:
    '''Compute the annualized Sharpe ratio.'''
    annualReturn = annualizedReturn(
        returns,
        tradingDays=tradingDays,
        method='arithmetic'
    )

    annualVolatility = annualizedVolatility(
        returns,
        tradingDays=tradingDays
    )

    if np.isclose(annualVolatility, 0):
        return np.nan

    return (
        annualReturn - riskFreeRate
    ) / annualVolatility


def downsideDeviation(
    returns: pd.Series,
    targetReturn: float = 0,
    tradingDays: int = TRADING_DAYS
) -> float:
    '''Compute annualized downside deviation below a target return.'''
    if returns.empty:
        raise ValueError('returns cannot be empty.')

    if tradingDays <= 0:
        raise ValueError('tradingDays must be positive.')

    returns = returns.dropna().astype(float)

    if returns.empty:
        raise ValueError(
            'returns must contain valid observations.'
        )

    dailyTarget = (
        (1 + targetReturn) ** (1 / tradingDays) - 1
    )

    downsideReturns = np.minimum(
        returns - dailyTarget,
        0
    )

    downsideVariance = np.mean(
        downsideReturns ** 2
    )

    return float(
        np.sqrt(downsideVariance) * np.sqrt(tradingDays)
    )


def sortinoRatio(
    returns: pd.Series,
    riskFreeRate: float = 0,
    targetReturn: float = 0,
    tradingDays: int = TRADING_DAYS
) -> float:
    '''Compute the annualized Sortino ratio.'''
    annualReturn = annualizedReturn(
        returns,
        tradingDays=tradingDays,
        method='arithmetic'
    )

    downsideRisk = downsideDeviation(
        returns,
        targetReturn=targetReturn,
        tradingDays=tradingDays
    )

    if np.isclose(downsideRisk, 0):
        return np.nan

    return (
        annualReturn - riskFreeRate
    ) / downsideRisk


def drawdownSeries(
    returns: pd.Series
) -> pd.Series:
    '''Compute the portfolio drawdown series.'''
    if returns.empty:
        raise ValueError('returns cannot be empty.')

    returns = returns.dropna().astype(float)

    wealthIndex = (
        1 + returns
    ).cumprod()

    runningPeak = wealthIndex.cummax()

    drawdowns = (
        wealthIndex / runningPeak
    ) - 1

    drawdowns.name = 'Drawdown'

    return drawdowns


def maxDrawdown(
    returns: pd.Series
) -> float:
    '''Return the maximum historical drawdown.'''
    drawdowns = drawdownSeries(returns)

    return float(
        drawdowns.min()
    )


def calmarRatio(
    returns: pd.Series,
    tradingDays: int = TRADING_DAYS
) -> float:
    '''Compute the Calmar ratio.'''
    annualReturn = annualizedReturn(
        returns,
        tradingDays=tradingDays,
        method='geometric'
    )

    maximumDrawdown = abs(
        maxDrawdown(returns)
    )

    if np.isclose(maximumDrawdown, 0):
        return np.nan

    return annualReturn / maximumDrawdown


def historicalVaR(
    returns: pd.Series,
    confidenceLevel: float = 0.95,
    portfolioValue: float | None = None
) -> float:
    '''Compute historical Value at Risk.'''
    if returns.empty:
        raise ValueError('returns cannot be empty.')

    if not 0 < confidenceLevel < 1:
        raise ValueError(
            'confidenceLevel must be between zero and one.'
        )

    returns = returns.dropna().astype(float)

    if returns.empty:
        raise ValueError(
            'returns must contain valid observations.'
        )

    percentile = np.quantile(
        returns,
        1 - confidenceLevel
    )

    valueAtRisk = max(
        -float(percentile),
        0
    )

    if portfolioValue is not None:
        if portfolioValue <= 0:
            raise ValueError(
                'portfolioValue must be positive.'
            )

        valueAtRisk *= portfolioValue

    return valueAtRisk


def historicalCVaR(
    returns: pd.Series,
    confidenceLevel: float = 0.95,
    portfolioValue: float | None = None
) -> float:
    '''Compute historical Conditional Value at Risk.'''
    if returns.empty:
        raise ValueError('returns cannot be empty.')

    if not 0 < confidenceLevel < 1:
        raise ValueError(
            'confidenceLevel must be between zero and one.'
        )

    returns = returns.dropna().astype(float)

    threshold = np.quantile(
        returns,
        1 - confidenceLevel
    )

    tailReturns = returns[
        returns <= threshold
    ]

    if tailReturns.empty:
        return 0

    conditionalValueAtRisk = max(
        -float(tailReturns.mean()),
        0
    )

    if portfolioValue is not None:
        if portfolioValue <= 0:
            raise ValueError(
                'portfolioValue must be positive.'
            )

        conditionalValueAtRisk *= portfolioValue

    return conditionalValueAtRisk


def parametricVaR(
    returns: pd.Series,
    confidenceLevel: float = 0.95,
    portfolioValue: float | None = None
) -> float:
    '''Compute normal-distribution parametric Value at Risk.'''
    if returns.empty:
        raise ValueError('returns cannot be empty.')

    if not 0 < confidenceLevel < 1:
        raise ValueError(
            'confidenceLevel must be between zero and one.'
        )

    returns = returns.dropna().astype(float)

    meanReturn = returns.mean()
    volatility = returns.std(ddof=1)

    zScore = {
        0.90: 1.281552,
        0.95: 1.644854,
        0.975: 1.959964,
        0.99: 2.326348
    }.get(round(confidenceLevel, 3))

    if zScore is None:
        raise ValueError(
            'Supported confidence levels are 0.90, 0.95, 0.975 and 0.99.'
        )

    valueAtRisk = max(
        -(meanReturn - zScore * volatility),
        0
    )

    if portfolioValue is not None:
        if portfolioValue <= 0:
            raise ValueError(
                'portfolioValue must be positive.'
            )

        valueAtRisk *= portfolioValue

    return float(valueAtRisk)


def beta(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series
) -> float:
    '''Compute portfolio beta relative to a benchmark.'''
    alignedReturns = pd.concat(
        [
            portfolioReturns.rename('Portfolio'),
            benchmarkReturns.rename('Benchmark')
        ],
        axis=1
    ).dropna()

    if len(alignedReturns) < 2:
        raise ValueError(
            'At least two aligned observations are required.'
        )

    benchmarkVariance = alignedReturns[
        'Benchmark'
    ].var(ddof=1)

    if np.isclose(benchmarkVariance, 0):
        return np.nan

    covariance = alignedReturns.cov().loc[
        'Portfolio',
        'Benchmark'
    ]

    return float(
        covariance / benchmarkVariance
    )


def alpha(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series,
    riskFreeRate: float = 0,
    tradingDays: int = TRADING_DAYS
) -> float:
    '''Compute annualized Jensen alpha.'''
    portfolioAnnualReturn = annualizedReturn(
        portfolioReturns,
        tradingDays=tradingDays,
        method='arithmetic'
    )

    benchmarkAnnualReturn = annualizedReturn(
        benchmarkReturns,
        tradingDays=tradingDays,
        method='arithmetic'
    )

    portfolioBeta = beta(
        portfolioReturns,
        benchmarkReturns
    )

    if np.isnan(portfolioBeta):
        return np.nan

    expectedReturn = (
        riskFreeRate
        + portfolioBeta
        * (benchmarkAnnualReturn - riskFreeRate)
    )

    return (
        portfolioAnnualReturn
        - expectedReturn
    )