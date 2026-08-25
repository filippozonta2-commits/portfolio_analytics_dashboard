from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize


TRADING_DAYS = 252


def validateOptimizationInputs(
    meanReturns: pd.Series | np.ndarray,
    covarianceMatrix: pd.DataFrame | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    '''Validate and convert optimization inputs.'''
    meanReturns = np.asarray(
        meanReturns,
        dtype=float
    )

    covarianceMatrix = np.asarray(
        covarianceMatrix,
        dtype=float
    )

    if meanReturns.ndim != 1:
        raise ValueError(
            'meanReturns must be one-dimensional.'
        )

    if covarianceMatrix.ndim != 2:
        raise ValueError(
            'covarianceMatrix must be two-dimensional.'
        )

    if covarianceMatrix.shape[0] != covarianceMatrix.shape[1]:
        raise ValueError(
            'covarianceMatrix must be square.'
        )

    if covarianceMatrix.shape[0] != len(meanReturns):
        raise ValueError(
            'meanReturns and covarianceMatrix dimensions must match.'
        )

    if not np.isfinite(meanReturns).all():
        raise ValueError(
            'meanReturns must contain finite values.'
        )

    if not np.isfinite(covarianceMatrix).all():
        raise ValueError(
            'covarianceMatrix must contain finite values.'
        )

    return meanReturns, covarianceMatrix


def validateBounds(
    numberOfAssets: int,
    minimumWeight: float = 0,
    maximumWeight: float = 1
) -> tuple[tuple[float, float], ...]:
    '''Validate and build portfolio optimization bounds.'''
    if numberOfAssets <= 0:
        raise ValueError(
            'numberOfAssets must be positive.'
        )

    if minimumWeight > maximumWeight:
        raise ValueError(
            'minimumWeight cannot exceed maximumWeight.'
        )

    if minimumWeight * numberOfAssets > 1:
        raise ValueError(
            'minimumWeight is infeasible for the number of assets.'
        )

    if maximumWeight * numberOfAssets < 1:
        raise ValueError(
            'maximumWeight is infeasible for the number of assets.'
        )

    return tuple(
        (minimumWeight, maximumWeight)
        for _ in range(numberOfAssets)
    )


def portfolioReturn(
    weights: np.ndarray,
    meanReturns: np.ndarray,
    tradingDays: int = TRADING_DAYS
) -> float:
    '''Compute annualized expected portfolio return.'''
    if tradingDays <= 0:
        raise ValueError(
            'tradingDays must be positive.'
        )

    return float(
        np.dot(weights, meanReturns)
        * tradingDays
    )


def portfolioVolatility(
    weights: np.ndarray,
    covarianceMatrix: np.ndarray,
    tradingDays: int = TRADING_DAYS
) -> float:
    '''Compute annualized portfolio volatility.'''
    if tradingDays <= 0:
        raise ValueError(
            'tradingDays must be positive.'
        )

    variance = float(
        weights.T
        @ covarianceMatrix
        @ weights
    )

    return float(
        np.sqrt(max(variance, 0))
        * np.sqrt(tradingDays)
    )


def portfolioSharpeRatio(
    weights: np.ndarray,
    meanReturns: np.ndarray,
    covarianceMatrix: np.ndarray,
    riskFreeRate: float = 0,
    tradingDays: int = TRADING_DAYS
) -> float:
    '''Compute annualized portfolio Sharpe ratio.'''
    expectedReturn = portfolioReturn(
        weights,
        meanReturns,
        tradingDays=tradingDays
    )

    volatility = portfolioVolatility(
        weights,
        covarianceMatrix,
        tradingDays=tradingDays
    )

    if np.isclose(volatility, 0):
        return np.nan

    return (
        expectedReturn
        - riskFreeRate
    ) / volatility


def equalWeights(
    numberOfAssets: int
) -> np.ndarray:
    '''Return equal portfolio weights.'''
    if numberOfAssets <= 0:
        raise ValueError(
            'numberOfAssets must be positive.'
        )

    return np.repeat(
        1 / numberOfAssets,
        numberOfAssets
    )


def minimumVariancePortfolio(
    meanReturns: pd.Series | np.ndarray,
    covarianceMatrix: pd.DataFrame | np.ndarray,
    minimumWeight: float = 0,
    maximumWeight: float = 1,
    tradingDays: int = TRADING_DAYS
) -> dict:
    '''Optimize the global minimum-variance portfolio.'''
    meanReturns, covarianceMatrix = validateOptimizationInputs(
        meanReturns,
        covarianceMatrix
    )

    numberOfAssets = len(meanReturns)

    bounds = validateBounds(
        numberOfAssets,
        minimumWeight=minimumWeight,
        maximumWeight=maximumWeight
    )

    constraints = {
        'type': 'eq',
        'fun': lambda weights: weights.sum() - 1
    }

    initialWeights = equalWeights(
        numberOfAssets
    )

    result = minimize(
        fun=lambda weights: portfolioVolatility(
            weights,
            covarianceMatrix,
            tradingDays=tradingDays
        ),
        x0=initialWeights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise RuntimeError(
            f'Minimum-variance optimization failed: {result.message}'
        )

    weights = result.x

    return {
        'weights': weights,
        'expectedReturn': portfolioReturn(
            weights,
            meanReturns,
            tradingDays=tradingDays
        ),
        'volatility': portfolioVolatility(
            weights,
            covarianceMatrix,
            tradingDays=tradingDays
        ),
        'sharpeRatio': portfolioSharpeRatio(
            weights,
            meanReturns,
            covarianceMatrix,
            tradingDays=tradingDays
        ),
        'success': result.success,
        'message': result.message
    }


def maximumSharpePortfolio(
    meanReturns: pd.Series | np.ndarray,
    covarianceMatrix: pd.DataFrame | np.ndarray,
    riskFreeRate: float = 0,
    minimumWeight: float = 0,
    maximumWeight: float = 1,
    tradingDays: int = TRADING_DAYS
) -> dict:
    '''Optimize the maximum-Sharpe portfolio.'''
    meanReturns, covarianceMatrix = validateOptimizationInputs(
        meanReturns,
        covarianceMatrix
    )

    numberOfAssets = len(meanReturns)

    bounds = validateBounds(
        numberOfAssets,
        minimumWeight=minimumWeight,
        maximumWeight=maximumWeight
    )

    constraints = {
        'type': 'eq',
        'fun': lambda weights: weights.sum() - 1
    }

    initialWeights = equalWeights(
        numberOfAssets
    )

    result = minimize(
        fun=lambda weights: -portfolioSharpeRatio(
            weights,
            meanReturns,
            covarianceMatrix,
            riskFreeRate=riskFreeRate,
            tradingDays=tradingDays
        ),
        x0=initialWeights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise RuntimeError(
            f'Maximum-Sharpe optimization failed: {result.message}'
        )

    weights = result.x

    return {
        'weights': weights,
        'expectedReturn': portfolioReturn(
            weights,
            meanReturns,
            tradingDays=tradingDays
        ),
        'volatility': portfolioVolatility(
            weights,
            covarianceMatrix,
            tradingDays=tradingDays
        ),
        'sharpeRatio': portfolioSharpeRatio(
            weights,
            meanReturns,
            covarianceMatrix,
            riskFreeRate=riskFreeRate,
            tradingDays=tradingDays
        ),
        'success': result.success,
        'message': result.message
    }


def targetReturnPortfolio(
    meanReturns: pd.Series | np.ndarray,
    covarianceMatrix: pd.DataFrame | np.ndarray,
    targetReturn: float,
    minimumWeight: float = 0,
    maximumWeight: float = 1,
    tradingDays: int = TRADING_DAYS
) -> dict:
    '''Optimize minimum volatility for a target annual return.'''
    meanReturns, covarianceMatrix = validateOptimizationInputs(
        meanReturns,
        covarianceMatrix
    )

    numberOfAssets = len(meanReturns)

    bounds = validateBounds(
        numberOfAssets,
        minimumWeight=minimumWeight,
        maximumWeight=maximumWeight
    )

    constraints = (
        {
            'type': 'eq',
            'fun': lambda weights: weights.sum() - 1
        },
        {
            'type': 'eq',
            'fun': lambda weights: portfolioReturn(
                weights,
                meanReturns,
                tradingDays=tradingDays
            ) - targetReturn
        }
    )

    initialWeights = equalWeights(
        numberOfAssets
    )

    result = minimize(
        fun=lambda weights: portfolioVolatility(
            weights,
            covarianceMatrix,
            tradingDays=tradingDays
        ),
        x0=initialWeights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise RuntimeError(
            f'Target-return optimization failed: {result.message}'
        )

    weights = result.x

    return {
        'weights': weights,
        'expectedReturn': portfolioReturn(
            weights,
            meanReturns,
            tradingDays=tradingDays
        ),
        'volatility': portfolioVolatility(
            weights,
            covarianceMatrix,
            tradingDays=tradingDays
        ),
        'sharpeRatio': portfolioSharpeRatio(
            weights,
            meanReturns,
            covarianceMatrix,
            tradingDays=tradingDays
        ),
        'success': result.success,
        'message': result.message
    }


def targetVolatilityPortfolio(
    meanReturns: pd.Series | np.ndarray,
    covarianceMatrix: pd.DataFrame | np.ndarray,
    targetVolatility: float,
    minimumWeight: float = 0,
    maximumWeight: float = 1,
    tradingDays: int = TRADING_DAYS
) -> dict:
    '''Optimize maximum return for a target annual volatility.'''
    if targetVolatility <= 0:
        raise ValueError(
            'targetVolatility must be positive.'
        )

    meanReturns, covarianceMatrix = validateOptimizationInputs(
        meanReturns,
        covarianceMatrix
    )

    numberOfAssets = len(meanReturns)

    bounds = validateBounds(
        numberOfAssets,
        minimumWeight=minimumWeight,
        maximumWeight=maximumWeight
    )

    constraints = (
        {
            'type': 'eq',
            'fun': lambda weights: weights.sum() - 1
        },
        {
            'type': 'eq',
            'fun': lambda weights: portfolioVolatility(
                weights,
                covarianceMatrix,
                tradingDays=tradingDays
            ) - targetVolatility
        }
    )

    initialWeights = equalWeights(
        numberOfAssets
    )

    result = minimize(
        fun=lambda weights: -portfolioReturn(
            weights,
            meanReturns,
            tradingDays=tradingDays
        ),
        x0=initialWeights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise RuntimeError(
            f'Target-volatility optimization failed: {result.message}'
        )

    weights = result.x

    return {
        'weights': weights,
        'expectedReturn': portfolioReturn(
            weights,
            meanReturns,
            tradingDays=tradingDays
        ),
        'volatility': portfolioVolatility(
            weights,
            covarianceMatrix,
            tradingDays=tradingDays
        ),
        'sharpeRatio': portfolioSharpeRatio(
            weights,
            meanReturns,
            covarianceMatrix,
            tradingDays=tradingDays
        ),
        'success': result.success,
        'message': result.message
    }


def efficientFrontier(
    meanReturns: pd.Series | np.ndarray,
    covarianceMatrix: pd.DataFrame | np.ndarray,
    points: int = 50,
    minimumWeight: float = 0,
    maximumWeight: float = 1,
    tradingDays: int = TRADING_DAYS
) -> pd.DataFrame:
    '''Compute portfolios along the efficient frontier.'''
    if points < 2:
        raise ValueError(
            'points must be at least two.'
        )

    meanReturns, covarianceMatrix = validateOptimizationInputs(
        meanReturns,
        covarianceMatrix
    )

    assetAnnualReturns = meanReturns * tradingDays

    minimumVariance = minimumVariancePortfolio(
        meanReturns=meanReturns,
        covarianceMatrix=covarianceMatrix,
        minimumWeight=minimumWeight,
        maximumWeight=maximumWeight,
        tradingDays=tradingDays
    )

    minimumTarget = float(
        minimumVariance['expectedReturn']
    )

    maximumTarget = float(
        assetAnnualReturns.max()
    )

    targetReturns = np.linspace(
        minimumTarget,
        maximumTarget,
        points
    )

    frontier = []

    for targetReturn in targetReturns:
        try:
            portfolio = targetReturnPortfolio(
                meanReturns=meanReturns,
                covarianceMatrix=covarianceMatrix,
                targetReturn=targetReturn,
                minimumWeight=minimumWeight,
                maximumWeight=maximumWeight,
                tradingDays=tradingDays
            )

            frontier.append({
                'Expected Return': portfolio['expectedReturn'],
                'Volatility': portfolio['volatility'],
                'Sharpe Ratio': portfolio['sharpeRatio'],
                'Weights': portfolio['weights']
            })
        except RuntimeError:
            continue

    if not frontier:
        raise RuntimeError(
            'Unable to compute the efficient frontier.'
        )

    frontierData = pd.DataFrame(frontier)
    frontierData = frontierData.sort_values(
        'Expected Return'
    ).drop_duplicates(
        subset=['Expected Return'],
        keep='first'
    )

    efficientRows = []
    previousVolatility = -np.inf

    for _, row in frontierData.iterrows():
        volatility = float(row['Volatility'])

        if volatility > previousVolatility + 1e-10:
            efficientRows.append(row)
            previousVolatility = volatility

    if len(efficientRows) < 2:
        raise RuntimeError(
            'Unable to compute enough efficient-frontier points.'
        )

    return pd.DataFrame(efficientRows).reset_index(drop=True)


def randomPortfolios(
    meanReturns: pd.Series | np.ndarray,
    covarianceMatrix: pd.DataFrame | np.ndarray,
    portfolios: int = 5000,
    riskFreeRate: float = 0,
    tradingDays: int = TRADING_DAYS,
    randomSeed: int | None = None
) -> pd.DataFrame:
    '''Generate random long-only portfolios.'''
    if portfolios <= 0:
        raise ValueError(
            'portfolios must be positive.'
        )

    meanReturns, covarianceMatrix = validateOptimizationInputs(
        meanReturns,
        covarianceMatrix
    )

    numberOfAssets = len(meanReturns)

    randomGenerator = np.random.default_rng(
        randomSeed
    )

    weights = randomGenerator.dirichlet(
        np.ones(numberOfAssets),
        size=portfolios
    )

    expectedReturns = (
        weights
        @ meanReturns
        * tradingDays
    )

    variances = np.einsum(
        'ij,jk,ik->i',
        weights,
        covarianceMatrix,
        weights
    )

    volatilities = (
        np.sqrt(
            np.maximum(
                variances,
                0
            )
        )
        * np.sqrt(tradingDays)
    )

    sharpeRatios = np.divide(
        expectedReturns - riskFreeRate,
        volatilities,
        out=np.full(
            portfolios,
            np.nan
        ),
        where=~np.isclose(
            volatilities,
            0
        )
    )

    return pd.DataFrame({
        'Expected Return': expectedReturns,
        'Volatility': volatilities,
        'Sharpe Ratio': sharpeRatios,
        'Weights': list(weights)
    })


def optimizationWeights(
    optimizationResult: dict,
    assetNames: list[str] | pd.Index
) -> pd.Series:
    '''Return labeled weights from an optimization result.'''
    if 'weights' not in optimizationResult:
        raise ValueError(
            'optimizationResult must contain weights.'
        )

    weights = np.asarray(
        optimizationResult['weights'],
        dtype=float
    )

    assetNames = list(
        assetNames
    )

    if len(weights) != len(assetNames):
        raise ValueError(
            'assetNames length must match optimized weights.'
        )

    weights[np.abs(weights) < 1e-10] = 0.0

    totalWeight = weights.sum()

    if (
        not np.isclose(totalWeight, 0)
        and np.isclose(totalWeight, 1.0, atol=1e-8)
    ):
        weights = weights / totalWeight

    return pd.Series(
        weights,
        index=assetNames,
        name='Weight'
    )


def optimizationSummary(
    optimizationResult: dict,
    assetNames: list[str] | pd.Index
) -> pd.Series:
    '''Return a formatted optimization result summary.'''
    weights = optimizationWeights(
        optimizationResult,
        assetNames
    )

    summary = {
        'Expected Return': optimizationResult[
            'expectedReturn'
        ],
        'Volatility': optimizationResult[
            'volatility'
        ],
        'Sharpe Ratio': optimizationResult[
            'sharpeRatio'
        ]
    }

    for asset, weight in weights.items():
        summary[f'Weight {asset}'] = weight

    return pd.Series(
        summary,
        name='Optimized Portfolio'
    )


def optimizePortfolio(
    method: Literal[
        'minimum_variance',
        'maximum_sharpe',
        'target_return',
        'target_volatility'
    ],
    meanReturns: pd.Series | np.ndarray,
    covarianceMatrix: pd.DataFrame | np.ndarray,
    riskFreeRate: float = 0,
    targetReturn: float | None = None,
    targetVolatility: float | None = None,
    minimumWeight: float = 0,
    maximumWeight: float = 1,
    tradingDays: int = TRADING_DAYS
) -> dict:
    '''Run the selected portfolio optimization method.'''
    if method == 'minimum_variance':
        return minimumVariancePortfolio(
            meanReturns=meanReturns,
            covarianceMatrix=covarianceMatrix,
            minimumWeight=minimumWeight,
            maximumWeight=maximumWeight,
            tradingDays=tradingDays
        )

    if method == 'maximum_sharpe':
        return maximumSharpePortfolio(
            meanReturns=meanReturns,
            covarianceMatrix=covarianceMatrix,
            riskFreeRate=riskFreeRate,
            minimumWeight=minimumWeight,
            maximumWeight=maximumWeight,
            tradingDays=tradingDays
        )

    if method == 'target_return':
        if targetReturn is None:
            raise ValueError(
                'targetReturn is required for target_return.'
            )

        return targetReturnPortfolio(
            meanReturns=meanReturns,
            covarianceMatrix=covarianceMatrix,
            targetReturn=targetReturn,
            minimumWeight=minimumWeight,
            maximumWeight=maximumWeight,
            tradingDays=tradingDays
        )

    if method == 'target_volatility':
        if targetVolatility is None:
            raise ValueError(
                'targetVolatility is required for target_volatility.'
            )

        return targetVolatilityPortfolio(
            meanReturns=meanReturns,
            covarianceMatrix=covarianceMatrix,
            targetVolatility=targetVolatility,
            minimumWeight=minimumWeight,
            maximumWeight=maximumWeight,
            tradingDays=tradingDays
        )

    raise ValueError(
        'method must be minimum_variance, maximum_sharpe, target_return or target_volatility.'
    )
