import numpy as np
import pandas as pd


TRADING_DAYS = 252


def simulatePortfolios(
    assetReturns,
    simulations=5000,
    riskFreeRate=0.0,
    seed=42
):
    """
    Simulate random portfolios to approximate the Efficient Frontier.
    """

    rng = np.random.default_rng(seed)

    meanReturns = assetReturns.mean() * TRADING_DAYS
    covarianceMatrix = assetReturns.cov() * TRADING_DAYS

    tickers = assetReturns.columns
    numberOfAssets = len(tickers)

    results = []

    for _ in range(simulations):

        weights = rng.random(numberOfAssets)
        weights = weights / weights.sum()

        portfolioReturn = np.dot(
            weights,
            meanReturns
        )

        portfolioVolatility = np.sqrt(
            weights.T @ covarianceMatrix @ weights
        )

        if np.isclose(portfolioVolatility, 0):
            sharpeRatio = np.nan
        else:
            sharpeRatio = (
                portfolioReturn - riskFreeRate
            ) / portfolioVolatility

        result = {
            "Return": portfolioReturn,
            "Volatility": portfolioVolatility,
            "Sharpe Ratio": sharpeRatio
        }

        for ticker, weight in zip(tickers, weights):
            result[ticker] = weight

        results.append(result)

    return pd.DataFrame(results)


def maximumSharpePortfolio(portfolioResults):
    """
    Return the portfolio with the highest Sharpe Ratio.
    """

    validResults = portfolioResults.dropna(
        subset=["Sharpe Ratio"]
    )

    if validResults.empty:
        raise ValueError(
            "No valid portfolio found."
        )

    bestIndex = validResults["Sharpe Ratio"].idxmax()

    return validResults.loc[bestIndex]


def minimumVolatilityPortfolio(portfolioResults):
    """
    Return the portfolio with the lowest volatility.
    """

    validResults = portfolioResults.dropna(
        subset=["Volatility"]
    )

    if validResults.empty:
        raise ValueError(
            "No valid portfolio found."
        )

    bestIndex = validResults["Volatility"].idxmin()

    return validResults.loc[bestIndex]

def riskSummary(
    portfolioReturns: pd.Series,
    confidenceLevels: tuple[float, ...] = (0.95, 0.99),
    portfolioValue: float | None = None
) -> pd.DataFrame:
    '''Return portfolio risk metrics across multiple confidence levels.'''
    if portfolioReturns.empty:
        raise ValueError(
            'portfolioReturns cannot be empty.'
        )

    rows = []

    for confidenceLevel in confidenceLevels:
        rows.append({
            'Confidence Level': confidenceLevel,
            'Historical VaR': historicalVaR(
                portfolioReturns,
                confidenceLevel=confidenceLevel,
                portfolioValue=portfolioValue
            ),
            'Historical CVaR': historicalCVaR(
                portfolioReturns,
                confidenceLevel=confidenceLevel,
                portfolioValue=portfolioValue
            ),
            'Parametric VaR': parametricVaR(
                portfolioReturns,
                confidenceLevel=confidenceLevel,
                portfolioValue=portfolioValue
            )
        })

    summary = pd.DataFrame(rows)
    summary = summary.set_index('Confidence Level')

    return summary


def performanceSummary(
    portfolioReturns: pd.Series,
    riskFreeRate: float = 0,
    tradingDays: int = TRADING_DAYS
) -> pd.Series:
    '''Return the main portfolio performance metrics.'''
    if portfolioReturns.empty:
        raise ValueError(
            'portfolioReturns cannot be empty.'
        )

    summary = {
        'Annual Return': annualizedReturn(
            portfolioReturns,
            tradingDays=tradingDays,
            method='geometric'
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
        )
    }

    return pd.Series(
        summary,
        name='Portfolio'
    )


def benchmarkSummary(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series,
    riskFreeRate: float = 0,
    tradingDays: int = TRADING_DAYS
) -> pd.Series:
    '''Return portfolio metrics relative to a benchmark.'''
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
            'Portfolio and benchmark have no aligned observations.'
        )

    portfolioReturn = alignedReturns['Portfolio']
    benchmarkReturn = alignedReturns['Benchmark']

    activeReturns = (
        portfolioReturn
        - benchmarkReturn
    )

    trackingError = (
        activeReturns.std(ddof=1)
        * tradingDays ** 0.5
    )

    annualActiveReturn = (
        activeReturns.mean()
        * tradingDays
    )

    informationRatio = (
        annualActiveReturn / trackingError
        if trackingError != 0
        else float('nan')
    )

    summary = {
        'Portfolio Return': annualizedReturn(
            portfolioReturn,
            tradingDays=tradingDays,
            method='geometric'
        ),
        'Benchmark Return': annualizedReturn(
            benchmarkReturn,
            tradingDays=tradingDays,
            method='geometric'
        ),
        'Active Return': annualActiveReturn,
        'Tracking Error': trackingError,
        'Information Ratio': informationRatio,
        'Beta': beta(
            portfolioReturn,
            benchmarkReturn
        ),
        'Alpha': alpha(
            portfolioReturn,
            benchmarkReturn,
            riskFreeRate=riskFreeRate,
            tradingDays=tradingDays
        )
    }

    return pd.Series(
        summary,
        name='Benchmark Comparison'
    )


def rollingSummary(
    portfolioReturns: pd.Series,
    window: int = TRADING_DAYS,
    riskFreeRate: float = 0,
    tradingDays: int = TRADING_DAYS
) -> pd.DataFrame:
    '''Return rolling portfolio return, volatility and Sharpe ratio.'''
    if portfolioReturns.empty:
        raise ValueError(
            'portfolioReturns cannot be empty.'
        )

    if window <= 1:
        raise ValueError(
            'window must be greater than one.'
        )

    if len(portfolioReturns) < window:
        raise ValueError(
            'Not enough observations for the selected rolling window.'
        )

    returns = portfolioReturns.dropna().astype(float)

    rollingReturn = (
        (1 + returns)
        .rolling(window)
        .apply(
            lambda values: values.prod() ** (
                tradingDays / window
            ) - 1,
            raw=True
        )
    )

    rollingVolatility = (
        returns
        .rolling(window)
        .std(ddof=1)
        * tradingDays ** 0.5
    )

    rollingSharpe = (
        rollingReturn - riskFreeRate
    ) / rollingVolatility

    summary = pd.DataFrame({
        'Rolling Return': rollingReturn,
        'Rolling Volatility': rollingVolatility,
        'Rolling Sharpe': rollingSharpe
    })

    return summary.dropna(how='all')


def comparisonSummary(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series,
    riskFreeRate: float = 0,
    confidenceLevel: float = 0.95,
    tradingDays: int = TRADING_DAYS
) -> pd.DataFrame:
    '''Compare portfolio and benchmark performance and risk metrics.'''
    if portfolioReturns.empty or benchmarkReturns.empty:
        raise ValueError(
            'Portfolio and benchmark returns cannot be empty.'
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
            'Portfolio and benchmark have no aligned observations.'
        )

    summaries = {}

    for column in alignedReturns.columns:
        returns = alignedReturns[column]

        summaries[column] = {
            'Annual Return': annualizedReturn(
                returns,
                tradingDays=tradingDays,
                method='geometric'
            ),
            'Annual Volatility': annualizedVolatility(
                returns,
                tradingDays=tradingDays
            ),
            'Sharpe Ratio': sharpeRatio(
                returns,
                riskFreeRate=riskFreeRate,
                tradingDays=tradingDays
            ),
            'Sortino Ratio': sortinoRatio(
                returns,
                riskFreeRate=riskFreeRate,
                tradingDays=tradingDays
            ),
            'Maximum Drawdown': maxDrawdown(
                returns
            ),
            f'VaR {confidenceLevel:.0%}': historicalVaR(
                returns,
                confidenceLevel=confidenceLevel
            ),
            f'CVaR {confidenceLevel:.0%}': historicalCVaR(
                returns,
                confidenceLevel=confidenceLevel
            )
        }

    return pd.DataFrame(summaries)


def formatSummary(
    summary: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    '''Format summary values for dashboard display.'''
    percentageMetrics = {
        'Annual Return',
        'Annual Volatility',
        'Maximum Drawdown',
        'Portfolio Return',
        'Benchmark Return',
        'Active Return',
        'Tracking Error'
    }

    def formatValue(metric, value):
        if pd.isna(value):
            return 'N/A'

        if metric in percentageMetrics:
            return f'{value:.2%}'

        if 'VaR' in str(metric) or 'CVaR' in str(metric):
            return f'{value:.2%}'

        return f'{value:.2f}'

    if isinstance(summary, pd.Series):
        return pd.Series({
            metric: formatValue(metric, value)
            for metric, value in summary.items()
        })

    formatted = summary.copy()

    for index in formatted.index:
        for column in formatted.columns:
            metric = index
            formatted.loc[index, column] = formatValue(
                metric,
                formatted.loc[index, column]
            )

    return formatted