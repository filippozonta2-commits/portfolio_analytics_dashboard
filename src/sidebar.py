from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


DEFAULT_TICKERS = [
    'AAPL',
    'MSFT',
    'GOOGL',
    'AMZN'
]

DEFAULT_BENCHMARK = '^GSPC'
DEFAULT_LOOKBACK_YEARS = 5
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_ROLLING_WINDOW = 63
DEFAULT_SIMULATIONS = 5000
DEFAULT_HORIZON_DAYS = 252
DEFAULT_INITIAL_VALUE = 10000.0


def parseTickers(
    tickerInput: str
) -> list[str]:
    '''Parse and clean ticker input.'''
    tickers = [
        ticker.strip().upper()
        for ticker in tickerInput.replace(
            ';',
            ','
        ).split(',')
        if ticker.strip()
    ]

    return list(
        dict.fromkeys(tickers)
    )


def validateTickers(
    tickers: list[str],
    minimumTickers: int = 1,
    maximumTickers: int = 20
) -> list[str]:
    '''Validate the selected ticker universe.'''
    if len(tickers) < minimumTickers:
        raise ValueError(
            f'At least {minimumTickers} ticker is required.'
        )

    if len(tickers) > maximumTickers:
        raise ValueError(
            f'A maximum of {maximumTickers} tickers is allowed.'
        )

    invalidTickers = [
        ticker
        for ticker in tickers
        if not ticker.replace(
            '-',
            ''
        ).replace(
            '.',
            ''
        ).replace(
            '^',
            ''
        ).isalnum()
    ]

    if invalidTickers:
        raise ValueError(
            'Invalid ticker symbols: '
            + ', '.join(invalidTickers)
        )

    return tickers


def validateDateRange(
    startDate: date,
    endDate: date
) -> tuple[date, date]:
    '''Validate the selected data range.'''
    if startDate >= endDate:
        raise ValueError(
            'Start date must be earlier than end date.'
        )

    if endDate > date.today():
        raise ValueError(
            'End date cannot be in the future.'
        )

    return startDate, endDate


def validateConfidenceLevel(
    confidenceLevel: float
) -> float:
    '''Validate a confidence level.'''
    if not 0 < confidenceLevel < 1:
        raise ValueError(
            'Confidence level must be between zero and one.'
        )

    return float(confidenceLevel)


def normalizeWeights(
    weights: np.ndarray
) -> np.ndarray:
    '''Normalize weights so they sum to one.'''
    weights = np.asarray(
        weights,
        dtype=float
    )

    if weights.ndim != 1:
        raise ValueError(
            'weights must be one-dimensional.'
        )

    if not np.isfinite(weights).all():
        raise ValueError(
            'weights must contain finite values.'
        )

    totalWeight = weights.sum()

    if np.isclose(totalWeight, 0):
        raise ValueError(
            'weights cannot sum to zero.'
        )

    return weights / totalWeight


def validateWeights(
    weights: np.ndarray,
    tickerCount: int,
    allowShortSelling: bool = False
) -> np.ndarray:
    '''Validate portfolio weights.'''
    weights = np.asarray(
        weights,
        dtype=float
    )

    if len(weights) != tickerCount:
        raise ValueError(
            'The number of weights must match the number of tickers.'
        )

    if not np.isfinite(weights).all():
        raise ValueError(
            'weights must contain finite values.'
        )

    if (
        not allowShortSelling
        and np.any(weights < 0)
    ):
        raise ValueError(
            'Negative weights require short selling to be enabled.'
        )

    if np.isclose(weights.sum(), 0):
        raise ValueError(
            'weights cannot sum to zero.'
        )

    return normalizeWeights(weights)


def equalWeights(
    tickerCount: int
) -> np.ndarray:
    '''Return equal portfolio weights.'''
    if tickerCount <= 0:
        raise ValueError(
            'tickerCount must be positive.'
        )

    return np.repeat(
        1 / tickerCount,
        tickerCount
    )


def weightsEditor(
    tickers: list[str],
    defaultWeights: np.ndarray | None = None,
    allowShortSelling: bool = False
) -> np.ndarray:
    '''Render an editable portfolio weights table.'''
    if defaultWeights is None:
        defaultWeights = equalWeights(
            len(tickers)
        )

    defaultWeights = validateWeights(
        defaultWeights,
        tickerCount=len(tickers),
        allowShortSelling=allowShortSelling
    )

    weightData = pd.DataFrame({
        'Ticker': tickers,
        'Weight': defaultWeights
    })

    editedWeights = st.sidebar.data_editor(
        weightData,
        hide_index=True,
        use_container_width=True,
        disabled=['Ticker'],
        column_config={
            'Ticker': st.column_config.TextColumn(
                'Ticker'
            ),
            'Weight': st.column_config.NumberColumn(
                'Weight',
                min_value=(
                    -1.0
                    if allowShortSelling
                    else 0.0
                ),
                max_value=1.0,
                step=0.01,
                format='%.4f'
            )
        },
        key='portfolioWeightsEditor'
    )

    weights = editedWeights[
        'Weight'
    ].to_numpy(
        dtype=float
    )

    return validateWeights(
        weights,
        tickerCount=len(tickers),
        allowShortSelling=allowShortSelling
    )


def renderUniverseSettings() -> dict[str, Any]:
    '''Render ticker and date settings.'''
    st.sidebar.subheader(
        'Portfolio Universe'
    )

    tickerInput = st.sidebar.text_area(
        'Ticker symbols',
        value=', '.join(DEFAULT_TICKERS),
        help='Enter ticker symbols separated by commas.',
        key='tickerInput'
    )

    tickers = parseTickers(
        tickerInput
    )

    tickers = validateTickers(
        tickers
    )

    defaultEndDate = date.today()
    defaultStartDate = (
        defaultEndDate
        - timedelta(
            days=365 * DEFAULT_LOOKBACK_YEARS
        )
    )

    startDate = st.sidebar.date_input(
        'Start date',
        value=defaultStartDate,
        max_value=defaultEndDate,
        key='startDate'
    )

    endDate = st.sidebar.date_input(
        'End date',
        value=defaultEndDate,
        max_value=defaultEndDate,
        key='endDate'
    )

    startDate, endDate = validateDateRange(
        startDate,
        endDate
    )

    priceField = st.sidebar.selectbox(
        'Price field',
        options=[
            'Close',
            'Adj Close'
        ],
        index=1,
        key='priceField'
    )

    return {
        'tickers': tickers,
        'startDate': startDate,
        'endDate': endDate,
        'priceField': priceField
    }


def renderPortfolioSettings(
    tickers: list[str]
) -> dict[str, Any]:
    '''Render portfolio construction settings.'''
    st.sidebar.subheader(
        'Portfolio Construction'
    )

    weightingMethod = st.sidebar.selectbox(
        'Weighting method',
        options=[
            'Equal Weight',
            'Custom Weight',
            'Minimum Variance',
            'Maximum Sharpe'
        ],
        index=0,
        key='weightingMethod'
    )

    allowShortSelling = st.sidebar.checkbox(
        'Allow short selling',
        value=False,
        key='allowShortSelling'
    )

    weights = None

    if weightingMethod == 'Equal Weight':
        weights = equalWeights(
            len(tickers)
        )

    elif weightingMethod == 'Custom Weight':
        weights = weightsEditor(
            tickers,
            allowShortSelling=allowShortSelling
        )

        st.sidebar.caption(
            f'Normalized total: {weights.sum():.2%}'
        )

    rebalanceFrequency = st.sidebar.selectbox(
        'Rebalancing frequency',
        options=[
            'None',
            'Monthly',
            'Quarterly',
            'Semiannual',
            'Annual'
        ],
        index=0,
        key='rebalanceFrequency'
    )

    return {
        'weightingMethod': weightingMethod,
        'weights': weights,
        'allowShortSelling': allowShortSelling,
        'rebalanceFrequency': rebalanceFrequency
    }


def renderBenchmarkSettings() -> dict[str, Any]:
    '''Render benchmark analysis settings.'''
    st.sidebar.subheader(
        'Benchmark'
    )

    benchmarkEnabled = st.sidebar.checkbox(
        'Enable benchmark comparison',
        value=True,
        key='benchmarkEnabled'
    )

    benchmarkTicker = DEFAULT_BENCHMARK

    if benchmarkEnabled:
        benchmarkTicker = st.sidebar.text_input(
            'Benchmark ticker',
            value=DEFAULT_BENCHMARK,
            key='benchmarkTicker'
        ).strip().upper()

        if not benchmarkTicker:
            raise ValueError(
                'Benchmark ticker cannot be empty.'
            )

    rollingWindow = st.sidebar.number_input(
        'Rolling window',
        min_value=20,
        max_value=504,
        value=DEFAULT_ROLLING_WINDOW,
        step=1,
        key='rollingWindow'
    )

    return {
        'benchmarkEnabled': benchmarkEnabled,
        'benchmarkTicker': benchmarkTicker,
        'rollingWindow': int(rollingWindow)
    }


def renderRiskFreeSettings() -> dict[str, Any]:
    '''Render risk-free rate settings.'''
    st.sidebar.subheader(
        'Risk-Free Rate'
    )

    riskFreeMode = st.sidebar.radio(
        'Rate source',
        options=[
            'Automatic',
            'Manual'
        ],
        index=0,
        horizontal=True,
        key='riskFreeMode'
    )

    riskFreeMethod = 'match'
    manualRiskFreeRate = None

    if riskFreeMode == 'Automatic':
        riskFreeMethodLabel = st.sidebar.selectbox(
            'Treasury selection',
            options=[
                'Match investment horizon',
                'Latest 3-month Treasury',
                'Latest 1-year Treasury',
                'Latest 2-year Treasury',
                'Latest 5-year Treasury',
                'Latest 10-year Treasury',
                'Latest 30-year Treasury'
            ],
            index=0,
            key='riskFreeMethodLabel'
        )

        methodMap = {
            'Match investment horizon': 'match',
            'Latest 3-month Treasury': '3m',
            'Latest 1-year Treasury': '1y',
            'Latest 2-year Treasury': '2y',
            'Latest 5-year Treasury': '5y',
            'Latest 10-year Treasury': '10y',
            'Latest 30-year Treasury': '30y'
        }

        riskFreeMethod = methodMap[
            riskFreeMethodLabel
        ]

    else:
        manualRiskFreeRate = (
            st.sidebar.number_input(
                'Annual risk-free rate (%)',
                min_value=-10.0,
                max_value=30.0,
                value=4.0,
                step=0.05,
                key='manualRiskFreeRate'
            )
            / 100
        )

    return {
        'riskFreeMode': riskFreeMode.lower(),
        'riskFreeMethod': riskFreeMethod,
        'manualRiskFreeRate': manualRiskFreeRate
    }


def renderRiskSettings() -> dict[str, Any]:
    '''Render risk metric settings.'''
    st.sidebar.subheader(
        'Risk Analytics'
    )

    confidenceLabel = st.sidebar.select_slider(
        'Confidence level',
        options=[
            0.90,
            0.95,
            0.975,
            0.99
        ],
        value=DEFAULT_CONFIDENCE_LEVEL,
        format_func=lambda value: f'{value:.1%}',
        key='confidenceLevel'
    )

    confidenceLevel = validateConfidenceLevel(
        confidenceLabel
    )

    annualizationFactor = st.sidebar.number_input(
        'Trading days per year',
        min_value=1,
        max_value=366,
        value=252,
        step=1,
        key='annualizationFactor'
    )

    return {
        'confidenceLevel': confidenceLevel,
        'annualizationFactor': int(
            annualizationFactor
        )
    }


def renderOptimizationSettings() -> dict[str, Any]:
    '''Render portfolio optimization settings.'''
    st.sidebar.subheader(
        'Optimization'
    )

    optimizationEnabled = st.sidebar.checkbox(
        'Enable optimization',
        value=True,
        key='optimizationEnabled'
    )

    optimizationMethod = st.sidebar.selectbox(
        'Optimization objective',
        options=[
            'Maximum Sharpe',
            'Minimum Variance',
            'Target Return',
            'Target Volatility'
        ],
        index=0,
        disabled=not optimizationEnabled,
        key='optimizationMethod'
    )

    targetReturn = None
    targetVolatility = None

    if (
        optimizationEnabled
        and optimizationMethod == 'Target Return'
    ):
        targetReturn = (
            st.sidebar.number_input(
                'Target annual return (%)',
                min_value=-50.0,
                max_value=100.0,
                value=10.0,
                step=0.5,
                key='targetReturn'
            )
            / 100
        )

    if (
        optimizationEnabled
        and optimizationMethod
        == 'Target Volatility'
    ):
        targetVolatility = (
            st.sidebar.number_input(
                'Target annual volatility (%)',
                min_value=0.1,
                max_value=100.0,
                value=15.0,
                step=0.5,
                key='targetVolatility'
            )
            / 100
        )

    frontierPoints = st.sidebar.slider(
        'Efficient frontier points',
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        disabled=not optimizationEnabled,
        key='frontierPoints'
    )

    randomPortfolioCount = st.sidebar.slider(
        'Random portfolios',
        min_value=100,
        max_value=20000,
        value=5000,
        step=100,
        disabled=not optimizationEnabled,
        key='randomPortfolioCount'
    )

    return {
        'optimizationEnabled': optimizationEnabled,
        'optimizationMethod': optimizationMethod,
        'targetReturn': targetReturn,
        'targetVolatility': targetVolatility,
        'frontierPoints': int(frontierPoints),
        'randomPortfolioCount': int(
            randomPortfolioCount
        )
    }


def renderSimulationSettings() -> dict[str, Any]:
    '''Render portfolio simulation settings.'''
    st.sidebar.subheader(
        'Simulation'
    )

    simulationEnabled = st.sidebar.checkbox(
        'Enable simulation',
        value=True,
        key='simulationEnabled'
    )

    simulationMethod = st.sidebar.selectbox(
        'Simulation method',
        options=[
            'Monte Carlo',
            'GBM',
            'Bootstrap'
        ],
        index=0,
        disabled=not simulationEnabled,
        key='simulationMethod'
    )

    simulationCount = st.sidebar.slider(
        'Number of simulations',
        min_value=100,
        max_value=50000,
        value=DEFAULT_SIMULATIONS,
        step=100,
        disabled=not simulationEnabled,
        key='simulationCount'
    )

    horizonDays = st.sidebar.slider(
        'Investment horizon',
        min_value=21,
        max_value=2520,
        value=DEFAULT_HORIZON_DAYS,
        step=21,
        disabled=not simulationEnabled,
        key='horizonDays'
    )

    initialValue = st.sidebar.number_input(
        'Initial portfolio value',
        min_value=1.0,
        value=DEFAULT_INITIAL_VALUE,
        step=1000.0,
        disabled=not simulationEnabled,
        key='initialValue'
    )

    targetEnabled = st.sidebar.checkbox(
        'Set terminal target',
        value=True,
        disabled=not simulationEnabled,
        key='targetEnabled'
    )

    targetValue = None

    if targetEnabled and simulationEnabled:
        targetValue = st.sidebar.number_input(
            'Target portfolio value',
            min_value=1.0,
            value=float(initialValue * 1.10),
            step=1000.0,
            key='targetValue'
        )

    randomSeedEnabled = st.sidebar.checkbox(
        'Use fixed random seed',
        value=True,
        disabled=not simulationEnabled,
        key='randomSeedEnabled'
    )

    randomSeed = None

    if randomSeedEnabled and simulationEnabled:
        randomSeed = st.sidebar.number_input(
            'Random seed',
            min_value=0,
            max_value=1000000,
            value=42,
            step=1,
            key='randomSeed'
        )

    maximumDisplayedPaths = st.sidebar.slider(
        'Displayed simulation paths',
        min_value=10,
        max_value=1000,
        value=200,
        step=10,
        disabled=not simulationEnabled,
        key='maximumDisplayedPaths'
    )

    return {
        'simulationEnabled': simulationEnabled,
        'simulationMethod': simulationMethod,
        'simulationCount': int(simulationCount),
        'horizonDays': int(horizonDays),
        'initialValue': float(initialValue),
        'targetValue': (
            float(targetValue)
            if targetValue is not None
            else None
        ),
        'randomSeed': (
            int(randomSeed)
            if randomSeed is not None
            else None
        ),
        'maximumDisplayedPaths': int(
            maximumDisplayedPaths
        )
    }


def renderFundamentalSettings() -> dict[str, Any]:
    '''Render fundamental analysis settings.'''
    st.sidebar.subheader(
        'Fundamentals'
    )

    fundamentalsEnabled = st.sidebar.checkbox(
        'Enable fundamental analysis',
        value=True,
        key='fundamentalsEnabled'
    )

    fundamentalTickerLimit = st.sidebar.slider(
        'Maximum companies',
        min_value=1,
        max_value=20,
        value=10,
        step=1,
        disabled=not fundamentalsEnabled,
        key='fundamentalTickerLimit'
    )

    return {
        'fundamentalsEnabled': fundamentalsEnabled,
        'fundamentalTickerLimit': int(
            fundamentalTickerLimit
        )
    }


def renderDisplaySettings() -> dict[str, Any]:
    '''Render dashboard display settings.'''
    st.sidebar.subheader(
        'Display'
    )

    showRawData = st.sidebar.checkbox(
        'Show raw data',
        value=False,
        key='showRawData'
    )

    showCorrelation = st.sidebar.checkbox(
        'Show correlation matrix',
        value=True,
        key='showCorrelation'
    )

    showCovariance = st.sidebar.checkbox(
        'Show covariance matrix',
        value=False,
        key='showCovariance'
    )

    decimalPlaces = st.sidebar.selectbox(
        'Decimal places',
        options=[
            2,
            3,
            4
        ],
        index=2,
        key='decimalPlaces'
    )

    return {
        'showRawData': showRawData,
        'showCorrelation': showCorrelation,
        'showCovariance': showCovariance,
        'decimalPlaces': int(decimalPlaces)
    }


def validateSidebarSettings(
    settings: dict[str, Any]
) -> dict[str, Any]:
    '''Validate the complete sidebar configuration.'''
    validateTickers(
        settings['tickers']
    )

    validateDateRange(
        settings['startDate'],
        settings['endDate']
    )

    validateConfidenceLevel(
        settings['confidenceLevel']
    )

    if (
        settings['weights'] is not None
    ):
        settings['weights'] = validateWeights(
            settings['weights'],
            tickerCount=len(
                settings['tickers']
            ),
            allowShortSelling=settings[
                'allowShortSelling'
            ]
        )

    if settings['simulationCount'] <= 0:
        raise ValueError(
            'simulationCount must be positive.'
        )

    if settings['horizonDays'] <= 0:
        raise ValueError(
            'horizonDays must be positive.'
        )

    if settings['initialValue'] <= 0:
        raise ValueError(
            'initialValue must be positive.'
        )

    if (
        settings['targetValue'] is not None
        and settings['targetValue'] <= 0
    ):
        raise ValueError(
            'targetValue must be positive.'
        )

    if settings['annualizationFactor'] <= 0:
        raise ValueError(
            'annualizationFactor must be positive.'
        )

    if settings['rollingWindow'] <= 1:
        raise ValueError(
            'rollingWindow must be greater than one.'
        )

    return settings


def renderSidebar() -> dict[str, Any]:
    '''Render the full application sidebar.'''
    st.sidebar.title(
        'Portfolio Settings'
    )

    st.sidebar.caption(
        'Configure assets, portfolio construction, '
        'risk analysis and simulations.'
    )

    with st.sidebar.expander(
        'Data and Universe',
        expanded=True
    ):
        universeSettings = renderUniverseSettings()

    with st.sidebar.expander(
        'Portfolio',
        expanded=True
    ):
        portfolioSettings = renderPortfolioSettings(
            universeSettings['tickers']
        )

    with st.sidebar.expander(
        'Benchmark',
        expanded=False
    ):
        benchmarkSettings = renderBenchmarkSettings()

    with st.sidebar.expander(
        'Risk-Free Rate',
        expanded=False
    ):
        riskFreeSettings = renderRiskFreeSettings()

    with st.sidebar.expander(
        'Risk Analytics',
        expanded=False
    ):
        riskSettings = renderRiskSettings()

    with st.sidebar.expander(
        'Optimization',
        expanded=False
    ):
        optimizationSettings = (
            renderOptimizationSettings()
        )

    with st.sidebar.expander(
        'Simulation',
        expanded=False
    ):
        simulationSettings = (
            renderSimulationSettings()
        )

    with st.sidebar.expander(
        'Fundamentals',
        expanded=False
    ):
        fundamentalSettings = (
            renderFundamentalSettings()
        )

    with st.sidebar.expander(
        'Display',
        expanded=False
    ):
        displaySettings = renderDisplaySettings()

    settings = {
        **universeSettings,
        **portfolioSettings,
        **benchmarkSettings,
        **riskFreeSettings,
        **riskSettings,
        **optimizationSettings,
        **simulationSettings,
        **fundamentalSettings,
        **displaySettings
    }

    return validateSidebarSettings(
        settings
    )

