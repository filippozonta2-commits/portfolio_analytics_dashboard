from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import charts
from src.analytics import (
    alpha,
    annualizedReturn,
    annualizedVolatility,
    beta,
    calmarRatio,
    cumulativeReturns,
    drawdownSeries,
    historicalCVaR,
    historicalVaR,
    maxDrawdown,
    parametricVaR,
    portfolioReturns as computePortfolioReturns,
    sharpeRatio,
    sortinoRatio
)
from src.benchmark import (
    activeReturns as computeActiveReturns,
    benchmarkReturns as computeBenchmarkReturns,
    benchmarkSummary as computeBenchmarkSummary,
    captureRatio,
    downCaptureRatio,
    downloadBenchmarkPrices,
    relativePerformance as computeRelativePerformance,
    upCaptureRatio
)
from src.data import (
    assetSummary,
    computeReturns,
    correlationMatrix,
    covarianceMatrix,
    getData,
    getRiskFreeRate,
    normalizePrices,
    pricePerformance
)
from src.fundamentals import (
    formatFundamentals,
    multipleFundamentalsSummary
)
from src.metrics import portfolioSummary
from src.sidebar import renderSidebar
from src.simulation import (
    efficientFrontier,
    maximumSharpePortfolio,
    minimumVariancePortfolio,
    optimizationSummary,
    optimizationWeights,
    optimizePortfolio,
    randomPortfolios,
    targetReturnPortfolio,
    targetVolatilityPortfolio
)
from src.utils import (
    csvDownloadButton,
    formatCurrency,
    formatPercent,
    renderDataFrame,
    renderEmptyState,
    renderMetricRow,
    renderSectionTitle,
    rollingAnnualizedReturn,
    rollingAnnualizedVolatility,
    rollingSharpeRatio
)


BUILD_VERSION = 'runtime-fix-2026-08-26.2'


st.set_page_config(
    page_title='PortfolioLab | Portfolio Analytics',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown(
    '''
    <style>
        .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
        [data-testid="stMetric"] {
            background: #111827;
            border: 1px solid #263244;
            border-radius: 12px;
            padding: 14px 16px;
        }
        [data-testid="stMetricLabel"] {color: #94A3B8;}
        [data-testid="stMetricValue"] {color: #F8FAFC;}
        div[data-testid="stDataFrame"] {
            border: 1px solid #263244;
            border-radius: 10px;
            overflow: hidden;
        }
        .portfolio-subtitle {color: #94A3B8; margin-top: -0.6rem;}
        .portfolio-badge {
            display: inline-block;
            background: #172033;
            color: #93C5FD;
            border: 1px solid #263B5E;
            border-radius: 999px;
            padding: 0.2rem 0.65rem;
            margin-right: 0.35rem;
            font-size: 0.78rem;
        }
    </style>
    ''',
    unsafe_allow_html=True
)


# --------------------------------------------------------------------------
# Monte Carlo simulation engine
# --------------------------------------------------------------------------
# No price-path simulation engine was provided in the backend modules
# (simulation.py only covers portfolio optimization), so it is implemented
# here directly to support the Simulation tab configured in the sidebar.

def runSimulation(
    portfolioReturnSeries: pd.Series,
    method: str,
    simulations: int,
    horizonDays: int,
    initialValue: float,
    randomSeed: int | None
) -> pd.DataFrame:
    '''Simulate future portfolio value paths.'''
    returnsArray = portfolioReturnSeries.dropna().to_numpy()

    if len(returnsArray) < 2:
        raise ValueError(
            'Not enough historical observations to run a simulation.'
        )

    rng = np.random.default_rng(randomSeed)

    if method == 'Bootstrap':
        sampledReturns = rng.choice(
            returnsArray,
            size=(horizonDays, simulations),
            replace=True
        )

    elif method == 'GBM':
        drift = returnsArray.mean()
        volatility = returnsArray.std(ddof=1)

        sampledReturns = rng.normal(
            loc=drift - 0.5 * volatility ** 2,
            scale=volatility,
            size=(horizonDays, simulations)
        )

    else:
        drift = returnsArray.mean()
        volatility = returnsArray.std(ddof=1)

        sampledReturns = rng.normal(
            loc=drift,
            scale=volatility,
            size=(horizonDays, simulations)
        )

    growthFactors = 1 + sampledReturns
    growthFactors = np.clip(growthFactors, a_min=1e-6, a_max=None)

    paths = initialValue * np.cumprod(growthFactors, axis=0)

    pathColumns = [f'Path {i + 1}' for i in range(simulations)]

    return pd.DataFrame(
        paths,
        columns=pathColumns,
        index=pd.RangeIndex(1, horizonDays + 1, name='Simulation Day')
    )


def simulationPercentiles(
    simulatedPaths: pd.DataFrame,
    percentiles: tuple[int, ...] = (5, 10, 25, 50, 75, 90, 95)
) -> pd.DataFrame:
    '''Compute percentile bands across all simulated paths.'''
    percentilePaths = {}

    for percentile in percentiles:
        percentilePaths[f'{percentile}%'] = np.percentile(
            simulatedPaths.values,
            percentile,
            axis=1
        )

    return pd.DataFrame(
        percentilePaths,
        index=simulatedPaths.index
    )


def simulationOutcomeSummary(
    terminalValues: pd.Series,
    initialValue: float,
    targetValue: float | None
) -> pd.Series:
    '''Summarize terminal simulation outcomes.'''
    summary = {
        'Initial Value': initialValue,
        'Expected Terminal Value': float(terminalValues.mean()),
        'Median Terminal Value': float(terminalValues.median()),
        'Minimum Terminal Value': float(terminalValues.min()),
        'Maximum Terminal Value': float(terminalValues.max()),
        'Probability of Loss': float(
            (terminalValues < initialValue).mean()
        )
    }

    if targetValue is not None:
        summary['Target Value'] = targetValue
        summary['Probability of Reaching Target'] = float(
            (terminalValues >= targetValue).mean()
        )

    return pd.Series(summary, name='Simulation Summary')


# --------------------------------------------------------------------------
# Portfolio weight resolution
# --------------------------------------------------------------------------

def resolveWeights(
    settings: dict,
    returns: pd.DataFrame
) -> pd.Series:
    '''Resolve portfolio weights for the selected construction method.'''
    tickers = settings['tickers']

    if settings['weights'] is not None:
        return pd.Series(
            settings['weights'],
            index=tickers,
            name='Weight'
        )

    meanReturns = returns.mean()
    covariance = returns.cov()
    maximumWeight = settings.get('maximumOptimizationWeight', 1.0)

    if settings['weightingMethod'] == 'Minimum Variance':
        result = minimumVariancePortfolio(
            meanReturns=meanReturns,
            covarianceMatrix=covariance,
            minimumWeight=(
                -maximumWeight if settings['allowShortSelling'] else 0.0
            ),
            maximumWeight=maximumWeight,
            tradingDays=settings['annualizationFactor']
        )

    elif settings['weightingMethod'] == 'Maximum Sharpe':
        result = maximumSharpePortfolio(
            meanReturns=meanReturns,
            covarianceMatrix=covariance,
            riskFreeRate=st.session_state.get(
                'resolvedRiskFreeRate',
                0.0
            ),
            minimumWeight=(
                -maximumWeight if settings['allowShortSelling'] else 0.0
            ),
            maximumWeight=maximumWeight,
            tradingDays=settings['annualizationFactor']
        )

    else:
        raise ValueError(
            f'Unsupported weighting method: {settings["weightingMethod"]}'
        )

    return optimizationWeights(result, tickers)


# --------------------------------------------------------------------------
# Tab renderers
# --------------------------------------------------------------------------

def renderOverviewTab(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    weights: pd.Series,
    portfolioReturnSeries: pd.Series,
    benchmarkReturnSeries: pd.Series | None,
    riskFreeRate: float,
    settings: dict
) -> None:
    renderSectionTitle(
        'Portfolio Overview',
        'Key performance indicators for the current portfolio.'
    )

    decimals = settings['decimalPlaces']
    tradingDays = settings['annualizationFactor']

    totalReturn = float(
        (1 + portfolioReturnSeries).prod() - 1
    )

    benchmarkDelta = None
    benchmarkTotalReturn = None

    if benchmarkReturnSeries is not None:
        aligned = pd.concat(
            [
                portfolioReturnSeries.rename('Portfolio'),
                benchmarkReturnSeries.rename('Benchmark')
            ],
            axis=1,
            join='inner'
        ).dropna()

        if not aligned.empty:
            benchmarkTotalReturn = float(
                (1 + aligned['Benchmark']).prod() - 1
            )
            portfolioAlignedReturn = float(
                (1 + aligned['Portfolio']).prod() - 1
            )
            benchmarkDelta = formatPercent(
                portfolioAlignedReturn - benchmarkTotalReturn,
                decimals=decimals
            )

    renderMetricRow([
        {
            'label': 'Total Return',
            'value': formatPercent(totalReturn, decimals=decimals),
            'delta': benchmarkDelta,
            'help': (
                'Difference versus the selected benchmark over the '
                'common observation period.'
                if benchmarkDelta is not None
                else 'Cumulative return over the selected period.'
            )
        },
        {
            'label': 'Annual Return',
            'value': formatPercent(
                annualizedReturn(
                    portfolioReturnSeries,
                    tradingDays=tradingDays
                ),
                decimals=decimals
            )
        },
        {
            'label': 'Annual Volatility',
            'value': formatPercent(
                annualizedVolatility(
                    portfolioReturnSeries,
                    tradingDays=tradingDays
                ),
                decimals=decimals
            )
        },
        {
            'label': 'Sharpe Ratio',
            'value': f'{sharpeRatio(portfolioReturnSeries, riskFreeRate=riskFreeRate, tradingDays=tradingDays):.2f}'
        },
        {
            'label': 'Max Drawdown',
            'value': formatPercent(
                maxDrawdown(portfolioReturnSeries),
                decimals=decimals
            )
        }
    ])

    st.divider()

    st.caption(
        f'Data through {prices.index.max():%B %d, %Y} · '
        f'{len(prices):,} aligned trading-day observations'
    )

    leftColumn, rightColumn = st.columns([2, 1])

    with leftColumn:
        if benchmarkReturnSeries is not None:
            st.plotly_chart(
                charts.cumulativeBenchmarkComparisonChart(
                    portfolioReturnSeries,
                    benchmarkReturnSeries
                ),
                use_container_width=True,
                key='overview_cumulative_benchmark_chart'
            )
        else:
            cumulativeChart = charts.cumulativeReturnsChart(
                cumulativeReturns(
                    portfolioReturnSeries
                ).rename('Portfolio')
            )
            st.plotly_chart(cumulativeChart, use_container_width=True)

    with rightColumn:
        allocationChart = charts.allocationPieChart(
            weights,
            title='Current Portfolio Allocation'
        )
        st.plotly_chart(allocationChart, use_container_width=True)

    with st.expander('Asset-level performance', expanded=False):
        normalizedChart = charts.normalizedPricesChart(
            normalizePrices(prices)
        )
        st.plotly_chart(normalizedChart, use_container_width=True)

    if settings['showRawData']:
        renderSectionTitle('Asset Summary')
        summary = assetSummary(prices, tradingDays=tradingDays)
        renderDataFrame(summary)


def renderPerformanceRiskTab(
    portfolioReturnSeries: pd.Series,
    benchmarkReturnSeries: pd.Series | None,
    riskFreeRate: float,
    settings: dict
) -> None:
    renderSectionTitle(
        'Performance and Risk',
        'Historical performance, drawdowns and downside risk.'
    )

    summary = portfolioSummary(
        portfolioReturnSeries,
        benchmarkReturns=benchmarkReturnSeries,
        riskFreeRate=riskFreeRate,
        confidenceLevel=settings['confidenceLevel'],
        tradingDays=settings['annualizationFactor']
    )

    renderDataFrame(summary.to_frame(name='Value'))

    drawdown = drawdownSeries(portfolioReturnSeries)
    st.plotly_chart(
        charts.drawdownChart(drawdown.rename('Portfolio')),
        use_container_width=True
    )

    histogramColumn, varColumn = st.columns(2)

    with histogramColumn:
        st.plotly_chart(
            charts.histogramChart(portfolioReturnSeries),
            use_container_width=True
        )

    with varColumn:
        st.plotly_chart(
            charts.valueAtRiskComparisonChart(
                portfolioReturnSeries,
                historicalVar=historicalVaR(
                    portfolioReturnSeries,
                    confidenceLevel=settings['confidenceLevel']
                ),
                parametricVar=parametricVaR(
                    portfolioReturnSeries,
                    confidenceLevel=settings['confidenceLevel']
                )
            ),
            use_container_width=True
        )

    st.plotly_chart(
        charts.qqPlot(portfolioReturnSeries),
        use_container_width=True
    )

    renderSectionTitle('Rolling Metrics')

    window = settings['rollingWindow']

    if len(portfolioReturnSeries) <= window:
        renderEmptyState(
            'Not enough data',
            'Select a longer date range to view rolling metrics.'
        )
        return

    rollingReturn = rollingAnnualizedReturn(
        portfolioReturnSeries,
        window=window,
        annualizationFactor=settings['annualizationFactor']
    )

    rollingVolatility = rollingAnnualizedVolatility(
        portfolioReturnSeries,
        window=window,
        annualizationFactor=settings['annualizationFactor']
    )

    rollingSharpe = rollingSharpeRatio(
        portfolioReturnSeries,
        riskFreeRate=riskFreeRate,
        window=window,
        annualizationFactor=settings['annualizationFactor']
    )

    rollingReturnColumn, rollingVolatilityColumn = st.columns(2)

    with rollingReturnColumn:
        st.plotly_chart(
            charts.rollingMetricChart(
                rollingReturn.dropna().rename('Rolling Return'),
                title='Rolling Annualized Return'
            ),
            use_container_width=True
        )

    with rollingVolatilityColumn:
        st.plotly_chart(
            charts.rollingVolatilityChart(
                rollingVolatility.dropna().rename('Rolling Volatility')
            ),
            use_container_width=True
        )

    st.plotly_chart(
        charts.rollingSharpeChart(
            rollingSharpe.dropna().rename('Rolling Sharpe')
        ),
        use_container_width=True
    )


def renderBenchmarkTab(
    portfolioReturnSeries: pd.Series,
    benchmarkReturnSeries: pd.Series,
    benchmarkTicker: str,
    riskFreeRate: float,
    settings: dict
) -> None:
    renderSectionTitle(
        'Benchmark Comparison',
        f'Portfolio performance relative to {benchmarkTicker}.'
    )

    summary = computeBenchmarkSummary(
        portfolioReturnSeries,
        benchmarkReturnSeries,
        tradingDays=settings['annualizationFactor']
    )

    portfolioMetrics = portfolioSummary(
        portfolioReturnSeries,
        benchmarkReturns=benchmarkReturnSeries,
        riskFreeRate=riskFreeRate,
        confidenceLevel=settings['confidenceLevel'],
        tradingDays=settings['annualizationFactor']
    )

    metricColumn, benchmarkColumn = st.columns(2)

    with metricColumn:
        st.caption('Beta and alpha vs. benchmark')
        renderDataFrame(
            portfolioMetrics.loc[['Beta', 'Alpha']].to_frame(name='Value')
        )

    with benchmarkColumn:
        st.caption('Active return metrics')
        renderDataFrame(summary.to_frame(name='Value'))

    st.plotly_chart(
        charts.cumulativeBenchmarkComparisonChart(
            portfolioReturnSeries,
            benchmarkReturnSeries
        ),
        use_container_width=True,
        key='benchmark_cumulative_comparison_chart'
    )

    relativePerformance = computeRelativePerformance(
        portfolioReturnSeries,
        benchmarkReturnSeries
    )

    st.plotly_chart(
        charts.relativePerformanceChart(
            relativePerformance['Relative Performance']
        ),
        use_container_width=True
    )

    activeReturnColumn, captureColumn = st.columns(2)

    with activeReturnColumn:
        activeReturnSeries = computeActiveReturns(
            portfolioReturnSeries,
            benchmarkReturnSeries
        )

        st.plotly_chart(
            charts.activeReturnsChart(activeReturnSeries),
            use_container_width=True
        )

    with captureColumn:
        try:
            captureFigure = charts.captureRatiosChart(
                upCaptureRatio(
                    portfolioReturnSeries,
                    benchmarkReturnSeries
                ),
                downCaptureRatio(
                    portfolioReturnSeries,
                    benchmarkReturnSeries
                )
            )
            st.plotly_chart(captureFigure, use_container_width=True)
        except ValueError as error:
            renderEmptyState(
                'Capture ratios unavailable',
                str(error)
            )


def renderCorrelationTab(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    settings: dict
) -> None:
    renderSectionTitle(
        'Correlation and Covariance',
        'Relationships between the selected assets.'
    )

    if settings['showCorrelation']:
        correlation = correlationMatrix(prices)
        st.plotly_chart(
            charts.correlationHeatmap(correlation),
            use_container_width=True
        )

    if settings['showCovariance']:
        covariance = covarianceMatrix(
            prices,
            annualized=True,
            tradingDays=settings['annualizationFactor']
        )
        st.plotly_chart(
            charts.covarianceHeatmap(covariance),
            use_container_width=True
        )

    if not settings['showCorrelation'] and not settings['showCovariance']:
        renderEmptyState(
            'Nothing to display',
            'Enable correlation or covariance matrices in the sidebar '
            'Display settings.'
        )


def renderOptimizationTab(
    returns: pd.DataFrame,
    currentWeights: pd.Series,
    riskFreeRate: float,
    settings: dict
) -> None:
    renderSectionTitle(
        'Portfolio Optimization',
        'Efficient frontier and optimized allocations for the current '
        'asset universe.'
    )

    if not settings['optimizationEnabled']:
        renderEmptyState(
            'Optimization disabled',
            'Enable optimization in the sidebar to view this tab.'
        )
        return

    if len(settings['tickers']) < 2:
        renderEmptyState(
            'Not enough assets',
            'Optimization requires at least two tickers.'
        )
        return

    meanReturns = returns.mean()
    covariance = returns.cov()
    tradingDays = settings['annualizationFactor']
    maximumWeight = settings.get('maximumOptimizationWeight', 1.0)
    minimumWeight = (
        -maximumWeight if settings['allowShortSelling'] else 0.0
    )

    with st.spinner('Computing efficient frontier...'):
        try:
            frontier = efficientFrontier(
                meanReturns=meanReturns,
                covarianceMatrix=covariance,
                points=settings['frontierPoints'],
                minimumWeight=minimumWeight,
                maximumWeight=maximumWeight,
                tradingDays=tradingDays
            )
        except RuntimeError as error:
            renderEmptyState('Optimization failed', str(error))
            return

        randomPortfolioSet = randomPortfolios(
            meanReturns=meanReturns,
            covarianceMatrix=covariance,
            portfolios=settings['randomPortfolioCount'],
            riskFreeRate=riskFreeRate,
            tradingDays=tradingDays,
            randomSeed=settings['randomSeed']
        )

        minimumVariance = minimumVariancePortfolio(
            meanReturns=meanReturns,
            covarianceMatrix=covariance,
            minimumWeight=minimumWeight,
            maximumWeight=maximumWeight,
            tradingDays=tradingDays
        )

        maximumSharpe = maximumSharpePortfolio(
            meanReturns=meanReturns,
            covarianceMatrix=covariance,
            riskFreeRate=riskFreeRate,
            minimumWeight=minimumWeight,
            maximumWeight=maximumWeight,
            tradingDays=tradingDays
        )

    st.caption(
        f'Hard constraint: maximum {maximumWeight:.0%} per asset.'
    )

    st.plotly_chart(
        charts.efficientFrontierChart(
            frontier,
            randomPortfolios=randomPortfolioSet,
            minimumVariance=minimumVariance,
            maximumSharpe=maximumSharpe,
            riskFreeRate=riskFreeRate
        ),
        use_container_width=True
    )

    optimizationResult = None

    if settings['optimizationMethod'] == 'Maximum Sharpe':
        optimizationResult = maximumSharpe
    elif settings['optimizationMethod'] == 'Minimum Variance':
        optimizationResult = minimumVariance
    elif settings['optimizationMethod'] == 'Target Return':
        if settings['targetReturn'] is not None:
            try:
                optimizationResult = targetReturnPortfolio(
                    meanReturns=meanReturns,
                    covarianceMatrix=covariance,
                    targetReturn=settings['targetReturn'],
                    minimumWeight=minimumWeight,
                    maximumWeight=maximumWeight,
                    tradingDays=tradingDays
                )
            except RuntimeError as error:
                renderEmptyState('Optimization failed', str(error))
    elif settings['optimizationMethod'] == 'Target Volatility':
        if settings['targetVolatility'] is not None:
            try:
                optimizationResult = targetVolatilityPortfolio(
                    meanReturns=meanReturns,
                    covarianceMatrix=covariance,
                    targetVolatility=settings['targetVolatility'],
                    minimumWeight=minimumWeight,
                    maximumWeight=maximumWeight,
                    tradingDays=tradingDays
                )
            except RuntimeError as error:
                renderEmptyState('Optimization failed', str(error))

    if optimizationResult is None:
        return

    st.divider()
    renderSectionTitle(
        f'Optimized Portfolio — {settings["optimizationMethod"]}'
    )

    optimizedWeights = optimizationWeights(
        optimizationResult,
        settings['tickers']
    )

    summaryColumn, allocationColumn = st.columns([2, 1])

    with summaryColumn:
        renderMetricRow([
            {
                'label': 'Expected Return',
                'value': f'{optimizationResult["expectedReturn"]:.2%}'
            },
            {
                'label': 'Volatility',
                'value': f'{optimizationResult["volatility"]:.2%}'
            },
            {
                'label': 'Sharpe Ratio',
                'value': f'{optimizationResult["sharpeRatio"]:.3f}'
            }
        ])

        allocationComparison = pd.DataFrame({
            'Asset': settings['tickers'],
            'Current Weight': currentWeights.reindex(
                settings['tickers']
            ).values,
            'Optimized Weight': optimizedWeights.reindex(
                settings['tickers']
            ).values
        })
        allocationComparison['Change'] = (
            allocationComparison['Optimized Weight']
            - allocationComparison['Current Weight']
        )
        percentageColumns = [
            'Current Weight',
            'Optimized Weight',
            'Change'
        ]
        allocationComparison[percentageColumns] = (
            allocationComparison[percentageColumns] * 100
        )

        st.dataframe(
            allocationComparison,
            hide_index=True,
            use_container_width=True,
            column_config={
                'Asset': st.column_config.TextColumn('Asset'),
                'Current Weight': st.column_config.NumberColumn(
                    'Current',
                    format='%.2f%%'
                ),
                'Optimized Weight': st.column_config.NumberColumn(
                    'Optimized',
                    format='%.2f%%'
                ),
                'Change': st.column_config.NumberColumn(
                    'Change',
                    format='%+.2f%%'
                )
            }
        )

    with allocationColumn:
        st.plotly_chart(
            charts.allocationPieChart(
                optimizedWeights,
                title='Optimized Allocation'
            ),
            use_container_width=True
        )

    comparison = pd.DataFrame({
        'Current': currentWeights,
        'Optimized': optimizedWeights
    })

    st.plotly_chart(
        charts.allocationComparisonChart(comparison),
        use_container_width=True
    )


def renderSimulationTab(
    portfolioReturnSeries: pd.Series,
    settings: dict
) -> None:
    renderSectionTitle(
        'Portfolio Simulation',
        'Forward-looking simulation of portfolio value based on '
        'historical return behavior.'
    )

    if not settings['simulationEnabled']:
        renderEmptyState(
            'Simulation disabled',
            'Enable simulation in the sidebar to view this tab.'
        )
        return

    with st.spinner('Running simulation...'):
        simulatedPaths = runSimulation(
            portfolioReturnSeries,
            method=settings['simulationMethod'],
            simulations=settings['simulationCount'],
            horizonDays=settings['horizonDays'],
            initialValue=settings['initialValue'],
            randomSeed=settings['randomSeed']
        )

    terminalValues = simulatedPaths.iloc[-1]
    terminalValues.name = 'Terminal Value'

    summary = simulationOutcomeSummary(
        terminalValues,
        initialValue=settings['initialValue'],
        targetValue=settings['targetValue']
    )

    metrics = [
        {
            'label': 'Expected Terminal Value',
            'value': formatCurrency(summary['Expected Terminal Value'])
        },
        {
            'label': 'Median Terminal Value',
            'value': formatCurrency(summary['Median Terminal Value'])
        },
        {
            'label': 'Probability of Loss',
            'value': formatPercent(summary['Probability of Loss'])
        }
    ]

    if settings['targetValue'] is not None:
        metrics.append({
            'label': 'Probability of Reaching Target',
            'value': formatPercent(
                summary['Probability of Reaching Target']
            )
        })

    renderMetricRow(metrics)

    st.plotly_chart(
        charts.monteCarloPathsChart(
            simulatedPaths,
            maximumPaths=settings['maximumDisplayedPaths']
        ),
        use_container_width=True
    )

    percentilePaths = simulationPercentiles(simulatedPaths)

    st.plotly_chart(
        charts.percentileFanChart(percentilePaths),
        use_container_width=True
    )

    distributionColumn, probabilityColumn = st.columns(2)

    with distributionColumn:
        st.plotly_chart(
            charts.terminalDistributionChart(
                terminalValues,
                initialValue=settings['initialValue'],
                targetValue=settings['targetValue'],
                confidenceLevel=settings['confidenceLevel']
            ),
            use_container_width=True
        )

    with probabilityColumn:
        st.plotly_chart(
            charts.lossProbabilityChart(
                terminalValues,
                initialValue=settings['initialValue']
            ),
            use_container_width=True
        )

    if settings['targetValue'] is not None:
        st.plotly_chart(
            charts.targetProbabilityChart(
                terminalValues,
                targetValue=settings['targetValue']
            ),
            use_container_width=True
        )

    st.plotly_chart(
        charts.simulationSummaryChart(summary),
        use_container_width=True
    )


def renderFundamentalsTab(
    settings: dict,
    prices: pd.DataFrame,
    weights: pd.Series
) -> None:
    renderSectionTitle(
        'Fundamentals',
        'Company-level fundamental data from Yahoo Finance.'
    )

    if not settings['fundamentalsEnabled']:
        renderEmptyState(
            'Fundamentals disabled',
            'Enable fundamental analysis in the sidebar to view this tab.'
        )
        return

    tickers = settings['tickers'][:settings['fundamentalTickerLimit']]

    with st.spinner('Fetching fundamentals...'):
        try:
            fundamentals = multipleFundamentalsSummary(tickers)
        except Exception as error:
            renderEmptyState('Unable to load fundamentals', str(error))

            fundamentals = pd.DataFrame()

    priceRows = {
        ('Market', 'Current Price'): prices.iloc[-1],
        ('Market', 'Previous Close'): prices.iloc[-2],
        ('Market', '52 Week Low'): prices.tail(252).min(),
        ('Market', '52 Week High'): prices.tail(252).max(),
        ('Market', '50 Day Average'): prices.tail(50).mean(),
        ('Market', '200 Day Average'): prices.tail(200).mean()
    }

    priceFallback = pd.DataFrame(priceRows).T
    priceFallback.index = pd.MultiIndex.from_tuples(
        priceFallback.index,
        names=['Section', 'Metric']
    )
    priceFallback = priceFallback.reindex(columns=tickers)

    if fundamentals.empty:
        fundamentals = priceFallback
    else:
        fundamentals = fundamentals.combine_first(priceFallback)

    if fundamentals.empty:
        renderEmptyState(
            'No data available',
            'No fundamental data could be retrieved for these tickers.'
        )
        return

    st.subheader('Expected Portfolio Income')
    investmentAmount = st.number_input(
        'Investment amount',
        min_value=1.0,
        value=float(settings['initialValue']),
        step=1000.0,
        key='dividendInvestmentAmount',
        help='Capital used to estimate annual and monthly dividend income.'
    )

    dividendIndex = ('Dividends', 'Dividend Yield')

    if dividendIndex in fundamentals.index:
        dividendYields = pd.to_numeric(
            fundamentals.loc[dividendIndex].reindex(tickers),
            errors='coerce'
        )
        portfolioWeights = weights.reindex(tickers).fillna(0).astype(float)
        available = dividendYields.notna()
        coveredWeight = float(portfolioWeights[available].sum())
        weightedDividendYield = float(
            (portfolioWeights[available] * dividendYields[available]).sum()
        )
        annualDividendIncome = investmentAmount * weightedDividendYield

        renderMetricRow([
            {
                'label': 'Portfolio Dividend Yield',
                'value': formatPercent(weightedDividendYield, decimals=2)
            },
            {
                'label': 'Expected Annual Income',
                'value': formatCurrency(annualDividendIncome)
            },
            {
                'label': 'Expected Monthly Income',
                'value': formatCurrency(annualDividendIncome / 12)
            },
            {
                'label': 'Weight Coverage',
                'value': formatPercent(coveredWeight, decimals=1),
                'help': 'Portfolio weight for which dividend data is available.'
            }
        ])

        dividendBreakdown = pd.DataFrame({
            'Ticker': tickers,
            'Weight': portfolioWeights.values * 100,
            'Dividend Yield': dividendYields.values * 100
        })
        dividendBreakdown['Annual Income'] = (
            investmentAmount
            * portfolioWeights.values
            * dividendYields.fillna(0).values
        )

        with st.expander('Dividend income breakdown', expanded=False):
            st.dataframe(
                dividendBreakdown,
                hide_index=True,
                width='stretch',
                column_config={
                    'Ticker': st.column_config.TextColumn('Ticker'),
                    'Weight': st.column_config.NumberColumn(
                        'Weight', format='%.2f%%'
                    ),
                    'Dividend Yield': st.column_config.NumberColumn(
                        'Dividend Yield', format='%.2f%%'
                    ),
                    'Annual Income': st.column_config.NumberColumn(
                        'Annual Income', format='$%.2f'
                    )
                }
            )
    else:
        renderEmptyState(
            'Dividend estimate unavailable',
            'Yahoo Finance did not return dividend data for this portfolio.'
        )

    st.divider()

    selectedTicker = st.selectbox(
        'Company or fund',
        options=tickers,
        key='fundamentalSelectedTicker'
    )

    selectedFundamentals = fundamentals[[selectedTicker]]
    formatted = formatFundamentals(selectedFundamentals)

    quoteType = selectedFundamentals.loc[
        ('Company', 'Quote Type'),
        selectedTicker
    ] if ('Company', 'Quote Type') in selectedFundamentals.index else None

    if str(quoteType).upper() in {'ETF', 'MUTUALFUND'}:
        st.info(
            'This instrument is a fund. Company-specific accounting '
            'metrics may not apply and can appear as N/A.'
        )

    headlineMetrics = [
        ('Market', 'Current Price', 'Price'),
        ('Valuation', 'Market Cap', 'Market Cap'),
        ('Valuation', 'Trailing P/E', 'Trailing P/E'),
        ('Growth', 'Earnings Per Share', 'EPS'),
        ('Dividends', 'Dividend Yield', 'Dividend Yield')
    ]

    metricCards = []

    for section, metric, label in headlineMetrics:
        index = (section, metric)
        value = (
            formatted.loc[index, selectedTicker]
            if index in formatted.index
            else 'N/A'
        )
        metricCards.append({'label': label, 'value': str(value)})

    renderMetricRow(metricCards)

    sections = [
        'Company',
        'Market',
        'Valuation',
        'Growth',
        'Profitability',
        'Financial Health',
        'Dividends',
        'Analysts'
    ]
    sectionTabs = st.tabs(sections)

    for section, sectionTab in zip(sections, sectionTabs):
        with sectionTab:
            if section not in formatted.index.get_level_values(0):
                renderEmptyState(
                    'No data available',
                    f'{section} data is unavailable for {selectedTicker}.'
                )
                continue

            sectionData = formatted.xs(
                section,
                level='Section'
            ).rename(columns={selectedTicker: 'Value'})
            renderDataFrame(sectionData, hideIndex=False)

    csvDownloadButton(
        fundamentals,
        fileName='fundamentals.csv',
        label='Download Fundamentals CSV',
        key='fundamentalsDownload'
    )


def renderMethodologyTab(
    settings: dict,
    riskFreeInfo: dict
) -> None:
    renderSectionTitle(
        'Methodology and Model Notes',
        'Definitions, assumptions and limitations behind the analysis.'
    )

    st.markdown(
        f'''
        ### Performance and risk

        - Returns use adjusted daily closing prices and are annualized with
          **{settings['annualizationFactor']} trading days**.
        - Volatility is the annualized standard deviation of daily returns.
        - Sharpe and Sortino ratios use an annual risk-free rate of
          **{riskFreeInfo['rate']:.2%}** from **{riskFreeInfo['source']}**
          ({riskFreeInfo['maturity']}).
        - Maximum drawdown measures the largest peak-to-trough decline.
        - Historical VaR/CVaR use the empirical return distribution at a
          **{settings['confidenceLevel']:.1%} confidence level**.

        ### Optimization

        - The efficient frontier is estimated from historical mean returns
          and the sample covariance matrix.
        - Optimization is long-only unless short selling is explicitly
          enabled. Weights sum to 100%.
        - The maximum-Sharpe portfolio is sensitive to expected-return
          estimates and should be interpreted as a scenario, not a forecast.

        ### Simulation

        - Monte Carlo, Gaussian GBM and historical bootstrap paths are
          available over a **{settings['horizonDays']}-trading-day horizon**.
        - Simulations assume the historical sample is informative about the
          future and do not model taxes, fees, liquidity or market impact.

        ### Data and limitations

        Market and company data come from Yahoo Finance; Treasury yields come
        from FRED. Availability and update times depend on those providers.
        Results are historical, model-dependent and provided for educational
        purposes only—not as investment advice.
        '''
    )


def renderRawDataTab(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    portfolioReturnSeries: pd.Series
) -> None:
    renderSectionTitle('Raw Data')

    priceTab, returnTab, portfolioTab = st.tabs(
        ['Prices', 'Asset Returns', 'Portfolio Returns']
    )

    with priceTab:
        renderDataFrame(prices)
        csvDownloadButton(
            prices,
            fileName='prices.csv',
            label='Download Prices CSV',
            key='pricesDownload'
        )

    with returnTab:
        renderDataFrame(returns)
        csvDownloadButton(
            returns,
            fileName='returns.csv',
            label='Download Returns CSV',
            key='returnsDownload'
        )

    with portfolioTab:
        renderDataFrame(portfolioReturnSeries.to_frame())
        csvDownloadButton(
            portfolioReturnSeries.to_frame(),
            fileName='portfolio_returns.csv',
            label='Download Portfolio Returns CSV',
            key='portfolioReturnsDownload'
        )


# --------------------------------------------------------------------------
# Application entry point
# --------------------------------------------------------------------------

def main() -> None:
    st.title('PortfolioLab')
    st.markdown(
        '<p class="portfolio-subtitle">Portfolio analytics, risk, '
        'optimization and scenario simulation.</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<span class="portfolio-badge">Python</span>'
        '<span class="portfolio-badge">Streamlit</span>'
        '<span class="portfolio-badge">SciPy</span>'
        '<span class="portfolio-badge">Plotly</span>',
        unsafe_allow_html=True
    )

    try:
        settings = renderSidebar()
    except ValueError as error:
        st.sidebar.error(str(error))
        st.stop()

    try:
        with st.spinner('Downloading market data...'):
            prices = getData(
                settings['tickers'],
                settings['startDate'],
                settings['endDate']
            )

            returns = computeReturns(prices)

        riskFreeInfo = getRiskFreeRate(
            mode=settings['riskFreeMode'],
            method=settings['riskFreeMethod'],
            horizonDays=settings['horizonDays'],
            manualRate=settings['manualRiskFreeRate']
        )
        riskFreeRate = riskFreeInfo['rate']
        st.session_state['resolvedRiskFreeRate'] = riskFreeRate

        weights = resolveWeights(settings, returns)

        portfolioReturnSeries = computePortfolioReturns(
            returns,
            weights
        )

        benchmarkReturnSeries = None

        if settings['benchmarkEnabled']:
            try:
                benchmarkPrices = downloadBenchmarkPrices(
                    settings['benchmarkTicker'],
                    settings['startDate'],
                    settings['endDate']
                )
                benchmarkReturnSeries = computeBenchmarkReturns(
                    benchmarkPrices
                )
            except (ValueError, RuntimeError) as error:
                st.sidebar.warning(
                    f'Benchmark unavailable: {error}'
                )

    except (ValueError, RuntimeError) as error:
        st.error(f'Unable to load market data: {error}')
        st.stop()

    st.sidebar.divider()
    st.sidebar.caption(
        f'Risk-free rate: {formatPercent(riskFreeRate)} '
        f'({riskFreeInfo["maturity"]}, {riskFreeInfo["source"]})'
    )

    tabNames = [
        'Overview',
        'Performance & Risk',
        'Benchmark',
        'Correlation',
        'Optimization',
        'Simulation',
        'Fundamentals',
        'Methodology',
        'Raw Data'
    ]

    (
        overviewTab,
        performanceTab,
        benchmarkTab,
        correlationTab,
        optimizationTab,
        simulationTab,
        fundamentalsTab,
        methodologyTab,
        rawDataTab
    ) = st.tabs(tabNames)

    with overviewTab:
        renderOverviewTab(
            prices,
            returns,
            weights,
            portfolioReturnSeries,
            benchmarkReturnSeries,
            riskFreeRate,
            settings
        )

    with performanceTab:
        renderPerformanceRiskTab(
            portfolioReturnSeries,
            benchmarkReturnSeries,
            riskFreeRate,
            settings
        )

    with benchmarkTab:
        if settings['benchmarkEnabled'] and benchmarkReturnSeries is not None:
            renderBenchmarkTab(
                portfolioReturnSeries,
                benchmarkReturnSeries,
                settings['benchmarkTicker'],
                riskFreeRate,
                settings
            )
        else:
            renderEmptyState(
                'Benchmark disabled',
                'Enable benchmark comparison in the sidebar to view this tab.'
            )

    with correlationTab:
        renderCorrelationTab(prices, returns, settings)

    with optimizationTab:
        renderOptimizationTab(
            returns,
            weights,
            riskFreeRate,
            settings
        )

    with simulationTab:
        renderSimulationTab(portfolioReturnSeries, settings)

    with fundamentalsTab:
        renderFundamentalsTab(settings, prices, weights)

    with methodologyTab:
        renderMethodologyTab(settings, riskFreeInfo)

    with rawDataTab:
        renderRawDataTab(prices, returns, portfolioReturnSeries)

    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: #6B7280; padding: 0.5rem 0 1rem;">
            Built by <strong>Filippo Zonta, MSc</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption(f'Build: {BUILD_VERSION}')


if __name__ == '__main__':
    main()
