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


st.set_page_config(
    page_title='Portfolio Analytics Dashboard',
    page_icon='📊',
    layout='wide'
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

    if settings['weightingMethod'] == 'Minimum Variance':
        result = minimumVariancePortfolio(
            meanReturns=meanReturns,
            covarianceMatrix=covariance,
            minimumWeight=(
                -1.0 if settings['allowShortSelling'] else 0.0
            ),
            maximumWeight=1.0,
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
                -1.0 if settings['allowShortSelling'] else 0.0
            ),
            maximumWeight=1.0,
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

    renderMetricRow([
        {
            'label': 'Total Return',
            'value': formatPercent(totalReturn, decimals=decimals)
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
            'value': f'{sharpeRatio(portfolioReturnSeries, tradingDays=tradingDays):.2f}'
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

    leftColumn, rightColumn = st.columns([2, 1])

    with leftColumn:
        normalizedChart = charts.normalizedPricesChart(
            normalizePrices(prices)
        )
        st.plotly_chart(normalizedChart, use_container_width=True)

    with rightColumn:
        allocationChart = charts.allocationPieChart(
            weights,
            title='Current Portfolio Allocation'
        )
        st.plotly_chart(allocationChart, use_container_width=True)

    cumulativeChart = charts.cumulativeReturnsChart(
        cumulativeReturns(portfolioReturnSeries).rename('Portfolio')
    )
    st.plotly_chart(cumulativeChart, use_container_width=True)

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
        use_container_width=True
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

    with st.spinner('Computing efficient frontier...'):
        try:
            frontier = efficientFrontier(
                meanReturns=meanReturns,
                covarianceMatrix=covariance,
                points=settings['frontierPoints'],
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
            tradingDays=tradingDays
        )

        maximumSharpe = maximumSharpePortfolio(
            meanReturns=meanReturns,
            covarianceMatrix=covariance,
            riskFreeRate=riskFreeRate,
            tradingDays=tradingDays
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
        summarySeries = optimizationSummary(
            optimizationResult,
            settings['tickers']
        )

        formattedSummary = summarySeries.astype(object)

        for metric in formattedSummary.index:
            value = float(summarySeries[metric])

            if metric == 'Sharpe Ratio':
                formattedSummary[metric] = f'{value:.3f}'
            else:
                formattedSummary[metric] = f'{value:.2%}'

        renderDataFrame(
            formattedSummary.to_frame(name='Value')
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
    prices: pd.DataFrame
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

    formatted = formatFundamentals(fundamentals)
    renderDataFrame(formatted, height=600)

    csvDownloadButton(
        fundamentals,
        fileName='fundamentals.csv',
        label='Download Fundamentals CSV',
        key='fundamentalsDownload'
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
    st.title('📊 Portfolio Analytics Dashboard')
    st.caption(
        'Build, analyze, optimize and simulate a multi-asset portfolio.'
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
        rawDataTab
    ) = st.tabs(tabNames)

    with overviewTab:
        renderOverviewTab(
            prices,
            returns,
            weights,
            portfolioReturnSeries,
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
        renderFundamentalsTab(settings, prices)

    with rawDataTab:
        renderRawDataTab(prices, returns, portfolioReturnSeries)


if __name__ == '__main__':
    main()
