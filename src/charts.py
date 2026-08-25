from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DEFAULT_HEIGHT = 500
DEFAULT_WIDTH = None


def applyLayout(
    figure: go.Figure,
    title: str,
    xaxisTitle: str = '',
    yaxisTitle: str = '',
    height: int = DEFAULT_HEIGHT,
    width: int | None = DEFAULT_WIDTH
) -> go.Figure:
    '''Apply a consistent layout to Plotly figures.'''
    figure.update_layout(
        template='plotly_white',
        title=title,
        xaxis_title=xaxisTitle,
        yaxis_title=yaxisTitle,
        hovermode='x unified',
        height=height,
        width=width,
        legend_title='',
        margin=dict(
            l=40,
            r=20,
            t=60,
            b=40
        )
    )

    return figure


def lineChart(
    data: pd.DataFrame | pd.Series,
    title: str = '',
    xaxisTitle: str = '',
    yaxisTitle: str = ''
) -> go.Figure:
    '''Return an interactive line chart.'''
    if isinstance(data, pd.Series):
        data = data.to_frame()

    figure = px.line(
        data,
        x=data.index,
        y=data.columns
    )

    return applyLayout(
        figure,
        title,
        xaxisTitle,
        yaxisTitle
    )


def cumulativeReturnsChart(
    cumulativeReturns: pd.DataFrame | pd.Series
) -> go.Figure:
    '''Plot cumulative returns.'''
    return lineChart(
        cumulativeReturns,
        title='Cumulative Returns',
        xaxisTitle='Date',
        yaxisTitle='Return'
    )


def normalizedPricesChart(
    normalizedPrices: pd.DataFrame
) -> go.Figure:
    '''Plot normalized asset prices.'''
    return lineChart(
        normalizedPrices,
        title='Normalized Prices',
        xaxisTitle='Date',
        yaxisTitle='Normalized Price'
    )


def dailyReturnsChart(
    returns: pd.DataFrame | pd.Series
) -> go.Figure:
    '''Plot daily returns.'''
    return lineChart(
        returns,
        title='Daily Returns',
        xaxisTitle='Date',
        yaxisTitle='Return'
    )


def drawdownChart(
    drawdown: pd.Series | pd.DataFrame
) -> go.Figure:
    '''Plot portfolio drawdown.'''
    if isinstance(drawdown, pd.Series):
        drawdown = drawdown.to_frame()

    figure = go.Figure()

    for column in drawdown.columns:
        figure.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown[column],
                mode='lines',
                fill='tozeroy',
                name=column
            )
        )

    return applyLayout(
        figure,
        title='Drawdown',
        xaxisTitle='Date',
        yaxisTitle='Drawdown'
    )


def rollingMetricChart(
    metric: pd.Series | pd.DataFrame,
    title: str
) -> go.Figure:
    '''Plot a rolling metric.'''
    return lineChart(
        metric,
        title=title,
        xaxisTitle='Date',
        yaxisTitle=title
    )


def benchmarkComparisonChart(
    cumulativePerformance: pd.DataFrame
) -> go.Figure:
    '''Compare cumulative portfolio and benchmark performance.'''
    return lineChart(
        cumulativePerformance,
        title='Portfolio vs Benchmark',
        xaxisTitle='Date',
        yaxisTitle='Cumulative Return'
    )

def histogramChart(
    returns: pd.Series,
    bins: int = 50
) -> go.Figure:
    '''Plot the return distribution histogram.'''
    if returns.empty:
        raise ValueError(
            'returns cannot be empty.'
        )

    figure = go.Figure()

    figure.add_trace(
        go.Histogram(
            x=returns.dropna(),
            nbinsx=bins,
            name='Returns'
        )
    )

    return applyLayout(
        figure,
        title='Return Distribution',
        xaxisTitle='Return',
        yaxisTitle='Frequency'
    )


def densityChart(
    returns: pd.Series
) -> go.Figure:
    '''Plot the return density estimate.'''
    if returns.empty:
        raise ValueError(
            'returns cannot be empty.'
        )

    figure = px.histogram(
        returns.dropna(),
        marginal='violin',
        histnorm='probability density'
    )

    return applyLayout(
        figure,
        title='Return Density',
        xaxisTitle='Return',
        yaxisTitle='Density'
    )


def returnsBoxPlot(
    returns: pd.Series | pd.DataFrame
) -> go.Figure:
    '''Plot return boxplots.'''
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    figure = px.box(
        returns,
        points='outliers'
    )

    return applyLayout(
        figure,
        title='Return Distribution',
        yaxisTitle='Return'
    )


def valueAtRiskChart(
    returns: pd.Series,
    confidenceLevel: float = 0.95
) -> go.Figure:
    '''Plot the historical Value at Risk.'''
    if returns.empty:
        raise ValueError(
            'returns cannot be empty.'
        )

    var = returns.quantile(
        1 - confidenceLevel
    )

    figure = go.Figure()

    figure.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns'
        )
    )

    figure.add_vline(
        x=var,
        line_dash='dash',
        annotation_text=f'VaR {confidenceLevel:.0%}'
    )

    return applyLayout(
        figure,
        title='Historical Value at Risk',
        xaxisTitle='Return',
        yaxisTitle='Frequency'
    )


def valueAtRiskComparisonChart(
    returns: pd.Series,
    historicalVar: float,
    parametricVar: float
) -> go.Figure:
    '''Compare historical and parametric VaR.'''
    figure = go.Figure()

    figure.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns'
        )
    )

    figure.add_vline(
        x=historicalVar,
        line_dash='dash',
        annotation_text='Historical VaR'
    )

    figure.add_vline(
        x=parametricVar,
        line_dash='dot',
        annotation_text='Parametric VaR'
    )

    return applyLayout(
        figure,
        title='VaR Comparison',
        xaxisTitle='Return',
        yaxisTitle='Frequency'
    )


def rollingVolatilityChart(
    rollingVolatility: pd.Series
) -> go.Figure:
    '''Plot rolling volatility.'''
    return lineChart(
        rollingVolatility,
        title='Rolling Volatility',
        xaxisTitle='Date',
        yaxisTitle='Volatility'
    )


def rollingSharpeChart(
    rollingSharpe: pd.Series
) -> go.Figure:
    '''Plot rolling Sharpe ratio.'''
    return lineChart(
        rollingSharpe,
        title='Rolling Sharpe Ratio',
        xaxisTitle='Date',
        yaxisTitle='Sharpe Ratio'
    )


def rollingDrawdownChart(
    drawdown: pd.Series
) -> go.Figure:
    '''Plot rolling drawdown.'''
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            fill='tozeroy',
            mode='lines',
            name='Drawdown'
        )
    )

    return applyLayout(
        figure,
        title='Rolling Drawdown',
        xaxisTitle='Date',
        yaxisTitle='Drawdown'
    )


def correlationHeatmap(
    correlationMatrix: pd.DataFrame
) -> go.Figure:
    '''Plot the correlation matrix heatmap.'''
    if correlationMatrix.empty:
        raise ValueError(
            'correlationMatrix cannot be empty.'
        )

    figure = px.imshow(
        correlationMatrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1
    )

    return applyLayout(
        figure,
        title='Correlation Matrix'
    )


def covarianceHeatmap(
    covarianceMatrix: pd.DataFrame
) -> go.Figure:
    '''Plot the covariance matrix heatmap.'''
    if covarianceMatrix.empty:
        raise ValueError(
            'covarianceMatrix cannot be empty.'
        )

    figure = px.imshow(
        covarianceMatrix,
        text_auto='.4f',
        aspect='auto'
    )

    return applyLayout(
        figure,
        title='Covariance Matrix'
    )


def qqPlot(
    returns: pd.Series
) -> go.Figure:
    '''Create a normal Q-Q plot.'''
    if returns.empty:
        raise ValueError(
            'returns cannot be empty.'
        )

    from scipy import stats

    theoretical, observed = stats.probplot(
        returns.dropna(),
        dist='norm'
    )[0]

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=theoretical,
            y=observed,
            mode='markers',
            name='Observations'
        )
    )

    minimum = min(
        theoretical.min(),
        observed.min()
    )

    maximum = max(
        theoretical.max(),
        observed.max()
    )

    figure.add_trace(
        go.Scatter(
            x=[minimum, maximum],
            y=[minimum, maximum],
            mode='lines',
            name='Normal'
        )
    )

    return applyLayout(
        figure,
        title='Normal Q-Q Plot',
        xaxisTitle='Theoretical Quantiles',
        yaxisTitle='Sample Quantiles'
    )

def allocationPieChart(
    weights: pd.Series,
    title: str = 'Portfolio Allocation'
) -> go.Figure:
    '''Plot portfolio allocation as a pie chart.'''
    if weights.empty:
        raise ValueError('weights cannot be empty.')

    weights = weights.dropna().astype(float)

    if np.isclose(weights.abs().sum(), 0):
        raise ValueError('weights cannot sum to zero.')

    figure = go.Figure(
        data=[
            go.Pie(
                labels=weights.index.astype(str),
                values=weights.values,
                textinfo='label+percent',
                hovertemplate=(
                    '<b>%{label}</b><br>'
                    'Weight: %{value:.2%}<br>'
                    '<extra></extra>'
                )
            )
        ]
    )

    figure.update_layout(
        template='plotly_white',
        title=title,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),
        legend_title=''
    )

    return figure


def allocationDonutChart(
    weights: pd.Series,
    title: str = 'Portfolio Allocation'
) -> go.Figure:
    '''Plot portfolio allocation as a donut chart.'''
    if weights.empty:
        raise ValueError('weights cannot be empty.')

    weights = weights.dropna().astype(float)

    if np.isclose(weights.abs().sum(), 0):
        raise ValueError('weights cannot sum to zero.')

    figure = go.Figure(
        data=[
            go.Pie(
                labels=weights.index.astype(str),
                values=weights.values,
                hole=0.55,
                textinfo='label+percent',
                hovertemplate=(
                    '<b>%{label}</b><br>'
                    'Weight: %{value:.2%}<br>'
                    '<extra></extra>'
                )
            )
        ]
    )

    figure.update_layout(
        template='plotly_white',
        title=title,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),
        legend_title=''
    )

    return figure


def allocationBarChart(
    weights: pd.Series,
    title: str = 'Portfolio Weights',
    horizontal: bool = True
) -> go.Figure:
    '''Plot portfolio weights as a bar chart.'''
    if weights.empty:
        raise ValueError('weights cannot be empty.')

    weights = weights.dropna().astype(float).sort_values()

    if horizontal:
        figure = go.Figure(
            data=[
                go.Bar(
                    x=weights.values,
                    y=weights.index.astype(str),
                    orientation='h',
                    text=[
                        f'{weight:.2%}'
                        for weight in weights.values
                    ],
                    textposition='outside',
                    hovertemplate=(
                        '<b>%{y}</b><br>'
                        'Weight: %{x:.2%}<br>'
                        '<extra></extra>'
                    )
                )
            ]
        )

        return applyLayout(
            figure,
            title=title,
            xaxisTitle='Weight',
            yaxisTitle='Asset'
        )

    figure = go.Figure(
        data=[
            go.Bar(
                x=weights.index.astype(str),
                y=weights.values,
                text=[
                    f'{weight:.2%}'
                    for weight in weights.values
                ],
                textposition='outside',
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    'Weight: %{y:.2%}<br>'
                    '<extra></extra>'
                )
            )
        ]
    )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Asset',
        yaxisTitle='Weight'
    )


def allocationComparisonChart(
    allocations: pd.DataFrame,
    title: str = 'Portfolio Allocation Comparison'
) -> go.Figure:
    '''Compare weights across multiple portfolios.'''
    if allocations.empty:
        raise ValueError('allocations cannot be empty.')

    allocations = allocations.dropna(how='all').astype(float)

    figure = go.Figure()

    for column in allocations.columns:
        figure.add_trace(
            go.Bar(
                x=allocations.index.astype(str),
                y=allocations[column],
                name=str(column),
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    f'{column}: '
                    '%{y:.2%}<br>'
                    '<extra></extra>'
                )
            )
        )

    figure.update_layout(
        barmode='group'
    )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Asset',
        yaxisTitle='Weight'
    )


def stackedAllocationChart(
    allocations: pd.DataFrame,
    title: str = 'Allocation Breakdown'
) -> go.Figure:
    '''Plot portfolio allocations as stacked bars.'''
    if allocations.empty:
        raise ValueError('allocations cannot be empty.')

    allocations = allocations.dropna(how='all').astype(float)

    figure = go.Figure()

    for column in allocations.columns:
        figure.add_trace(
            go.Bar(
                x=allocations.index.astype(str),
                y=allocations[column],
                name=str(column),
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    f'{column}: '
                    '%{y:.2%}<br>'
                    '<extra></extra>'
                )
            )
        )

    figure.update_layout(
        barmode='stack'
    )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Portfolio',
        yaxisTitle='Weight'
    )


def allocationTreemap(
    weights: pd.Series,
    title: str = 'Portfolio Allocation'
) -> go.Figure:
    '''Plot portfolio allocation as a treemap.'''
    if weights.empty:
        raise ValueError('weights cannot be empty.')

    weights = weights.dropna().astype(float)

    positiveWeights = weights[
        weights > 0
    ]

    if positiveWeights.empty:
        raise ValueError(
            'Treemap requires at least one positive weight.'
        )

    data = pd.DataFrame({
        'Asset': positiveWeights.index.astype(str),
        'Weight': positiveWeights.values
    })

    figure = px.treemap(
        data,
        path=['Asset'],
        values='Weight'
    )

    figure.update_traces(
        texttemplate=(
            '<b>%{label}</b><br>'
            '%{value:.2%}'
        ),
        hovertemplate=(
            '<b>%{label}</b><br>'
            'Weight: %{value:.2%}<br>'
            '<extra></extra>'
        )
    )

    figure.update_layout(
        template='plotly_white',
        title=title,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    return figure

def sectorAllocationChart(
    assetWeights: pd.Series,
    sectors: pd.Series | dict[str, str],
    title: str = 'Sector Allocation'
) -> go.Figure:
    '''Aggregate and plot portfolio weights by sector.'''
    if assetWeights.empty:
        raise ValueError('assetWeights cannot be empty.')

    sectorSeries = pd.Series(
        sectors,
        dtype=object
    )

    aligned = pd.concat(
        [
            assetWeights.rename('Weight'),
            sectorSeries.rename('Sector')
        ],
        axis=1
    ).dropna(subset=['Weight'])

    if aligned.empty:
        raise ValueError(
            'No asset weights could be aligned with sectors.'
        )

    aligned['Sector'] = aligned[
        'Sector'
    ].fillna('Unknown')

    sectorWeights = (
        aligned.groupby('Sector')['Weight']
        .sum()
        .sort_values(ascending=False)
    )

    return allocationDonutChart(
        sectorWeights,
        title=title
    )


def sectorTreemap(
    assetWeights: pd.Series,
    sectors: pd.Series | dict[str, str],
    title: str = 'Sector and Asset Allocation'
) -> go.Figure:
    '''Plot portfolio weights by sector and asset.'''
    if assetWeights.empty:
        raise ValueError('assetWeights cannot be empty.')

    sectorSeries = pd.Series(
        sectors,
        dtype=object
    )

    aligned = pd.concat(
        [
            assetWeights.rename('Weight'),
            sectorSeries.rename('Sector')
        ],
        axis=1
    ).dropna(subset=['Weight'])

    aligned = aligned[
        aligned['Weight'] > 0
    ]

    if aligned.empty:
        raise ValueError(
            'Treemap requires positive aligned weights.'
        )

    aligned['Sector'] = aligned[
        'Sector'
    ].fillna('Unknown')

    aligned['Asset'] = aligned.index.astype(str)

    figure = px.treemap(
        aligned.reset_index(drop=True),
        path=['Sector', 'Asset'],
        values='Weight'
    )

    figure.update_traces(
        texttemplate=(
            '<b>%{label}</b><br>'
            '%{value:.2%}'
        ),
        hovertemplate=(
            '<b>%{label}</b><br>'
            'Weight: %{value:.2%}<br>'
            '<extra></extra>'
        )
    )

    figure.update_layout(
        template='plotly_white',
        title=title,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    return figure


def contributionBarChart(
    contributions: pd.Series,
    title: str = 'Asset Contributions',
    yaxisTitle: str = 'Contribution'
) -> go.Figure:
    '''Plot asset-level portfolio contributions.'''
    if contributions.empty:
        raise ValueError('contributions cannot be empty.')

    contributions = (
        contributions.dropna()
        .astype(float)
        .sort_values()
    )

    figure = go.Figure(
        data=[
            go.Bar(
                x=contributions.values,
                y=contributions.index.astype(str),
                orientation='h',
                text=[
                    f'{value:.2%}'
                    for value in contributions.values
                ],
                textposition='outside',
                hovertemplate=(
                    '<b>%{y}</b><br>'
                    'Contribution: %{x:.2%}<br>'
                    '<extra></extra>'
                )
            )
        ]
    )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle=yaxisTitle,
        yaxisTitle='Asset'
    )


def riskContributionChart(
    riskContributions: pd.Series
) -> go.Figure:
    '''Plot asset contributions to portfolio risk.'''
    return contributionBarChart(
        riskContributions,
        title='Risk Contributions',
        yaxisTitle='Risk Contribution'
    )


def returnContributionChart(
    returnContributions: pd.Series
) -> go.Figure:
    '''Plot asset contributions to portfolio return.'''
    return contributionBarChart(
        returnContributions,
        title='Return Contributions',
        yaxisTitle='Return Contribution'
    )

def efficientFrontierChart(
    frontier: pd.DataFrame,
    randomPortfolios: pd.DataFrame | None = None,
    minimumVariance: dict | None = None,
    maximumSharpe: dict | None = None,
    riskFreeRate: float | None = None,
    title: str = 'Efficient Frontier'
) -> go.Figure:
    '''Plot the efficient frontier and optional portfolio sets.'''
    requiredColumns = {
        'Expected Return',
        'Volatility'
    }

    if frontier.empty:
        raise ValueError('frontier cannot be empty.')

    if not requiredColumns.issubset(frontier.columns):
        raise ValueError(
            'frontier must contain Expected Return and Volatility.'
        )

    figure = go.Figure()

    if randomPortfolios is not None:
        if not randomPortfolios.empty:
            if not requiredColumns.issubset(
                randomPortfolios.columns
            ):
                raise ValueError(
                    'randomPortfolios must contain Expected Return '
                    'and Volatility.'
                )

            marker = {}

            if 'Sharpe Ratio' in randomPortfolios.columns:
                marker = {
                    'color': randomPortfolios['Sharpe Ratio'],
                    'colorscale': 'Viridis',
                    'showscale': True,
                    'colorbar': {
                        'title': 'Sharpe'
                    },
                    'size': 6,
                    'opacity': 0.6
                }

            figure.add_trace(
                go.Scatter(
                    x=randomPortfolios['Volatility'],
                    y=randomPortfolios['Expected Return'],
                    mode='markers',
                    name='Random Portfolios',
                    marker=marker,
                    customdata=(
                        randomPortfolios[['Sharpe Ratio']].values
                        if 'Sharpe Ratio'
                        in randomPortfolios.columns
                        else None
                    ),
                    hovertemplate=(
                        'Volatility: %{x:.2%}<br>'
                        'Return: %{y:.2%}<br>'
                        + (
                            'Sharpe: %{customdata[0]:.2f}<br>'
                            if 'Sharpe Ratio'
                            in randomPortfolios.columns
                            else ''
                        )
                        + '<extra></extra>'
                    )
                )
            )

    frontier = frontier.sort_values(
        'Volatility'
    )

    figure.add_trace(
        go.Scatter(
            x=frontier['Volatility'],
            y=frontier['Expected Return'],
            mode='lines+markers',
            name='Efficient Frontier',
            line={
                'width': 3
            },
            marker={
                'size': 6
            },
            hovertemplate=(
                'Volatility: %{x:.2%}<br>'
                'Return: %{y:.2%}<br>'
                '<extra></extra>'
            )
        )
    )

    if minimumVariance is not None:
        figure.add_trace(
            optimalPortfolioTrace(
                minimumVariance,
                name='Minimum Variance',
                symbol='diamond'
            )
        )

    if maximumSharpe is not None:
        figure.add_trace(
            optimalPortfolioTrace(
                maximumSharpe,
                name='Maximum Sharpe',
                symbol='star'
            )
        )

    if (
        riskFreeRate is not None
        and maximumSharpe is not None
    ):
        capitalMarketLine = capitalMarketLineData(
            maximumSharpe,
            riskFreeRate=riskFreeRate,
            maximumVolatility=float(
                frontier['Volatility'].max()
            )
        )

        figure.add_trace(
            go.Scatter(
                x=capitalMarketLine['Volatility'],
                y=capitalMarketLine['Expected Return'],
                mode='lines',
                name='Capital Market Line',
                line={
                    'dash': 'dash'
                },
                hovertemplate=(
                    'Volatility: %{x:.2%}<br>'
                    'Return: %{y:.2%}<br>'
                    '<extra></extra>'
                )
            )
        )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Annualized Volatility',
        yaxisTitle='Annualized Expected Return',
        height=600
    )


def optimalPortfolioTrace(
    portfolio: dict,
    name: str,
    symbol: str = 'star'
) -> go.Scatter:
    '''Return a Plotly trace for an optimized portfolio.'''
    requiredKeys = {
        'expectedReturn',
        'volatility'
    }

    if not requiredKeys.issubset(portfolio):
        raise ValueError(
            'portfolio must contain expectedReturn and volatility.'
        )

    sharpeRatio = portfolio.get(
        'sharpeRatio',
        np.nan
    )

    return go.Scatter(
        x=[portfolio['volatility']],
        y=[portfolio['expectedReturn']],
        mode='markers',
        name=name,
        marker={
            'size': 16,
            'symbol': symbol,
            'line': {
                'width': 1
            }
        },
        customdata=[[sharpeRatio]],
        hovertemplate=(
            f'<b>{name}</b><br>'
            'Volatility: %{x:.2%}<br>'
            'Return: %{y:.2%}<br>'
            'Sharpe: %{customdata[0]:.2f}<br>'
            '<extra></extra>'
        )
    )


def randomPortfoliosChart(
    portfolios: pd.DataFrame,
    highlightMaximumSharpe: bool = True,
    highlightMinimumVariance: bool = True,
    title: str = 'Random Portfolios'
) -> go.Figure:
    '''Plot simulated portfolios in risk-return space.'''
    requiredColumns = {
        'Expected Return',
        'Volatility'
    }

    if portfolios.empty:
        raise ValueError('portfolios cannot be empty.')

    if not requiredColumns.issubset(portfolios.columns):
        raise ValueError(
            'portfolios must contain Expected Return and Volatility.'
        )

    figure = go.Figure()

    marker = {
        'size': 7,
        'opacity': 0.7
    }

    if 'Sharpe Ratio' in portfolios.columns:
        marker.update({
            'color': portfolios['Sharpe Ratio'],
            'colorscale': 'Viridis',
            'showscale': True,
            'colorbar': {
                'title': 'Sharpe'
            }
        })

    figure.add_trace(
        go.Scatter(
            x=portfolios['Volatility'],
            y=portfolios['Expected Return'],
            mode='markers',
            name='Portfolios',
            marker=marker,
            customdata=(
                portfolios[['Sharpe Ratio']].values
                if 'Sharpe Ratio' in portfolios.columns
                else None
            ),
            hovertemplate=(
                'Volatility: %{x:.2%}<br>'
                'Return: %{y:.2%}<br>'
                + (
                    'Sharpe: %{customdata[0]:.2f}<br>'
                    if 'Sharpe Ratio' in portfolios.columns
                    else ''
                )
                + '<extra></extra>'
            )
        )
    )

    if (
        highlightMaximumSharpe
        and 'Sharpe Ratio' in portfolios.columns
    ):
        maximumSharpeIndex = portfolios[
            'Sharpe Ratio'
        ].idxmax()

        maximumSharpePortfolio = portfolios.loc[
            maximumSharpeIndex
        ]

        figure.add_trace(
            go.Scatter(
                x=[
                    maximumSharpePortfolio[
                        'Volatility'
                    ]
                ],
                y=[
                    maximumSharpePortfolio[
                        'Expected Return'
                    ]
                ],
                mode='markers',
                name='Maximum Sharpe',
                marker={
                    'size': 16,
                    'symbol': 'star'
                },
                hovertemplate=(
                    '<b>Maximum Sharpe</b><br>'
                    'Volatility: %{x:.2%}<br>'
                    'Return: %{y:.2%}<br>'
                    '<extra></extra>'
                )
            )
        )

    if highlightMinimumVariance:
        minimumVarianceIndex = portfolios[
            'Volatility'
        ].idxmin()

        minimumVariancePortfolio = portfolios.loc[
            minimumVarianceIndex
        ]

        figure.add_trace(
            go.Scatter(
                x=[
                    minimumVariancePortfolio[
                        'Volatility'
                    ]
                ],
                y=[
                    minimumVariancePortfolio[
                        'Expected Return'
                    ]
                ],
                mode='markers',
                name='Minimum Variance',
                marker={
                    'size': 14,
                    'symbol': 'diamond'
                },
                hovertemplate=(
                    '<b>Minimum Variance</b><br>'
                    'Volatility: %{x:.2%}<br>'
                    'Return: %{y:.2%}<br>'
                    '<extra></extra>'
                )
            )
        )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Annualized Volatility',
        yaxisTitle='Annualized Expected Return',
        height=600
    )


def capitalMarketLineData(
    maximumSharpePortfolio: dict,
    riskFreeRate: float,
    maximumVolatility: float,
    points: int = 100
) -> pd.DataFrame:
    '''Return Capital Market Line coordinates.'''
    if maximumVolatility <= 0:
        raise ValueError(
            'maximumVolatility must be positive.'
        )

    if points < 2:
        raise ValueError(
            'points must be at least two.'
        )

    requiredKeys = {
        'expectedReturn',
        'volatility'
    }

    if not requiredKeys.issubset(
        maximumSharpePortfolio
    ):
        raise ValueError(
            'maximumSharpePortfolio must contain '
            'expectedReturn and volatility.'
        )

    portfolioVolatility = float(
        maximumSharpePortfolio['volatility']
    )

    portfolioReturn = float(
        maximumSharpePortfolio['expectedReturn']
    )

    if np.isclose(portfolioVolatility, 0):
        raise ValueError(
            'Maximum-Sharpe volatility cannot be zero.'
        )

    slope = (
        portfolioReturn
        - riskFreeRate
    ) / portfolioVolatility

    volatilities = np.linspace(
        0,
        maximumVolatility,
        points
    )

    expectedReturns = (
        riskFreeRate
        + slope * volatilities
    )

    return pd.DataFrame({
        'Volatility': volatilities,
        'Expected Return': expectedReturns
    })


def capitalMarketLineChart(
    maximumSharpePortfolio: dict,
    riskFreeRate: float,
    maximumVolatility: float | None = None,
    points: int = 100
) -> go.Figure:
    '''Plot the Capital Market Line.'''
    if maximumVolatility is None:
        maximumVolatility = (
            float(
                maximumSharpePortfolio[
                    'volatility'
                ]
            )
            * 1.75
        )

    lineData = capitalMarketLineData(
        maximumSharpePortfolio,
        riskFreeRate=riskFreeRate,
        maximumVolatility=maximumVolatility,
        points=points
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=lineData['Volatility'],
            y=lineData['Expected Return'],
            mode='lines',
            name='Capital Market Line',
            hovertemplate=(
                'Volatility: %{x:.2%}<br>'
                'Return: %{y:.2%}<br>'
                '<extra></extra>'
            )
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[0],
            y=[riskFreeRate],
            mode='markers',
            name='Risk-Free Asset',
            marker={
                'size': 12,
                'symbol': 'circle'
            },
            hovertemplate=(
                '<b>Risk-Free Asset</b><br>'
                'Return: %{y:.2%}<br>'
                '<extra></extra>'
            )
        )
    )

    figure.add_trace(
        optimalPortfolioTrace(
            maximumSharpePortfolio,
            name='Tangency Portfolio',
            symbol='star'
        )
    )

    return applyLayout(
        figure,
        title='Capital Market Line',
        xaxisTitle='Annualized Volatility',
        yaxisTitle='Annualized Expected Return',
        height=600
    )


def frontierWeightsChart(
    frontier: pd.DataFrame,
    assetNames: list[str] | pd.Index,
    title: str = 'Efficient Frontier Weights'
) -> go.Figure:
    '''Plot asset weights across the efficient frontier.'''
    if frontier.empty:
        raise ValueError('frontier cannot be empty.')

    if 'Weights' not in frontier.columns:
        raise ValueError(
            'frontier must contain a Weights column.'
        )

    assetNames = list(assetNames)

    weights = np.vstack(
        frontier['Weights'].to_numpy()
    )

    if weights.shape[1] != len(assetNames):
        raise ValueError(
            'assetNames length must match frontier weights.'
        )

    xValues = (
        frontier['Expected Return']
        if 'Expected Return' in frontier.columns
        else frontier.index
    )

    figure = go.Figure()

    for assetIndex, assetName in enumerate(
        assetNames
    ):
        figure.add_trace(
            go.Scatter(
                x=xValues,
                y=weights[:, assetIndex],
                mode='lines',
                stackgroup='weights',
                name=str(assetName),
                hovertemplate=(
                    f'<b>{assetName}</b><br>'
                    'Target Return: %{x:.2%}<br>'
                    'Weight: %{y:.2%}<br>'
                    '<extra></extra>'
                )
            )
        )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Expected Return',
        yaxisTitle='Portfolio Weight',
        height=550
    )


def optimizationWeightsChart(
    optimizationResults: dict[str, dict],
    assetNames: list[str] | pd.Index,
    title: str = 'Optimized Portfolio Weights'
) -> go.Figure:
    '''Compare allocations across optimized portfolios.'''
    if not optimizationResults:
        raise ValueError(
            'optimizationResults cannot be empty.'
        )

    assetNames = list(assetNames)
    allocationData = {}

    for portfolioName, result in optimizationResults.items():
        if 'weights' not in result:
            raise ValueError(
                f'{portfolioName} does not contain weights.'
            )

        weights = np.asarray(
            result['weights'],
            dtype=float
        )

        if len(weights) != len(assetNames):
            raise ValueError(
                f'{portfolioName} weights do not match assetNames.'
            )

        allocationData[portfolioName] = weights

    allocations = pd.DataFrame(
        allocationData,
        index=assetNames
    )

    return allocationComparisonChart(
        allocations,
        title=title
    )


def riskReturnComparisonChart(
    portfolios: pd.DataFrame,
    title: str = 'Portfolio Risk-Return Comparison'
) -> go.Figure:
    '''Compare named portfolios in risk-return space.'''
    requiredColumns = {
        'Expected Return',
        'Volatility'
    }

    if portfolios.empty:
        raise ValueError(
            'portfolios cannot be empty.'
        )

    if not requiredColumns.issubset(
        portfolios.columns
    ):
        raise ValueError(
            'portfolios must contain Expected Return and Volatility.'
        )

    figure = go.Figure()

    for portfolioName, portfolio in portfolios.iterrows():
        sharpeRatio = portfolio.get(
            'Sharpe Ratio',
            np.nan
        )

        figure.add_trace(
            go.Scatter(
                x=[portfolio['Volatility']],
                y=[portfolio['Expected Return']],
                mode='markers+text',
                name=str(portfolioName),
                text=[str(portfolioName)],
                textposition='top center',
                marker={
                    'size': 14
                },
                customdata=[[sharpeRatio]],
                hovertemplate=(
                    f'<b>{portfolioName}</b><br>'
                    'Volatility: %{x:.2%}<br>'
                    'Return: %{y:.2%}<br>'
                    'Sharpe: %{customdata[0]:.2f}<br>'
                    '<extra></extra>'
                )
            )
        )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Annualized Volatility',
        yaxisTitle='Annualized Expected Return',
        height=550
    )

def monteCarloPathsChart(
    simulatedPaths: pd.DataFrame,
    maximumPaths: int = 200,
    title: str = 'Monte Carlo Portfolio Paths'
) -> go.Figure:
    '''Plot a sample of simulated portfolio paths.'''
    if simulatedPaths.empty:
        raise ValueError('simulatedPaths cannot be empty.')

    if maximumPaths <= 0:
        raise ValueError('maximumPaths must be positive.')

    selectedColumns = simulatedPaths.columns[
        :min(maximumPaths, simulatedPaths.shape[1])
    ]

    figure = go.Figure()

    for column in selectedColumns:
        figure.add_trace(
            go.Scatter(
                x=simulatedPaths.index,
                y=simulatedPaths[column],
                mode='lines',
                name=str(column),
                showlegend=False,
                opacity=0.25,
                hovertemplate=(
                    'Day: %{x}<br>'
                    'Portfolio Value: %{y:,.2f}<br>'
                    '<extra></extra>'
                )
            )
        )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Simulation Day',
        yaxisTitle='Portfolio Value',
        height=600
    )


def percentileFanChart(
    percentilePaths: pd.DataFrame,
    medianColumn: str = '50%',
    title: str = 'Simulation Percentile Fan'
) -> go.Figure:
    '''Plot simulation percentile bands and median path.'''
    if percentilePaths.empty:
        raise ValueError('percentilePaths cannot be empty.')

    if medianColumn not in percentilePaths.columns:
        raise ValueError(
            f'{medianColumn} must exist in percentilePaths.'
        )

    percentageColumns = []

    for column in percentilePaths.columns:
        try:
            percentile = float(
                str(column).replace('%', '')
            ) / 100

            percentageColumns.append(
                (percentile, column)
            )
        except ValueError:
            continue

    if len(percentageColumns) < 3:
        raise ValueError(
            'percentilePaths must contain at least three percentile columns.'
        )

    percentageColumns.sort(
        key=lambda item: item[0]
    )

    figure = go.Figure()

    lowerIndex = 0
    upperIndex = len(percentageColumns) - 1

    while lowerIndex < upperIndex:
        lowerPercentile, lowerColumn = percentageColumns[
            lowerIndex
        ]
        upperPercentile, upperColumn = percentageColumns[
            upperIndex
        ]

        if lowerColumn == medianColumn:
            lowerIndex += 1
            continue

        if upperColumn == medianColumn:
            upperIndex -= 1
            continue

        bandName = (
            f'{lowerPercentile:.0%}–'
            f'{upperPercentile:.0%}'
        )

        figure.add_trace(
            go.Scatter(
                x=percentilePaths.index,
                y=percentilePaths[upperColumn],
                mode='lines',
                line={
                    'width': 0
                },
                name=bandName,
                showlegend=False,
                hoverinfo='skip'
            )
        )

        figure.add_trace(
            go.Scatter(
                x=percentilePaths.index,
                y=percentilePaths[lowerColumn],
                mode='lines',
                line={
                    'width': 0
                },
                fill='tonexty',
                name=bandName,
                hovertemplate=(
                    f'{bandName}<br>'
                    'Day: %{x}<br>'
                    'Value: %{y:,.2f}<br>'
                    '<extra></extra>'
                )
            )
        )

        lowerIndex += 1
        upperIndex -= 1

    figure.add_trace(
        go.Scatter(
            x=percentilePaths.index,
            y=percentilePaths[medianColumn],
            mode='lines',
            name='Median',
            line={
                'width': 3
            },
            hovertemplate=(
                'Day: %{x}<br>'
                'Median Value: %{y:,.2f}<br>'
                '<extra></extra>'
            )
        )
    )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Simulation Day',
        yaxisTitle='Portfolio Value',
        height=600
    )


def terminalDistributionChart(
    terminalValues: pd.Series,
    initialValue: float | None = None,
    targetValue: float | None = None,
    confidenceLevel: float = 0.95,
    bins: int = 60,
    title: str = 'Terminal Portfolio Value Distribution'
) -> go.Figure:
    '''Plot the simulated terminal-value distribution.'''
    if terminalValues.empty:
        raise ValueError('terminalValues cannot be empty.')

    if bins <= 0:
        raise ValueError('bins must be positive.')

    if not 0 < confidenceLevel < 1:
        raise ValueError(
            'confidenceLevel must be between zero and one.'
        )

    values = terminalValues.dropna().astype(float)

    if values.empty:
        raise ValueError(
            'terminalValues contains no valid observations.'
        )

    figure = go.Figure()

    figure.add_trace(
        go.Histogram(
            x=values,
            nbinsx=bins,
            name='Terminal Values',
            hovertemplate=(
                'Value: %{x:,.2f}<br>'
                'Frequency: %{y}<br>'
                '<extra></extra>'
            )
        )
    )

    percentileValue = float(
        values.quantile(
            1 - confidenceLevel
        )
    )

    figure.add_vline(
        x=percentileValue,
        line_dash='dash',
        annotation_text=(
            f'{1 - confidenceLevel:.0%} Percentile'
        )
    )

    figure.add_vline(
        x=float(values.mean()),
        line_dash='dot',
        annotation_text='Mean'
    )

    figure.add_vline(
        x=float(values.median()),
        line_dash='dashdot',
        annotation_text='Median'
    )

    if initialValue is not None:
        figure.add_vline(
            x=initialValue,
            annotation_text='Initial Value'
        )

    if targetValue is not None:
        figure.add_vline(
            x=targetValue,
            line_dash='longdash',
            annotation_text='Target Value'
        )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Terminal Portfolio Value',
        yaxisTitle='Frequency',
        height=550
    )


def terminalReturnDistributionChart(
    terminalValues: pd.Series,
    initialValue: float,
    bins: int = 60,
    title: str = 'Terminal Return Distribution'
) -> go.Figure:
    '''Plot simulated terminal portfolio returns.'''
    if initialValue <= 0:
        raise ValueError('initialValue must be positive.')

    if terminalValues.empty:
        raise ValueError('terminalValues cannot be empty.')

    terminalReturns = (
        terminalValues.dropna().astype(float)
        / initialValue
        - 1
    )

    figure = go.Figure()

    figure.add_trace(
        go.Histogram(
            x=terminalReturns,
            nbinsx=bins,
            name='Terminal Returns',
            hovertemplate=(
                'Return: %{x:.2%}<br>'
                'Frequency: %{y}<br>'
                '<extra></extra>'
            )
        )
    )

    figure.add_vline(
        x=0,
        line_dash='dash',
        annotation_text='Break-Even'
    )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Terminal Return',
        yaxisTitle='Frequency',
        height=550
    )


def lossProbabilityChart(
    terminalValues: pd.Series,
    initialValue: float
) -> go.Figure:
    '''Plot simulated profitable and loss outcomes.'''
    if initialValue <= 0:
        raise ValueError('initialValue must be positive.')

    if terminalValues.empty:
        raise ValueError('terminalValues cannot be empty.')

    values = terminalValues.dropna().astype(float)

    outcomes = pd.Series({
        'Loss': float(
            (values < initialValue).mean()
        ),
        'No Loss': float(
            (values >= initialValue).mean()
        )
    })

    figure = go.Figure(
        data=[
            go.Pie(
                labels=outcomes.index,
                values=outcomes.values,
                hole=0.55,
                textinfo='label+percent',
                hovertemplate=(
                    '<b>%{label}</b><br>'
                    'Probability: %{value:.2%}<br>'
                    '<extra></extra>'
                )
            )
        ]
    )

    figure.update_layout(
        template='plotly_white',
        title='Probability of Loss',
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),
        legend_title=''
    )

    return figure


def targetProbabilityChart(
    terminalValues: pd.Series,
    targetValue: float
) -> go.Figure:
    '''Plot target achievement probability.'''
    if targetValue <= 0:
        raise ValueError('targetValue must be positive.')

    if terminalValues.empty:
        raise ValueError('terminalValues cannot be empty.')

    values = terminalValues.dropna().astype(float)

    outcomes = pd.Series({
        'Target Reached': float(
            (values >= targetValue).mean()
        ),
        'Target Missed': float(
            (values < targetValue).mean()
        )
    })

    figure = go.Figure(
        data=[
            go.Pie(
                labels=outcomes.index,
                values=outcomes.values,
                hole=0.55,
                textinfo='label+percent',
                hovertemplate=(
                    '<b>%{label}</b><br>'
                    'Probability: %{value:.2%}<br>'
                    '<extra></extra>'
                )
            )
        ]
    )

    figure.update_layout(
        template='plotly_white',
        title='Probability of Reaching Target',
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),
        legend_title=''
    )

    return figure


def simulationSummaryChart(
    summary: pd.Series,
    title: str = 'Simulation Summary'
) -> go.Figure:
    '''Plot selected simulation summary values.'''
    if summary.empty:
        raise ValueError('summary cannot be empty.')

    metricOrder = [
        'Initial Value',
        'Expected Terminal Value',
        'Median Terminal Value',
        'Minimum Terminal Value',
        'Maximum Terminal Value',
        'Target Value'
    ]

    availableMetrics = [
        metric
        for metric in metricOrder
        if metric in summary.index
        and pd.notna(summary[metric])
    ]

    if not availableMetrics:
        raise ValueError(
            'summary contains no plottable value metrics.'
        )

    values = summary.loc[
        availableMetrics
    ].astype(float)

    figure = go.Figure(
        data=[
            go.Bar(
                x=availableMetrics,
                y=values,
                text=[
                    f'{value:,.2f}'
                    for value in values
                ],
                textposition='outside',
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    'Value: %{y:,.2f}<br>'
                    '<extra></extra>'
                )
            )
        ]
    )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Metric',
        yaxisTitle='Portfolio Value',
        height=550
    )


def simulationRiskChart(
    summary: pd.Series,
    title: str = 'Simulation Risk Metrics'
) -> go.Figure:
    '''Plot probability and terminal risk metrics.'''
    if summary.empty:
        raise ValueError('summary cannot be empty.')

    probabilityMetrics = [
        metric
        for metric in summary.index
        if metric in {
            'Probability of Loss',
            'Probability of Target'
        }
    ]

    riskMetrics = [
        metric
        for metric in summary.index
        if 'Terminal VaR' in str(metric)
        or 'Terminal CVaR' in str(metric)
    ]

    figure = go.Figure()

    if probabilityMetrics:
        probabilityValues = summary.loc[
            probabilityMetrics
        ].astype(float)

        figure.add_trace(
            go.Bar(
                x=probabilityMetrics,
                y=probabilityValues,
                name='Probability',
                yaxis='y',
                text=[
                    f'{value:.2%}'
                    for value in probabilityValues
                ],
                textposition='outside',
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    'Probability: %{y:.2%}<br>'
                    '<extra></extra>'
                )
            )
        )

    if riskMetrics:
        riskValues = summary.loc[
            riskMetrics
        ].astype(float)

        figure.add_trace(
            go.Bar(
                x=riskMetrics,
                y=riskValues,
                name='Value at Risk',
                yaxis='y2',
                text=[
                    f'{value:,.2f}'
                    for value in riskValues
                ],
                textposition='outside',
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    'Amount: %{y:,.2f}<br>'
                    '<extra></extra>'
                )
            )
        )

    if not probabilityMetrics and not riskMetrics:
        raise ValueError(
            'summary contains no simulation risk metrics.'
        )

    figure.update_layout(
        yaxis={
            'title': 'Probability',
            'tickformat': '.0%'
        },
        yaxis2={
            'title': 'Risk Amount',
            'overlaying': 'y',
            'side': 'right'
        },
        barmode='group'
    )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Metric',
        height=550
    )


def pathEndingScatterChart(
    simulatedPaths: pd.DataFrame,
    title: str = 'Simulation Terminal Outcomes'
) -> go.Figure:
    '''Plot each simulation terminal value by simulation number.'''
    if simulatedPaths.empty:
        raise ValueError('simulatedPaths cannot be empty.')

    values = simulatedPaths.iloc[-1].astype(float)

    figure = go.Figure(
        data=[
            go.Scatter(
                x=np.arange(
                    1,
                    len(values) + 1
                ),
                y=values.values,
                mode='markers',
                name='Terminal Values',
                marker={
                    'size': 6,
                    'opacity': 0.6
                },
                hovertemplate=(
                    'Simulation: %{x}<br>'
                    'Terminal Value: %{y:,.2f}<br>'
                    '<extra></extra>'
                )
            )
        ]
    )

    figure.add_hline(
        y=float(values.mean()),
        line_dash='dash',
        annotation_text='Mean'
    )

    figure.add_hline(
        y=float(values.median()),
        line_dash='dot',
        annotation_text='Median'
    )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Simulation',
        yaxisTitle='Terminal Portfolio Value',
        height=550
    )

def activeReturnsChart(
    activeReturns: pd.Series,
    title: str = 'Active Returns'
) -> go.Figure:
    '''Plot portfolio returns relative to the benchmark.'''
    if activeReturns.empty:
        raise ValueError('activeReturns cannot be empty.')

    activeReturns = activeReturns.dropna().astype(float)

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=activeReturns.index,
            y=activeReturns.values,
            name='Active Return',
            hovertemplate=(
                'Date: %{x}<br>'
                'Active Return: %{y:.2%}<br>'
                '<extra></extra>'
            )
        )
    )

    figure.add_hline(
        y=0,
        line_dash='dash'
    )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Date',
        yaxisTitle='Active Return'
    )


def cumulativeActiveReturnsChart(
    activeReturns: pd.Series,
    title: str = 'Cumulative Active Return'
) -> go.Figure:
    '''Plot cumulative active portfolio performance.'''
    if activeReturns.empty:
        raise ValueError('activeReturns cannot be empty.')

    cumulativeActiveReturns = (
        1 + activeReturns.dropna().astype(float)
    ).cumprod() - 1

    return lineChart(
        cumulativeActiveReturns,
        title=title,
        xaxisTitle='Date',
        yaxisTitle='Cumulative Active Return'
    )


def relativePerformanceChart(
    relativePerformance: pd.Series,
    title: str = 'Relative Performance'
) -> go.Figure:
    '''Plot portfolio wealth relative to benchmark wealth.'''
    if relativePerformance.empty:
        raise ValueError(
            'relativePerformance cannot be empty.'
        )

    relativePerformance = (
        relativePerformance.dropna().astype(float)
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=relativePerformance.index,
            y=relativePerformance.values,
            mode='lines',
            name='Relative Performance',
            hovertemplate=(
                'Date: %{x}<br>'
                'Relative Value: %{y:.4f}<br>'
                '<extra></extra>'
            )
        )
    )

    figure.add_hline(
        y=1,
        line_dash='dash',
        annotation_text='Equal Performance'
    )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Date',
        yaxisTitle='Portfolio / Benchmark'
    )


def portfolioBenchmarkReturnsChart(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series,
    title: str = 'Portfolio and Benchmark Returns'
) -> go.Figure:
    '''Compare periodic portfolio and benchmark returns.'''
    aligned = pd.concat(
        [
            portfolioReturns.rename('Portfolio'),
            benchmarkReturns.rename('Benchmark')
        ],
        axis=1
    ).dropna()

    if aligned.empty:
        raise ValueError(
            'Portfolio and benchmark returns have no overlapping data.'
        )

    return lineChart(
        aligned,
        title=title,
        xaxisTitle='Date',
        yaxisTitle='Return'
    )


def cumulativeBenchmarkComparisonChart(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series,
    title: str = 'Portfolio vs Benchmark'
) -> go.Figure:
    '''Compare cumulative portfolio and benchmark returns.'''
    aligned = pd.concat(
        [
            portfolioReturns.rename('Portfolio'),
            benchmarkReturns.rename('Benchmark')
        ],
        axis=1
    ).dropna()

    if aligned.empty:
        raise ValueError(
            'Portfolio and benchmark returns have no overlapping data.'
        )

    cumulativePerformance = (
        1 + aligned.astype(float)
    ).cumprod() - 1

    return lineChart(
        cumulativePerformance,
        title=title,
        xaxisTitle='Date',
        yaxisTitle='Cumulative Return'
    )


def rollingBetaChart(
    rollingBeta: pd.Series,
    title: str = 'Rolling Beta'
) -> go.Figure:
    '''Plot rolling portfolio beta.'''
    if rollingBeta.empty:
        raise ValueError('rollingBeta cannot be empty.')

    figure = lineChart(
        rollingBeta.dropna(),
        title=title,
        xaxisTitle='Date',
        yaxisTitle='Beta'
    )

    figure.add_hline(
        y=1,
        line_dash='dash',
        annotation_text='Market Beta'
    )

    return figure


def rollingAlphaChart(
    rollingAlpha: pd.Series,
    title: str = 'Rolling Alpha'
) -> go.Figure:
    '''Plot rolling annualized portfolio alpha.'''
    if rollingAlpha.empty:
        raise ValueError('rollingAlpha cannot be empty.')

    figure = lineChart(
        rollingAlpha.dropna(),
        title=title,
        xaxisTitle='Date',
        yaxisTitle='Alpha'
    )

    figure.add_hline(
        y=0,
        line_dash='dash'
    )

    return figure


def rollingCorrelationChart(
    rollingCorrelation: pd.Series,
    title: str = 'Rolling Benchmark Correlation'
) -> go.Figure:
    '''Plot rolling correlation with the benchmark.'''
    if rollingCorrelation.empty:
        raise ValueError(
            'rollingCorrelation cannot be empty.'
        )

    figure = lineChart(
        rollingCorrelation.dropna(),
        title=title,
        xaxisTitle='Date',
        yaxisTitle='Correlation'
    )

    figure.update_yaxes(
        range=[-1, 1]
    )

    figure.add_hline(
        y=0,
        line_dash='dash'
    )

    return figure


def rollingTrackingErrorChart(
    rollingTrackingError: pd.Series,
    title: str = 'Rolling Tracking Error'
) -> go.Figure:
    '''Plot annualized rolling tracking error.'''
    if rollingTrackingError.empty:
        raise ValueError(
            'rollingTrackingError cannot be empty.'
        )

    return lineChart(
        rollingTrackingError.dropna(),
        title=title,
        xaxisTitle='Date',
        yaxisTitle='Tracking Error'
    )


def rollingInformationRatioChart(
    rollingInformationRatio: pd.Series,
    title: str = 'Rolling Information Ratio'
) -> go.Figure:
    '''Plot the rolling information ratio.'''
    if rollingInformationRatio.empty:
        raise ValueError(
            'rollingInformationRatio cannot be empty.'
        )

    figure = lineChart(
        rollingInformationRatio.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna(),
        title=title,
        xaxisTitle='Date',
        yaxisTitle='Information Ratio'
    )

    figure.add_hline(
        y=0,
        line_dash='dash'
    )

    return figure


def rollingActiveReturnChart(
    rollingActiveReturn: pd.Series,
    title: str = 'Rolling Active Return'
) -> go.Figure:
    '''Plot annualized rolling active return.'''
    if rollingActiveReturn.empty:
        raise ValueError(
            'rollingActiveReturn cannot be empty.'
        )

    figure = lineChart(
        rollingActiveReturn.dropna(),
        title=title,
        xaxisTitle='Date',
        yaxisTitle='Active Return'
    )

    figure.add_hline(
        y=0,
        line_dash='dash'
    )

    return figure


def captureRatiosChart(
    upCaptureRatio: float,
    downCaptureRatio: float,
    title: str = 'Benchmark Capture Ratios'
) -> go.Figure:
    '''Plot upside and downside capture ratios.'''
    values = pd.Series({
        'Up Capture': float(upCaptureRatio),
        'Down Capture': float(downCaptureRatio)
    })

    if not np.isfinite(values).all():
        raise ValueError(
            'Capture ratios must be finite.'
        )

    figure = go.Figure(
        data=[
            go.Bar(
                x=values.index,
                y=values.values,
                text=[
                    f'{value:.2%}'
                    for value in values.values
                ],
                textposition='outside',
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    'Capture Ratio: %{y:.2%}<br>'
                    '<extra></extra>'
                )
            )
        ]
    )

    figure.add_hline(
        y=1,
        line_dash='dash',
        annotation_text='100% Capture'
    )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Metric',
        yaxisTitle='Capture Ratio'
    )


def captureRatioComparisonChart(
    captureResults: pd.DataFrame,
    title: str = 'Capture Ratio Comparison'
) -> go.Figure:
    '''Compare capture ratios across portfolios or benchmarks.'''
    requiredColumns = {
        'Up Capture',
        'Down Capture'
    }

    if captureResults.empty:
        raise ValueError(
            'captureResults cannot be empty.'
        )

    if not requiredColumns.issubset(
        captureResults.columns
    ):
        raise ValueError(
            'captureResults must contain Up Capture '
            'and Down Capture.'
        )

    figure = go.Figure()

    for column in [
        'Up Capture',
        'Down Capture'
    ]:
        figure.add_trace(
            go.Bar(
                x=captureResults.index.astype(str),
                y=captureResults[column],
                name=column,
                text=[
                    f'{value:.2%}'
                    for value in captureResults[column]
                ],
                textposition='outside',
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    f'{column}: '
                    '%{y:.2%}<br>'
                    '<extra></extra>'
                )
            )
        )

    figure.update_layout(
        barmode='group'
    )

    figure.add_hline(
        y=1,
        line_dash='dash',
        annotation_text='100% Capture'
    )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Portfolio or Benchmark',
        yaxisTitle='Capture Ratio'
    )


def benchmarkScatterChart(
    portfolioReturns: pd.Series,
    benchmarkReturns: pd.Series,
    betaValue: float | None = None,
    alphaValue: float | None = None,
    title: str = 'Portfolio Sensitivity to Benchmark'
) -> go.Figure:
    '''Plot portfolio returns against benchmark returns.'''
    aligned = pd.concat(
        [
            portfolioReturns.rename('Portfolio'),
            benchmarkReturns.rename('Benchmark')
        ],
        axis=1
    ).dropna()

    if aligned.empty:
        raise ValueError(
            'Portfolio and benchmark returns have no overlapping data.'
        )

    if betaValue is None:
        benchmarkVariance = aligned[
            'Benchmark'
        ].var()

        if np.isclose(benchmarkVariance, 0):
            raise ValueError(
                'Benchmark variance cannot be zero.'
            )

        betaValue = (
            aligned['Portfolio'].cov(
                aligned['Benchmark']
            )
            / benchmarkVariance
        )

    if alphaValue is None:
        alphaValue = (
            aligned['Portfolio'].mean()
            - betaValue
            * aligned['Benchmark'].mean()
        )

    benchmarkRange = np.linspace(
        aligned['Benchmark'].min(),
        aligned['Benchmark'].max(),
        100
    )

    regressionLine = (
        alphaValue
        + betaValue * benchmarkRange
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=aligned['Benchmark'],
            y=aligned['Portfolio'],
            mode='markers',
            name='Observations',
            marker={
                'size': 7,
                'opacity': 0.6
            },
            hovertemplate=(
                'Benchmark: %{x:.2%}<br>'
                'Portfolio: %{y:.2%}<br>'
                '<extra></extra>'
            )
        )
    )

    figure.add_trace(
        go.Scatter(
            x=benchmarkRange,
            y=regressionLine,
            mode='lines',
            name='Regression Line',
            hovertemplate=(
                'Benchmark: %{x:.2%}<br>'
                'Estimated Portfolio: %{y:.2%}<br>'
                '<extra></extra>'
            )
        )
    )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Benchmark Return',
        yaxisTitle='Portfolio Return',
        height=550
    )


def rollingBenchmarkMetricsChart(
    rollingMetrics: pd.DataFrame,
    title: str = 'Rolling Benchmark Metrics'
) -> go.Figure:
    '''Plot multiple rolling benchmark metrics.'''
    if rollingMetrics.empty:
        raise ValueError(
            'rollingMetrics cannot be empty.'
        )

    figure = go.Figure()

    for column in rollingMetrics.columns:
        values = rollingMetrics[
            column
        ].replace(
            [np.inf, -np.inf],
            np.nan
        )

        figure.add_trace(
            go.Scatter(
                x=rollingMetrics.index,
                y=values,
                mode='lines',
                name=str(column),
                hovertemplate=(
                    'Date: %{x}<br>'
                    f'{column}: '
                    '%{y:.4f}<br>'
                    '<extra></extra>'
                )
            )
        )

    figure.add_hline(
        y=0,
        line_dash='dash'
    )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Date',
        yaxisTitle='Metric Value',
        height=550
    )


def benchmarkSummaryChart(
    summary: pd.Series,
    title: str = 'Benchmark Analysis Summary'
) -> go.Figure:
    '''Plot selected benchmark comparison metrics.'''
    if summary.empty:
        raise ValueError('summary cannot be empty.')

    preferredMetrics = [
        'Beta',
        'Alpha',
        'Correlation',
        'Tracking Error',
        'Information Ratio',
        'Annualized Active Return',
        'Up Capture Ratio',
        'Down Capture Ratio',
        'Capture Ratio'
    ]

    availableMetrics = [
        metric
        for metric in preferredMetrics
        if metric in summary.index
        and pd.notna(summary[metric])
        and np.isfinite(summary[metric])
    ]

    if not availableMetrics:
        raise ValueError(
            'summary contains no plottable benchmark metrics.'
        )

    values = summary.loc[
        availableMetrics
    ].astype(float)

    figure = go.Figure(
        data=[
            go.Bar(
                x=availableMetrics,
                y=values,
                text=[
                    f'{value:.3f}'
                    for value in values
                ],
                textposition='outside',
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    'Value: %{y:.4f}<br>'
                    '<extra></extra>'
                )
            )
        ]
    )

    return applyLayout(
        figure,
        title=title,
        xaxisTitle='Metric',
        yaxisTitle='Value',
        height=550
    )