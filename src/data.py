from __future__ import annotations

from io import StringIO
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


TRADING_DAYS = 252
FRED_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}'

TREASURY_SERIES = {
    '1M': {
        'series': 'DGS1MO',
        'days': 30,
        'name': '1-Month Treasury'
    },
    '3M': {
        'series': 'DGS3MO',
        'days': 90,
        'name': '3-Month Treasury'
    },
    '6M': {
        'series': 'DGS6MO',
        'days': 180,
        'name': '6-Month Treasury'
    },
    '1Y': {
        'series': 'DGS1',
        'days': 365,
        'name': '1-Year Treasury'
    },
    '2Y': {
        'series': 'DGS2',
        'days': 730,
        'name': '2-Year Treasury'
    },
    '3Y': {
        'series': 'DGS3',
        'days': 1095,
        'name': '3-Year Treasury'
    },
    '5Y': {
        'series': 'DGS5',
        'days': 1825,
        'name': '5-Year Treasury'
    },
    '7Y': {
        'series': 'DGS7',
        'days': 2555,
        'name': '7-Year Treasury'
    },
    '10Y': {
        'series': 'DGS10',
        'days': 3650,
        'name': '10-Year Treasury'
    },
    '20Y': {
        'series': 'DGS20',
        'days': 7300,
        'name': '20-Year Treasury'
    },
    '30Y': {
        'series': 'DGS30',
        'days': 10950,
        'name': '30-Year Treasury'
    }
}


def _cleanTickers(tickers: Iterable[str]) -> list[str]:
    '''Clean, standardize and remove duplicate ticker symbols.'''
    cleanedTickers = []

    for ticker in tickers:
        ticker = str(ticker).strip().upper()

        if ticker and ticker not in cleanedTickers:
            cleanedTickers.append(ticker)

    if not cleanedTickers:
        raise ValueError('Enter at least one valid ticker symbol.')

    return cleanedTickers


def _validateDateRange(
    startDate,
    endDate
) -> tuple[pd.Timestamp, pd.Timestamp]:
    '''Validate and standardize the selected date range.'''
    startDate = pd.Timestamp(startDate)
    endDate = pd.Timestamp(endDate)

    if pd.isna(startDate) or pd.isna(endDate):
        raise ValueError('Start date and end date must be valid dates.')

    if startDate >= endDate:
        raise ValueError('Start date must be earlier than end date.')

    return startDate, endDate


def _extractClosePrices(
    data: pd.DataFrame,
    tickers: list[str]
) -> pd.DataFrame:
    '''Extract adjusted closing prices from a yfinance response.'''
    if data.empty:
        raise ValueError(
            'No market data returned. Check tickers and dates.'
        )

    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' not in data.columns.get_level_values(0):
            raise ValueError(
                'Yahoo Finance did not return a Close price field.'
            )

        prices = data['Close'].copy()
    else:
        if 'Close' not in data.columns:
            raise ValueError(
                'Yahoo Finance did not return a Close price field.'
            )

        prices = data[['Close']].copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    if len(tickers) == 1 and list(prices.columns) == ['Close']:
        prices.columns = tickers

    prices.columns = [
        str(column).strip().upper()
        for column in prices.columns
    ]

    return prices


def _matchTreasurySeries(horizonDays: int) -> dict:
    '''Return the Treasury maturity closest to the simulation horizon.'''
    if horizonDays <= 0:
        raise ValueError('horizonDays must be positive.')

    maturities = list(TREASURY_SERIES.values())

    return min(
        maturities,
        key=lambda item: abs(item['days'] - horizonDays)
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _downloadTreasuryRate(
    series: str
) -> tuple[float, pd.Timestamp]:
    '''Download the latest available Treasury yield from FRED.'''
    response = requests.get(
        FRED_URL.format(series=series),
        timeout=15
    )

    response.raise_for_status()

    data = pd.read_csv(StringIO(response.text))

    if 'DATE' not in data.columns or series not in data.columns:
        raise ValueError('Unexpected FRED response format.')

    data['DATE'] = pd.to_datetime(
        data['DATE'],
        errors='coerce'
    )

    data[series] = pd.to_numeric(
        data[series],
        errors='coerce'
    )

    data = data.dropna(
        subset=['DATE', series]
    )

    if data.empty:
        raise ValueError(
            f'No valid observations returned for {series}.'
        )

    latest = data.iloc[-1]

    rate = float(latest[series]) / 100
    observationDate = pd.Timestamp(latest['DATE'])

    return rate, observationDate


@st.cache_data(ttl=3600, show_spinner=False)
def getData(
    tickers: Iterable[str],
    startDate,
    endDate
) -> pd.DataFrame:
    '''Download adjusted daily closing prices from Yahoo Finance.'''
    tickers = _cleanTickers(tickers)

    startDate, endDate = _validateDateRange(
        startDate,
        endDate
    )

    try:
        data = yf.download(
            tickers=tickers,
            start=startDate.strftime('%Y-%m-%d'),
            end=endDate.strftime('%Y-%m-%d'),
            auto_adjust=True,
            progress=False,
            group_by='column',
            threads=True
        )
    except Exception as error:
        raise RuntimeError(
            f'Market-data download failed: {error}'
        ) from error

    prices = _extractClosePrices(
        data,
        tickers
    )

    missingTickers = [
        ticker
        for ticker in tickers
        if ticker not in prices.columns
    ]

    if missingTickers:
        missing = ', '.join(missingTickers)

        raise ValueError(
            f'No valid price series returned for: {missing}'
        )

    prices = prices[tickers]
    prices = prices.replace(
        [np.inf, -np.inf],
        np.nan
    )

    prices = prices.dropna(how='all')
    prices = prices.ffill()
    prices = prices.dropna(how='any')

    if prices.empty:
        raise ValueError(
            'No common price history is available across the selected assets.'
        )

    if len(prices) < 2:
        raise ValueError(
            'At least two price observations are required.'
        )

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = 'Date'

    return prices.astype(float)

def getRiskFreeRate(
    mode: Literal['automatic', 'manual'] = 'automatic',
    method: str = 'match',
    horizonDays: int = 252,
    manualRate: float | None = None,
    fallbackRate: float = 0.04
) -> dict:
    '''Return a manual or automatically selected annual risk-free rate.'''
    mode = mode.lower()
    method = method.lower()

    if mode == 'manual':
        if manualRate is None:
            raise ValueError(
                'manualRate is required when mode is manual.'
            )

        if manualRate < 0:
            raise ValueError(
                'manualRate cannot be negative.'
            )

        return {
            'rate': float(manualRate),
            'series': None,
            'maturity': 'Manual Rate',
            'observationDate': None,
            'source': 'User Input',
            'mode': 'Manual',
            'method': 'Manual'
        }

    if mode != 'automatic':
        raise ValueError(
            'mode must be either automatic or manual.'
        )

    if method == 'match':
        treasury = _matchTreasurySeries(horizonDays)
        methodName = 'Match Simulation Horizon'

    else:
        maturityMap = {
            '1m': '1M',
            '3m': '3M',
            '6m': '6M',
            '1y': '1Y',
            '2y': '2Y',
            '3y': '3Y',
            '5y': '5Y',
            '7y': '7Y',
            '10y': '10Y',
            '20y': '20Y',
            '30y': '30Y'
        }

        if method not in maturityMap:
            validMethods = ', '.join(
                ['match', *maturityMap]
            )
            raise ValueError(
                f'Unsupported Treasury selection. Use: {validMethods}.'
            )

        treasury = TREASURY_SERIES[
            maturityMap[method]
        ]
        methodName = treasury['name']

    try:
        rate, observationDate = _downloadTreasuryRate(
            treasury['series']
        )

        source = 'FRED'

    except Exception:
        rate = float(fallbackRate)
        observationDate = pd.Timestamp.today().normalize()
        source = 'Fallback'

    return {
        'rate': rate,
        'series': treasury['series'],
        'maturity': treasury['name'],
        'observationDate': observationDate,
        'source': source,
        'mode': 'Automatic',
        'method': methodName
    }


def computeReturns(
    prices: pd.DataFrame | pd.Series,
    returnType: Literal['simple', 'log'] = 'simple'
) -> pd.DataFrame | pd.Series:
    '''Compute daily simple or logarithmic returns.'''
    if prices.empty:
        raise ValueError(
            'Price data cannot be empty.'
        )

    prices = prices.astype(float)
    prices = prices.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if returnType == 'simple':
        returns = prices.pct_change(
            fill_method=None
        )

    elif returnType == 'log':
        if isinstance(prices, pd.DataFrame):
            hasNonPositive = (prices <= 0).any().any()
        else:
            hasNonPositive = (prices <= 0).any()

        if hasNonPositive:
            raise ValueError(
                'Log returns require strictly positive prices.'
            )

        returns = np.log(
            prices / prices.shift(1)
        )

    else:
        raise ValueError(
            'returnType must be either simple or log.'
        )

    returns = returns.replace(
        [np.inf, -np.inf],
        np.nan
    )

    returns = returns.dropna(
        how='any'
    )

    if returns.empty:
        raise ValueError(
            'Unable to compute returns from the supplied price data.'
        )

    return returns


def normalizePrices(
    prices: pd.DataFrame | pd.Series
) -> pd.DataFrame | pd.Series:
    '''Normalize prices so that every asset starts at one.'''
    if prices.empty:
        raise ValueError(
            'Price data cannot be empty.'
        )

    firstObservation = prices.iloc[0]

    if np.any(
        np.isclose(firstObservation, 0)
    ):
        raise ValueError(
            'Initial prices cannot contain zero values.'
        )

    return prices / firstObservation


def assetSummary(
    prices: pd.DataFrame,
    tradingDays: int = TRADING_DAYS
) -> pd.DataFrame:
    '''Build a historical performance summary for every asset.'''
    if tradingDays <= 0:
        raise ValueError(
            'tradingDays must be positive.'
        )

    returns = computeReturns(prices)
    years = len(returns) / tradingDays

    if years <= 0:
        raise ValueError(
            'Not enough observations to calculate an asset summary.'
        )

    totalReturn = (
        prices.iloc[-1] / prices.iloc[0] - 1
    )

    cagr = (
        prices.iloc[-1] / prices.iloc[0]
    ) ** (1 / years) - 1

    summary = pd.DataFrame({
        'Annual Return':
            returns.mean() * tradingDays,

        'CAGR':
            cagr,

        'Annual Volatility':
            returns.std(ddof=1) * np.sqrt(tradingDays),

        'Total Return':
            totalReturn,

        'Best Day':
            returns.max(),

        'Worst Day':
            returns.min(),

        'Positive Days':
            (returns > 0).mean()
    })

    summary.index.name = 'Ticker'

    return summary


def correlationMatrix(
    prices: pd.DataFrame,
    method: Literal[
        'pearson',
        'spearman',
        'kendall'
    ] = 'pearson'
) -> pd.DataFrame:
    '''Calculate the correlation matrix of daily asset returns.'''
    returns = computeReturns(prices)

    return returns.corr(
        method=method
    )


def covarianceMatrix(
    prices: pd.DataFrame,
    annualized: bool = False,
    tradingDays: int = TRADING_DAYS
) -> pd.DataFrame:
    '''Calculate the daily or annualized covariance matrix.'''
    if tradingDays <= 0:
        raise ValueError(
            'tradingDays must be positive.'
        )

    returns = computeReturns(prices)
    covariance = returns.cov()

    if annualized:
        covariance = covariance * tradingDays

    return covariance


def latestPrices(
    prices: pd.DataFrame
) -> pd.Series:
    '''Return the latest available price for every asset.'''
    if prices.empty:
        raise ValueError(
            'Price data cannot be empty.'
        )

    latest = prices.iloc[-1].copy()
    latest.name = prices.index[-1]

    return latest


def latestReturns(
    prices: pd.DataFrame,
    periods: int = 1
) -> pd.Series:
    '''Return the latest percentage return for every asset.'''
    if periods <= 0:
        raise ValueError(
            'periods must be positive.'
        )

    returns = prices.pct_change(periods=periods)
    returns = returns.dropna(how='all')

    if returns.empty:
        raise ValueError(
            'Unable to calculate returns.'
        )

    latest = returns.iloc[-1].copy()
    latest.name = returns.index[-1]

    return latest


def pricePerformance(
    prices: pd.DataFrame
) -> pd.Series:
    '''Return cumulative performance since the first observation.'''
    if prices.empty:
        raise ValueError(
            'Price data cannot be empty.'
        )

    performance = prices.iloc[-1] / prices.iloc[0] - 1
    performance.name = 'Performance'

    return performance


def tradingYears(
    prices: pd.DataFrame,
    tradingDays: int = TRADING_DAYS
) -> float:
    '''Return the sample length expressed in trading years.'''
    if tradingDays <= 0:
        raise ValueError(
            'tradingDays must be positive.'
        )

    returns = computeReturns(prices)

    return len(returns) / tradingDays


def annualizationFactor(
    tradingDays: int = TRADING_DAYS
) -> float:
    '''Return the square-root annualization factor.'''
    if tradingDays <= 0:
        raise ValueError(
            'tradingDays must be positive.'
        )

    return np.sqrt(tradingDays)
