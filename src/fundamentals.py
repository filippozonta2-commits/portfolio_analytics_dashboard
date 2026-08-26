from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import streamlit as st
import yfinance as yf


YAHOO_QUOTE_URLS = (
    'https://query1.finance.yahoo.com/v7/finance/quote',
    'https://query2.finance.yahoo.com/v7/finance/quote'
)


def validateTicker(ticker: str) -> str:
    '''Validate and standardize a ticker symbol.'''
    ticker = str(ticker).strip().upper()

    if not ticker:
        raise ValueError('ticker cannot be empty.')

    return ticker


def getTicker(ticker: str) -> yf.Ticker:
    '''Return a validated yfinance ticker object.'''
    return yf.Ticker(validateTicker(ticker))


def _fastInfoFallback(tickerObject: yf.Ticker) -> dict[str, Any]:
    '''Return normalized fields from yfinance fast_info when available.'''
    try:
        fastInfo = dict(tickerObject.fast_info)
    except Exception:
        return {}

    fieldMap = {
        'currency': 'currency',
        'day_high': 'dayHigh',
        'day_low': 'dayLow',
        'exchange': 'exchange',
        'fifty_day_average': 'fiftyDayAverage',
        'last_price': 'currentPrice',
        'market_cap': 'marketCap',
        'open': 'open',
        'previous_close': 'previousClose',
        'quote_type': 'quoteType',
        'three_month_average_volume': 'averageVolume',
        'two_hundred_day_average': 'twoHundredDayAverage',
        'year_high': 'fiftyTwoWeekHigh',
        'year_low': 'fiftyTwoWeekLow'
    }

    return {
        destination: fastInfo.get(source)
        for source, destination in fieldMap.items()
        if fastInfo.get(source) is not None
    }


def _quoteApiFallback(ticker: str) -> dict[str, Any]:
    '''Retrieve core quote fields from Yahoo's lightweight quote endpoint.'''
    fieldMap = {
        'averageDailyVolume3Month': 'averageVolume',
        'beta': 'beta',
        'currency': 'currency',
        'dividendYield': 'dividendYield',
        'epsForward': 'forwardEps',
        'epsTrailingTwelveMonths': 'trailingEps',
        'exchange': 'exchange',
        'fiftyDayAverage': 'fiftyDayAverage',
        'fiftyTwoWeekHigh': 'fiftyTwoWeekHigh',
        'fiftyTwoWeekLow': 'fiftyTwoWeekLow',
        'forwardPE': 'forwardPE',
        'fullExchangeName': 'exchange',
        'longName': 'longName',
        'marketCap': 'marketCap',
        'priceToBook': 'priceToBook',
        'quoteType': 'quoteType',
        'regularMarketDayHigh': 'dayHigh',
        'regularMarketDayLow': 'dayLow',
        'regularMarketOpen': 'open',
        'regularMarketPreviousClose': 'previousClose',
        'regularMarketPrice': 'currentPrice',
        'shortName': 'shortName',
        'trailingAnnualDividendYield': 'trailingAnnualDividendYield',
        'trailingPE': 'trailingPE',
        'twoHundredDayAverage': 'twoHundredDayAverage'
    }

    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0 PortfolioAnalyticsDashboard/1.0'
    }

    for url in YAHOO_QUOTE_URLS:
        try:
            response = requests.get(
                url,
                params={'symbols': ticker},
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            results = response.json()['quoteResponse']['result']

            if not results:
                continue

            quote = results[0]

            return {
                destination: quote.get(source)
                for source, destination in fieldMap.items()
                if quote.get(source) is not None
            }
        except (KeyError, TypeError, ValueError, requests.RequestException):
            continue

    return {}


@st.cache_data(ttl=21600, show_spinner=False)
def companyInfo(ticker: str) -> dict[str, Any]:
    '''Return company information.'''
    ticker = validateTicker(ticker)
    tickerObject = getTicker(ticker)
    info: dict[str, Any] = {}

    try:
        downloadedInfo = tickerObject.get_info()
        if downloadedInfo:
            info.update(downloadedInfo)
    except Exception:
        try:
            downloadedInfo = tickerObject.info
            if downloadedInfo:
                info.update(downloadedInfo)
        except Exception:
            pass

    for field, value in _quoteApiFallback(ticker).items():
        if info.get(field) is None:
            info[field] = value

    for field, value in _fastInfoFallback(tickerObject).items():
        if info.get(field) is None:
            info[field] = value

    info.setdefault('symbol', ticker)
    info.setdefault('shortName', ticker)

    if len(info) <= 2:
        raise ValueError(
            f'No company information available for {ticker}.'
        )

    return info


def companySummary(
    ticker: str,
    info: dict[str, Any] | None = None
) -> pd.Series:
    '''Return a concise company summary.'''
    ticker = validateTicker(ticker)
    info = companyInfo(ticker) if info is None else info

    return pd.Series(
        {
            'Company': info.get('longName'),
            'Short Name': info.get('shortName'),
            'Sector': info.get('sector'),
            'Industry': info.get('industry'),
            'Country': info.get('country'),
            'City': info.get('city'),
            'Employees': info.get('fullTimeEmployees'),
            'Website': info.get('website'),
            'Exchange': info.get('exchange'),
            'Quote Type': info.get('quoteType'),
            'Currency': info.get('currency')
        },
        name=ticker
    )


def marketMetrics(
    ticker: str,
    info: dict[str, Any] | None = None
) -> pd.Series:
    '''Return market and trading metrics.'''
    ticker = validateTicker(ticker)
    info = companyInfo(ticker) if info is None else info

    return pd.Series(
        {
            'Current Price': info.get('currentPrice'),
            'Previous Close': info.get('previousClose'),
            'Open': info.get('open'),
            'Day Low': info.get('dayLow'),
            'Day High': info.get('dayHigh'),
            '52 Week Low': info.get('fiftyTwoWeekLow'),
            '52 Week High': info.get('fiftyTwoWeekHigh'),
            '50 Day Average': info.get('fiftyDayAverage'),
            '200 Day Average': info.get('twoHundredDayAverage'),
            'Average Volume': info.get('averageVolume'),
            'Beta': info.get('beta')
        },
        name=ticker
    )


def valuationMetrics(
    ticker: str,
    info: dict[str, Any] | None = None
) -> pd.Series:
    '''Return valuation metrics.'''
    ticker = validateTicker(ticker)
    info = companyInfo(ticker) if info is None else info

    return pd.Series(
        {
            'Market Cap': info.get('marketCap'),
            'Enterprise Value': info.get('enterpriseValue'),
            'Trailing P/E': info.get('trailingPE'),
            'Forward P/E': info.get('forwardPE'),
            'PEG Ratio': info.get('pegRatio'),
            'Price to Book': info.get('priceToBook'),
            'Price to Sales': info.get(
                'priceToSalesTrailing12Months'
            ),
            'Enterprise to Revenue': info.get(
                'enterpriseToRevenue'
            ),
            'Enterprise to EBITDA': info.get(
                'enterpriseToEbitda'
            )
        },
        name=ticker
    )


def growthMetrics(
    ticker: str,
    info: dict[str, Any] | None = None
) -> pd.Series:
    '''Return company growth metrics.'''
    ticker = validateTicker(ticker)
    info = companyInfo(ticker) if info is None else info

    return pd.Series(
        {
            'Revenue Growth': info.get('revenueGrowth'),
            'Earnings Growth': info.get('earningsGrowth'),
            'Earnings Quarterly Growth': info.get(
                'earningsQuarterlyGrowth'
            ),
            'Revenue Per Share': info.get('revenuePerShare'),
            'Earnings Per Share': info.get('trailingEps'),
            'Forward Earnings Per Share': info.get(
                'forwardEps'
            )
        },
        name=ticker
    )


def profitabilityMetrics(
    ticker: str,
    info: dict[str, Any] | None = None
) -> pd.Series:
    '''Return profitability metrics.'''
    ticker = validateTicker(ticker)
    info = companyInfo(ticker) if info is None else info

    return pd.Series(
        {
            'Gross Margin': info.get('grossMargins'),
            'Operating Margin': info.get('operatingMargins'),
            'EBITDA Margin': info.get('ebitdaMargins'),
            'Profit Margin': info.get('profitMargins'),
            'Return on Equity': info.get('returnOnEquity'),
            'Return on Assets': info.get('returnOnAssets'),
            'Gross Profit': info.get('grossProfits'),
            'EBITDA': info.get('ebitda'),
            'Net Income': info.get('netIncomeToCommon')
        },
        name=ticker
    )


def financialHealth(
    ticker: str,
    info: dict[str, Any] | None = None
) -> pd.Series:
    '''Return balance-sheet and liquidity metrics.'''
    ticker = validateTicker(ticker)
    info = companyInfo(ticker) if info is None else info

    return pd.Series(
        {
            'Current Ratio': info.get('currentRatio'),
            'Quick Ratio': info.get('quickRatio'),
            'Debt to Equity': info.get('debtToEquity'),
            'Total Cash': info.get('totalCash'),
            'Cash Per Share': info.get('totalCashPerShare'),
            'Total Debt': info.get('totalDebt'),
            'Operating Cash Flow': info.get(
                'operatingCashflow'
            ),
            'Free Cash Flow': info.get('freeCashflow')
        },
        name=ticker
    )


def dividendMetrics(
    ticker: str,
    info: dict[str, Any] | None = None
) -> pd.Series:
    '''Return dividend metrics.'''
    ticker = validateTicker(ticker)
    info = companyInfo(ticker) if info is None else info

    dividendRate = info.get('dividendRate')
    currentPrice = info.get('currentPrice')
    trailingDividendRate = info.get('trailingAnnualDividendRate')
    previousClose = info.get('previousClose')

    def normalizeYield(value: Any) -> float | None:
        '''Normalize Yahoo yield ratios, percent values, or basis points.'''
        if value is None:
            return None

        normalized = float(value)

        # Yahoo has returned the same yield as 0.0035, 0.35, and 35
        # across different endpoints/releases. Convert to a decimal ratio.
        while abs(normalized) > 0.20:
            normalized /= 100

        return normalized

    # Calculate yields from cash dividends and prices whenever possible.
    # This avoids Yahoo's inconsistent yield units across API endpoints.
    if dividendRate is not None and currentPrice not in (None, 0):
        dividendYield = normalizeYield(
            float(dividendRate) / float(currentPrice)
        )
    else:
        dividendYield = normalizeYield(info.get('dividendYield'))

    if trailingDividendRate is not None and previousClose not in (None, 0):
        trailingYield = normalizeYield(
            float(trailingDividendRate) / float(previousClose)
        )
    else:
        trailingYield = normalizeYield(
            info.get('trailingAnnualDividendYield')
        )

    fiveYearAverageYield = info.get('fiveYearAvgDividendYield')

    if fiveYearAverageYield is not None:
        fiveYearAverageYield = normalizeYield(fiveYearAverageYield)

    exDividendDate = info.get('exDividendDate')

    if exDividendDate is not None:
        try:
            exDividendDate = pd.to_datetime(
                exDividendDate,
                unit='s'
            )
        except (TypeError, ValueError, OverflowError):
            pass

    return pd.Series(
        {
            'Dividend Yield': dividendYield,
            'Dividend Rate': dividendRate,
            'Trailing Annual Dividend Yield': trailingYield,
            'Trailing Annual Dividend Rate': trailingDividendRate,
            'Payout Ratio': info.get('payoutRatio'),
            'Five Year Average Dividend Yield': info.get(
                'fiveYearAvgDividendYield'
            ) if fiveYearAverageYield is None else fiveYearAverageYield,
            'Ex Dividend Date': exDividendDate
        },
        name=ticker
    )


def analystMetrics(
    ticker: str,
    info: dict[str, Any] | None = None
) -> pd.Series:
    '''Return analyst consensus metrics.'''
    ticker = validateTicker(ticker)
    info = companyInfo(ticker) if info is None else info

    return pd.Series(
        {
            'Recommendation': info.get('recommendationKey'),
            'Recommendation Mean': info.get(
                'recommendationMean'
            ),
            'Target Mean Price': info.get('targetMeanPrice'),
            'Target Median Price': info.get(
                'targetMedianPrice'
            ),
            'Target High Price': info.get('targetHighPrice'),
            'Target Low Price': info.get('targetLowPrice'),
            'Number of Analysts': info.get(
                'numberOfAnalystOpinions'
            )
        },
        name=ticker
    )


def incomeStatement(
    ticker: str,
    quarterly: bool = False
) -> pd.DataFrame:
    '''Return the annual or quarterly income statement.'''
    tickerObject = getTicker(ticker)

    statement = (
        tickerObject.quarterly_income_stmt
        if quarterly
        else tickerObject.income_stmt
    )

    return statement.copy()


def balanceSheet(
    ticker: str,
    quarterly: bool = False
) -> pd.DataFrame:
    '''Return the annual or quarterly balance sheet.'''
    tickerObject = getTicker(ticker)

    statement = (
        tickerObject.quarterly_balance_sheet
        if quarterly
        else tickerObject.balance_sheet
    )

    return statement.copy()


def cashFlowStatement(
    ticker: str,
    quarterly: bool = False
) -> pd.DataFrame:
    '''Return the annual or quarterly cash-flow statement.'''
    tickerObject = getTicker(ticker)

    statement = (
        tickerObject.quarterly_cash_flow
        if quarterly
        else tickerObject.cash_flow
    )

    return statement.copy()


def earningsDates(
    ticker: str,
    limit: int = 12
) -> pd.DataFrame:
    '''Return historical and expected earnings dates.'''
    if limit <= 0:
        raise ValueError('limit must be positive.')

    try:
        dates = getTicker(ticker).get_earnings_dates(
            limit=limit
        )
    except Exception as error:
        raise RuntimeError(
            f'Unable to retrieve earnings dates for '
            f'{validateTicker(ticker)}.'
        ) from error

    if dates is None:
        return pd.DataFrame()

    return dates.copy()


def recommendations(
    ticker: str
) -> pd.DataFrame:
    '''Return analyst recommendation history.'''
    try:
        data = getTicker(ticker).recommendations
    except Exception as error:
        raise RuntimeError(
            f'Unable to retrieve recommendations for '
            f'{validateTicker(ticker)}.'
        ) from error

    if data is None:
        return pd.DataFrame()

    return data.copy()


def institutionalHolders(
    ticker: str
) -> pd.DataFrame:
    '''Return major institutional holders.'''
    try:
        holders = getTicker(ticker).institutional_holders
    except Exception as error:
        raise RuntimeError(
            f'Unable to retrieve institutional holders for '
            f'{validateTicker(ticker)}.'
        ) from error

    if holders is None:
        return pd.DataFrame()

    return holders.copy()


def insiderTransactions(
    ticker: str
) -> pd.DataFrame:
    '''Return reported insider transactions.'''
    try:
        transactions = getTicker(ticker).insider_transactions
    except Exception as error:
        raise RuntimeError(
            f'Unable to retrieve insider transactions for '
            f'{validateTicker(ticker)}.'
        ) from error

    if transactions is None:
        return pd.DataFrame()

    return transactions.copy()


def fundamentalsSummary(ticker: str) -> pd.DataFrame:
    '''Return the main company fundamental metrics.'''
    ticker = validateTicker(ticker)
    info = companyInfo(ticker)

    sections = {
        'Company': companySummary(ticker, info),
        'Market': marketMetrics(ticker, info),
        'Valuation': valuationMetrics(ticker, info),
        'Growth': growthMetrics(ticker, info),
        'Profitability': profitabilityMetrics(ticker, info),
        'Financial Health': financialHealth(ticker, info),
        'Dividends': dividendMetrics(ticker, info),
        'Analysts': analystMetrics(ticker, info)
    }

    rows = []

    for section, metrics in sections.items():
        for metric, value in metrics.items():
            rows.append({
                'Section': section,
                'Metric': metric,
                'Value': value
            })

    return pd.DataFrame(rows).set_index(
        ['Section', 'Metric']
    )


def multipleFundamentalsSummary(
    tickers: list[str]
) -> pd.DataFrame:
    '''Return comparable fundamentals for multiple tickers.'''
    tickers = [
        validateTicker(ticker)
        for ticker in tickers
    ]

    if not tickers:
        raise ValueError(
            'Enter at least one ticker.'
        )

    summaries = {}

    for ticker in dict.fromkeys(tickers):
        try:
            summary = fundamentalsSummary(ticker)
            summaries[ticker] = summary['Value']
        except (ValueError, RuntimeError):
            summaries[ticker] = pd.Series(
                dtype=object
            )

    if not summaries:
        return pd.DataFrame()

    return pd.concat(
        summaries,
        axis=1
    )


def formatFundamentals(
    fundamentals: pd.DataFrame
) -> pd.DataFrame:
    '''Format fundamental data for dashboard display.'''
    if fundamentals.empty:
        return fundamentals.copy()

    percentageMetrics = {
        'Dividend Yield',
        'Trailing Annual Dividend Yield',
        'Five Year Average Dividend Yield',
        'Payout Ratio',
        'Revenue Growth',
        'Earnings Growth',
        'Earnings Quarterly Growth',
        'Gross Margin',
        'Operating Margin',
        'EBITDA Margin',
        'Profit Margin',
        'Return on Equity',
        'Return on Assets'
    }

    currencyMetrics = {
        'Current Price',
        'Previous Close',
        'Open',
        'Day Low',
        'Day High',
        '52 Week Low',
        '52 Week High',
        '50 Day Average',
        '200 Day Average',
        'Dividend Rate',
        'Trailing Annual Dividend Rate',
        'Target Mean Price',
        'Target Median Price',
        'Target High Price',
        'Target Low Price',
        'Cash Per Share',
        'Revenue Per Share',
        'Earnings Per Share',
        'Forward Earnings Per Share'
    }

    largeNumberMetrics = {
        'Market Cap',
        'Enterprise Value',
        'Average Volume',
        'Employees',
        'Gross Profit',
        'EBITDA',
        'Net Income',
        'Total Cash',
        'Total Debt',
        'Operating Cash Flow',
        'Free Cash Flow'
    }

    def formatValue(
        metric: str,
        value: Any
    ) -> Any:
        if value is None or (
            not isinstance(value, str)
            and pd.isna(value)
        ):
            return 'N/A'

        if metric in percentageMetrics:
            return f'{float(value):.2%}'

        if metric in currencyMetrics:
            return f'{float(value):,.2f}'

        if metric in largeNumberMetrics:
            value = float(value)

            if abs(value) >= 1_000_000_000_000:
                return f'{value / 1_000_000_000_000:.2f}T'

            if abs(value) >= 1_000_000_000:
                return f'{value / 1_000_000_000:.2f}B'

            if abs(value) >= 1_000_000:
                return f'{value / 1_000_000:.2f}M'

            if abs(value) >= 1_000:
                return f'{value / 1_000:.2f}K'

            return f'{value:,.0f}'

        if isinstance(value, float):
            return f'{value:.2f}'

        return value

    formatted = fundamentals.copy().astype(object)

    if isinstance(formatted.index, pd.MultiIndex):
        for index in formatted.index:
            metric = str(index[-1])

            for column in formatted.columns:
                formatted.loc[index, column] = formatValue(
                    metric,
                    formatted.loc[index, column]
                )

        return formatted

    for index in formatted.index:
        metric = str(index)

        for column in formatted.columns:
            formatted.loc[index, column] = formatValue(
                metric,
                formatted.loc[index, column]
            )

    return formatted
