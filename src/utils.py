from __future__ import annotations

import io
import json

from datetime import date, datetime
from functools import wraps
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar

import numpy as np
import pandas as pd
import streamlit as st


def validateNotNone(
    value: Any,
    name: str = 'value'
) -> Any:
    '''Validate that a value is not None.'''
    if value is None:
        raise ValueError(
            f'{name} cannot be None.'
        )

    return value


def validateBoolean(
    value: bool,
    name: str = 'value'
) -> bool:
    '''Validate a boolean value.'''
    if not isinstance(
        value,
        bool
    ):
        raise TypeError(
            f'{name} must be a boolean.'
        )

    return value


def validateString(
    value: str,
    name: str = 'value',
    allowEmpty: bool = False
) -> str:
    '''Validate a string value.'''
    if not isinstance(
        value,
        str
    ):
        raise TypeError(
            f'{name} must be a string.'
        )

    cleanedValue = value.strip()

    if not allowEmpty and not cleanedValue:
        raise ValueError(
            f'{name} cannot be empty.'
        )

    return cleanedValue


def validateIterable(
    values: Iterable[Any],
    name: str = 'values',
    allowEmpty: bool = False
) -> list[Any]:
    '''Validate and return an iterable as a list.'''
    if isinstance(
        values,
        (str, bytes)
    ):
        raise TypeError(
            f'{name} must not be a string.'
        )

    try:
        values = list(values)
    except TypeError as error:
        raise TypeError(
            f'{name} must be iterable.'
        ) from error

    if not allowEmpty and not values:
        raise ValueError(
            f'{name} cannot be empty.'
        )

    return values


def validateSequence(
    values: Sequence[Any],
    name: str = 'values',
    minimumLength: int | None = None,
    maximumLength: int | None = None
) -> Sequence[Any]:
    '''Validate a sequence and its length.'''
    if isinstance(
        values,
        (str, bytes)
    ):
        raise TypeError(
            f'{name} must not be a string.'
        )

    if not isinstance(
        values,
        Sequence
    ):
        raise TypeError(
            f'{name} must be a sequence.'
        )

    if (
        minimumLength is not None
        and len(values) < minimumLength
    ):
        raise ValueError(
            f'{name} must contain at least '
            f'{minimumLength} elements.'
        )

    if (
        maximumLength is not None
        and len(values) > maximumLength
    ):
        raise ValueError(
            f'{name} must contain at most '
            f'{maximumLength} elements.'
        )

    return values


def validateNumber(
    value: Real,
    name: str = 'value',
    minimum: float | None = None,
    maximum: float | None = None,
    allowInfinite: bool = False
) -> float:
    '''Validate a numeric value and optional bounds.'''
    if isinstance(
        value,
        bool
    ) or not isinstance(
        value,
        Real
    ):
        raise TypeError(
            f'{name} must be numeric.'
        )

    value = float(value)

    if np.isnan(value):
        raise ValueError(
            f'{name} cannot be NaN.'
        )

    if (
        not allowInfinite
        and not np.isfinite(value)
    ):
        raise ValueError(
            f'{name} must be finite.'
        )

    if (
        minimum is not None
        and value < minimum
    ):
        raise ValueError(
            f'{name} must be greater than '
            f'or equal to {minimum}.'
        )

    if (
        maximum is not None
        and value > maximum
    ):
        raise ValueError(
            f'{name} must be less than '
            f'or equal to {maximum}.'
        )

    return value


def validateInteger(
    value: int,
    name: str = 'value',
    minimum: int | None = None,
    maximum: int | None = None
) -> int:
    '''Validate an integer value and optional bounds.'''
    if isinstance(
        value,
        bool
    ) or not isinstance(
        value,
        (int, np.integer)
    ):
        raise TypeError(
            f'{name} must be an integer.'
        )

    value = int(value)

    if (
        minimum is not None
        and value < minimum
    ):
        raise ValueError(
            f'{name} must be greater than '
            f'or equal to {minimum}.'
        )

    if (
        maximum is not None
        and value > maximum
    ):
        raise ValueError(
            f'{name} must be less than '
            f'or equal to {maximum}.'
        )

    return value


def validatePositiveNumber(
    value: Real,
    name: str = 'value',
    allowZero: bool = False
) -> float:
    '''Validate a positive numeric value.'''
    minimum = 0.0

    value = validateNumber(
        value,
        name=name,
        minimum=minimum
    )

    if (
        not allowZero
        and np.isclose(value, 0)
    ):
        raise ValueError(
            f'{name} must be greater than zero.'
        )

    return value


def validateProbability(
    probability: Real,
    name: str = 'probability',
    inclusive: bool = True
) -> float:
    '''Validate a probability value.'''
    probability = validateNumber(
        probability,
        name=name
    )

    if inclusive:
        valid = 0 <= probability <= 1
    else:
        valid = 0 < probability < 1

    if not valid:
        interval = (
            '[0, 1]'
            if inclusive
            else '(0, 1)'
        )

        raise ValueError(
            f'{name} must be in {interval}.'
        )

    return probability


def validateConfidenceLevel(
    confidenceLevel: Real
) -> float:
    '''Validate a confidence level.'''
    return validateProbability(
        confidenceLevel,
        name='confidenceLevel',
        inclusive=False
    )


def validateAnnualizationFactor(
    annualizationFactor: int
) -> int:
    '''Validate the annualization factor.'''
    return validateInteger(
        annualizationFactor,
        name='annualizationFactor',
        minimum=1
    )


def validateSeries(
    series: pd.Series,
    name: str = 'series',
    allowEmpty: bool = False,
    allowMissing: bool = True,
    numeric: bool = False
) -> pd.Series:
    '''Validate a pandas Series.'''
    if not isinstance(
        series,
        pd.Series
    ):
        raise TypeError(
            f'{name} must be a pandas Series.'
        )

    if not allowEmpty and series.empty:
        raise ValueError(
            f'{name} cannot be empty.'
        )

    if (
        not allowMissing
        and series.isna().any()
    ):
        raise ValueError(
            f'{name} cannot contain missing values.'
        )

    if (
        numeric
        and not pd.api.types.is_numeric_dtype(
            series
        )
    ):
        raise TypeError(
            f'{name} must contain numeric values.'
        )

    return series


def validateDataFrame(
    dataFrame: pd.DataFrame,
    name: str = 'dataFrame',
    allowEmpty: bool = False,
    allowMissing: bool = True,
    numeric: bool = False,
    requiredColumns: Iterable[str] | None = None
) -> pd.DataFrame:
    '''Validate a pandas DataFrame.'''
    if not isinstance(
        dataFrame,
        pd.DataFrame
    ):
        raise TypeError(
            f'{name} must be a pandas DataFrame.'
        )

    if not allowEmpty and dataFrame.empty:
        raise ValueError(
            f'{name} cannot be empty.'
        )

    if (
        not allowMissing
        and dataFrame.isna().any().any()
    ):
        raise ValueError(
            f'{name} cannot contain missing values.'
        )

    if numeric:
        nonNumericColumns = [
            column
            for column in dataFrame.columns
            if not pd.api.types.is_numeric_dtype(
                dataFrame[column]
            )
        ]

        if nonNumericColumns:
            raise TypeError(
                f'{name} contains non-numeric columns: '
                + ', '.join(
                    map(
                        str,
                        nonNumericColumns
                    )
                )
            )

    if requiredColumns is not None:
        requiredColumns = set(
            requiredColumns
        )

        missingColumns = (
            requiredColumns
            - set(dataFrame.columns)
        )

        if missingColumns:
            raise ValueError(
                f'{name} is missing required columns: '
                + ', '.join(
                    sorted(
                        map(
                            str,
                            missingColumns
                        )
                    )
                )
            )

    return dataFrame


def validateArray(
    values: np.ndarray | Sequence[Real],
    name: str = 'values',
    dimensions: int | None = None,
    allowEmpty: bool = False,
    allowMissing: bool = False
) -> np.ndarray:
    '''Validate and return a NumPy array.'''
    try:
        array = np.asarray(
            values,
            dtype=float
        )
    except (
        TypeError,
        ValueError
    ) as error:
        raise TypeError(
            f'{name} must contain numeric values.'
        ) from error

    if (
        dimensions is not None
        and array.ndim != dimensions
    ):
        raise ValueError(
            f'{name} must have '
            f'{dimensions} dimensions.'
        )

    if (
        not allowEmpty
        and array.size == 0
    ):
        raise ValueError(
            f'{name} cannot be empty.'
        )

    if (
        not allowMissing
        and not np.isfinite(array).all()
    ):
        raise ValueError(
            f'{name} must contain finite values.'
        )

    return array


def validateVector(
    values: np.ndarray | Sequence[Real],
    name: str = 'values',
    expectedLength: int | None = None
) -> np.ndarray:
    '''Validate a one-dimensional numeric vector.'''
    vector = validateArray(
        values,
        name=name,
        dimensions=1
    )

    if (
        expectedLength is not None
        and len(vector) != expectedLength
    ):
        raise ValueError(
            f'{name} must contain exactly '
            f'{expectedLength} elements.'
        )

    return vector


def validateMatrix(
    matrix: np.ndarray | pd.DataFrame,
    name: str = 'matrix',
    square: bool = False,
    symmetric: bool = False,
    tolerance: float = 1e-10
) -> np.ndarray:
    '''Validate a numeric matrix.'''
    values = validateArray(
        matrix,
        name=name,
        dimensions=2
    )

    rows, columns = values.shape

    if square and rows != columns:
        raise ValueError(
            f'{name} must be square.'
        )

    if symmetric:
        if rows != columns:
            raise ValueError(
                f'{name} must be square '
                'to be symmetric.'
            )

        if not np.allclose(
            values,
            values.T,
            atol=tolerance,
            rtol=0
        ):
            raise ValueError(
                f'{name} must be symmetric.'
            )

    return values


def validateSquareMatrix(
    matrix: np.ndarray | pd.DataFrame,
    name: str = 'matrix',
    symmetric: bool = False
) -> np.ndarray:
    '''Validate a square numeric matrix.'''
    return validateMatrix(
        matrix,
        name=name,
        square=True,
        symmetric=symmetric
    )


def validateCovarianceMatrix(
    covarianceMatrix: np.ndarray | pd.DataFrame,
    tolerance: float = 1e-8
) -> np.ndarray:
    '''Validate a covariance matrix.'''
    covarianceMatrix = validateSquareMatrix(
        covarianceMatrix,
        name='covarianceMatrix',
        symmetric=True
    )

    eigenvalues = np.linalg.eigvalsh(
        covarianceMatrix
    )

    if np.any(
        eigenvalues < -tolerance
    ):
        raise ValueError(
            'covarianceMatrix must be '
            'positive semidefinite.'
        )

    return covarianceMatrix


def validateCorrelationMatrix(
    correlationMatrix: np.ndarray | pd.DataFrame,
    tolerance: float = 1e-8
) -> np.ndarray:
    '''Validate a correlation matrix.'''
    correlationMatrix = validateSquareMatrix(
        correlationMatrix,
        name='correlationMatrix',
        symmetric=True
    )

    if not np.allclose(
        np.diag(correlationMatrix),
        1,
        atol=tolerance
    ):
        raise ValueError(
            'correlationMatrix diagonal '
            'must contain ones.'
        )

    if np.any(
        correlationMatrix < -1 - tolerance
    ) or np.any(
        correlationMatrix > 1 + tolerance
    ):
        raise ValueError(
            'correlationMatrix values must '
            'be between -1 and 1.'
        )

    return correlationMatrix


def validateWeights(
    weights: np.ndarray | pd.Series | Sequence[Real],
    assetCount: int | None = None,
    allowShortSelling: bool = False,
    normalize: bool = False,
    tolerance: float = 1e-8
) -> np.ndarray:
    '''Validate portfolio weights.'''
    weights = validateVector(
        weights,
        name='weights',
        expectedLength=assetCount
    )

    if (
        not allowShortSelling
        and np.any(weights < -tolerance)
    ):
        raise ValueError(
            'weights cannot be negative when '
            'short selling is disabled.'
        )

    totalWeight = float(
        weights.sum()
    )

    if np.isclose(
        totalWeight,
        0,
        atol=tolerance
    ):
        raise ValueError(
            'weights cannot sum to zero.'
        )

    if normalize:
        weights = (
            weights
            / totalWeight
        )

    elif not np.isclose(
        totalWeight,
        1,
        atol=tolerance
    ):
        raise ValueError(
            'weights must sum to one.'
        )

    return weights


def validateBounds(
    bounds: Sequence[
        tuple[Real, Real]
    ],
    assetCount: int | None = None
) -> list[tuple[float, float]]:
    '''Validate optimization bounds.'''
    bounds = validateIterable(
        bounds,
        name='bounds'
    )

    if (
        assetCount is not None
        and len(bounds) != assetCount
    ):
        raise ValueError(
            'The number of bounds must match '
            'the number of assets.'
        )

    validatedBounds = []

    for index, bound in enumerate(bounds):
        if (
            not isinstance(
                bound,
                Sequence
            )
            or len(bound) != 2
        ):
            raise ValueError(
                f'bounds[{index}] must contain '
                'a lower and upper bound.'
            )

        lowerBound = validateNumber(
            bound[0],
            name=f'bounds[{index}][0]'
        )

        upperBound = validateNumber(
            bound[1],
            name=f'bounds[{index}][1]'
        )

        if lowerBound > upperBound:
            raise ValueError(
                f'bounds[{index}] lower bound '
                'cannot exceed the upper bound.'
            )

        validatedBounds.append(
            (
                lowerBound,
                upperBound
            )
        )

    return validatedBounds


def validateDate(
    value: date | datetime | pd.Timestamp,
    name: str = 'date'
) -> date:
    '''Validate and return a date object.'''
    if isinstance(
        value,
        pd.Timestamp
    ):
        value = value.date()

    elif isinstance(
        value,
        datetime
    ):
        value = value.date()

    if not isinstance(
        value,
        date
    ):
        raise TypeError(
            f'{name} must be a date.'
        )

    return value


def validateDateRange(
    startDate: date | datetime | pd.Timestamp,
    endDate: date | datetime | pd.Timestamp,
    allowEqual: bool = False,
    allowFuture: bool = False
) -> tuple[date, date]:
    '''Validate a date interval.'''
    startDate = validateDate(
        startDate,
        name='startDate'
    )

    endDate = validateDate(
        endDate,
        name='endDate'
    )

    if allowEqual:
        invalidRange = startDate > endDate
    else:
        invalidRange = startDate >= endDate

    if invalidRange:
        relation = (
            'later than'
            if allowEqual
            else 'earlier than'
        )

        raise ValueError(
            f'startDate must be {relation} endDate.'
        )

    if (
        not allowFuture
        and endDate > date.today()
    ):
        raise ValueError(
            'endDate cannot be in the future.'
        )

    return startDate, endDate


def validateDatetimeIndex(
    data: pd.Series | pd.DataFrame,
    name: str = 'data'
) -> pd.Series | pd.DataFrame:
    '''Validate that data uses a DatetimeIndex.'''
    if not isinstance(
        data,
        (pd.Series, pd.DataFrame)
    ):
        raise TypeError(
            f'{name} must be a Series or DataFrame.'
        )

    if not isinstance(
        data.index,
        pd.DatetimeIndex
    ):
        raise TypeError(
            f'{name} must use a DatetimeIndex.'
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            f'{name} index must be sorted '
            'in ascending order.'
        )

    if data.index.has_duplicates:
        raise ValueError(
            f'{name} index cannot contain duplicates.'
        )

    return data


def validateMatchingLength(
    first: Sequence[Any] | np.ndarray,
    second: Sequence[Any] | np.ndarray,
    firstName: str = 'first',
    secondName: str = 'second'
) -> None:
    '''Validate that two objects have matching length.'''
    if len(first) != len(second):
        raise ValueError(
            f'{firstName} and {secondName} '
            'must have matching length.'
        )


def validateMatchingIndex(
    first: pd.Series | pd.DataFrame,
    second: pd.Series | pd.DataFrame,
    firstName: str = 'first',
    secondName: str = 'second'
) -> None:
    '''Validate that two pandas objects share the same index.'''
    if not first.index.equals(
        second.index
    ):
        raise ValueError(
            f'{firstName} and {secondName} '
            'must have matching indexes.'
        )


def validateMatchingColumns(
    first: pd.DataFrame,
    second: pd.DataFrame,
    firstName: str = 'first',
    secondName: str = 'second'
) -> None:
    '''Validate that two DataFrames share the same columns.'''
    if not first.columns.equals(
        second.columns
    ):
        raise ValueError(
            f'{firstName} and {secondName} '
            'must have matching columns.'
        )
    
def formatPercent(
    value: Real | None,
    decimals: int = 2,
    missingValue: str = 'N/A'
) -> str:
    '''Format a decimal value as a percentage.'''
    if value is None or pd.isna(value):
        return missingValue

    value = validateNumber(
        value,
        name='value'
    )

    decimals = validateInteger(
        decimals,
        name='decimals',
        minimum=0
    )

    return f'{value:.{decimals}%}'


def formatCurrency(
    value: Real | None,
    currencySymbol: str = '$',
    decimals: int = 2,
    missingValue: str = 'N/A'
) -> str:
    '''Format a numeric value as currency.'''
    if value is None or pd.isna(value):
        return missingValue

    value = validateNumber(
        value,
        name='value'
    )

    currencySymbol = validateString(
        currencySymbol,
        name='currencySymbol',
        allowEmpty=True
    )

    decimals = validateInteger(
        decimals,
        name='decimals',
        minimum=0
    )

    return (
        f'{currencySymbol}'
        f'{value:,.{decimals}f}'
    )


def formatNumber(
    value: Real | None,
    decimals: int = 2,
    missingValue: str = 'N/A',
    thousandsSeparator: bool = True
) -> str:
    '''Format a numeric value.'''
    if value is None or pd.isna(value):
        return missingValue

    value = validateNumber(
        value,
        name='value'
    )

    decimals = validateInteger(
        decimals,
        name='decimals',
        minimum=0
    )

    formatSpecification = (
        f',.{decimals}f'
        if thousandsSeparator
        else f'.{decimals}f'
    )

    return format(
        value,
        formatSpecification
    )


def formatInteger(
    value: Real | None,
    missingValue: str = 'N/A'
) -> str:
    '''Format a numeric value as an integer.'''
    if value is None or pd.isna(value):
        return missingValue

    value = validateNumber(
        value,
        name='value'
    )

    return f'{value:,.0f}'


def formatLargeNumber(
    value: Real | None,
    decimals: int = 2,
    missingValue: str = 'N/A'
) -> str:
    '''Format a large numeric value using suffixes.'''
    if value is None or pd.isna(value):
        return missingValue

    value = validateNumber(
        value,
        name='value'
    )

    decimals = validateInteger(
        decimals,
        name='decimals',
        minimum=0
    )

    absoluteValue = abs(value)

    suffixes = [
        (1e12, 'T'),
        (1e9, 'B'),
        (1e6, 'M'),
        (1e3, 'K')
    ]

    for threshold, suffix in suffixes:
        if absoluteValue >= threshold:
            scaledValue = value / threshold

            return (
                f'{scaledValue:.{decimals}f}'
                f'{suffix}'
            )

    return f'{value:.{decimals}f}'


def formatRatio(
    value: Real | None,
    decimals: int = 2,
    missingValue: str = 'N/A'
) -> str:
    '''Format a ratio value.'''
    if value is None or pd.isna(value):
        return missingValue

    value = validateNumber(
        value,
        name='value',
        allowInfinite=True
    )

    if np.isposinf(value):
        return '∞'

    if np.isneginf(value):
        return '-∞'

    return f'{value:.{decimals}f}'


def formatDate(
    value: date | datetime | pd.Timestamp | None,
    dateFormat: str = '%Y-%m-%d',
    missingValue: str = 'N/A'
) -> str:
    '''Format a date value.'''
    if value is None or pd.isna(value):
        return missingValue

    if isinstance(
        value,
        pd.Timestamp
    ):
        value = value.to_pydatetime()

    if not isinstance(
        value,
        (date, datetime)
    ):
        raise TypeError(
            'value must be a date or datetime.'
        )

    return value.strftime(
        dateFormat
    )


def formatMetric(
    value: Any,
    metricType: str = 'number',
    decimals: int = 2,
    currencySymbol: str = '$',
    missingValue: str = 'N/A'
) -> str:
    '''Format a value according to a metric type.'''
    metricType = validateString(
        metricType,
        name='metricType'
    ).lower()

    formatterMap = {
        'percent': lambda item: formatPercent(
            item,
            decimals=decimals,
            missingValue=missingValue
        ),
        'currency': lambda item: formatCurrency(
            item,
            currencySymbol=currencySymbol,
            decimals=decimals,
            missingValue=missingValue
        ),
        'number': lambda item: formatNumber(
            item,
            decimals=decimals,
            missingValue=missingValue
        ),
        'integer': lambda item: formatInteger(
            item,
            missingValue=missingValue
        ),
        'large': lambda item: formatLargeNumber(
            item,
            decimals=decimals,
            missingValue=missingValue
        ),
        'ratio': lambda item: formatRatio(
            item,
            decimals=decimals,
            missingValue=missingValue
        )
    }

    if metricType not in formatterMap:
        raise ValueError(
            'metricType must be one of: '
            + ', '.join(
                formatterMap.keys()
            )
        )

    return formatterMap[metricType](
        value
    )


def formatSeries(
    series: pd.Series,
    metricType: str = 'number',
    decimals: int = 2,
    currencySymbol: str = '$',
    missingValue: str = 'N/A'
) -> pd.Series:
    '''Format every value in a Series.'''
    series = validateSeries(
        series,
        name='series',
        allowEmpty=True
    )

    return series.map(
        lambda value: formatMetric(
            value,
            metricType=metricType,
            decimals=decimals,
            currencySymbol=currencySymbol,
            missingValue=missingValue
        )
    )


def formatDataFrame(
    dataFrame: pd.DataFrame,
    columnFormats: dict[str, str] | None = None,
    decimals: int = 2,
    currencySymbol: str = '$',
    missingValue: str = 'N/A'
) -> pd.DataFrame:
    '''Format selected DataFrame columns.'''
    dataFrame = validateDataFrame(
        dataFrame,
        name='dataFrame',
        allowEmpty=True
    ).copy()

    if columnFormats is None:
        columnFormats = {
            column: 'number'
            for column in dataFrame.select_dtypes(
                include=np.number
            ).columns
        }

    for column, metricType in columnFormats.items():
        if column not in dataFrame.columns:
            continue

        dataFrame[column] = formatSeries(
            dataFrame[column],
            metricType=metricType,
            decimals=decimals,
            currencySymbol=currencySymbol,
            missingValue=missingValue
        )

    return dataFrame


def toSeries(
    values: pd.Series | pd.DataFrame | Sequence[Real] | np.ndarray,
    name: str | None = None,
    index: Sequence[Any] | pd.Index | None = None
) -> pd.Series:
    '''Convert supported data into a pandas Series.'''
    if isinstance(
        values,
        pd.Series
    ):
        series = values.copy()

    elif isinstance(
        values,
        pd.DataFrame
    ):
        if values.shape[1] != 1:
            raise ValueError(
                'DataFrame must contain exactly one column.'
            )

        series = values.iloc[:, 0].copy()

    else:
        array = validateVector(
            values,
            name='values'
        )

        if (
            index is not None
            and len(index) != len(array)
        ):
            raise ValueError(
                'index length must match values length.'
            )

        series = pd.Series(
            array,
            index=index
        )

    if name is not None:
        series.name = name

    return series


def toDataFrame(
    values: pd.DataFrame | pd.Series | np.ndarray | Sequence[Any],
    columns: Sequence[str] | None = None,
    index: Sequence[Any] | pd.Index | None = None
) -> pd.DataFrame:
    '''Convert supported data into a pandas DataFrame.'''
    if isinstance(
        values,
        pd.DataFrame
    ):
        dataFrame = values.copy()

    elif isinstance(
        values,
        pd.Series
    ):
        dataFrame = values.to_frame()

    else:
        array = np.asarray(
            values
        )

        if array.ndim == 1:
            array = array.reshape(
                -1,
                1
            )

        if array.ndim != 2:
            raise ValueError(
                'values must be one- or two-dimensional.'
            )

        dataFrame = pd.DataFrame(
            array,
            index=index,
            columns=columns
        )

    if columns is not None:
        if len(columns) != dataFrame.shape[1]:
            raise ValueError(
                'columns length must match DataFrame width.'
            )

        dataFrame.columns = list(
            columns
        )

    if index is not None:
        if len(index) != dataFrame.shape[0]:
            raise ValueError(
                'index length must match DataFrame height.'
            )

        dataFrame.index = index

    return dataFrame


def ensureDatetimeIndex(
    data: pd.Series | pd.DataFrame,
    sort: bool = True,
    removeDuplicates: bool = True
) -> pd.Series | pd.DataFrame:
    '''Convert the index to DatetimeIndex.'''
    if not isinstance(
        data,
        (pd.Series, pd.DataFrame)
    ):
        raise TypeError(
            'data must be a Series or DataFrame.'
        )

    result = data.copy()

    try:
        result.index = pd.to_datetime(
            result.index
        )
    except (
        TypeError,
        ValueError
    ) as error:
        raise ValueError(
            'Index could not be converted to datetime.'
        ) from error

    if removeDuplicates:
        result = result[
            ~result.index.duplicated(
                keep='last'
            )
        ]

    if sort:
        result = result.sort_index()

    return result


def cleanNumericData(
    data: pd.Series | pd.DataFrame,
    dropInfinite: bool = True,
    dropMissing: bool = True
) -> pd.Series | pd.DataFrame:
    '''Clean missing and infinite numeric observations.'''
    if not isinstance(
        data,
        (pd.Series, pd.DataFrame)
    ):
        raise TypeError(
            'data must be a Series or DataFrame.'
        )

    result = data.copy()

    if dropInfinite:
        result = result.replace(
            [np.inf, -np.inf],
            np.nan
        )

    if dropMissing:
        if isinstance(
            result,
            pd.Series
        ):
            result = result.dropna()
        else:
            result = result.dropna(
                how='all'
            )

    return result


def annualToPeriodicRate(
    annualRate: Real,
    periodsPerYear: int = 252,
    compounded: bool = True
) -> float:
    '''Convert an annual rate to a periodic rate.'''
    annualRate = validateNumber(
        annualRate,
        name='annualRate',
        minimum=-1
    )

    periodsPerYear = validateInteger(
        periodsPerYear,
        name='periodsPerYear',
        minimum=1
    )

    if compounded:
        return (
            1 + annualRate
        ) ** (
            1 / periodsPerYear
        ) - 1

    return annualRate / periodsPerYear


def periodicToAnnualRate(
    periodicRate: Real,
    periodsPerYear: int = 252,
    compounded: bool = True
) -> float:
    '''Convert a periodic rate to an annual rate.'''
    periodicRate = validateNumber(
        periodicRate,
        name='periodicRate',
        minimum=-1
    )

    periodsPerYear = validateInteger(
        periodsPerYear,
        name='periodsPerYear',
        minimum=1
    )

    if compounded:
        return (
            1 + periodicRate
        ) ** periodsPerYear - 1

    return periodicRate * periodsPerYear


def annualToDailyRate(
    annualRate: Real,
    tradingDays: int = 252,
    compounded: bool = True
) -> float:
    '''Convert an annual rate to a daily rate.'''
    return annualToPeriodicRate(
        annualRate,
        periodsPerYear=tradingDays,
        compounded=compounded
    )


def dailyToAnnualRate(
    dailyRate: Real,
    tradingDays: int = 252,
    compounded: bool = True
) -> float:
    '''Convert a daily rate to an annual rate.'''
    return periodicToAnnualRate(
        dailyRate,
        periodsPerYear=tradingDays,
        compounded=compounded
    )


def annualToPeriodicVolatility(
    annualVolatility: Real,
    periodsPerYear: int = 252
) -> float:
    '''Convert annual volatility to periodic volatility.'''
    annualVolatility = validatePositiveNumber(
        annualVolatility,
        name='annualVolatility',
        allowZero=True
    )

    periodsPerYear = validateInteger(
        periodsPerYear,
        name='periodsPerYear',
        minimum=1
    )

    return annualVolatility / np.sqrt(
        periodsPerYear
    )


def periodicToAnnualVolatility(
    periodicVolatility: Real,
    periodsPerYear: int = 252
) -> float:
    '''Convert periodic volatility to annual volatility.'''
    periodicVolatility = validatePositiveNumber(
        periodicVolatility,
        name='periodicVolatility',
        allowZero=True
    )

    periodsPerYear = validateInteger(
        periodsPerYear,
        name='periodsPerYear',
        minimum=1
    )

    return periodicVolatility * np.sqrt(
        periodsPerYear
    )


def simpleReturns(
    prices: pd.Series | pd.DataFrame,
    periods: int = 1,
    dropMissing: bool = True
) -> pd.Series | pd.DataFrame:
    '''Calculate simple percentage returns.'''
    if not isinstance(
        prices,
        (pd.Series, pd.DataFrame)
    ):
        raise TypeError(
            'prices must be a Series or DataFrame.'
        )

    periods = validateInteger(
        periods,
        name='periods',
        minimum=1
    )

    returns = prices.pct_change(
        periods=periods,
        fill_method=None
    )

    if dropMissing:
        returns = returns.dropna(
            how=(
                'all'
                if isinstance(
                    returns,
                    pd.DataFrame
                )
                else None
            )
        )

    return returns


def logReturns(
    prices: pd.Series | pd.DataFrame,
    periods: int = 1,
    dropMissing: bool = True
) -> pd.Series | pd.DataFrame:
    '''Calculate logarithmic returns.'''
    if not isinstance(
        prices,
        (pd.Series, pd.DataFrame)
    ):
        raise TypeError(
            'prices must be a Series or DataFrame.'
        )

    periods = validateInteger(
        periods,
        name='periods',
        minimum=1
    )

    if (
        prices.select_dtypes(
            include=np.number
        ).shape[1] != prices.shape[1]
        if isinstance(prices, pd.DataFrame)
        else not pd.api.types.is_numeric_dtype(
            prices
        )
    ):
        raise TypeError(
            'prices must contain numeric values.'
        )

    if (prices <= 0).any().any() if isinstance(
        prices,
        pd.DataFrame
    ) else (prices <= 0).any():
        raise ValueError(
            'prices must be positive for log returns.'
        )

    returns = np.log(
        prices / prices.shift(periods)
    )

    if dropMissing:
        returns = returns.dropna(
            how=(
                'all'
                if isinstance(
                    returns,
                    pd.DataFrame
                )
                else None
            )
        )

    return returns


def cumulativeReturns(
    returns: pd.Series | pd.DataFrame,
    initialValue: float = 1.0
) -> pd.Series | pd.DataFrame:
    '''Convert periodic returns into cumulative values.'''
    if not isinstance(
        returns,
        (pd.Series, pd.DataFrame)
    ):
        raise TypeError(
            'returns must be a Series or DataFrame.'
        )

    initialValue = validatePositiveNumber(
        initialValue,
        name='initialValue'
    )

    return (
        1 + returns.fillna(0)
    ).cumprod() * initialValue


def cumulativePerformance(
    returns: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    '''Calculate cumulative percentage performance.'''
    return cumulativeReturns(
        returns,
        initialValue=1.0
    ) - 1


def normalizePrices(
    prices: pd.Series | pd.DataFrame,
    baseValue: float = 100.0
) -> pd.Series | pd.DataFrame:
    '''Normalize prices to a common starting value.'''
    if not isinstance(
        prices,
        (pd.Series, pd.DataFrame)
    ):
        raise TypeError(
            'prices must be a Series or DataFrame.'
        )

    baseValue = validatePositiveNumber(
        baseValue,
        name='baseValue'
    )

    validPrices = prices.dropna(
        how=(
            'all'
            if isinstance(
                prices,
                pd.DataFrame
            )
            else None
        )
    )

    if validPrices.empty:
        raise ValueError(
            'prices contains no valid observations.'
        )

    if isinstance(
        validPrices,
        pd.Series
    ):
        firstValue = validPrices.iloc[0]

        if np.isclose(
            firstValue,
            0
        ):
            raise ValueError(
                'Initial price cannot be zero.'
            )

        return (
            validPrices
            / firstValue
            * baseValue
        )

    firstValues = validPrices.apply(
        lambda column: column.dropna().iloc[0]
    )

    if np.isclose(
        firstValues,
        0
    ).any():
        raise ValueError(
            'Initial prices cannot be zero.'
        )

    return validPrices.divide(
        firstValues,
        axis=1
    ) * baseValue


def alignPandasObjects(
    *objects: pd.Series | pd.DataFrame,
    join: str = 'inner',
    axis: int = 0,
    dropMissing: bool = True
) -> tuple[pd.Series | pd.DataFrame, ...]:
    '''Align multiple pandas objects.'''
    if len(objects) < 2:
        raise ValueError(
            'At least two objects are required.'
        )

    for item in objects:
        if not isinstance(
            item,
            (pd.Series, pd.DataFrame)
        ):
            raise TypeError(
                'All objects must be Series or DataFrames.'
            )

    combined = pd.concat(
        objects,
        axis=1,
        join=join,
        keys=range(len(objects))
    )

    if dropMissing:
        combined = combined.dropna()

    alignedObjects = []

    for objectIndex, originalObject in enumerate(
        objects
    ):
        extracted = combined[
            objectIndex
        ]

        if isinstance(
            originalObject,
            pd.Series
        ):
            extracted = extracted.iloc[
                :,
                0
            ]

            extracted.name = (
                originalObject.name
            )

        alignedObjects.append(
            extracted
        )

    return tuple(
        alignedObjects
    )


def flattenDictionary(
    dictionary: dict[str, Any],
    parentKey: str = '',
    separator: str = '.'
) -> dict[str, Any]:
    '''Flatten a nested dictionary.'''
    if not isinstance(
        dictionary,
        dict
    ):
        raise TypeError(
            'dictionary must be a dict.'
        )

    flattened = {}

    for key, value in dictionary.items():
        key = str(key)

        newKey = (
            f'{parentKey}{separator}{key}'
            if parentKey
            else key
        )

        if isinstance(
            value,
            dict
        ):
            flattened.update(
                flattenDictionary(
                    value,
                    parentKey=newKey,
                    separator=separator
                )
            )

        else:
            flattened[newKey] = value

    return flattened


def dictionaryToSeries(
    dictionary: dict[str, Any],
    flatten: bool = False,
    name: str | None = None
) -> pd.Series:
    '''Convert a dictionary into a Series.'''
    if not isinstance(
        dictionary,
        dict
    ):
        raise TypeError(
            'dictionary must be a dict.'
        )

    if flatten:
        dictionary = flattenDictionary(
            dictionary
        )

    return pd.Series(
        dictionary,
        name=name
    )


def recordsToDataFrame(
    records: Sequence[dict[str, Any]],
    index: str | None = None
) -> pd.DataFrame:
    '''Convert a sequence of records into a DataFrame.'''
    records = validateIterable(
        records,
        name='records',
        allowEmpty=True
    )

    dataFrame = pd.DataFrame.from_records(
        records
    )

    if (
        index is not None
        and index in dataFrame.columns
    ):
        dataFrame = dataFrame.set_index(
            index
        )

    return dataFrame


def reorderColumns(
    dataFrame: pd.DataFrame,
    firstColumns: Sequence[str]
) -> pd.DataFrame:
    '''Move selected columns to the front.'''
    dataFrame = validateDataFrame(
        dataFrame,
        name='dataFrame',
        allowEmpty=True
    )

    firstColumns = [
        column
        for column in firstColumns
        if column in dataFrame.columns
    ]

    remainingColumns = [
        column
        for column in dataFrame.columns
        if column not in firstColumns
    ]

    return dataFrame[
        firstColumns
        + remainingColumns
    ]


def percentageChange(
    currentValue: Real,
    previousValue: Real
) -> float:
    '''Calculate the percentage change between two values.'''
    currentValue = validateNumber(
        currentValue,
        name='currentValue'
    )

    previousValue = validateNumber(
        previousValue,
        name='previousValue'
    )

    if np.isclose(
        previousValue,
        0
    ):
        raise ZeroDivisionError(
            'previousValue cannot be zero.'
        )

    return (
        currentValue
        - previousValue
    ) / abs(previousValue)

def safeDivide(
    numerator: Real | np.ndarray | pd.Series,
    denominator: Real | np.ndarray | pd.Series,
    default: float = np.nan
) -> float | np.ndarray | pd.Series:
    '''Safely divide values while handling zero denominators.'''
    if isinstance(
        numerator,
        pd.Series
    ) or isinstance(
        denominator,
        pd.Series
    ):
        numeratorSeries = (
            numerator
            if isinstance(
                numerator,
                pd.Series
            )
            else pd.Series(
                numerator,
                index=denominator.index
            )
        )

        denominatorSeries = (
            denominator
            if isinstance(
                denominator,
                pd.Series
            )
            else pd.Series(
                denominator,
                index=numeratorSeries.index
            )
        )

        numeratorSeries, denominatorSeries = (
            numeratorSeries.align(
                denominatorSeries,
                join='outer'
            )
        )

        result = numeratorSeries.divide(
            denominatorSeries.replace(
                0,
                np.nan
            )
        )

        return result.fillna(
            default
        )

    numeratorArray = np.asarray(
        numerator,
        dtype=float
    )

    denominatorArray = np.asarray(
        denominator,
        dtype=float
    )

    numeratorArray, denominatorArray = (
        np.broadcast_arrays(
            numeratorArray,
            denominatorArray
        )
    )

    result = np.full(
        numeratorArray.shape,
        default,
        dtype=float
    )

    validMask = (
        np.isfinite(numeratorArray)
        & np.isfinite(denominatorArray)
        & ~np.isclose(
            denominatorArray,
            0
        )
    )

    np.divide(
        numeratorArray,
        denominatorArray,
        out=result,
        where=validMask
    )

    if result.ndim == 0:
        return float(result)

    return result


def clipValues(
    values: pd.Series | pd.DataFrame | np.ndarray,
    lowerBound: float | None = None,
    upperBound: float | None = None
) -> pd.Series | pd.DataFrame | np.ndarray:
    '''Clip values to optional lower and upper bounds.'''
    if (
        lowerBound is not None
        and upperBound is not None
        and lowerBound > upperBound
    ):
        raise ValueError(
            'lowerBound cannot exceed upperBound.'
        )

    if isinstance(
        values,
        (pd.Series, pd.DataFrame)
    ):
        return values.clip(
            lower=lowerBound,
            upper=upperBound
        )

    array = validateArray(
        values,
        name='values',
        allowMissing=True
    )

    return np.clip(
        array,
        a_min=lowerBound,
        a_max=upperBound
    )


def winsorizeSeries(
    series: pd.Series,
    lowerQuantile: float = 0.01,
    upperQuantile: float = 0.99
) -> pd.Series:
    '''Winsorize a Series using quantile limits.'''
    series = validateSeries(
        series,
        name='series',
        numeric=True
    )

    lowerQuantile = validateProbability(
        lowerQuantile,
        name='lowerQuantile'
    )

    upperQuantile = validateProbability(
        upperQuantile,
        name='upperQuantile'
    )

    if lowerQuantile >= upperQuantile:
        raise ValueError(
            'lowerQuantile must be lower than upperQuantile.'
        )

    lowerBound = series.quantile(
        lowerQuantile
    )

    upperBound = series.quantile(
        upperQuantile
    )

    return series.clip(
        lower=lowerBound,
        upper=upperBound
    )


def winsorizeDataFrame(
    dataFrame: pd.DataFrame,
    lowerQuantile: float = 0.01,
    upperQuantile: float = 0.99,
    columns: Sequence[str] | None = None
) -> pd.DataFrame:
    '''Winsorize selected numeric DataFrame columns.'''
    dataFrame = validateDataFrame(
        dataFrame,
        name='dataFrame'
    ).copy()

    if columns is None:
        columns = list(
            dataFrame.select_dtypes(
                include=np.number
            ).columns
        )

    for column in columns:
        if column not in dataFrame.columns:
            raise ValueError(
                f'Column not found: {column}.'
            )

        if not pd.api.types.is_numeric_dtype(
            dataFrame[column]
        ):
            raise TypeError(
                f'{column} must be numeric.'
            )

        dataFrame[column] = winsorizeSeries(
            dataFrame[column],
            lowerQuantile=lowerQuantile,
            upperQuantile=upperQuantile
        )

    return dataFrame


def removeOutliers(
    series: pd.Series,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.Series:
    '''Remove Series outliers using IQR or z-scores.'''
    series = validateSeries(
        series,
        name='series',
        numeric=True
    ).dropna()

    method = validateString(
        method,
        name='method'
    ).lower()

    threshold = validatePositiveNumber(
        threshold,
        name='threshold'
    )

    if method == 'iqr':
        firstQuartile = series.quantile(
            0.25
        )

        thirdQuartile = series.quantile(
            0.75
        )

        interquartileRange = (
            thirdQuartile
            - firstQuartile
        )

        lowerBound = (
            firstQuartile
            - threshold * interquartileRange
        )

        upperBound = (
            thirdQuartile
            + threshold * interquartileRange
        )

        return series[
            series.between(
                lowerBound,
                upperBound
            )
        ]

    if method == 'zscore':
        standardDeviation = series.std(
            ddof=0
        )

        if np.isclose(
            standardDeviation,
            0
        ):
            return series

        zScores = (
            series
            - series.mean()
        ) / standardDeviation

        return series[
            zScores.abs() <= threshold
        ]

    raise ValueError(
        'method must be iqr or zscore.'
    )


def isPositiveSemidefinite(
    matrix: np.ndarray | pd.DataFrame,
    tolerance: float = 1e-8
) -> bool:
    '''Check whether a matrix is positive semidefinite.'''
    matrix = validateSquareMatrix(
        matrix,
        name='matrix',
        symmetric=True
    )

    eigenvalues = np.linalg.eigvalsh(
        matrix
    )

    return bool(
        np.all(
            eigenvalues >= -tolerance
        )
    )


def isPositiveDefinite(
    matrix: np.ndarray | pd.DataFrame,
    tolerance: float = 1e-8
) -> bool:
    '''Check whether a matrix is positive definite.'''
    matrix = validateSquareMatrix(
        matrix,
        name='matrix',
        symmetric=True
    )

    eigenvalues = np.linalg.eigvalsh(
        matrix
    )

    return bool(
        np.all(
            eigenvalues > tolerance
        )
    )


def nearestPositiveSemidefinite(
    matrix: np.ndarray | pd.DataFrame,
    epsilon: float = 1e-10
) -> np.ndarray:
    '''Return a nearest positive semidefinite matrix.'''
    matrix = validateSquareMatrix(
        matrix,
        name='matrix'
    )

    epsilon = validatePositiveNumber(
        epsilon,
        name='epsilon',
        allowZero=True
    )

    symmetricMatrix = (
        matrix
        + matrix.T
    ) / 2

    eigenvalues, eigenvectors = np.linalg.eigh(
        symmetricMatrix
    )

    clippedEigenvalues = np.maximum(
        eigenvalues,
        epsilon
    )

    positiveMatrix = (
        eigenvectors
        @ np.diag(clippedEigenvalues)
        @ eigenvectors.T
    )

    positiveMatrix = (
        positiveMatrix
        + positiveMatrix.T
    ) / 2

    return positiveMatrix


def covarianceToCorrelation(
    covarianceMatrix: np.ndarray | pd.DataFrame
) -> np.ndarray | pd.DataFrame:
    '''Convert a covariance matrix into a correlation matrix.'''
    labels = (
        covarianceMatrix.index
        if isinstance(
            covarianceMatrix,
            pd.DataFrame
        )
        else None
    )

    covarianceValues = validateCovarianceMatrix(
        covarianceMatrix
    )

    standardDeviations = np.sqrt(
        np.diag(
            covarianceValues
        )
    )

    denominator = np.outer(
        standardDeviations,
        standardDeviations
    )

    correlationValues = safeDivide(
        covarianceValues,
        denominator,
        default=0.0
    )

    np.fill_diagonal(
        correlationValues,
        1.0
    )

    correlationValues = np.clip(
        correlationValues,
        -1,
        1
    )

    if labels is not None:
        return pd.DataFrame(
            correlationValues,
            index=labels,
            columns=labels
        )

    return correlationValues


def correlationToCovariance(
    correlationMatrix: np.ndarray | pd.DataFrame,
    volatilities: Sequence[Real] | np.ndarray | pd.Series
) -> np.ndarray | pd.DataFrame:
    '''Convert correlation and volatility into covariance.'''
    labels = (
        correlationMatrix.index
        if isinstance(
            correlationMatrix,
            pd.DataFrame
        )
        else None
    )

    correlationValues = validateCorrelationMatrix(
        correlationMatrix
    )

    volatilityValues = validateVector(
        volatilities,
        name='volatilities',
        expectedLength=correlationValues.shape[0]
    )

    if np.any(
        volatilityValues < 0
    ):
        raise ValueError(
            'volatilities cannot be negative.'
        )

    covarianceValues = (
        correlationValues
        * np.outer(
            volatilityValues,
            volatilityValues
        )
    )

    if labels is not None:
        return pd.DataFrame(
            covarianceValues,
            index=labels,
            columns=labels
        )

    return covarianceValues


def regularizeCovarianceMatrix(
    covarianceMatrix: np.ndarray | pd.DataFrame,
    shrinkage: float = 0.01
) -> np.ndarray | pd.DataFrame:
    '''Regularize a covariance matrix toward its diagonal.'''
    labels = (
        covarianceMatrix.index
        if isinstance(
            covarianceMatrix,
            pd.DataFrame
        )
        else None
    )

    covarianceValues = validateSquareMatrix(
        covarianceMatrix,
        name='covarianceMatrix',
        symmetric=True
    )

    shrinkage = validateProbability(
        shrinkage,
        name='shrinkage'
    )

    diagonalMatrix = np.diag(
        np.diag(
            covarianceValues
        )
    )

    regularizedMatrix = (
        1 - shrinkage
    ) * covarianceValues + (
        shrinkage
        * diagonalMatrix
    )

    regularizedMatrix = nearestPositiveSemidefinite(
        regularizedMatrix
    )

    if labels is not None:
        return pd.DataFrame(
            regularizedMatrix,
            index=labels,
            columns=labels
        )

    return regularizedMatrix


def matrixConditionNumber(
    matrix: np.ndarray | pd.DataFrame
) -> float:
    '''Calculate the matrix condition number.'''
    matrix = validateSquareMatrix(
        matrix,
        name='matrix'
    )

    return float(
        np.linalg.cond(
            matrix
        )
    )


def stableMatrixInverse(
    matrix: np.ndarray | pd.DataFrame,
    usePseudoInverse: bool = True,
    conditionThreshold: float = 1e12
) -> np.ndarray:
    '''Calculate a stable matrix inverse.'''
    matrix = validateSquareMatrix(
        matrix,
        name='matrix'
    )

    conditionThreshold = validatePositiveNumber(
        conditionThreshold,
        name='conditionThreshold'
    )

    conditionNumber = matrixConditionNumber(
        matrix
    )

    if (
        not np.isfinite(conditionNumber)
        or conditionNumber > conditionThreshold
    ):
        if usePseudoInverse:
            return np.linalg.pinv(
                matrix
            )

        raise np.linalg.LinAlgError(
            'Matrix is singular or ill-conditioned.'
        )

    return np.linalg.inv(
        matrix
    )


def weightedAverage(
    values: Sequence[Real] | np.ndarray | pd.Series,
    weights: Sequence[Real] | np.ndarray | pd.Series,
    normalizeWeights: bool = True
) -> float:
    '''Calculate a weighted average.'''
    valuesArray = validateVector(
        values,
        name='values'
    )

    weightsArray = validateVector(
        weights,
        name='weights',
        expectedLength=len(
            valuesArray
        )
    )

    if normalizeWeights:
        totalWeight = weightsArray.sum()

        if np.isclose(
            totalWeight,
            0
        ):
            raise ValueError(
                'weights cannot sum to zero.'
            )

        weightsArray = (
            weightsArray
            / totalWeight
        )

    return float(
        np.dot(
            valuesArray,
            weightsArray
        )
    )


def weightedCovariance(
    values: pd.DataFrame,
    weights: Sequence[Real] | np.ndarray | pd.Series,
    ddof: int = 0
) -> pd.DataFrame:
    '''Calculate a weighted covariance matrix.'''
    values = validateDataFrame(
        values,
        name='values',
        numeric=True
    ).dropna()

    weightsArray = validateVector(
        weights,
        name='weights',
        expectedLength=len(
            values
        )
    )

    ddof = validateInteger(
        ddof,
        name='ddof',
        minimum=0
    )

    weightSum = weightsArray.sum()

    if np.isclose(
        weightSum,
        0
    ):
        raise ValueError(
            'weights cannot sum to zero.'
        )

    normalizedWeights = (
        weightsArray
        / weightSum
    )

    valuesArray = values.to_numpy(
        dtype=float
    )

    weightedMean = np.average(
        valuesArray,
        axis=0,
        weights=normalizedWeights
    )

    centeredValues = (
        valuesArray
        - weightedMean
    )

    denominator = (
        1
        - ddof
        * np.sum(
            normalizedWeights ** 2
        )
    )

    if denominator <= 0:
        raise ValueError(
            'Invalid weighted covariance denominator.'
        )

    covarianceMatrix = (
        centeredValues.T
        @ (
            centeredValues
            * normalizedWeights[:, None]
        )
    ) / denominator

    return pd.DataFrame(
        covarianceMatrix,
        index=values.columns,
        columns=values.columns
    )


def expandingZScore(
    series: pd.Series,
    minimumPeriods: int = 20
) -> pd.Series:
    '''Calculate expanding z-scores.'''
    series = validateSeries(
        series,
        name='series',
        numeric=True
    )

    minimumPeriods = validateInteger(
        minimumPeriods,
        name='minimumPeriods',
        minimum=2
    )

    expandingMean = series.expanding(
        min_periods=minimumPeriods
    ).mean()

    expandingStandardDeviation = series.expanding(
        min_periods=minimumPeriods
    ).std()

    return safeDivide(
        series - expandingMean,
        expandingStandardDeviation
    )


def rollingZScore(
    series: pd.Series,
    window: int = 63,
    minimumPeriods: int | None = None
) -> pd.Series:
    '''Calculate rolling z-scores.'''
    series = validateSeries(
        series,
        name='series',
        numeric=True
    )

    window = validateInteger(
        window,
        name='window',
        minimum=2
    )

    if minimumPeriods is None:
        minimumPeriods = window

    minimumPeriods = validateInteger(
        minimumPeriods,
        name='minimumPeriods',
        minimum=2,
        maximum=window
    )

    rollingMean = series.rolling(
        window=window,
        min_periods=minimumPeriods
    ).mean()

    rollingStandardDeviation = series.rolling(
        window=window,
        min_periods=minimumPeriods
    ).std()

    return safeDivide(
        series - rollingMean,
        rollingStandardDeviation
    )


def rollingAnnualizedReturn(
    returns: pd.Series,
    window: int = 252,
    annualizationFactor: int = 252
) -> pd.Series:
    '''Calculate rolling annualized compounded returns.'''
    returns = validateSeries(
        returns,
        name='returns',
        numeric=True
    )

    window = validateInteger(
        window,
        name='window',
        minimum=2
    )

    annualizationFactor = validateAnnualizationFactor(
        annualizationFactor
    )

    def annualizeWindow(
        windowReturns: np.ndarray
    ) -> float:
        validReturns = windowReturns[
            np.isfinite(
                windowReturns
            )
        ]

        if len(validReturns) == 0:
            return np.nan

        cumulativeReturn = np.prod(
            1 + validReturns
        )

        if cumulativeReturn <= 0:
            return np.nan

        return (
            cumulativeReturn ** (
                annualizationFactor
                / len(validReturns)
            )
            - 1
        )

    return returns.rolling(
        window=window,
        min_periods=window
    ).apply(
        annualizeWindow,
        raw=True
    )


def rollingAnnualizedVolatility(
    returns: pd.Series,
    window: int = 63,
    annualizationFactor: int = 252
) -> pd.Series:
    '''Calculate rolling annualized volatility.'''
    returns = validateSeries(
        returns,
        name='returns',
        numeric=True
    )

    window = validateInteger(
        window,
        name='window',
        minimum=2
    )

    annualizationFactor = validateAnnualizationFactor(
        annualizationFactor
    )

    return returns.rolling(
        window=window,
        min_periods=window
    ).std() * np.sqrt(
        annualizationFactor
    )


def rollingSharpeRatio(
    returns: pd.Series,
    riskFreeRate: float = 0.0,
    window: int = 63,
    annualizationFactor: int = 252
) -> pd.Series:
    '''Calculate a rolling annualized Sharpe ratio.'''
    returns = validateSeries(
        returns,
        name='returns',
        numeric=True
    )

    riskFreeRate = validateNumber(
        riskFreeRate,
        name='riskFreeRate',
        minimum=-1
    )

    window = validateInteger(
        window,
        name='window',
        minimum=2
    )

    annualizationFactor = validateAnnualizationFactor(
        annualizationFactor
    )

    periodicRiskFreeRate = annualToPeriodicRate(
        riskFreeRate,
        periodsPerYear=annualizationFactor
    )

    excessReturns = (
        returns
        - periodicRiskFreeRate
    )

    rollingMean = excessReturns.rolling(
        window=window,
        min_periods=window
    ).mean()

    rollingStandardDeviation = excessReturns.rolling(
        window=window,
        min_periods=window
    ).std()

    return safeDivide(
        rollingMean,
        rollingStandardDeviation
    ) * np.sqrt(
        annualizationFactor
    )


def inferPeriodsPerYear(
    index: pd.DatetimeIndex
) -> int:
    '''Infer an annualization factor from a DatetimeIndex.'''
    if not isinstance(
        index,
        pd.DatetimeIndex
    ):
        raise TypeError(
            'index must be a DatetimeIndex.'
        )

    if len(index) < 2:
        raise ValueError(
            'index must contain at least two dates.'
        )

    sortedIndex = index.sort_values()

    medianDays = np.median(
        np.diff(
            sortedIndex.values
        ).astype(
            'timedelta64[D]'
        ).astype(
            float
        )
    )

    if medianDays <= 1.5:
        return 252

    if medianDays <= 8:
        return 52

    if medianDays <= 16:
        return 26

    if medianDays <= 35:
        return 12

    if medianDays <= 100:
        return 4

    return 1


def inferDataFrequency(
    index: pd.DatetimeIndex
) -> str:
    '''Infer a human-readable data frequency.'''
    periodsPerYear = inferPeriodsPerYear(
        index
    )

    frequencyMap = {
        252: 'daily',
        52: 'weekly',
        26: 'biweekly',
        12: 'monthly',
        4: 'quarterly',
        1: 'annual'
    }

    return frequencyMap[
        periodsPerYear
    ]


def countTradingDays(
    startDate: date | datetime | pd.Timestamp,
    endDate: date | datetime | pd.Timestamp,
    inclusive: str = 'both'
) -> int:
    '''Count business days between two dates.'''
    startDate, endDate = validateDateRange(
        startDate,
        endDate,
        allowEqual=True,
        allowFuture=True
    )

    validInclusiveValues = {
        'both',
        'neither',
        'left',
        'right'
    }

    if inclusive not in validInclusiveValues:
        raise ValueError(
            'inclusive must be both, neither, left or right.'
        )

    return int(
        len(
            pd.bdate_range(
                start=startDate,
                end=endDate,
                inclusive=inclusive
            )
        )
    )


def yearsBetween(
    startDate: date | datetime | pd.Timestamp,
    endDate: date | datetime | pd.Timestamp,
    dayCountBasis: float = 365.25
) -> float:
    '''Calculate fractional years between two dates.'''
    startDate, endDate = validateDateRange(
        startDate,
        endDate,
        allowEqual=True,
        allowFuture=True
    )

    dayCountBasis = validatePositiveNumber(
        dayCountBasis,
        name='dayCountBasis'
    )

    elapsedDays = (
        endDate
        - startDate
    ).days

    return elapsedDays / dayCountBasis


def businessDaysToYears(
    businessDays: int,
    tradingDaysPerYear: int = 252
) -> float:
    '''Convert business days into fractional years.'''
    businessDays = validateInteger(
        businessDays,
        name='businessDays',
        minimum=0
    )

    tradingDaysPerYear = validateInteger(
        tradingDaysPerYear,
        name='tradingDaysPerYear',
        minimum=1
    )

    return businessDays / tradingDaysPerYear


def yearsToBusinessDays(
    years: Real,
    tradingDaysPerYear: int = 252
) -> int:
    '''Convert fractional years into business days.'''
    years = validatePositiveNumber(
        years,
        name='years',
        allowZero=True
    )

    tradingDaysPerYear = validateInteger(
        tradingDaysPerYear,
        name='tradingDaysPerYear',
        minimum=1
    )

    return int(
        round(
            years
            * tradingDaysPerYear
        )
    )


def calculateDrawdownSeries(
    values: pd.Series
) -> pd.Series:
    '''Calculate drawdowns from a wealth or price Series.'''
    values = validateSeries(
        values,
        name='values',
        numeric=True
    )

    if (
        values.dropna() <= 0
    ).any():
        raise ValueError(
            'values must be positive.'
        )

    runningMaximum = values.cummax()

    return (
        values
        / runningMaximum
        - 1
    )


def maximumDrawdown(
    values: pd.Series
) -> float:
    '''Calculate maximum drawdown from values.'''
    drawdowns = calculateDrawdownSeries(
        values
    )

    return float(
        drawdowns.min()
    )


def drawdownDuration(
    values: pd.Series
) -> pd.Series:
    '''Calculate the duration of each drawdown period.'''
    drawdowns = calculateDrawdownSeries(
        values
    )

    underwater = drawdowns < 0

    groups = (
        ~underwater
    ).cumsum()

    durations = underwater.groupby(
        groups
    ).cumsum()

    return durations.astype(
        int
    )


def longestDrawdownDuration(
    values: pd.Series
) -> int:
    '''Return the longest drawdown duration.'''
    durations = drawdownDuration(
        values
    )

    if durations.empty:
        return 0

    return int(
        durations.max()
    )

FunctionType = TypeVar(
    'FunctionType',
    bound=Callable[..., Any]
)


def dataFrameToCsv(
    dataFrame: pd.DataFrame,
    includeIndex: bool = True,
    encoding: str = 'utf-8'
) -> bytes:
    '''Convert a DataFrame into CSV bytes.'''
    dataFrame = validateDataFrame(
        dataFrame,
        name='dataFrame',
        allowEmpty=True
    )

    validateString(
        encoding,
        name='encoding'
    )

    return dataFrame.to_csv(
        index=includeIndex
    ).encode(
        encoding
    )


def seriesToCsv(
    series: pd.Series,
    includeIndex: bool = True,
    encoding: str = 'utf-8'
) -> bytes:
    '''Convert a Series into CSV bytes.'''
    series = validateSeries(
        series,
        name='series',
        allowEmpty=True
    )

    return series.to_csv(
        index=includeIndex
    ).encode(
        encoding
    )


def dictionaryToJson(
    dictionary: dict[str, Any],
    indent: int = 2,
    sortKeys: bool = False
) -> str:
    '''Convert a dictionary into JSON text.'''
    if not isinstance(
        dictionary,
        dict
    ):
        raise TypeError(
            'dictionary must be a dict.'
        )

    indent = validateInteger(
        indent,
        name='indent',
        minimum=0
    )

    return json.dumps(
        dictionary,
        indent=indent,
        sort_keys=sortKeys,
        default=jsonSerializer
    )


def jsonSerializer(
    value: Any
) -> Any:
    '''Serialize common NumPy and pandas objects.'''
    if isinstance(
        value,
        (
            np.integer,
            np.int8,
            np.int16,
            np.int32,
            np.int64
        )
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float16,
            np.float32,
            np.float64
        )
    ):
        if np.isnan(value):
            return None

        return float(value)

    if isinstance(
        value,
        np.ndarray
    ):
        return value.tolist()

    if isinstance(
        value,
        pd.Series
    ):
        return value.to_dict()

    if isinstance(
        value,
        pd.DataFrame
    ):
        return value.to_dict(
            orient='records'
        )

    if isinstance(
        value,
        (
            pd.Timestamp,
            datetime,
            date
        )
    ):
        return value.isoformat()

    if isinstance(
        value,
        Path
    ):
        return str(value)

    if pd.isna(value):
        return None

    raise TypeError(
        f'Object of type {type(value).__name__} '
        'is not JSON serializable.'
    )


def dataFrameToJson(
    dataFrame: pd.DataFrame,
    orientation: str = 'records',
    indent: int = 2
) -> str:
    '''Convert a DataFrame into JSON text.'''
    dataFrame = validateDataFrame(
        dataFrame,
        name='dataFrame',
        allowEmpty=True
    )

    validOrientations = {
        'split',
        'records',
        'index',
        'columns',
        'values',
        'table'
    }

    if orientation not in validOrientations:
        raise ValueError(
            'Invalid JSON orientation.'
        )

    return dataFrame.to_json(
        orient=orientation,
        indent=indent,
        date_format='iso'
    )


def objectToBytes(
    value: Any,
    encoding: str = 'utf-8'
) -> bytes:
    '''Convert supported objects into bytes.'''
    validateString(
        encoding,
        name='encoding'
    )

    if isinstance(
        value,
        bytes
    ):
        return value

    if isinstance(
        value,
        str
    ):
        return value.encode(
            encoding
        )

    if isinstance(
        value,
        pd.DataFrame
    ):
        return dataFrameToCsv(
            value,
            encoding=encoding
        )

    if isinstance(
        value,
        pd.Series
    ):
        return seriesToCsv(
            value,
            encoding=encoding
        )

    if isinstance(
        value,
        dict
    ):
        return dictionaryToJson(
            value
        ).encode(
            encoding
        )

    raise TypeError(
        'Unsupported object type.'
    )


def saveTextFile(
    content: str,
    filePath: str | Path,
    encoding: str = 'utf-8'
) -> Path:
    '''Save text content to a file.'''
    if not isinstance(
        content,
        str
    ):
        raise TypeError(
            'content must be a string.'
        )

    filePath = Path(
        filePath
    )

    filePath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    filePath.write_text(
        content,
        encoding=encoding
    )

    return filePath


def saveDataFrame(
    dataFrame: pd.DataFrame,
    filePath: str | Path,
    includeIndex: bool = True
) -> Path:
    '''Save a DataFrame as CSV.'''
    dataFrame = validateDataFrame(
        dataFrame,
        name='dataFrame',
        allowEmpty=True
    )

    filePath = Path(
        filePath
    )

    filePath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataFrame.to_csv(
        filePath,
        index=includeIndex
    )

    return filePath


def ensureDirectory(
    directoryPath: str | Path
) -> Path:
    '''Create a directory when it does not exist.'''
    directoryPath = Path(
        directoryPath
    )

    directoryPath.mkdir(
        parents=True,
        exist_ok=True
    )

    return directoryPath


def fileExists(
    filePath: str | Path
) -> bool:
    '''Check whether a file exists.'''
    return Path(
        filePath
    ).is_file()


def directoryExists(
    directoryPath: str | Path
) -> bool:
    '''Check whether a directory exists.'''
    return Path(
        directoryPath
    ).is_dir()


def downloadButton(
    data: Any,
    fileName: str,
    label: str = 'Download',
    mimeType: str | None = None,
    key: str | None = None,
    includeIndex: bool = True,
    useContainerWidth: bool = True
) -> bool:
    '''Render a Streamlit download button.'''
    fileName = validateString(
        fileName,
        name='fileName'
    )

    label = validateString(
        label,
        name='label'
    )

    fileExtension = Path(
        fileName
    ).suffix.lower()

    if isinstance(
        data,
        pd.DataFrame
    ):
        downloadData = dataFrameToCsv(
            data,
            includeIndex=includeIndex
        )

        mimeType = (
            mimeType
            or 'text/csv'
        )

    elif isinstance(
        data,
        pd.Series
    ):
        downloadData = seriesToCsv(
            data,
            includeIndex=includeIndex
        )

        mimeType = (
            mimeType
            or 'text/csv'
        )

    elif isinstance(
        data,
        dict
    ):
        downloadData = dictionaryToJson(
            data
        )

        mimeType = (
            mimeType
            or 'application/json'
        )

    elif isinstance(
        data,
        str
    ):
        downloadData = data

        if mimeType is None:
            mimeType = (
                'application/json'
                if fileExtension == '.json'
                else 'text/plain'
            )

    elif isinstance(
        data,
        bytes
    ):
        downloadData = data

        mimeType = (
            mimeType
            or 'application/octet-stream'
        )

    else:
        raise TypeError(
            'Unsupported download data type.'
        )

    return st.download_button(
        label=label,
        data=downloadData,
        file_name=fileName,
        mime=mimeType,
        key=key,
        use_container_width=useContainerWidth
    )


def csvDownloadButton(
    dataFrame: pd.DataFrame,
    fileName: str = 'data.csv',
    label: str = 'Download CSV',
    key: str | None = None,
    includeIndex: bool = True
) -> bool:
    '''Render a CSV download button.'''
    return downloadButton(
        data=dataFrame,
        fileName=fileName,
        label=label,
        mimeType='text/csv',
        key=key,
        includeIndex=includeIndex
    )


def jsonDownloadButton(
    dictionary: dict[str, Any],
    fileName: str = 'data.json',
    label: str = 'Download JSON',
    key: str | None = None
) -> bool:
    '''Render a JSON download button.'''
    return downloadButton(
        data=dictionary,
        fileName=fileName,
        label=label,
        mimeType='application/json',
        key=key
    )


def cacheData(
    ttl: int | None = None,
    maximumEntries: int | None = None,
    showSpinner: bool | str = False
) -> Callable[[FunctionType], FunctionType]:
    '''Create a Streamlit data cache decorator.'''
    if ttl is not None:
        ttl = validateInteger(
            ttl,
            name='ttl',
            minimum=1
        )

    if maximumEntries is not None:
        maximumEntries = validateInteger(
            maximumEntries,
            name='maximumEntries',
            minimum=1
        )

    return st.cache_data(
        ttl=ttl,
        max_entries=maximumEntries,
        show_spinner=showSpinner
    )


def cacheResource(
    ttl: int | None = None,
    maximumEntries: int | None = None,
    showSpinner: bool | str = False
) -> Callable[[FunctionType], FunctionType]:
    '''Create a Streamlit resource cache decorator.'''
    if ttl is not None:
        ttl = validateInteger(
            ttl,
            name='ttl',
            minimum=1
        )

    if maximumEntries is not None:
        maximumEntries = validateInteger(
            maximumEntries,
            name='maximumEntries',
            minimum=1
        )

    return st.cache_resource(
        ttl=ttl,
        max_entries=maximumEntries,
        show_spinner=showSpinner
    )


def clearDataCache() -> None:
    '''Clear Streamlit data cache.'''
    st.cache_data.clear()


def clearResourceCache() -> None:
    '''Clear Streamlit resource cache.'''
    st.cache_resource.clear()


def clearAllCaches() -> None:
    '''Clear all Streamlit caches.'''
    clearDataCache()
    clearResourceCache()


def renderCacheClearButton(
    label: str = 'Clear cache',
    key: str = 'clearCacheButton'
) -> bool:
    '''Render a button that clears all caches.'''
    clicked = st.button(
        label,
        key=key,
        use_container_width=True
    )

    if clicked:
        clearAllCaches()

        st.success(
            'Cache cleared successfully.'
        )

    return clicked


def initializeSessionState(
    defaults: dict[str, Any]
) -> None:
    '''Initialize missing Streamlit session-state values.'''
    if not isinstance(
        defaults,
        dict
    ):
        raise TypeError(
            'defaults must be a dict.'
        )

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def getSessionState(
    key: str,
    default: Any = None
) -> Any:
    '''Return a session-state value.'''
    key = validateString(
        key,
        name='key'
    )

    return st.session_state.get(
        key,
        default
    )


def setSessionState(
    key: str,
    value: Any
) -> None:
    '''Set a session-state value.'''
    key = validateString(
        key,
        name='key'
    )

    st.session_state[key] = value


def removeSessionState(
    key: str
) -> None:
    '''Remove a session-state value.'''
    key = validateString(
        key,
        name='key'
    )

    if key in st.session_state:
        del st.session_state[key]


def resetSessionState(
    preserveKeys: Sequence[str] | None = None
) -> None:
    '''Reset Streamlit session state.'''
    preserveKeys = set(
        preserveKeys
        or []
    )

    keysToRemove = [
        key
        for key in st.session_state.keys()
        if key not in preserveKeys
    ]

    for key in keysToRemove:
        del st.session_state[key]


def displayException(
    error: Exception,
    title: str = 'An error occurred',
    showDetails: bool = False
) -> None:
    '''Display a formatted Streamlit exception.'''
    if not isinstance(
        error,
        Exception
    ):
        raise TypeError(
            'error must be an Exception.'
        )

    st.error(
        f'{title}: {error}'
    )

    if showDetails:
        st.exception(
            error
        )


def handleStreamlitError(
    title: str = 'An error occurred',
    showDetails: bool = False,
    stopExecution: bool = False
) -> Callable[[FunctionType], FunctionType]:
    '''Handle exceptions raised by a Streamlit function.'''
    def decorator(
        function: FunctionType
    ) -> FunctionType:
        @wraps(
            function
        )
        def wrapper(
            *args: Any,
            **kwargs: Any
        ) -> Any:
            try:
                return function(
                    *args,
                    **kwargs
                )

            except Exception as error:
                displayException(
                    error,
                    title=title,
                    showDetails=showDetails
                )

                if stopExecution:
                    st.stop()

                return None

        return wrapper

    return decorator


def executeSafely(
    function: Callable[..., Any],
    *args: Any,
    default: Any = None,
    errorMessage: str | None = None,
    showError: bool = False,
    **kwargs: Any
) -> Any:
    '''Execute a callable and return a default on failure.'''
    if not callable(
        function
    ):
        raise TypeError(
            'function must be callable.'
        )

    try:
        return function(
            *args,
            **kwargs
        )

    except Exception as error:
        if showError:
            message = (
                errorMessage
                or str(error)
            )

            st.error(
                message
            )

        return default


def renderLoadingMessage(
    message: str = 'Loading...'
) -> Any:
    '''Create a Streamlit spinner context.'''
    message = validateString(
        message,
        name='message'
    )

    return st.spinner(
        message
    )


def renderEmptyState(
    title: str,
    message: str,
    icon: str = 'ℹ️'
) -> None:
    '''Render a simple empty-state message.'''
    title = validateString(
        title,
        name='title'
    )

    message = validateString(
        message,
        name='message'
    )

    st.info(
        f'{icon} {title}\n\n{message}'
    )


def renderMetricRow(
    metrics: Sequence[dict[str, Any]]
) -> None:
    '''Render multiple Streamlit metrics in columns.'''
    metrics = validateIterable(
        metrics,
        name='metrics'
    )

    columns = st.columns(
        len(metrics)
    )

    for column, metric in zip(
        columns,
        metrics
    ):
        if not isinstance(
            metric,
            dict
        ):
            raise TypeError(
                'Each metric must be a dict.'
            )

        with column:
            st.metric(
                label=metric.get(
                    'label',
                    ''
                ),
                value=metric.get(
                    'value',
                    'N/A'
                ),
                delta=metric.get(
                    'delta'
                ),
                help=metric.get(
                    'help'
                )
            )


def renderDataFrame(
    dataFrame: pd.DataFrame,
    height: int | None = None,
    hideIndex: bool = False,
    useContainerWidth: bool = True
) -> None:
    '''Render a DataFrame with consistent settings.'''
    dataFrame = validateDataFrame(
        dataFrame,
        name='dataFrame',
        allowEmpty=True
    )

    st.dataframe(
        dataFrame,
        height=height if height is not None else 'content',
        hide_index=hideIndex,
        use_container_width=useContainerWidth
    )


def renderSectionTitle(
    title: str,
    description: str | None = None
) -> None:
    '''Render a section title and optional description.'''
    title = validateString(
        title,
        name='title'
    )

    st.subheader(
        title
    )

    if description:
        st.caption(
            description
        )


def validateUploadedFile(
    uploadedFile: Any,
    allowedExtensions: Sequence[str] | None = None,
    maximumSizeMb: float | None = None
) -> Any:
    '''Validate a Streamlit uploaded file.'''
    if uploadedFile is None:
        raise ValueError(
            'No file was uploaded.'
        )

    if allowedExtensions is not None:
        allowedExtensions = [
            extension.lower().lstrip(
                '.'
            )
            for extension in allowedExtensions
        ]

        fileExtension = Path(
            uploadedFile.name
        ).suffix.lower().lstrip(
            '.'
        )

        if fileExtension not in allowedExtensions:
            raise ValueError(
                'Unsupported file extension.'
            )

    if maximumSizeMb is not None:
        maximumSizeMb = validatePositiveNumber(
            maximumSizeMb,
            name='maximumSizeMb'
        )

        fileSizeMb = uploadedFile.size / (
            1024 ** 2
        )

        if fileSizeMb > maximumSizeMb:
            raise ValueError(
                f'File exceeds {maximumSizeMb:.2f} MB.'
            )

    return uploadedFile


def uploadedCsvToDataFrame(
    uploadedFile: Any,
    indexColumn: str | int | None = None,
    parseDates: bool | list[str] = False
) -> pd.DataFrame:
    '''Read an uploaded CSV file into a DataFrame.'''
    uploadedFile = validateUploadedFile(
        uploadedFile,
        allowedExtensions=[
            'csv'
        ]
    )

    try:
        return pd.read_csv(
            uploadedFile,
            index_col=indexColumn,
            parse_dates=parseDates
        )

    except Exception as error:
        raise ValueError(
            'Unable to read uploaded CSV file.'
        ) from error


def uploadedJsonToDictionary(
    uploadedFile: Any
) -> dict[str, Any]:
    '''Read an uploaded JSON file into a dictionary.'''
    uploadedFile = validateUploadedFile(
        uploadedFile,
        allowedExtensions=[
            'json'
        ]
    )

    try:
        content = uploadedFile.read()

        if isinstance(
            content,
            bytes
        ):
            content = content.decode(
                'utf-8'
            )

        result = json.loads(
            content
        )

    except Exception as error:
        raise ValueError(
            'Unable to read uploaded JSON file.'
        ) from error

    if not isinstance(
        result,
        dict
    ):
        raise ValueError(
            'Uploaded JSON must contain an object.'
        )

    return result


def bytesBuffer(
    data: bytes | str,
    encoding: str = 'utf-8'
) -> io.BytesIO:
    '''Create an in-memory bytes buffer.'''
    if isinstance(
        data,
        str
    ):
        data = data.encode(
            encoding
        )

    if not isinstance(
        data,
        bytes
    ):
        raise TypeError(
            'data must be bytes or a string.'
        )

    return io.BytesIO(
        data
    )