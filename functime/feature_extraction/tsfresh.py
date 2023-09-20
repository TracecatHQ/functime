from typing import List, Mapping, Union

import polars as pl

TIME_SERIES_T = Union[pl.Series, pl.Expr]


def absolute_energy(x: TIME_SERIES_T) -> float:
    """
    Compute the absolute energy of a time series.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float
    """
    return x.dot(x)


def absolute_maximum(x: TIME_SERIES_T) -> float:
    """
    Compute the absolute maximum of a time series.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float
    """
    return x.abs().max()


def absolute_sum_of_changes(x: TIME_SERIES_T) -> float:
    """
    Compute the absolute sum of changes of a time series.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float
    """
    return x.diff(n=1, null_behavior="drop").abs().sum()


def linear_trend(x: TIME_SERIES_T) -> Mapping[str, float]:
    """
    Compute the slope, intercept, and RSS of the linear trend.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    dict
        A dictionary containing OLS results.
    """
    x_range = pl.int_range(1, x.len() + 1)
    beta = pl.cov(x, x_range) / x.var()
    alpha = x.mean() - beta * x_range.mean()
    resid = x - beta * x_range + alpha
    rss = resid.pow(2).sum()
    return {"slope": beta, "intercept": alpha, "rss": rss}


def change_quantiles(
    x: TIME_SERIES_T, ql: float, qh: float, is_abs: bool
) -> List[float]:
    """First fixes a corridor given by the quantiles ql and qh of the distribution of x.
    Then calculates the average, absolute value of consecutive changes of the series x inside this corridor.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        A single time-series.
    ql : float
        The lower quantile of the corridor. Must be less than `qh`.
    qh : float
        The upper quantile of the corridor. Must be greater than `ql`.
    is_abs : bool
        If True, takes absolute difference.

    Returns
    -------
    """
    x = x.diff()
    if is_abs:
        x = x.abs()
    x = x.filter(
        x.is_between(
            x.quantile(ql, interpolation="linear"),
            x.quantile(qh, interpolation="linear"),
        ).and_(
            x.is_between(
                x.quantile(ql, interpolation="linear"),
                x.quantile(qh, interpolation="linear"),
            ).shift_and_fill(fill_value=False, periods=1)
        )
    )
    return x


def mean_abs_change(x: TIME_SERIES_T) -> float:
    """
    Compute mean absolute change.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        A single time-series.

    Returns
    -------
    """
    return x.diff(null_behavior="drop").abs().mean()


def mean_change(x: TIME_SERIES_T) -> float:
    """
    Compute mean change.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        A single time-series.

    Returns
    -------
    """
    return x.diff(null_behavior="drop").mean()


def number_crossing_m(x: TIME_SERIES_T, m: float) -> float:
    """
    Calculates the number of crossings of x on m. A crossing is defined as two sequential values where the first value
    is lower than m and the next is greater, or vice-versa. If you set m to zero, you will get the number of zero
    crossings.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        A single time-series.
    m : float
        The crossing value.

    Returns
    -------
    """
    return x.gt(m).cast(pl.Int8).diff(null_behavior="drop").abs().eq(1).sum()


def var_greater_than_std(x: TIME_SERIES_T) -> bool:
    """
    Is variance higher than the standard deviation?

    Boolean variable denoting if the variance of x is greater than its standard deviation. Is equal to variance of x
    being larger than 1

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    """
    y = x.var(ddof=0)
    return y > y.sqrt()


def first_location_of_maximum(x: TIME_SERIES_T) -> float:
    """
    Returns the first location of the maximum value of x.
    The position is calculated relatively to the length of x.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    """
    return x.arg_max() / x.len()


def last_location_of_maximum(x: TIME_SERIES_T) -> float:
    """
    Returns the last location of the maximum value of x.
    The position is calculated relatively to the length of x.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    """
    return (x.len() - x.reverse().arg_max()) / x.len()


def first_location_of_minimum(x: TIME_SERIES_T) -> float:
    """
    Returns the first location of the minimum value of x.
    The position is calculated relatively to the length of x.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    """
    return x.arg_min() / x.len()


def last_location_of_minimum(x: TIME_SERIES_T) -> float:
    """
    Returns the last location of the minimum value of x.
    The position is calculated relatively to the length of x.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    """
    return (x.len() - x.reverse().arg_min()) / x.len()


def autocorr(x: TIME_SERIES_T, lag: int) -> float:
    """Calculate the autocorrelation for a specified lag.

    The autocorrelation measures the linear dependence between a time-series and a lagged version of itself.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    lag : int
        The lag at which to calculate the autocorrelation. Must be a non-negative integer.

    Returns
    -------
    float | None
        Autocorrelation at the given lag. Returns None, if lag is less than 0.

    Notes
    -----
    - This function calculates the autocorrelation using https://en.wikipedia.org/wiki/Autocorrelation#Estimation
    - If `lag` is 0, the autocorrelation is always 1.0, as it represents the correlation of the timeseries with itself.
    """
    if lag < 0:
        return None

    if lag == 0:
        return pl.lit(1.0)

    return (
        x.shift(periods=-lag)
        .drop_nulls()
        .sub(x.mean())
        .dot(x.shift(periods=lag).drop_nulls().sub(x.mean()))
        .truediv((x.count() - lag).mul(x.var(ddof=0)))
    )


def count_below(x: TIME_SERIES_T, t: float) -> float:
    """Calculate the percentage of values below or equal to a threshold.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    t : float
        The threshold value for comparison.

    Returns
    -------
    float
        The percentage of values in `x` that are less than or equal to `t`.
    """
    return x.filter(x <= t).count().truediv(x.count()).mul(100)


def count_above(x: TIME_SERIES_T, t: float) -> float:
    """Calculate the percentage of values above or equal to a threshold.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    t : float
        The threshold value for comparison.

    Returns
    -------
    float
        The percentage of values in `x` that are greater than or equal to `t`.
    """
    return x.filter(x >= t).count().truediv(x.count()).mul(100)


def count_below_mean(x: pl.Expr) -> int:
    """Count the number of values that are below the mean.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    int
        The count of values in `x` that are below the mean.
    """
    return x.filter(x < x.mean()).count()


def count_above_mean(x: TIME_SERIES_T) -> int:
    """Count the number of values that are above the mean.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    int
        The count of values in `x` that are above the mean.
    """
    return x.filter(x > x.mean()).count()


def has_duplicate(x: TIME_SERIES_T) -> bool:
    """Check if the time-series contains any duplicate values.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    bool
        A boolean indicating whether there are duplicate values in `x`.
        Returns True if duplicates exist, otherwise False.
    """
    return x.is_duplicated().any()


def _has_duplicate_of_value(x: TIME_SERIES_T, t: float) -> bool:
    """Check if a value exists as a duplicate.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    t : float
        The value to check for duplicates of.

    Returns
    -------
    bool
    """
    return x.filter(x == t).is_duplicated().any()


def has_duplicate_max(x: TIME_SERIES_T) -> bool:
    """Check if the time-series contains any duplicate values equal to its maximum value.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    bool
    """
    return _has_duplicate_of_value(x, x.max())


def has_duplicate_min(x: TIME_SERIES_T) -> pl.Expr:
    """Check if the time-series contains duplicate values equal to its minimum value.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    bool
    """
    return _has_duplicate_of_value(x, x.min())


def benfords_correlation(x: TIME_SERIES_T) -> float:
    """Returns the correlation between the first digit distribution of the input time series and the Newcomb-Benford's Law distribution [1][2].

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float
        The value of this feature.

    Notes
    -----
    The Newcomb-Benford distribution for d that is the leading digit of the number {1, 2, 3, 4, 5, 6, 7, 8, 9} is given by:

    .. math::

        P(d) = \\log_{10}\\left(1 + \\frac{1}{d}\\right)

    References
    ----------
    [1] Hill, T. P. (1995). A Statistical Derivation of the Significant-Digit Law. Statistical Science.
    [2] Hill, T. P. (1995). The significant-digit phenomenon. The American Mathematical Monthly.
    [3] Benford, F. (1938). The law of anomalous numbers. Proceedings of the American philosophical society.
    [4] Newcomb, S. (1881). Note on the frequency of use of the different digits in natural numbers. American Journal of
        mathematics.
    """
    # TODO: Can be sped up using df.select(pl.col("value").cast(pl.Utf8).str.slice(0,1).cast(pl.UInt8))
    X = (x / (10 ** x.abs().log10().floor())).abs().floor()
    df_corr = pl.DataFrame(
        [
            [X.eq(i).mean() for i in pl.int_range(1, 10, eager=True)],
            (1 + 1 / pl.int_range(1, 10, eager=True)).log10(),
        ]
    ).corr()
    return df_corr[0, 1]


def _get_length_sequences_where(x: pl.Series) -> pl.Series:
    """Calculates the length of all sub-sequences where the series x is either True or 1.

    Parameters
    ----------
    x : pl.Series
        A series containing only 1, True, 0 and False values.

    Returns
    -------
    pl.Series
        A series with the length of all sub-sequences where the series is either True or False.
        If no ones or Trues contained, the list [0] is returned.
    """
    lengths = (
        x.alias("orig")
        .to_frame()
        .with_columns(shift=pl.col("orig").shift(periods=1))
        .with_columns(mask=pl.col("orig").ne(pl.col("shift")).fill_null(0).cumsum())
        .filter(pl.col("orig") == 1)
        .group_by(pl.col("mask"), maintain_order=True)
        .count()
        .get_column("count")
    )
    return lengths


def longest_strike_below_mean(x: pl.Series) -> float:
    """Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x.

    Parameters
    ----------
    x : pl.Series
        Input time-series.

    Returns
    -------
    float
    """
    lengths = _get_length_sequences_where(x.cast(pl.Float64) < x.mean())
    strike = lengths.max() if lengths.len() > 0 else 0
    return strike


def longest_strike_above_mean(x: pl.Series) -> float:
    """
    Returns the length of the longest consecutive subsequence in x that is greater than the mean of x.

    Parameters
    ----------
    x : pl.Series
        Input time-series.

    Returns
    -------
    float
    """
    lengths = _get_length_sequences_where(x.cast(pl.Float64) > x.mean())
    strike = lengths.max() if lengths.len() > 0 else 0
    return strike


def mean_n_absolute_max(x: pl.Series, n_maxima: int) -> float:
    """
    Calculates the arithmetic mean of the n absolute maximum values of the time series.

    Parameters
    ----------
    x : pl.Series
        Input time-series.
    n_maxima : int
        The number of maxima to consider.

    Returns
    -------
    float
        The value of this feature.
    """
    if n_maxima <= 0:
        raise ValueError("The number of maxima should be > 0.")
    return (
        x.abs().sort(descending=True)[:n_maxima].mean() if x.len() > n_maxima else None
    )


def percent_reocurring_points(x: pl.Series) -> float:
    """
    Returns the percentage of non-unique data points in the time series. Non-unique data points are those that occur
    more than once in the time series.

    The percentage is calculated as follows:

        # of data points occurring more than once / # of all data points

    This means the ratio is normalized to the number of data points in the time series, in contrast to the
    `percent_recoccuring_values` function.

    Parameters
    ----------
    x : pl.Series
        Input time-series.

    Returns
    -------
    float
        The value of this feature.
    """
    counts = x.value_counts().filter(pl.col("counts") > 1).sum()
    return counts.item(0, "counts") / x.len()


def percent_recoccuring_values(x: pl.Series) -> float:
    """
    Returns the percentage of values that are present in the time series more than once.

    The percentage is calculated as follows:

        len(different values occurring more than once) / len(different values)

    This means the percentage is normalized to the number of unique values in the time series, in contrast to the
    `percent_reocurring_points` function.

    Parameters
    ----------
    x : pl.Series
        Input time-series.

    Returns
    -------
    float
        The value of this feature.
    """
    counts = x.value_counts().filter(pl.col("counts") > 1)
    return counts.shape[0] / x.n_unique()


def sum_reocurring_points(x: pl.Series) -> float:
    """
    Returns the sum of all data points that are present in the time series more than once.

    For example, `sum_reocurring_points(pl.Series([2, 2, 2, 2, 1]))` returns 8, as 2 is a reoccurring value, so all 2's
    are summed up.

    This is in contrast to the `sum_reocurring_values` function, where each reoccuring value is only counted once.

    Parameters
    ----------
    x : pl.Series
        Input time-series.

    Returns
    -------
    float
        The value of this feature.
    """
    counts = x.value_counts().filter(pl.col("counts") > 1)
    return counts.get_column(counts.columns[0]).dot(
        counts.get_column(counts.columns[0])
    )


def sum_reocurring_values(x: pl.Series) -> float:
    """
    Returns the sum of all values that are present in the time series more than once.

    For example, `sum_reocurring_values(pl.Series([2, 2, 2, 2, 1]))` returns 2, as 2 is a reoccurring value, so it is
    summed up with all other reoccuring values (there is none), so the result is 2.

    This is in contrast to the `sum_reocurring_points` function, where each reoccuring value is only counted as often
    as it is present in the data.

    Parameters
    ----------
    x : pl.Series
        Input time-series.

    Returns
    -------
    float
        The value of this feature.
    """
    X = x.value_counts().filter(pl.col("counts") > 1).sum()
    return X.item()


def mean_second_derivative_central(x: pl.Series) -> float:
    """
    Returns the mean value of a central approximation of the second derivative.

    .. math::

    \\frac{1}{2(n-2)} \\sum_{i=1}^{n-1} 0.5 (x_{i+2} - 2 x_{i+1} + x_i)

    where n is the length of the time series x

    Parameters
    ----------
    x : pl.Series
        A time series to calculate the feature of.

    Returns
    -------
    float
        The value of the central approximation of the second derivative of x.
    """
    return (x[-1] - x[-2] - x[1] + x[0]) / (2 * (len(x) - 2))


def symmetry_looking(x: pl.Series, r: float) -> bool:
    """Check if the distribution of x looks symmetric.

    A distribution is considered symmetric if:

    .. math::

    | mean(X)-median(X) | < r * (max(X)-min(X))

    Parameters
    ----------
    x : polars.Series
        Input time-series.
    r : float
        Multiplier on distance between max and min.

    Returns
    -------
    bool
    """
    mean_median_difference = abs(x.mean() - x.median())
    max_min_difference = x.max() - x.min()
    return mean_median_difference < r["r"] * max_min_difference


def time_reversal_asymmetry_statistic(x: pl.Series, lag: int) -> float:
    """
    Returns the time reversal asymmetry statistic.

    This function calculates the value of:

    .. math::

        \\frac{1}{n-2lag} \\sum_{i=1}^{n-2lag} x_{i + 2 \\cdot lag}^2 \\cdot x_{i + lag} - x_{i + lag} \\cdot  x_{i}^2

    which is

    .. math::

        \\mathbb{E}[L^2(X)^2 \\cdot L(X) - L(X) \\cdot X^2]

    where :math:`\\mathbb{E}` is the mean and :math:`L` is the lag operator. It was proposed in [1] as a
    promising feature to extract from time series.

    Parameters
    ----------
    x : pl.Series
        Input time-series.
    lag : int
        The lag that should be used in the calculation of the feature.

    Returns
    -------
    float

    References
    ----------
    [1] Fulcher, B.D., Jones, N.S. (2014). Highly comparative feature-based time-series classification.
        Knowledge and Data Engineering, IEEE Transactions on 26, 3026–3037.
    """
    n = len(x)
    one_lag = x.shift(-lag)
    two_lag = x.shift(-2 * lag)
    result = (two_lag * two_lag * one_lag - one_lag * x * x).head(n - 2 * lag).mean()
    return result
