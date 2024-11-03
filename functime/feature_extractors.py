from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union

import numpy as np
import polars as pl
from polars.type_aliases import ClosedInterval

# from numpy.linalg import lstsq
from scipy.linalg import lstsq
from scipy.signal import find_peaks_cwt, ricker, welch
from scipy.spatial import KDTree

from functime._compat import register_plugin_function, rle_fields
from functime._functime_rust import rs_faer_lstsq1
from functime._utils import warn_is_unstable
from functime.type_aliases import DetrendMethod

# from functime.feature_extractor import FeatureExtractor  # noqa: F401

logger = logging.getLogger(__name__)

TIME_SERIES_T = Union[pl.Series, pl.Expr]
FLOAT_EXPR = Union[float, pl.Expr]
FLOAT_INT_EXPR = Union[int, float, pl.Expr]
INT_EXPR = Union[int, pl.Expr]
LIST_EXPR = Union[list, pl.Expr]
BOOL_EXPR = Union[bool, pl.Expr]
MAP_EXPR = Union[Mapping[str, float], pl.Expr]
MAP_LIST_EXPR = Union[Mapping[str, List[float]], pl.Expr]


# from polars.type_aliases import IntoExpr

try:
    from polars.utils.udfs import _get_shared_lib_location

    lib = _get_shared_lib_location(__file__)
except ImportError:
    lib = Path(__file__).parent


def absolute_energy(x: TIME_SERIES_T) -> FLOAT_INT_EXPR:
    """
    Compute the absolute energy of a time series.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float | Expr
    """
    if isinstance(x, pl.Series):
        if x.len() > 300_000:  # Would be different for different machines
            return np.dot(x, x)
    return x.dot(x)


def absolute_maximum(x: TIME_SERIES_T) -> FLOAT_INT_EXPR:
    """
    Compute the absolute maximum of a time series.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float | Expr
    """
    if isinstance(x, pl.Series):
        return np.max(np.abs([x.min(), x.max()]))
    else:
        return pl.max_horizontal(x.min().abs(), x.max().abs())


def absolute_sum_of_changes(x: TIME_SERIES_T) -> FLOAT_INT_EXPR:
    """
    Compute the absolute sum of changes of a time series.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float | Expr
    """
    return x.diff(n=1, null_behavior="drop").abs().sum()


def approximate_entropy(
    x: TIME_SERIES_T, run_length: int, filtering_level: float, scale_by_std: bool = True
) -> float:
    """
    Approximate sample entropies of a time series given the filtering level.
    This only works for Series input right now.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    run_length : int
        Length of compared run of data. This is `m` in the wikipedia article.
    filtering_level : float
        Filtering level, must be positive. This is `r` in the wikipedia article.
    scale_by_std : bool
        Whether to scale filter level by std of data. In most applications, this is the default
        behavior, but not in some other cases.

    Returns
    -------
    float
    """

    if filtering_level <= 0:
        raise ValueError("Filter level must be positive.")

    if scale_by_std:
        r = filtering_level * x.std()
    else:
        r = filtering_level

    if isinstance(x, pl.Series):
        if len(x) < run_length + 1:
            return 0

        frame = x.to_frame()
        # Shared between phi_m and phi_m+1
        data = frame.with_columns(
            pl.col(x.name).shift(-i).alias(str(i)) for i in range(1, run_length + 1)
        ).to_numpy()

        n1 = len(x) - run_length + 1
        data_m = data[:n1, :run_length]
        # Computes phi. Let's not make it into a separate function until we know this will be reused.
        tree = KDTree(data_m, leafsize=50)  # Can play with leafsize to fine tune perf
        nb_in_radius: np.ndarray = tree.query_ball_point(
            data_m, r, p=np.inf, workers=-1, return_length=True
        )
        phi_m = np.log(nb_in_radius / n1).sum() / n1

        n2 = n1 - 1
        data_mp1 = data[:n2, :]
        # Compute phi
        tree = KDTree(data_mp1, leafsize=50)
        nb_in_radius: np.ndarray = tree.query_ball_point(
            data_mp1, r, p=np.inf, workers=-1, return_length=True
        )
        phi_mp1 = np.log(nb_in_radius / n2).sum() / n2

        return np.abs(phi_m - phi_mp1)
    else:
        logger.info(
            "Expression version of approximate_entropy is not yet implemented due to "
            "technical difficulty regarding Polars Expression Plugins."
        )
        return NotImplemented


# An alias
ApEn = approximate_entropy


@warn_is_unstable
def augmented_dickey_fuller(x: TIME_SERIES_T, n_lags: int) -> float:
    """
    Calculates the Augmented Dickey-Fuller (ADF) test statistic. This only works for Series input right now.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    n_lags : int
        The number of lags to include in the test.

    Returns
    -------
    float
    """
    if isinstance(x, pl.Series):
        y = x.fill_null(0.0)
        length = y.len() - n_lags - 1
        data_x = y.to_frame().select(
            pl.col(y.name).slice(n_lags, length=length),
            *(
                pl.col(y.name)
                .diff(null_behavior="drop")
                .slice(n_lags - i, length=length)
                .alias(str(i))
                for i in range(0, n_lags + 1)
            ),
            pl.lit(1),
        )  # was a frame
        y = data_x.drop_in_place("0").to_numpy(zero_copy_only=True)
        data_x = data_x.to_numpy()  # to NumPy matrix

        coeffs, resids, _, _ = lstsq(data_x, y, cond=None)
        mse = np.sum(resids**2) / (length - data_x.shape[1])
        ys = data_x[:, 0] - np.mean(data_x[:, 0])
        ss = np.dot(ys, ys)
        stderr = np.sqrt(mse / ss)
        return coeffs[0] / stderr
    else:
        logger.info(
            "Expression version of augmented_dickey_fuller is not yet implemented due to "
            "technical difficulty regarding Polars Expression Plugins."
        )
        return NotImplemented


def autocorrelation(x: TIME_SERIES_T, n_lags: int) -> FLOAT_EXPR:
    """
    Calculate the autocorrelation for a specified lag.

    The autocorrelation measures the linear dependence between a time-series and a lagged version of itself.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    n_lags : int
        The lag at which to calculate the autocorrelation. Must be a non-negative integer.

    Returns
    -------
    float | Expr
        Autocorrelation at the given lag. Returns None, if lag is less than 0.
    """
    if n_lags < 0:
        raise ValueError("Input `n_lags` must be >= 0")
    elif n_lags == 0:
        return 1.0

    mean = x.mean()
    var = x.var(ddof=0)
    range_ = x.len() - n_lags
    y1 = x.slice(0, length=range_) - mean
    y2 = x.slice(n_lags, length=None) - mean
    return y1.dot(y2) / (var * range_)


def autoregressive_coefficients(x: TIME_SERIES_T, n_lags: int) -> List[float]:
    """
    Computes coefficients for an AR(`n_lags`) process. This only works for Series input
    right now. Caution: Any Null Value in Series will replaced by 0!

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    n_lags : int
        The number of lags in the autoregressive process.

    Returns
    -------
    list of float
    """
    if isinstance(x, pl.Series):
        length = len(x) - n_lags
        y = x.fill_null(0.0)
        data_x = (
            y.to_frame()
            .select(
                *(
                    pl.col(y.name).slice(n_lags - i, length=length).alias(str(i))
                    for i in range(1, n_lags + 1)
                ),
                pl.lit(1),
            )
            .to_numpy()
        )
        y_ = y.tail(length).to_numpy(zero_copy_only=True).reshape((-1, 1))
        out: np.ndarray = rs_faer_lstsq1(data_x, y_)
        return out.ravel()
    else:
        logger.info(
            "Expression version of autoregressive_coefficients is not yet implemented due to "
            "technical difficulty regarding Polars Expression Plugins."
        )
        return NotImplemented


_BENFORD_DIST_SERIES = (1 + 1 / pl.int_range(1, 10, eager=True)).log10()


def benford_correlation(x: TIME_SERIES_T) -> FLOAT_EXPR:
    """
    Returns the correlation between the first digit distribution of the input time series and
    the Newcomb-Benford's Law distribution.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float | Expr
    """

    if isinstance(x, pl.Series):
        counts = (
            pl.int_range(1, 10, eager=True, dtype=pl.UInt8)
            .cast(pl.Utf8)
            .append(
                x.cast(pl.Utf8)
                .str.strip_chars_start("-0.")
                .filter(x != 0)
                .str.slice(0, 1)
            )
            .unique_counts()
        )
        return np.corrcoef(counts - 1, _BENFORD_DIST_SERIES)[0, 1]
    else:
        counts = (
            pl.int_range(1, 10, eager=False)
            .cast(pl.Utf8)
            .append(
                x.cast(pl.Utf8)
                .str.strip_chars_start("-0.")
                .filter(x != 0)
                .str.slice(0, 1)
            )
            .unique_counts()
        )
        return pl.corr(counts - 1, pl.lit(_BENFORD_DIST_SERIES))


# def benford_correlation2(x: pl.Expr) -> pl.Expr:
#     """
#     Returns the correlation between the first digit distribution of the input time series and
#     the Newcomb-Benford's Law distribution. This version may hit some float point precision
#     issues for some rare numbers.

#     Parameters
#     ----------
#     x : pl.Expr | pl.Series
#         Input time-series.

#     Returns
#     -------
#     float | Expr
#     """
#     counts = (
#         # This part can be simplified once the log10(1000) precision issue is resolved.
#         pl.when(x.abs() == 1000)
#         .then(pl.lit(1))
#         .otherwise(x.abs() / (pl.lit(10).pow((x.abs().log10()).floor())))
#         .drop_nans()
#         .drop_nulls()
#         .cast(pl.UInt8)
#         .append(pl.int_range(1, 10, eager=False))
#         .value_counts()
#         .sort()
#         .struct.field("count")
#     )
#     return pl.corr(counts - 1, pl.lit(_BENFORD_DIST_SERIES))


def binned_entropy(x: TIME_SERIES_T, bin_count: int = 10) -> FLOAT_EXPR:
    """
    Calculates the entropy of a binned histogram for a given time series.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    bin_count : int, optional
        The number of bins to use in the histogram. Default is 10.

    Returns
    -------
    float | Expr
    """
    if isinstance(x, pl.Series):
        hist, _ = np.histogram(x, bins=bin_count)
        probs = hist / len(x)
        probs[probs == 0] = 1.0
        return -np.sum(probs * np.log(probs))
    else:
        return (
            (x - x.min())
            .floordiv(pl.lit(1e-12) + (x.max() - x.min()) / pl.lit(bin_count))
            .drop_nulls()
            .value_counts()
            .struct.field("count")
            .entropy()
        )


def c3(x: TIME_SERIES_T, n_lags: int) -> FLOAT_EXPR:
    """
    Measure of non-linearity in the time series using c3 statistics.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    n_lags : int
        The lag that should be used in the calculation of the feature.

    Returns
    -------
    float | Expr
    """
    twice_lag = 2 * n_lags
    if isinstance(x, pl.Series):
        if twice_lag >= x.len():
            return np.nan
        else:
            return (x * x.shift(n_lags) * x.shift(twice_lag)).sum() / (
                x.len() - twice_lag
            )
    else:
        return ((x.mul(x.shift(n_lags)).mul(x.shift(twice_lag))).sum()).truediv(
            x.len() - twice_lag
        )


def change_quantiles(
    x: TIME_SERIES_T, q_low: float, q_high: float, is_abs: bool
) -> LIST_EXPR:
    """
    First fixes a corridor given by the quantiles ql and qh of the distribution of x.
    It will return a list of changes coming from consecutive values that both lie within
    the quantile range. The user may optionally get abssolute value of the changes, and
    compute stats from these changes. If q_low >= q_high, it will return null.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        A single time-series.
    q_low : float
        The lower quantile of the corridor. Must be less than `q_high`.
    q_high : float
        The upper quantile of the corridor. Must be greater than `q_low`.
    is_abs : bool
        If True, takes absolute difference.

    Returns
    -------
    list of float | Expr
    """
    if q_high <= q_low:
        return None

    if isinstance(x, pl.Series):
        frame = x.to_frame()
        return frame.select(
            change_quantiles(pl.col(x.name), q_low, q_high, is_abs)
        ).item(0, 0)
    else:
        # Use linear to conform to NumPy
        y = x.is_between(
            x.quantile(q_low, "linear"),
            x.quantile(q_high, "linear"),
        )
        # I tested again, pl.all_horizontal is slightly faster than
        # pl.all_horizontal(y, y.shift_and_fill(False, periods=1))
        expr = x.diff().filter(pl.all_horizontal(y, y.shift(1, fill_value=False)))
        if is_abs:
            expr = expr.abs()

        return expr.implode()


# Revisit later.
def cid_ce(x: TIME_SERIES_T, normalize: bool = False) -> FLOAT_EXPR:
    """
    Computes estimate of time-series complexity[^1].

    A more complex time series has more peaks and valleys.
    This feature is calculated by:

    Parameters
    ----------
    x : pl.Expr | pl.Series
        A single time-series.
    normalize : bool, optional
        If True, z-normalizes the time-series before computing the feature.
        Default is False.

    Returns
    -------
    float | Expr
    """
    if normalize:
        y = (x - x.mean()) / x.std(ddof=0)
    else:
        y = x

    if isinstance(x, pl.Series):
        diff = np.diff(y)
        return np.sqrt(np.dot(diff, diff))
    else:
        z = y - y.shift(-1)
        return (z.dot(z)).sqrt()


def count_above(x: TIME_SERIES_T, threshold: float = 0.0) -> FLOAT_EXPR:
    """
    Calculate the percentage of values above or equal to a threshold.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    threshold : float
        The threshold value for comparison.

    Returns
    -------
    float | Expr
    """
    return 100 * (x >= threshold).sum() / x.len()


# Should this be percentage?
def count_above_mean(x: TIME_SERIES_T) -> INT_EXPR:
    """
    Count the number of values that are above the mean.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    int | Expr
    """
    return (x > x.mean()).sum()


def count_below(x: TIME_SERIES_T, threshold: float = 0.0) -> FLOAT_EXPR:
    """
    Calculate the percentage of values below or equal to a threshold.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    threshold : float
        The threshold value for comparison.

    Returns
    -------
    float | Expr
    """
    return 100 * (x <= threshold).sum() / x.len()


def count_below_mean(x: TIME_SERIES_T) -> INT_EXPR:
    """
    Count the number of values that are below the mean.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    int | Expr
    """
    return (x < x.mean()).sum()


def cwt_coefficients(
    x: TIME_SERIES_T, widths: Sequence[int] = (2, 5, 10, 20), n_coefficients: int = 14
) -> List[float]:
    """
    Calculates a Continuous wavelet transform for the Ricker wavelet.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    widths : Sequence[int], optional
        The widths of the Ricker wavelet to use for the CWT. Default is (2, 5, 10, 20).
    n_coefficients : int, optional
        The number of CWT coefficients to return. Default is 14.

    Returns
    -------
    list of float
    """
    if isinstance(x, pl.Series):
        convolution = np.empty((len(widths), x.len()), dtype=np.float32)
        for i, width in enumerate(widths):
            points = np.min([10 * width, x.len()])
            wavelet_x = np.conj(ricker(points, width)[::-1])
            convolution[i] = np.convolve(
                x.to_numpy(zero_copy_only=True), wavelet_x, mode="same"
            )
        coeffs = []
        for coeff_idx in range(min(n_coefficients, convolution.shape[1])):
            coeffs.extend(convolution[widths.index(w), coeff_idx] for w in widths)
        return coeffs
    else:
        logger.info(
            "Expression version of cwt_coefficients is not yet implemented due to "
            "technical difficulty regarding Polars Expression Plugins."
        )
        NotImplemented


def energy_ratios(x: TIME_SERIES_T, n_chunks: int = 10) -> LIST_EXPR:
    """
    Calculates sum of squares over the whole series for `n_chunks` equally segmented parts of the time-series.
    E.g. if n_chunks = 10, values are [0, 1, 2, 3, .. , 999], the first chunk will be [0, .. , 99].

    Parameters
    ----------
    x : list of float
        The time-series to be segmented and analyzed.
    n_chunks : int, optional
        The number of equally segmented parts to divide the time-series into. Default is 10.

    Returns
    -------
    list of float | Expr
    """
    # Unlike Tsfresh,
    # We calculate all 1,2,3,...,n_chunk at once
    if isinstance(x, pl.Series):
        r = x.len() % n_chunks
        if r == 0:
            y = x.pow(2)
        else:
            y = x.pow(2).extend_constant(0, n_chunks - r)
        seg_sum = y.reshape((n_chunks, -1)).list.sum()
        return (seg_sum / seg_sum.sum()).to_list()
    else:
        r = x.count().mod(n_chunks)
        y = x.pow(2).append(pl.repeat(0, n=r.eq(0).not_() * (pl.lit(n_chunks) - r)))
        seg_sum = y.reshape((n_chunks, -1)).list.sum()
        return (seg_sum / seg_sum.sum()).implode()


def first_location_of_maximum(x: TIME_SERIES_T) -> FLOAT_EXPR:
    """
    Returns the first location of the maximum value of x.
    The position is calculated relatively to the length of x.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float | Expr
    """
    return x.arg_max() / x.len()


def first_location_of_minimum(x: TIME_SERIES_T) -> FLOAT_EXPR:
    """
    Returns the first location of the minimum value of x.
    The position is calculated relatively to the length of x.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float | Expr
    """
    return x.arg_min() / x.len()


def fourier_entropy(x: TIME_SERIES_T, n_bins: int = 10) -> float:
    """
    Calculate the Fourier entropy of a time series. This only works for Series input right now.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    n_bins : int, optional
        The number of bins to use for the entropy calculation. Default is 10.

    Returns
    -------
    float
    """
    if isinstance(x, pl.Series):
        if len(x) == 1:
            return np.nan
        else:
            _, pxx = welch(x, nperseg=min(x.len(), 256))
            pxx_as_series = pl.Series(pxx)
            return binned_entropy(pxx_as_series / pxx_as_series.max(), n_bins)
    else:
        logger.info(
            "Expression version of fourier_entropy is not yet implemented due to "
            "technical difficulty regarding Polars Expression Plugins."
        )
        return NotImplemented


def friedrich_coefficients(
    x: TIME_SERIES_T, polynomial_order: int = 3, n_quantiles: int = 30
) -> LIST_EXPR:
    """
    Calculate the Friedrich coefficients of a time series.

    Parameters
    ----------
    x : TIME_SERIES_T
        The time series to calculate the Friedrich coefficients of.
    polynomial_order : int, optional
        The order of the polynomial to fit to the quantile means. Default is 3.
    n_quantiles : int, optional
        The number of quantiles to use for the calculation. Default is 30.

    Returns
    -------
    list of float
    """
    if isinstance(x, pl.Series):
        x_means = (
            x.alias("signal")
            .to_frame()
            .lazy()
            .with_columns(
                pl.col("signal").diff().alias("delta"),
                pl.col("signal")
                .qcut(q=n_quantiles, labels=[str(i) for i in range(n_quantiles)])
                .alias("quantile"),
            )
            .group_by("quantile")
            .agg(pl.all().mean())
            .drop_nulls()
            .collect()
        )
        # This is a Vandermonde matrix. May have shortcuts in our use case, as again length >> degree.
        # Not sure if NumPy is taking advantage of this though.
        return np.polyfit(
            x_means.get_column("signal").to_numpy(zero_copy_only=True),
            x_means.get_column("delta").to_numpy(zero_copy_only=True),
            deg=polynomial_order,
        )
    else:
        logger.info(
            "Expression version of friedrich_coefficients is not yet implemented due to "
            "technical difficulty regarding Polars Expression Plugins."
        )
        return NotImplemented


def has_duplicate(x: TIME_SERIES_T) -> BOOL_EXPR:
    """
    Check if the time-series contains any duplicate values.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    bool | Expr
    """
    return x.is_duplicated().any()


def has_duplicate_max(x: TIME_SERIES_T) -> BOOL_EXPR:
    """
    Check if the time-series contains any duplicate values equal to its maximum value.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    bool | Expr
    """
    return (x == x.max()).sum() > 1


def has_duplicate_min(x: TIME_SERIES_T) -> BOOL_EXPR:
    """
    Check if the time-series contains duplicate values equal to its minimum value.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    bool | Expr
    """
    return (x == x.min()).sum() > 1


def index_mass_quantile(x: TIME_SERIES_T, q: float) -> FLOAT_EXPR:
    """
    Calculates the relative index i of time series x where q% of the mass of x lies left of i.
    For example for q = 50% this feature calculator will return the mass center of the time series.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    q : float
        The quantile.

    Returns
    -------
    float | Expr
    """
    x_cumsum = x.abs().cum_sum().set_sorted()
    if isinstance(x, pl.Series):
        if x.dtype.is_integer():
            idx = x_cumsum.search_sorted(int(q * x_cumsum[-1]) + 1, "left")
        else:  # Search sorted is sensitive to dtype.
            idx = x_cumsum.search_sorted(q * x_cumsum[-1], "left")
        return (idx + 1) / x.len()
    else:
        # In lazy case we have to cast, which kind of wasts some time, but this is on
        # par with the old implementation. Use max here because... believe it or not
        # max (with fast track because I did set_sorted()) is faster than .last() ???
        idx = x_cumsum.cast(pl.Float64).search_sorted(q * x_cumsum.max(), "left")
        return (idx + 1) / x.len()


def large_standard_deviation(x: TIME_SERIES_T, ratio: float = 0.25) -> BOOL_EXPR:
    """
    Checks if the time-series has a large standard deviation: `std(x) > r * (max(X)-min(X))`.

    As a heuristic, the standard deviation should be a forth of the range of the values.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    ratio : float
        The ratio of the interval to compare with.

    Returns
    -------
    bool | Expr
    """
    x_std = x.std()
    x_interval = x.max() - x.min()
    return x_std > (ratio * x_interval)


def last_location_of_maximum(x: TIME_SERIES_T) -> FLOAT_EXPR:
    """
    Returns the last location of the maximum value of x.
    The position is calculated relatively to the length of x.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float | Expr
    """
    # if isinstance(x, pl.Series):
    #     return ((x == x.max()).arg_true()[-1] + 1) / x.len()
    # else:
    #     return ((x == x.max()).arg_true().last() + 1) / x.len()
    return 1.0 - (x.reverse().arg_max()) / x.len()


def last_location_of_minimum(x: TIME_SERIES_T) -> FLOAT_EXPR:
    """
    Returns the last location of the minimum value of x.
    The position is calculated relatively to the length of x.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float | Expr
    """
    # if isinstance(x, pl.Series):
    #     return ((x == x.min()).arg_true()[-1] + 1) / x.len()
    # else:
    #     return ((x == x.min()).arg_true().last() + 1) / x.len()
    return 1.0 - (x.reverse().arg_min()) / x.len()


def lempel_ziv_complexity(
    x: TIME_SERIES_T, threshold: Union[float, pl.Expr], as_ratio: bool = True
) -> FLOAT_EXPR:
    """
    Calculate a complexity estimate based on the Lempel-Ziv compression algorithm. The
    implementation here is currently a Rust rewrite of Lilian Besson'code. See the reference
    section below. Instead of return the complexity value, we return a ratio w.r.t the length of
    the input series. If null is encountered, it will be interpreted as 0 in the bit sequence.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    threshold: float | pl.Expr
        Either a number, or an expression representing a comparable quantity. If x > value,
        then it will be binarized as 1 and 0 otherwise. If x is eager, then value must also
        be eager as well.
    as_ratio: bool
        If true, return the complexity / length of sequence

    Returns
    -------
    float

    Reference
    ---------
    https://github.com/Naereen/Lempel-Ziv_Complexity/tree/master
    https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv_complexity
    """
    if isinstance(x, pl.Series):
        frame = x.to_frame()
        return frame.select(
            pl.col(x.name).ts.lempel_ziv_complexity(threshold, as_ratio)
        ).item(0, 0)
    else:
        return x.ts.lempel_ziv_complexity(threshold, as_ratio)


def linear_trend(x: TIME_SERIES_T) -> MAP_EXPR:
    """
    Compute the slope, intercept, and RSS of the linear trend.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    Mapping[str, float] | Expr
    """
    if isinstance(x, pl.Series):
        n = x.len()
        if n == 1:
            return {"slope": np.nan, "intercept": np.nan, "rss": np.nan}

        m = n - 1
        x_range_mean = m / 2
        x_range = np.arange(start=0, stop=n)
        x_mean = x.mean()
        beta = np.dot(x - x_mean, x_range - x_range_mean) / (
            m * np.var(x_range, ddof=1)
        )
        alpha = x_mean - beta * x_range_mean
        resid = x - (beta * x_range + alpha)
        return {"slope": beta, "intercept": alpha, "rss": np.dot(resid, resid)}
    else:
        m = x.len() - 1
        x_range = pl.int_range(0, x.len())
        x_range_mean = m / 2
        beta = (x - x.mean()).dot(x_range - x_range_mean) / (m * x_range.var())
        alpha = x.mean() - beta * x_range_mean
        resid = x - (beta * x_range + alpha)
        # When length = 1, the case is handled implicitly, as x_range.var = 0 which makes beta nan
        return pl.struct(
            beta.alias("slope"), alpha.alias("intercept"), resid.dot(resid).alias("rss")
        )


def longest_streak_above_mean(x: TIME_SERIES_T) -> INT_EXPR:
    """
    Returns the length of the longest consecutive subsequence in x that is > mean of x.
    If all values in x are null, 0 will be returned. Note: this does not measure consecutive
    changes in time series, only counts the streak based on the original time series, not the
    differences.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    int | Expr
    """
    y = (x > x.mean()).rle()
    if isinstance(x, pl.Series):
        result = (
            y.filter(y.struct.field(rle_fields["value"]))
            .struct.field(rle_fields["len"])
            .max()
        )
        return 0 if result is None else result
    else:
        return (
            y.filter(y.struct.field(rle_fields["value"]))
            .struct.field(rle_fields["len"])
            .max()
            .fill_null(0)
        )


def longest_streak_below_mean(x: TIME_SERIES_T) -> INT_EXPR:
    """
    Returns the length of the longest consecutive subsequence in x that is < mean of x.
    If all values in x are null, 0 will be returned. Note: this does not measure consecutive
    changes in time series, only counts the streak based on the original time series, not the
    differences.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    int | Expr
    """
    y = (x < x.mean()).rle()
    if isinstance(x, pl.Series):
        result = (
            y.filter(y.struct.field(rle_fields["value"]))
            .struct.field(rle_fields["len"])
            .max()
        )
        return 0 if result is None else result
    else:
        return (
            y.filter(y.struct.field(rle_fields["value"]))
            .struct.field(rle_fields["len"])
            .max()
            .fill_null(0)
        )


def mean_abs_change(x: TIME_SERIES_T) -> FLOAT_EXPR:
    """
    Compute mean absolute change.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        A single time-series.

    Returns
    -------
    float | Expr
    """
    return x.diff(null_behavior="drop").abs().mean()


def max_abs_change(x: TIME_SERIES_T) -> FLOAT_INT_EXPR:
    """
    Compute the maximum absolute change from X_t to X_t+1.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        A single time-series.

    Returns
    -------
    float | Expr
    """
    return absolute_maximum(x.diff(null_behavior="drop"))


def mean_change(x: TIME_SERIES_T) -> FLOAT_EXPR:
    """
    Compute mean change.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        A single time-series.

    Returns
    -------
    float | Expr
    """
    if isinstance(x, pl.Series):
        if len(x) <= 1:
            return 0
        return (x[-1] - x[0]) / (x.len() - 1)
    else:
        return (
            pl.when(x.len() > 1)
            .then((x.last() - x.first()) / (x.len() - 1))
            .otherwise(0)
        )


def mean_n_absolute_max(x: TIME_SERIES_T, n_maxima: int) -> FLOAT_EXPR:
    """
    Calculates the arithmetic mean of the n absolute maximum values of the time series.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    n_maxima : int
        The number of maxima to consider.

    Returns
    -------
    float | Expr
    """
    if n_maxima <= 0:
        raise ValueError("The number of maxima should be > 0.")
    return x.abs().top_k(n_maxima).mean()


def mean_second_derivative_central(x: TIME_SERIES_T) -> FLOAT_EXPR:
    """
    Returns the mean value of a central approximation of the second derivative.

    Parameters
    ----------
    x : pl.Series
        A time series to calculate the feature of.

    Returns
    -------
    pl.Series
    """
    if isinstance(x, pl.Series):
        if len(x) < 3:
            return np.nan
        return (x[-1] - x[-2] - x[1] + x[0]) / (2 * (x.len() - 2))
    else:
        return (
            pl.when(x.len() < 3)
            .then(np.nan)
            .otherwise(x.tail(2).sub(x.head(2)).diff().last() / (2 * (x.len() - 2)))
        )


def number_crossings(x: TIME_SERIES_T, crossing_value: float = 0.0) -> FLOAT_EXPR:
    """
    Calculates the number of crossings of x on m, where m is the crossing value.

    A crossing is defined as two sequential values where the first value is lower than m and the next is greater, or vice-versa.
    If you set m to zero, you will get the number of zero crossings.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        A single time-series.
    crossing_value : float
        The crossing value. Defaults to 0.0.

    Returns
    -------
    float | Expr
    """
    y = x > crossing_value
    return y.eq(y.shift(1)).not_().sum()


def number_cwt_peaks(x: TIME_SERIES_T, max_width: int = 5) -> float:
    """
    Number of different peaks in x.

    To estimate the numbers of peaks, x is smoothed by a ricker wavelet for widths ranging from 1 to n. This feature
    calculator returns the number of peaks that occur at enough width scales and with sufficiently high
    Signal-to-Noise-Ratio (SNR)

    Parameters
    ----------
    x : pl.Series
        A single time-series.

    max_width : int
        maximum width to consider


    Returns
    -------
    float
    """
    if isinstance(x, pl.Series):
        return len(
            find_peaks_cwt(
                vector=x.to_numpy(zero_copy_only=True),
                widths=np.array(list(range(1, max_width + 1))),
                wavelet=ricker,
            )
        )
    else:
        logger.info(
            "Expression version of number_cwt_peaks is not yet implemented due to "
            "technical difficulty regarding Polars Expression Plugins."
        )
        return NotImplemented


# def partial_autocorrelation(x: TIME_SERIES_T, n_lags: int) -> float:
#     return NotImplemented


def percent_reoccurring_points(x: TIME_SERIES_T) -> float:
    """
    Returns the percentage of non-unique data points in the time series. Non-unique data points are those that occur
    more than once in the time series.

    The percentage is calculated as follows:

        # of data points occurring more than once / # of all data points

    This means the ratio is normalized to the number of data points in the time series, in contrast to the
    `percent_reoccuring_values` function.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float
    """
    return 1 - x.is_unique().sum() / x.len()


def percent_reoccurring_values(x: TIME_SERIES_T) -> FLOAT_EXPR:
    """
    Returns the percentage of values that are present in the time series more than once.

    The percentage is calculated as follows:

        # (distinct values occurring more than once) / # of distinct values

    This means the percentage is normalized to the number of unique values in the time series, in contrast to the
    `percent_reoccurring_points` function.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float | Expr
    """
    count = x.unique_counts()
    return (count > 1).sum() / count.len()


def number_peaks(x: TIME_SERIES_T, support: int) -> INT_EXPR:
    """
    Calculates the number of peaks of at least support n in the time series x. A peak of support n is defined as a
    subsequence of x where a value occurs, which is bigger than its n neighbours to the left and to the right.

    Hence in the sequence

    x = [3, 0, 0, 4, 0, 0, 13]

    4 is a peak of support 1 and 2 because in the subsequences

    [0, 4, 0]
    [0, 0, 4, 0, 0]

    4 is still the highest value. Here, 4 is not a peak of support 3 because 13 is the 3th neighbour to the right of 4
    and its bigger than 4.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    support : int
        Support of the peak

    Returns
    -------
    int | Expr
    """
    if isinstance(x, pl.Series):
        # Cheating by calling lazy... Series may or may not have a better implementation.
        frame = x.to_frame()
        return frame.select(number_peaks(pl.col(x.name), support)).item(0, 0)
    else:
        return pl.all_horizontal(
            ((x > x.shift(-i)) & (x > x.shift(i))).fill_null(False)
            for i in range(1, support + 1)
        ).sum()


def permutation_entropy(
    x: TIME_SERIES_T,
    tau: int = 1,
    n_dims: int = 3,
    base: float = math.e,
) -> FLOAT_EXPR:
    """
    Computes permutation entropy.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    tau : int
        The embedding time delay which controls the number of time periods between elements
        of each of the new column vectors.
    n_dims : int, > 1
        The embedding dimension which controls the length of each of the new column vectors
    base : float
        The base for log in the entropy computation

    Returns
    -------
    float | Expr
    """
    if isinstance(x, pl.Series):
        # Complete this implementation by cheating (using exprs) for now.
        # There might be short cuts if we use eager. So improve this later.
        frame = x.to_frame()
        expr = permutation_entropy(pl.col(x.name), tau=tau, n_dims=n_dims, base=base)
        return frame.select(expr).item(0, 0)
    else:
        if tau == 1:  # Fast track the most common use case
            return (
                pl.concat_list(x, *(x.shift(-i) for i in range(1, n_dims)))
                .head(x.len() - n_dims + 1)
                .list.eval(pl.element().arg_sort())
                .value_counts()
                .struct.field("count")
                .entropy(base=base, normalize=True)
            )
        else:
            max_shift = -n_dims + 1
            return (
                pl.concat_list(
                    x.gather_every(tau),
                    *(x.shift(-i).gather_every(tau) for i in range(1, n_dims)),
                )
                .filter(  # This is the correct way to filter (with respect to tau)
                    pl.repeat(True, x.len())
                    .shift(max_shift, fill_value=False)
                    .gather_every(tau)
                )
                .list.eval(pl.element().arg_sort())
                .value_counts()
                .struct.field("count")
                .entropy(base=base, normalize=True)
            )


def range_count(
    x: TIME_SERIES_T, lower: float, upper: float, closed: ClosedInterval = "left"
) -> INT_EXPR:
    """
    Computes values of input expression that is between lower (inclusive) and upper (exclusive).

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    lower : float
        The lower bound, inclusive
    upper : float
        The upper bound, exclusive
    closed : ClosedInterval
        Whether or not the boundaries should be included/excluded

    Returns
    -------
    int | Expr
    """
    if upper < lower:
        raise ValueError("Upper must be greater than lower.")
    return x.is_between(lower_bound=lower, upper_bound=upper, closed=closed).sum()


def ratio_beyond_r_sigma(x: TIME_SERIES_T, ratio: float = 0.25) -> FLOAT_EXPR:
    """
    Returns the ratio of values in the series that is beyond r*std from mean on both sides.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.
    ratio : float
        The scaling factor for std

    Returns
    -------
    float | Expr
    """
    return (
        x.is_between(
            x.mean() - pl.lit(ratio) * x.std(ddof=0),
            x.mean() + pl.lit(ratio) * x.std(ddof=0),
            closed="both",
        )
        .not_()
        .sum()
        / x.len()
    )


def ratio_n_unique_to_length(x: TIME_SERIES_T) -> FLOAT_EXPR:
    """
    Calculate the ratio of the number of unique values to the length of the time-series.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float | Expr
    """
    return x.n_unique() / x.len()


def root_mean_square(x: TIME_SERIES_T) -> FLOAT_EXPR:
    """
    Calculate the root mean square.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float | Expr
    """
    if isinstance(x, pl.Series):
        return np.sqrt(np.dot(x, x) / len(x))
    else:
        return (x.dot(x) / x.count()).sqrt()


def _into_sequential_chunks(x: pl.Series, m: int) -> np.ndarray:
    name = x.name
    n_rows = len(x) - m + 1
    df = (
        x.to_frame()
        .select(
            pl.col(name),
            *(pl.col(name).shift(-i).name.suffix(str(i)) for i in range(1, m)),
        )
        .slice(0, n_rows)
    )
    return df.to_numpy()


# Only works on series
def sample_entropy(x: TIME_SERIES_T, ratio: float = 0.2, m: int = 2) -> FLOAT_EXPR:
    """
    Calculate the sample entropy of a time series.
    This only works for Series input right now.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        The input time series.
    ratio : float, optional
        The tolerance parameter. Default is 0.2.
    m : int
        Length of a run of data. Most common run length is 2.

    Returns
    -------
    float | Expr
    """
    # This is significantly faster than tsfresh
    if isinstance(x, pl.Series):
        if len(x) < m:
            return np.nan

        threshold = ratio * x.std(ddof=0)
        mat = _into_sequential_chunks(x, m)
        tree = KDTree(mat)
        b = (
            np.sum(
                tree.query_ball_point(
                    mat, r=threshold, p=np.inf, workers=-1, return_length=True
                )
            )
            - mat.shape[0]
        )
        mat = _into_sequential_chunks(x, m + 1)
        tree = KDTree(mat)
        a = (
            np.sum(
                tree.query_ball_point(
                    mat, r=threshold, p=np.inf, workers=-1, return_length=True
                )
            )
            - mat.shape[0]
        )
        return np.log(b / a)  # -ln(a/b) = ln(b/a)
    else:
        logger.info(
            "Expression version of sample_entropy is not yet implemented due to "
            "technical difficulty regarding Polars Expression Plugins."
        )
        return NotImplemented


# Need to improve the input arguments depending on use cases.
def spkt_welch_density(x: TIME_SERIES_T, n_coeffs: Optional[int] = None) -> LIST_EXPR:
    """
    This estimates the cross power spectral density of the time series x at different frequencies.
    This only works for Series input right now.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        The input time series.
    n_coeffs : Optional[int]
        The number of coefficients you want to take. If none, will take all, which will be a list
        as long as the input time series.

    Returns
    -------
    list of floats
    """
    if isinstance(x, pl.Series):
        if n_coeffs is None:
            last_idx = len(x)
        else:
            last_idx = n_coeffs
        _, pxx = welch(x, nperseg=min(len(x), 256))
        return pxx[:last_idx]
    else:
        logger.info(
            "Expression version of spkt_welch_density is not yet implemented due to "
            "technical difficulty regarding Polars Expression Plugins."
        )
        return NotImplemented


# Originally named: `sum_of_reoccurring_data_points`
def sum_reoccurring_points(x: TIME_SERIES_T) -> FLOAT_INT_EXPR:
    """
    Returns the sum of all data points that are present in the time series more than once.

    For example, `sum_reoccurring_points(pl.Series([2, 2, 2, 2, 1]))` returns 8, as 2 is a reoccurring value, so all 2's
    are summed up.

    This is in contrast to the `sum_reoccurring_values` function, where each reoccuring value is only counted once.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float | Expr
    """
    return x.filter(~x.is_unique()).sum()


# Originally named: `sum_of_reoccurring_values`
def sum_reoccurring_values(x: TIME_SERIES_T) -> FLOAT_INT_EXPR:
    """
    Returns the sum of all values that are present in the time series more than once.

    For example, `sum_reoccurring_values(pl.Series([2, 2, 2, 2, 1]))` returns 2, as 2 is a reoccurring value, so it is
    summed up with all other reoccuring values (there is none), so the result is 2.

    This is in contrast to the `sum_reoccurring_points` function, where each reoccuring value is only counted as often as it is present in the data.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time-series.

    Returns
    -------
    float | Expr
    """
    if isinstance(x, pl.Series):
        vc = x.value_counts()
        return vc.filter(pl.col("count") > 1).select(pl.col(x.name).sum()).item(0, 0)
    else:
        name = x.meta.output_name() or "_"
        vc = x.value_counts(sort=True)
        return vc.filter(vc.struct.field("count") > 1).struct.field(name).sum()


def symmetry_looking(x: TIME_SERIES_T, ratio: float = 0.25) -> BOOL_EXPR:
    """
    Check if the distribution of x looks symmetric.

    A distribution is considered symmetric if: `| mean(X)-median(X) | < ratio * (max(X)-min(X))`

    Parameters
    ----------
    x : polars.Series
        Input time-series.
    ratio : float
        Multiplier on distance between max and min.

    Returns
    -------
    bool | Expr
    """
    if isinstance(x, pl.Series):
        mean_median_difference = np.abs(x.mean() - x.median())
    else:
        mean_median_difference = (x.mean() - x.median()).abs()

    max_min_difference = x.max() - x.min()
    return mean_median_difference < ratio * max_min_difference


def time_reversal_asymmetry_statistic(x: TIME_SERIES_T, n_lags: int) -> FLOAT_EXPR:
    """
    Returns the time reversal asymmetry statistic.

    Parameters
    ----------
    x : pl.Series
        Input time-series.
    n_lags : int
        The lag that should be used in the calculation of the feature.

    Returns
    -------
    float | Expr
    """
    one_lag = x.shift(-n_lags)
    two_lag = x.shift(-2 * n_lags)
    return (one_lag * (two_lag + x) * (two_lag - x)).mean()


def variation_coefficient(x: TIME_SERIES_T) -> FLOAT_EXPR:
    """
    Calculate the coefficient of variation (CV).

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time series.

    Returns
    -------
    float | Expr
    """
    x_mean = x.mean()
    x_std = x.std(ddof=0)
    if isinstance(x, pl.Series):
        return np.divide(x_std, x_mean)
    return x_std / x_mean


def var_gt_std(x: TIME_SERIES_T, ddof: int = 1) -> BOOL_EXPR:
    """
    Is the variance >= std? In other words, is var >= 1?

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time series.
    ddof : int
        Delta Degrees of Freedom used when computing var.

    Returns
    -------
    bool | Expr
    """
    return x.var(ddof=ddof) >= 1


def harmonic_mean(x: TIME_SERIES_T) -> FLOAT_EXPR:
    """
    Returns the harmonic mean of the of the time series.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time series.

    Returns
    -------
    float | Expr
    """
    return x.len() / (1.0 / x).sum()


def range_over_mean(x: TIME_SERIES_T) -> FLOAT_EXPR:
    """
    Returns the range (max - min) over mean of the time series.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time series.

    Returns
    -------
    float | Expr
    """
    return (x.max() - x.min()) / x.mean()


def range_change(x: TIME_SERIES_T, percentage: bool = True) -> FLOAT_EXPR:
    """
    Returns the maximum value range. If percentage is true, will compute
    (max - min) / min, which only makes sense when x is always positive.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time series.
    percentage : bool
        compute the percentage if set to True

    Returns
    -------
    float | Expr
    """
    if percentage:
        return x.max() / x.min() - 1.0
    else:
        return x.max() - x.min()


def streak_length_stats(x: TIME_SERIES_T, above: bool, threshold: float) -> MAP_EXPR:
    """
    Returns some statistics of the length of the streaks of the time series. Note that the streaks here
    are about the changes for consecutive values in the time series, not the individual values.

    The statistics include: min length, max length, average length, std of length,
    10-percentile length, median length, 90-percentile length, and mode of the length. If input is Series,
    a dictionary will be returned. If input is an expression, the expression will evaluate to a struct
    with the fields ordered by the statistics.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time series.
    above: bool
        Above (>=) or below (<=) the given threshold
    threshold
        The threshold for the change (x_t+1 - x_t) to be counted

    Returns
    -------
    float | Expr
    """
    if above:
        y = (x.diff() >= threshold).rle()
    else:
        y = (x.diff() <= threshold).rle()

    y = y.filter(y.struct.field(rle_fields["value"])).struct.field(rle_fields["len"])
    if isinstance(x, pl.Series):
        return {
            "min": y.min() or 0,
            "max": y.max(),
            "mean": y.mean(),
            "std": y.std(),
            "10-percentile": y.quantile(0.1),
            "median": y.median(),
            "90-percentile": y.quantile(0.9),
            "mode": y.mode().sort()[0] if not y.is_empty() else None,
        }
    else:
        return pl.struct(
            pl.max_horizontal(y.min(), 0).alias("min"),
            y.max().alias("max"),
            y.mean().alias("mean"),
            y.std().alias("std"),
            y.quantile(0.1).alias("10-percentile"),
            y.median().alias("median"),
            y.quantile(0.9).alias("90-percentile"),
            y.mode().sort(nulls_last=True).first().alias("mode"),
        )


def longest_streak_above(x: TIME_SERIES_T, threshold: float) -> TIME_SERIES_T:
    """
    Returns the longest streak of changes >= threshold of the time series. A change
    is counted when (x_t+1 - x_t) >= threshold. Note that the streaks here
    are about the changes for consecutive values in the time series, not the individual values.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time series.
    threshold : float
        The threshold value for comparison.

    Returns
    -------
    float | Expr
    """

    y = (x.diff() >= threshold).rle()
    if isinstance(x, pl.Series):
        streak_max = (
            y.filter(y.struct.field(rle_fields["value"]))
            .struct.field(rle_fields["len"])
            .max()
        )
        return 0 if streak_max is None else streak_max
    else:
        return (
            y.filter(y.struct.field(rle_fields["value"]))
            .struct.field(rle_fields["len"])
            .max()
            .fill_null(0)
        )


def longest_streak_below(x: TIME_SERIES_T, threshold: float) -> TIME_SERIES_T:
    """
    Returns the longest streak of changes <= threshold of the time series. A change
    is counted when (x_t+1 - x_t) <= threshold. Note that the streaks here
    are about the changes for consecutive values in the time series, not the individual values.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time series.
    threshold : float
        The threshold value for comparison.

    Returns
    -------
    float | Expr
    """
    y = (x.diff() <= threshold).rle()
    if isinstance(x, pl.Series):
        streak_max = (
            y.filter(y.struct.field(rle_fields["value"]))
            .struct.field(rle_fields["len"])
            .max()
        )
        return 0 if streak_max is None else streak_max
    else:
        return (
            y.filter(y.struct.field(rle_fields["value"]))
            .struct.field(rle_fields["len"])
            .max()
            .fill_null(0)
        )


def longest_winning_streak(x: TIME_SERIES_T) -> TIME_SERIES_T:
    """
    Returns the longest winning streak of the time series. A win is counted when
    (x_t+1 - x_t) >= 0

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time series.

    Returns
    -------
    float | Expr
    """
    return longest_streak_above(x, threshold=0)


def longest_losing_streak(x: TIME_SERIES_T) -> TIME_SERIES_T:
    """
    Returns the longest losing streak of the time series. A loss is counted when
    (x_t+1 - x_t) <= 0

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time series.

    Returns
    -------
    float | Expr
    """
    return longest_streak_below(x, threshold=0)


# FFT Features


def fft_coefficients(x: TIME_SERIES_T) -> MAP_LIST_EXPR:
    """
    Calculates Fourier coefficients and phase angles of the the 1-D discrete Fourier Transform.
    This only works for Series input right now.

    Parameters
    ----------
    x : pl.Expr | pl.Series
        Input time series.
    n_threads : int
        Number of threads to use.
        If None, uses all threads available. Defaults to None.

    Returns
    -------
    dict of list of floats | Expr
    """

    fft = np.fft.rfft(x.to_numpy(zero_copy_only=True))
    real = fft.real
    imag = fft.imag
    angle = np.arctan2(real, imag)
    deg_angle = angle * 180 / np.pi
    return {
        "real": fft.real.tolist(),
        "imag": fft.imag.tolist(),
        "angle": deg_angle.tolist(),
    }


@pl.api.register_expr_namespace("ts")
class FeatureExtractor:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def absolute_energy(self) -> pl.Expr:
        """
        Compute the absolute energy of a time series.

        Returns
        -------
        An expression of the output
        """
        return absolute_energy(self._expr)

    def absolute_maximum(self) -> pl.Expr:
        """
        Compute the absolute maximum of a time series.

        Returns
        -------
        An expression of the output
        """
        return absolute_maximum(self._expr)

    def absolute_sum_of_changes(self) -> pl.Expr:
        """
        Compute the absolute sum of changes of a time series.

        Returns
        -------
        An expression of the output
        """
        return absolute_sum_of_changes(self._expr)

    def autocorrelation(self, n_lags: int) -> pl.Expr:
        """
        Calculate the autocorrelation for a specified lag. The autocorrelation measures the linear
        dependence between a time-series and a lagged version of itself.

        Parameters
        ----------
        n_lags : int
            The lag at which to calculate the autocorrelation. Must be a non-negative integer.

        Returns
        -------
        An expression of the output
        """
        return autocorrelation(self._expr, n_lags)

    def root_mean_square(self) -> pl.Expr:
        """
        Calculate the root mean square.

        Returns
        -------
        An expression of the output
        """
        return root_mean_square(self._expr)

    def benford_correlation(self) -> pl.Expr:
        """
        Returns the correlation between the first digit distribution of the input time series and
        the Newcomb-Benford's Law distribution.

        Returns
        -------
        An expression of the output
        """
        return benford_correlation(self._expr)

    def binned_entropy(self, bin_count: int = 10) -> pl.Expr:
        """
        Calculates the entropy of a binned histogram for a given time series. It is highly recommended
        that you impute the time series before calling this.

        Parameters
        ----------
        bin_count : int, optional
            The number of bins to use in the histogram. Default is 10.

        Returns
        -------
        An expression of the output
        """
        return binned_entropy(self._expr, bin_count)

    def c3(self, n_lags: int) -> pl.Expr:
        """
        Measure of non-linearity in the time series using c3 statistics.

        Parameters
        ----------
        n_lags : int
            The lag that should be used in the calculation of the feature.

        Returns
        -------
        An expression of the output
        """
        return c3(self._expr, n_lags)

    def change_quantiles(
        self, q_low: float, q_high: float, is_abs: bool = True
    ) -> pl.Expr:
        """
        First fixes a corridor given by the quantiles ql and qh of the distribution of x.
        Then calculates the average, absolute value of consecutive changes of the series x inside this corridor.

        Parameters
        ----------
        q_low : float
            The lower quantile of the corridor. Must be less than `q_high`.
        q_high : float
            The upper quantile of the corridor. Must be greater than `q_low`.
        is_abs : bool
            If True, takes absolute difference.

        Returns
        -------
        An expression of the output
        """
        return change_quantiles(self._expr, q_low, q_high, is_abs)

    def cid_ce(self, normalize: bool = False) -> pl.Expr:
        """
        Computes estimate of time-series complexity[^1].

        A more complex time series has more peaks and valleys.
        This feature is calculated by:

        Parameters
        ----------
        normalize : bool, optional
            If True, z-normalizes the time-series before computing the feature.
            Default is False.

        Returns
        -------
        An expression of the output
        """
        return cid_ce(self._expr, normalize)

    def count_above(self, threshold: float = 0.0) -> pl.Expr:
        """
        Calculate the percentage of values above or equal to a threshold.

        Parameters
        ----------
        threshold : float
            The threshold value for comparison.

        Returns
        -------
        An expression of the output
        """
        return count_above(self._expr, threshold)

    def count_above_mean(self) -> pl.Expr:
        """
        Count the number of values that are above the mean.

        Parameters
        ----------
        x : pl.Expr | pl.Series
            Input time-series.

        Returns
        -------
        An expression of the output
        """
        return count_above_mean(self._expr)

    def count_below(self, threshold: float = 0.0) -> pl.Expr:
        """
        Calculate the percentage of values below or equal to a threshold.

        Parameters
        ----------
        threshold : float
            The threshold value for comparison.

        Returns
        -------
        An expression of the output
        """
        return count_below(self._expr, threshold)

    def count_below_mean(self) -> pl.Expr:
        """
        Count the number of values that are below the mean.

        Returns
        -------
        An expression of the output
        """
        return count_below_mean(self._expr)

    def energy_ratios(self, n_chunks: int = 10) -> pl.Expr:
        """
        Calculates sum of squares over the whole series for `n_chunks` equally segmented parts of the time-series.
        All ratios for all chunks will be returned at once.

        Parameters
        ----------
        n_chunks : int, optional
            The number of equally segmented parts to divide the time-series into. Default is 10.

        Returns
        -------
        An expression of the output
        """
        return energy_ratios(self._expr, n_chunks)

    def first_location_of_maximum(self) -> pl.Expr:
        """
        Returns the first location of the maximum value of x.
        The position is calculated relatively to the length of x.

        Returns
        -------
        An expression of the output
        """
        return first_location_of_maximum(self._expr)

    def first_location_of_minimum(self) -> pl.Expr:
        """
        Returns the first location of the minimum value of x.
        The position is calculated relatively to the length of x.

        Returns
        -------
        An expression of the output
        """
        return first_location_of_minimum(self._expr)

    def has_duplicate(self) -> pl.Expr:
        """
        Check if the time-series contains any duplicate values.

        Returns
        -------
        An expression of the output
        """
        return has_duplicate(self._expr)

    def has_duplicate_max(self) -> pl.Expr:
        """
        Check if the time-series contains any duplicate values equal to its maximum value.

        Returns
        -------
        An expression of the output
        """
        return has_duplicate_max(self._expr)

    def has_duplicate_min(self) -> pl.Expr:
        """
        Check if the time-series contains duplicate values equal to its minimum value.

        Returns
        -------
        An expression of the output
        """
        return has_duplicate_min(self._expr)

    def index_mass_quantile(self, q: float) -> pl.Expr:
        """
        Calculates the relative index i of time series x where q% of the mass of x lies left of i.
        For example for q = 50% this feature calculator will return the mass center of the time series.

        Parameters
        ----------
        q : float
            The quantile.

        Returns
        -------
        An expression of the output
        """
        return index_mass_quantile(self._expr, q)

    def large_standard_deviation(self, ratio: float = 0.25) -> pl.Expr:
        """
        Checks if the time-series has a large standard deviation: `std(x) > r * (max(X)-min(X))`.

        As a heuristic, the standard deviation should be a forth of the range of the values.

        Parameters
        ----------
        ratio : float
            The ratio of the interval to compare with.

        Returns
        -------
        An expression of the output
        """
        return large_standard_deviation(self._expr, ratio)

    def last_location_of_maximum(self) -> pl.Expr:
        """
        Returns the last location of the maximum value of x.
        The position is calculated relatively to the length of x.

        Returns
        -------
        An expression of the output
        """
        return last_location_of_maximum(self._expr)

    def last_location_of_minimum(self) -> pl.Expr:
        """
        Returns the last location of the minimum value of x.
        The position is calculated relatively to the length of x.

        Returns
        -------
        An expression of the output
        """
        return last_location_of_minimum(self._expr)

    def lempel_ziv_complexity(
        self, threshold: Union[float, pl.Expr], as_ratio: bool = True
    ) -> pl.Expr:
        """
        Calculate a complexity estimate based on the Lempel-Ziv compression algorithm. The
        implementation here is currently a Rust rewrite of Lilian Besson'code. Instead of returning
        the complexity value, we return a ratio w.r.t the length of the input series. If null is
        encountered, it will be interpreted as 0 in the bit sequence.

        Parameters
        ----------
        x : pl.Expr | pl.Series
            Input time-series.
        threshold: float | pl.Expr
            Either a number, or an expression representing a comparable quantity. If x > threshold,
            then it will be binarized as 1 and 0 otherwise.
        as_ratio: bool
            If true, return the complexity divided by length of sequence

        Returns
        -------
        Expr

        Reference
        ---------
        https://github.com/Naereen/Lempel-Ziv_Complexity/tree/master
        https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv_complexity
        """
        out = register_plugin_function(
            args=[self._expr > threshold],
            plugin_path=lib,
            function_name="pl_lempel_ziv_complexity",
            is_elementwise=False,
            returns_scalar=True,
        )
        if as_ratio:
            return out / self._expr.len()
        return out

    def linear_trend(self) -> pl.Expr:
        """
        Compute the slope, intercept, and RSS of the linear trend.

        Returns
        -------
        An expression of the output
        """
        return linear_trend(self._expr)

    def detrend(self, method: DetrendMethod = "linear") -> pl.Expr:
        """
        Detrends the time series by either removing a fitted linear regression or by
        removing the mean. This assumes that data is in order.

        Parameters
        ----------
        method
            Either `linear` or `mean`

        Returns
        -------
        An expression representing detrend-ed column
        """

        if method == "linear":
            N = self._expr.count()
            x = pl.int_range(0, N, dtype=pl.Float64, eager=False)
            coeff = pl.cov(self._expr, x) / x.var()
            const = self._expr.mean() - coeff * (N - 1) / 2
            return self._expr - x * coeff - const
        elif method == "mean":
            return self._expr - self._expr.mean()
        else:
            raise ValueError(f"Unknown detrend method: {method}")

    def longest_streak_above_mean(self) -> pl.Expr:
        """
        Returns the length of the longest consecutive subsequence in x that is greater than the mean of x.

        Returns
        -------
        An expression of the output
        """
        return longest_streak_above_mean(self._expr)

    def longest_streak_below_mean(self) -> pl.Expr:
        """
        Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x.

        Returns
        -------
        An expression of the output
        """
        return longest_streak_below_mean(self._expr)

    def longest_streak_above(self, threshold: float) -> pl.Expr:
        """
        Returns the longest streak of changes >= threshold of the time series. A change
        is counted when (x_t+1 - x_t) >= threshold. Note that the streaks here
        are about the changes for consecutive values in the time series, not the individual values.

        Parameters
        ----------
        threshold : float
            The threshold value for comparison.

        Returns
        -------
        An expression of the output
        """
        return longest_streak_above(self._expr, threshold)

    def longest_streak_below(self, threshold: float) -> pl.Expr:
        """
        Returns the longest streak of changes <= threshold of the time series. A change
        is counted when (x_t+1 - x_t) <= threshold. Note that the streaks here
        are about the changes for consecutive values in the time series, not the individual values.

        Parameters
        ----------
        threshold : float
            The threshold value for comparison.

        Returns
        -------
        An expression of the output
        """
        return longest_streak_below(self._expr, threshold)

    def mean_abs_change(self) -> pl.Expr:
        """
        Compute mean absolute change.

        Returns
        -------
        An expression of the output
        """
        return mean_abs_change(self._expr)

    def max_abs_change(self) -> pl.Expr:
        """
        Compute the maximum absolute change from X_t to X_t+1.

        Returns
        -------
        An expression of the output
        """
        return max_abs_change(self._expr)

    def mean_change(self) -> pl.Expr:
        """
        Compute mean change.

        Returns
        -------
        An expression of the output
        """
        return mean_change(self._expr)

    def mean_n_absolute_max(self, n_maxima: int) -> pl.Expr:
        """
        Calculates the arithmetic mean of the n absolute maximum values of the time series.

        Parameters
        ----------
        n_maxima : int
            The number of maxima to consider.

        Returns
        -------
        An expression of the output
        """
        return mean_n_absolute_max(self._expr, n_maxima)

    def mean_second_derivative_central(self) -> pl.Expr:
        """
        Returns the mean value of a central approximation of the second derivative.

        Returns
        -------
        An expression of the output
        """
        return mean_second_derivative_central(self._expr)

    def number_crossings(self, crossing_value: float = 0.0) -> pl.Expr:
        """
        Calculates the number of crossings of x on m, where m is the crossing value.

        A crossing is defined as two sequential values where the first value is lower than m and the next is greater,
        or vice-versa. If you set m to zero, you will get the number of zero crossings.

        Parameters
        ----------
        crossing_value : float
            The crossing value. Defaults to 0.0.

        Returns
        -------
        An expression of the output
        """
        return number_crossings(self._expr, crossing_value)

    def percent_reoccurring_points(self) -> pl.Expr:
        """
        Returns the percentage of non-unique data points in the time series. Non-unique data points are those that occur
        more than once in the time series.

        The percentage is calculated as follows:

            # of data points occurring more than once / # of all data points

        This means the ratio is normalized to the number of data points in the time series, in contrast to the
        `percent_reoccuring_values` function.

        Returns
        -------
        An expression of the output
        """
        return percent_reoccurring_points(self._expr)

    def percent_reoccurring_values(self) -> pl.Expr:
        """
        Returns the percentage of values that are present in the time series more than once.

        The percentage is calculated as follows:

            len(different values occurring more than once) / len(different values)

        This means the percentage is normalized to the number of unique values in the time series, in contrast to the
        `percent_reoccurring_points` function.

        Returns
        -------
        An expression of the output
        """
        return percent_reoccurring_values(self._expr)

    def number_peaks(self, support: int) -> pl.Expr:
        """
        Calculates the number of peaks of at least support n in the time series x. A peak of support n is defined as a
        subsequence of x where a value occurs, which is bigger than its n neighbours to the left and to the right.

        Hence in the sequence

        x = [3, 0, 0, 4, 0, 0, 13]

        4 is a peak of support 1 and 2 because in the subsequences

        [0, 4, 0]
        [0, 0, 4, 0, 0]

        4 is still the highest value. Here, 4 is not a peak of support 3 because 13 is the 3th neighbour to the right of 4
        and its bigger than 4.

        Parameters
        ----------
        support : int
            Support of the peak

        Returns
        -------
        An expression of the output
        """
        return number_peaks(self._expr, support)

    def permutation_entropy(
        self,
        tau: int = 1,
        n_dims: int = 3,
        base: float = math.e,
    ) -> pl.Expr:
        """
        Computes permutation entropy. It is recommended that users should impute the time series
        before calling this.

        Parameters
        ----------
        tau : int
            The embedding time delay which controls the number of time periods between elements
            of each of the new column vectors. The recommended value is 1.
        n_dims : int, > 1
            The embedding dimension which controls the length of each of the new column vectors. The
            recommended range is 3-7.
        base : float
            The base for log in the entropy computation

        Returns
        -------
        An expression of the output
        """
        return permutation_entropy(self._expr, tau, n_dims, base)

    def range_count(
        self, lower: float, upper: float, closed: ClosedInterval = "left"
    ) -> pl.Expr:
        """
        Computes values of input expression that is between lower (inclusive) and upper (exclusive).

        Parameters
        ----------
        lower : float
            The lower bound, inclusive
        upper : float
            The upper bound, exclusive
        closed : ClosedInterval
            Whether or not the boundaries should be included/excluded

        Returns
        -------
        An expression of the output
        """
        return range_count(self._expr, lower, upper, closed)

    def ratio_beyond_r_sigma(self, ratio: float = 0.25) -> pl.Expr:
        """
        Returns the ratio of values in the series that is beyond r*std from mean on both sides.

        Parameters
        ----------
        ratio : float
            The scaling factor for std

        Returns
        -------
        An expression of the output
        """
        return ratio_beyond_r_sigma(self._expr, ratio)

    # Originally named: `sum_of_reoccurring_data_points`
    def sum_reoccurring_points(self) -> pl.Expr:
        """
        Returns the sum of all data points that are present in the time series more than once.

        For example, `sum_reoccurring_points(pl.Series([2, 2, 2, 2, 1]))` returns 8, as 2 is a reoccurring value, so all 2's
        are summed up.

        This is in contrast to the `sum_reoccurring_values` function, where each reoccuring value is only counted once.

        Returns
        -------
        An expression of the output
        """
        return sum_reoccurring_points(self._expr)

    # Originally named: `sum_of_reoccurring_values`
    def sum_reoccurring_values(self) -> pl.Expr:
        """
        Returns the sum of all values that are present in the time series more than once.

        For example, `sum_reoccurring_values(pl.Series([2, 2, 2, 2, 1]))` returns 2, as 2 is a reoccurring value, so it is
        summed up with all other reoccuring values (there is none), so the result is 2.

        This is in contrast to the `sum_reoccurring_points` function, where each reoccuring value is only counted as often
        as it is present in the data.

        Returns
        -------
        An expression of the output
        """
        return sum_reoccurring_values(self._expr)

    def symmetry_looking(self, ratio: float = 0.25) -> pl.Expr:
        """
        Check if the distribution of x looks symmetric.

        A distribution is considered symmetric if: `| mean(X)-median(X) | < ratio * (max(X)-min(X))`

        Parameters
        ----------
        ratio : float
            Multiplier on distance between max and min.

        Returns
        -------
        An expression of the output
        """
        return symmetry_looking(self._expr, ratio)

    def time_reversal_asymmetry_statistic(self, n_lags: int) -> pl.Expr:
        """
        Returns the time reversal asymmetry statistic.

        Parameters
        ----------
        n_lags : int
            The lag that should be used in the calculation of the feature.

        Returns
        -------
        An expression of the output
        """
        return time_reversal_asymmetry_statistic(self._expr, n_lags)

    def variation_coefficient(self) -> pl.Expr:
        """
        Calculate the coefficient of variation (CV).

        Returns
        -------
        An expression of the output
        """
        return variation_coefficient(self._expr)

    def var_gt_std(self, ddof: int = 1) -> pl.Expr:
        """
        Is the variance >= std? In other words, is var >= 1?

        Parameters
        ----------
        ddof : int
            Delta Degrees of Freedom used when computing var/std.

        Returns
        -------
        An expression of the output
        """
        return var_gt_std(self._expr, ddof)

    def harmonic_mean(self) -> pl.Expr:
        """
        Returns the harmonic mean of the expression

        Returns
        -------
        An expression of the output
        """
        return harmonic_mean(self._expr)

    def range_over_mean(self) -> pl.Expr:
        """
        Returns the range (max - min) over mean of the time series.

        Returns
        -------
        An expression of the output
        """
        return range_over_mean(self._expr)

    def range_change(self, percentage: bool = True) -> pl.Expr:
        """
        Returns the range (max - min) over mean of the time series.

        Parameters
        ----------
        percentage : bool
            Compute the percentage if set to True

        Returns
        -------
        An expression of the output
        """
        return range_change(self._expr, percentage)

    def streak_length_stats(self, above: bool, threshold: float) -> pl.Expr:
        """
        Returns some statistics of the length of the streaks of the time series. Note that the streaks here
        are about the changes for consecutive values in the time series, not the individual values.

        The statistics include: min length, max length, average length, std of length,
        10-percentile length, median length, 90-percentile length, and mode of the length. If input is Series,
        a dictionary will be returned. If input is an expression, the expression will evaluate to a struct
        with the fields ordered by the statistics.

        Parameters
        ----------
        above: bool
            Above (>=) or below (<=) the given threshold
        threshold
            The threshold for the change (x_t+1 - x_t) to be counted

        Returns
        -------
        An expression of the output
        """
        return streak_length_stats(self._expr, above, threshold)

    def longest_winning_streak(self) -> pl.Expr:
        """
        Returns the longest winning streak of the time series. A win is counted when
        (x_t+1 - x_t) >= 0

        Returns
        -------
        An expression of the output
        """
        return longest_winning_streak(self._expr)

    def longest_losing_streak(self) -> pl.Expr:
        """
        Returns the longest losing streak of the time series. A loss is counted when
        (x_t+1 - x_t) <= 0

        Returns
        -------
        An expression of the output
        """
        return longest_losing_streak(self._expr)

    def ratio_n_unique_to_length(self) -> pl.Expr:
        """
        Calculate the ratio of the number of unique values to the length of the time-series.

        Returns
        -------
        An expression of the output
        """
        return ratio_n_unique_to_length(self._expr)

    def cusum(
        self,
        threshold: float,
        warmup_period: int,
        drift: float = 0.0,
    ) -> pl.Expr:
        """
        Cumulative sum (CUSUM) filter to detect abrupt changes in data.

        The CUSUM filter is a quality control method, designed to detect a shift in the
        mean value of the measured quantity away from a target value.

        The general formula for the CUSUM filter can be found here:
        https://en.wikipedia.org/wiki/CUSUM

        And the original paper that introduces it can be found here:
        https://www.tandfonline.com/doi/abs/10.1080/00401706.1961.10489922

        Parameters
        ----------
        threshold : float
            The threshold for the change (x_t+1 - x_t) to be counted
        warmup_period : int
            The number of observations which are used to estimate the mean and standard
            deviation of the data.
        drift : float
            The drift coefficient for the CUSUM filter. Default value is 0.

        Returns
        -------
        An expression of the output
        """
        return register_plugin_function(
            args=[self._expr],
            plugin_path=lib,
            function_name="cusum",
            kwargs={
                "threshold": threshold,
                "drift": drift,
                "warmup_period": warmup_period,
            },
            is_elementwise=False,
            cast_to_supertype=True,
        )

    def frac_diff(
        self,
        d: float,
        min_weight: Optional[float] = None,
        window_size: Optional[int] = None,
    ) -> pl.Expr:
        """Compute the fractional differential of a time series.

        This particular functionality is referenced in Advances in Financial Machine
        Learning by Marcos Lopez de Prado (2018).

        For feature creation purposes, it is suggested that the minimum value of d
        is used that removes stationarity from the time series. This can be achieved
        by running the augmented dickey-fuller test on the time series for different
        values of d and selecting the minimum value that makes the time series
        stationary.

        Parameters
        ----------
        d : float
            The fractional order of the differencing operator.
        min_weight : float, optional
            The minimum weight to use for calculations. If specified, the window size is
            computed from this value and not needed.
        window_size : int, optional
            The window size of the fractional differencing operator.
            If specified, the minimum weight is not needed.
        """

        # Assert only one of min_weight or window_size is specified
        if min_weight is not None and window_size is not None:
            raise ValueError("Only one of min_weight or window_size can be specified.")

        # Assert either min_weight or window_size is specified
        if min_weight is None and window_size is None:
            raise ValueError("Either min_weight or window_size must be specified.")

        return register_plugin_function(
            args=[self._expr],
            plugin_path=lib,
            function_name="frac_diff",
            kwargs={
                "d": d,
                "min_weight": min_weight,
                "window_size": window_size,
            },
            is_elementwise=False,
            cast_to_supertype=True,
        )
