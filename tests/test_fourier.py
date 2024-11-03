from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from functime.cross_validation import train_test_split
from functime.seasonality import add_fourier_terms


@pytest.mark.parametrize(
    "freq,sp, use_date",
    [
        ("1h", 24, False),
        ("1d", 365, False),
        ("1w", 52, False),
        ("1d", 365, True),
        ("1w", 52, True),
    ],
)
def test_fourier_with_timestamps(freq: str, sp: int, use_date: bool):
    if use_date:
        timestamps = pl.date_range(
            date(2020, 1, 1), date(2021, 1, 1), interval=freq, eager=True
        )
    else:
        timestamps = pl.datetime_range(
            date(2020, 1, 1), date(2021, 1, 1), interval=freq, eager=True
        )
    n_timestamps = len(timestamps)
    idx_timestamps = timestamps.arg_sort() + 1
    entities = pl.concat(
        [
            pl.repeat("a", n_timestamps, eager=True),
            pl.repeat("b", n_timestamps, eager=True),
        ]
    )
    X0 = pl.DataFrame(
        {
            "entity": entities,
            "time": pl.concat([timestamps, timestamps]),
        }
    )
    X1 = pl.DataFrame(
        {
            "entity": entities,
            "time": pl.concat([idx_timestamps, idx_timestamps]),
        }
    )
    idx_cols = ["entity", "time"]

    # Train test split
    X0_train, X0_test = X0.pipe(train_test_split(test_size=6))
    X1_train, X1_test = X1.pipe(train_test_split(test_size=6))

    # Test train
    X0_train_fourier = X0_train.pipe(add_fourier_terms(sp=sp, K=4)).collect()
    X1_train_fourier = X1_train.pipe(add_fourier_terms(sp=sp, K=4)).collect()
    assert_frame_equal(
        X0_train_fourier.sort(idx_cols).drop(idx_cols),
        X1_train_fourier.sort(idx_cols).drop(idx_cols),
    )

    # Test test
    X0_test_fourier = X0_test.pipe(add_fourier_terms(sp=sp, K=4)).collect()
    X1_test_fourier = X1_test.pipe(add_fourier_terms(sp=sp, K=4)).collect()
    assert_frame_equal(
        X0_test_fourier.sort(idx_cols).drop(idx_cols),
        X1_test_fourier.sort(idx_cols).drop(idx_cols),
    )


@pytest.mark.benchmark
def test_fourier_compare_with_aeon():
    from aeon.transformations.series.fourier import FourierFeatures

    sp = 12
    K = 4
    y = pl.read_parquet("data/commodities.parquet")
    entity_col, time_col, target_col = y.columns
    result = (
        y.pipe(add_fourier_terms(sp=sp, K=K)).sort([entity_col, time_col]).collect()
    )
    aeon_add_fourier_terms = FourierFeatures(
        sp_list=[sp], fourier_terms_list=[K], keep_original_columns=False
    ).fit_transform
    expected = pl.from_pandas(
        y.to_pandas().groupby(entity_col)[target_col].apply(aeon_add_fourier_terms)
    )
    assert_frame_equal(result.select(expected.columns), expected)
