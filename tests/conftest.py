from __future__ import annotations

import logging
import os
from functools import partial
from typing import List

import numpy as np
import pandas as pd
import polars as pl
import pytest

from functime.cross_validation import train_test_split
from functime.offsets import freq_to_sp


@pytest.fixture(params=[50], ids=lambda x: f"n_periods({x})")
def n_periods(request):
    return request.param


@pytest.fixture(params=[10], ids=lambda x: f"n_entities({x})")
def n_entities(request):
    return request.param


@pytest.fixture
def pd_X(n_periods, n_entities):
    """Return panel pd.DataFrame with sin, cos, and tan columns and time,
    entity multi-index. Used to benchmark polars vs pandas.
    """
    entity_idx = [f"x{i}" for i in range(n_entities)]
    time_idx = pd.date_range("2020-01-01", periods=n_periods, freq="1D", name="time")
    multi_row = [(entity, timestamp) for entity in entity_idx for timestamp in time_idx]
    idx = pd.MultiIndex.from_tuples(multi_row, names=["series_id", "time"])
    sin_x = np.sin(np.arange(0, n_entities * n_periods))
    cos_x = np.cos(np.arange(0, n_entities * n_periods))
    tan_x = np.tan(np.arange(0, n_entities * n_periods))
    X = pd.DataFrame({"open": sin_x, "close": cos_x, "volume": tan_x}, index=idx)
    return X.sort_index()


@pytest.fixture
def pd_y(pd_X):
    return pd_X.loc[:, "close"]


@pytest.fixture
def pl_y(pd_y):
    return pl.from_pandas(pd_y.reset_index()).lazy()


@pytest.fixture
def m4_freq_to_sp():
    return {
        "1d": freq_to_sp("1d")[0],
        "1w": freq_to_sp("1w")[0],
        "1mo": freq_to_sp("1mo")[0],
        "3mo": freq_to_sp("3mo")[0],
        "1y": freq_to_sp("1y")[0],
    }


@pytest.fixture
def m4_freq_to_lags():
    return {
        "1d": 30,
        "1w": 14,
        "1mo": 12,
        "3mo": 6,
        "1y": 3,
    }


@pytest.fixture(
    params=[
        ("1d", 14, "data/m4_1d_train.parquet/*/*", "data/m4_1d_test.parquet"),
        ("1w", 13, "data/m4_1w_train.parquet", "data/m4_1w_test.parquet"),
        ("1mo", 18, "data/m4_1mo_train.parquet/*/*", "data/m4_1mo_test.parquet"),
        ("3mo", 8, "data/m4_3mo_train.parquet", "data/m4_3mo_test.parquet"),
        ("1y", 6, "data/m4_1y_train.parquet", "data/m4_1y_test.parquet"),
    ],
    ids=lambda x: f"freq_{x[0]}-fh_{x[1]}",
    scope="module",
)
def m4_dataset(request):
    def load_panel_data(path: str) -> pl.LazyFrame:
        return (
            pl.read_parquet(path)
            .with_columns(
                pl.col("series").str.replace(" ", "").cast(pl.Categorical),
                pl.col("time").cast(pl.Int16),
                pl.all().exclude(["series", "time"]).cast(pl.Float32),
            )
            .sort(["series", "time"])
        )

    def update_test_time_ranges(y_train, y_test):
        entity_col, time_col = y_train.columns[:2]
        cutoffs = y_train.group_by(entity_col).agg(
            pl.col(time_col).last().alias("cutoff")
        )
        y_test = (
            y_test.join(cutoffs, on=entity_col, how="left")
            .with_columns(pl.col(time_col) + pl.col("cutoff").alias(time_col))
            .drop("cutoff")
        )
        return y_test

    freq, fh, train_path, test_path = request.param
    y_train = load_panel_data(train_path)
    y_test = load_panel_data(test_path)
    y_test = update_test_time_ranges(y_train, y_test)

    # Check m4 dataset RAM usage
    logging.info("y_train mem: %s", f'{y_train.estimated_size("mb"):.4f} mb')
    logging.info("y_test mem: %s", f'{y_test.estimated_size("mb"):.4f} mb')
    # Preview
    logging.info("y_train preview: %s", y_train)
    logging.info("y_test preview: %s", y_test)

    return y_train.lazy(), y_test.lazy(), fh, freq


def prepare_m5_dataset(m5_train: pl.LazyFrame, m5_test: pl.LazyFrame):
    def filter_most_recent(
        X: pl.LazyFrame, max_eval_period: int, train_periods: int
    ) -> pl.LazyFrame:
        return X.filter(
            pl.col("d").str.slice(2).cast(pl.Int32) > max_eval_period - train_periods
        )

    def drop_leading_zeros(
        X: pl.LazyFrame, entity_col: str, target_col: str
    ) -> pl.LazyFrame:
        X_new = X.filter(
            ((pl.col(target_col) > 0).cast(pl.Int8).cummax().cast(pl.Boolean)).over(
                entity_col
            )
        )
        return X_new

    def preprocess(
        X: pl.LazyFrame,
        entity_col: str,
        sampled_entities: List[str],
        categorical_cols: List[str],
        boolean_cols: List[str],
    ) -> pl.LazyFrame:
        X_new = (
            X.select(
                pl.all().exclude(["d", "sell_price", "event_type_2", "event_name_2"])
            )  # Drop constant and unused columns
            .filter(pl.col(entity_col).is_in(sampled_entities))
            .with_columns(
                pl.col(categorical_cols).cast(pl.Utf8),
                pl.col(boolean_cols).cast(pl.Boolean),
            )
        )
        return X_new

    # Prepare M5 dataset
    # Specification
    sample_frac = 0.02  # 10% ~1.2 million rows
    entity_col = "id"
    time_col = "date"
    target_col = "quantity_sold"
    max_eval_period = 1914
    train_periods = 420
    categorical_cols = [
        "id",
        "state_id",
        "store_id",
        "dept_id",
        "cat_id",
        "item_id",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
    ]
    boolean_cols = ["snap_CA", "snap_TX", "snap_WI"]

    # Load train data and get entities sample
    # NOTE: Must sort, maintain order, and set seed to prevent flaky test
    sampled_entities = (
        m5_train.select(entity_col)
        .collect()
        .get_column(entity_col)
        .sort()
        .unique(maintain_order=True)
        .sample(fraction=sample_frac, seed=42)
    )

    preprocess_transform = partial(
        preprocess,
        entity_col=entity_col,
        sampled_entities=sampled_entities,
        categorical_cols=categorical_cols,
        boolean_cols=boolean_cols,
    )

    X_y_train = (
        m5_train.pipe(
            filter_most_recent,
            max_eval_period=max_eval_period,
            train_periods=train_periods,
        )
        .pipe(drop_leading_zeros, entity_col=entity_col, target_col=target_col)
        .pipe(preprocess_transform)
        .collect()
    )
    X_y_test = m5_test.pipe(preprocess_transform).collect()

    # Train test split
    endog_cols = [entity_col, time_col, target_col]
    exog_cols = pl.all().exclude(target_col)
    y_train = X_y_train.select(endog_cols)
    X_train = X_y_train.select(exog_cols)
    y_test = X_y_test.select(endog_cols)
    X_test = X_y_test.select(exog_cols)

    return y_train, X_train, y_test, X_test


@pytest.fixture
def m5_dataset():
    """M5 competition Walmart dataset grouped by stores."""

    # Specification
    fh = 28
    max_lags = 64
    freq = "1d"
    n_samples = 30

    # Load data
    y_train = pl.read_parquet("data/m5_y_train_sample.parquet")
    X_train = pl.read_parquet("data/m5_X_train_sample.parquet")
    y_test = pl.read_parquet("data/m5_y_test_sample.parquet")
    X_test = pl.read_parquet("data/m5_X_test_sample.parquet")

    # Check series lengths
    entity_col, time_col, value_col = y_train.columns[:3]
    short_ts_counts = (
        y_train.group_by(entity_col)
        .agg(pl.count(time_col).alias("count"))
        .filter(pl.col("count") <= max_lags)
        .sort(by="count")
    )

    # Preview short series (not supported by lags >= 64) in M5 dataset
    with pl.Config(tbl_rows=-1):
        entity_col, time_col = y_train.columns[:2]
        logging.info(
            "%s short time-series (<= %s): %s",
            len(short_ts_counts),
            max_lags,
            short_ts_counts,
        )

    # Sample if FUNCTIME__TEST_MODE=true env var is set
    if os.environ.get("FUNCTIME__TEST_MODE", "").lower() == "true":
        # Get top N top sellers
        top_sellers = (
            y_train.groupby(entity_col)
            .agg(pl.col(value_col).sum())
            .top_k(n_samples, by=value_col)
            .get_column(entity_col)
        )
        y_train = y_train.filter(pl.col(entity_col).is_in(top_sellers))
        X_train = X_train.filter(pl.col(entity_col).is_in(top_sellers))
        y_test = y_test.filter(pl.col(entity_col).is_in(top_sellers))
        X_test = X_test.filter(pl.col(entity_col).is_in(top_sellers))

    # Check m5 dataset RAM usage
    logging.info("y_train mem: %s", f'{y_train.estimated_size("mb"):.4f} mb')
    logging.info("X_train mem: %s", f'{X_train.estimated_size("mb"):.4f} mb')
    logging.info("y_test mem: %s", f'{y_test.estimated_size("mb"):.4f} mb')
    logging.info("X_test mem: %s", f'{X_test.estimated_size("mb"):.4f} mb')

    # Preview
    logging.info("y_train preview: %s", y_train)
    logging.info("X_train preview: %s", X_train)
    logging.info("y_test preview: %s", y_test)
    logging.info("X_test preview: %s", X_test)

    return y_train.lazy(), X_train.lazy(), y_test.lazy(), X_test.lazy(), fh, freq


@pytest.fixture
def dunnhumby_retail():
    """Dunn Humby: The Complete Journey retail dataset.

    https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey
    """

    entity_col = "household_key__PRODUCT_ID"
    time_col = "DAY"
    feature_cols = [
        "UNIT_PRICE",
        "STORE_ID",
        "RETAIL_DISC",
        "TRANS_TIME",
        "WEEK_NO",
        "COUPON_DISC",
        "COUPON_MATCH_DISC",
        # NOTE: Too many dummy variables problem in demand forecasting
        # We can use prod2vec to cluster similar products
        # We can also experiment using prod2vec to cluster households
        # by the cluster of products they are most likely to purchase
        # "household_key",
        # "PRODUCT_ID",
    ]
    target_col = "QUANTITY"
    fh = 12
    freq = "1d"

    data = pl.read_parquet("data/dunnhumby.parquet").with_columns(
        # Create UNIT_PRICE column
        (pl.col("SALES_VALUE") / pl.col("QUANTITY"))
        .round(2)
        .cast(pl.Int16)
        .alias("UNIT_PRICE"),
        # Concat entity keys to make entity col
        pl.concat_str(
            [pl.col("household_key"), pl.col("PRODUCT_ID")], separator="__"
        ).alias(entity_col),
    )
    y = data.select([entity_col, time_col, target_col])
    X = data.select([entity_col, time_col, *feature_cols])

    # Train test split
    y_train, y_test = y.pipe(train_test_split(test_size=fh))
    X_train, X_test = X.pipe(train_test_split(test_size=fh))

    return y_train.lazy(), X_train.lazy(), y_test.lazy(), X_test.lazy(), fh, freq


# if __name__ == "__main__":
# y_train.collect().write_parquet("m5_y_train.parquet")
# X_train.collect().write_parquet("m5_X_train.parquet")
# y_test.collect().write_parquet("m5_y_test.parquet")
# X_test.collect().write_parquet("m5_X_test.parquet")
