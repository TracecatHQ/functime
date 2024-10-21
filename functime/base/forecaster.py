from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Callable, List, Literal, Mapping, Optional, Tuple, TypeVar, Union

import polars as pl

from functime.base.model import Model, ModelState
from functime.base.transformer import Transformer
from functime.ranges import make_future_ranges

if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec
else:
    from typing import ParamSpec

# The parameters of the Model
P = ParamSpec("P")
# The return type of the estimator's curried function
R = Tuple[TypeVar("fit", bound=Callable), TypeVar("predict", bound=Callable)]

FORECAST_STRATEGIES = Optional[Literal["direct", "recursive", "naive"]]
DF_TYPE = Union[pl.LazyFrame, pl.DataFrame]


SUPPORTED_FREQ = [
    "1i",
    "1m",
    "5m",
    "10m",
    "15m",
    "30m",
    "1h",
    "1d",
    "1w",
    "1mo",
    "3mo",
    "1y",
]


def check_backtest_lengths(
    y: DF_TYPE,
    max_lags: int,
    test_size: int,
    drop_short: bool,
    drop_tolerance: float,
):
    # Heuristic check that the shortest time series
    # in the panel dataset is sufficiently long
    # for the given cross-validation parameters
    y = y.lazy()
    entity_col, time_col = y.columns[:2]
    lengths = (
        y.group_by(entity_col)
        .agg(pl.col(time_col).len().alias("len"))
        .collect(streaming=True)
    )
    n_entities = len(lengths)
    min_length = int(max_lags + test_size * drop_tolerance)
    short_series = lengths.filter(pl.col("len") < min_length + 1)
    n_short = len(short_series)
    if n_short > 0:
        if drop_short:
            logging.warning(
                "Dropping %s / %s entities with insufficient length."
                " Expected all individual time-series to have length > %s.",
                n_short,
                n_entities,
                min_length,
            )
            y = y.filter(~pl.col(entity_col).is_in(short_series.get_column(entity_col)))
        else:
            raise ValueError(
                f"Found {n_short} / {n_entities} entities with insufficient length."
                f" Expected all individual time-series to have length > {min_length}."
            )
    return y


@dataclass(frozen=True)
class ForecastState(ModelState):
    target: str
    target_schema: Mapping[str, pl.DataType]
    strategy: Optional[str] = "naive"
    features: Optional[List[str]] = None


class Forecaster(Model):
    """Autoregressive forecaster.

    Parameters
    ----------
    freq : str
        Offset alias supported by Polars.
    lags : int
        Number of lagged target variables.
    max_horizons: Optional[int]
        Maximum number of horizons to predict directly.
        Only applied if `strategy` equals "direct" or "ensemble".
    strategy : Optional[str]
        Forecasting strategy. Currently supports "recursive", "direct",
        and "ensemble" of both recursive and direct strategies.
    target_transform : Optional[Transformer]
        functime transformer to apply to `y` before fit. The transform is inverted at predict time.
    feature_transform : Optional[Transformer]
        functime transformer to apply to `X` before fit and predict.
    **kwargs : Mapping[str, Any]
        Additional keyword arguments passed into underlying sklearn-compatible regressor.
    """

    def __init__(
        self,
        freq: Union[str, None],
        lags: int,
        max_horizons: Optional[int] = None,
        strategy: FORECAST_STRATEGIES = None,
        target_transform: Optional[Union[Transformer, List[Transformer]]] = None,
        feature_transform: Optional[Union[Transformer, List[Transformer]]] = None,
        **kwargs,
    ):
        if freq not in SUPPORTED_FREQ:
            raise ValueError(f"Offset {freq} not supported")

        self.freq = freq
        self.lags = lags
        self.max_horizons = max_horizons
        self.strategy = strategy
        self.target_transform = target_transform
        self.feature_transform = feature_transform
        self.fitted_target_transform = []
        self.kwargs = kwargs
        super().__init__()

    def __call__(
        self,
        y: DF_TYPE,
        fh: int,
        X: Optional[DF_TYPE] = None,
        X_future: Optional[DF_TYPE] = None,
    ) -> pl.DataFrame:
        self.fit(y=y, X=X)
        return self.predict(fh=fh, X=X_future)

    def __repr__(self):
        return f"{self.__class__.__name__}(strategy={self.strategy})"

    def _transform_y(self, y: DF_TYPE):
        fitted_transformers = []
        target_transform = self.target_transform
        if not isinstance(target_transform, List):
            target_transform = [target_transform]
        for transf in target_transform:
            y = y.pipe(transf)
            fitted_transformers.append(transf)
        y_new = (
            y.collect().lazy()
        )  # Disable streaming (still buggy with group_by as of v0.19.1 polars)
        self.fitted_target_transform = fitted_transformers
        return y_new

    def _transform_X(self, y: DF_TYPE, X: Optional[DF_TYPE] = None):
        feature_transform = self.feature_transform
        if X is None:
            X = y.select(y.columns[:2])
        if not isinstance(feature_transform, List):
            feature_transform = [feature_transform]
        for transf in feature_transform:
            X = X.pipe(transf)
        X_new = X.collect().lazy()
        return X_new

    def fit(self, y: DF_TYPE, X: Optional[DF_TYPE] = None):
        # Prepare y
        y: pl.LazyFrame = self._set_string_cache(y.lazy().collect()).lazy()
        if self.target_transform is not None:
            y = self._transform_y(y=y)
        # Prepare X
        if X is not None:
            if X.columns[0] == y.columns[0]:
                X = self._enforce_string_cache(X.lazy().collect())
            X = X.lazy()
        # Feature transform
        if self.feature_transform is not None:
            X = self._transform_X(X=X, y=y)
        # Fit AR forecaster
        artifacts = self._fit(y=y, X=X)
        # Prepare artifacts
        cutoffs = y.group_by(y.columns[0]).agg(pl.col(y.columns[1]).max().alias("low"))
        # BUG: Regression after changing arange to int_ranges
        # artifacts["__cutoffs"] = cutoffs.collect(streaming=True)
        artifacts["__cutoffs"] = cutoffs.collect()
        state = ForecastState(
            entity=y.columns[0],
            time=y.columns[1],
            artifacts=artifacts,
            target=y.columns[-1],
            target_schema=y.schema,
            strategy=self.strategy or "recursive",
            features=X.columns[2:] if X is not None else None,
        )
        self.state = state
        return self

    def predict(self, fh: int, X: Optional[DF_TYPE] = None) -> pl.DataFrame:
        from functime.forecasting._ar import predict_autoreg

        state = self.state
        entity_col = state.entity
        time_col = state.time
        target_col = state.target
        schema = state.target_schema
        # Cutoffs cannot be lazy
        cutoffs: pl.DataFrame = state.artifacts["__cutoffs"]
        # Get entity, time "index" for forecast
        future_ranges = make_future_ranges(
            time_col=state.time,
            cutoffs=cutoffs,
            fh=fh,
            freq=self.freq,
        )
        # Prepare X
        if X is not None:
            X = X.lazy()
            # Coerce X (can be panel / time series / cross sectional) into panel
            # and aggregate feature columns into lists
            has_entity = X.columns[0] == state.entity
            has_time = X.columns[1] == state.time

            if has_entity:
                X = self._enforce_string_cache(X.lazy().collect()).lazy()

            if has_entity and not has_time:
                X = future_ranges.lazy().join(X, on=entity_col, how="left")
            elif has_time and not has_entity:
                X = future_ranges.lazy().join(X, on=time_col, how="left")

        # Feature transform
        if self.feature_transform is not None:
            X = self._transform_X(
                X=X, y=future_ranges.explode(pl.all().exclude(entity_col))
            )

        y_pred_vals = predict_autoreg(self.state, fh=fh, X=X)
        # BUG: Exploding list[date] errogenously casts
        # into integer series but only for LazyFrame explode
        y_pred = y_pred_vals.join(future_ranges, on=entity_col, how="left").select(
            [entity_col, time_col, target_col]
        )
        if fh > 1:
            y_pred = y_pred.explode(pl.all().exclude(entity_col))
        else:
            # BUG: Cannot explode list[date] and list[f32] with only single element
            # Raises: exploded columns must have matching element counts
            y_pred = y_pred.with_columns(
                [
                    pl.col(time_col).list.last(),
                    pl.col(target_col).list.last(),
                ]
            )

        # Invert target transforms
        if self.target_transform is not None:
            for transf in reversed(self.fitted_target_transform):
                y_pred = y_pred.with_columns(
                    pl.col(time_col).cast(schema[time_col])
                ).pipe(transf.invert)
        y_pred = y_pred.lazy().collect().pipe(self._reset_string_cache)
        return y_pred

    def backtest(
        self,
        y: DF_TYPE,
        X: Optional[pl.DataFrame] = None,
        test_size: int = 1,
        step_size: int = 1,
        n_splits: int = 5,
        window_size: int = 10,
        strategy: Literal["expanding", "sliding"] = "expanding",
        drop_short: bool = False,
        drop_tolerance: float = 2.0,
    ):
        from functime.backtesting import backtest
        from functime.cross_validation import (
            expanding_window_split,
            sliding_window_split,
        )

        y = check_backtest_lengths(
            y,
            max_lags=self.lags,
            test_size=test_size,
            drop_short=drop_short,
            drop_tolerance=drop_tolerance,
        )
        if strategy == "expanding":
            cv = expanding_window_split(
                test_size=test_size,
                step_size=step_size,
                n_splits=n_splits,
            )
        else:
            cv = sliding_window_split(
                test_size=test_size,
                step_size=step_size,
                n_splits=n_splits,
                window_size=window_size,
            )
        y_preds, y_resids = backtest(
            forecaster=self,
            fh=test_size,
            y=y,
            X=X,
            cv=cv,
            residualize=True,
        )
        return y_preds, y_resids

    def conformalize(
        self,
        fh: int,
        y: pl.DataFrame,
        X: Optional[DF_TYPE] = None,
        X_future: Optional[DF_TYPE] = None,
        alphas: Optional[List[float]] = None,
        test_size: int = 1,
        step_size: int = 1,
        n_splits: int = 5,
        window_size: int = 10,
        strategy: Literal["expanding", "sliding"] = "expanding",
        drop_short: bool = False,
        drop_tolerance: float = 5.0,
        return_results: bool = False,
    ) -> pl.DataFrame:
        from functime.conformal import conformalize

        self.fit(y=y, X=X)

        y_pred = self.predict(fh=fh, X=X_future)

        y_preds, y_resids = self.backtest(
            y=y,
            X=X,
            test_size=test_size,
            step_size=step_size,
            n_splits=n_splits,
            window_size=window_size,
            strategy=strategy,
            drop_short=drop_short,
            drop_tolerance=drop_tolerance,
        )

        y_pred_qnts = conformalize(
            y_pred=y_pred,
            y_preds=y_preds,
            y_resids=y_resids,
            alphas=alphas,
        )

        if return_results:
            return y_pred, y_pred_qnts, y_preds, y_resids
        return y_pred_qnts
