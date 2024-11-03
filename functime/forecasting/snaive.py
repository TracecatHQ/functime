from __future__ import annotations

from typing import Optional, Union

import polars as pl

from functime.base import Forecaster
from functime.ranges import make_future_ranges


class snaive(Forecaster):
    """Seasonal naive forecaster.

    Parameters
    ----------
    freq : str
        Offset alias supported by Polars.
    sp : int
        Seasonal periods.
    """

    def __init__(self, freq: str, sp: int):
        self.sp = sp
        super().__init__(freq=freq, lags=1)

    def _fit(self, y: pl.LazyFrame, X: Optional[pl.LazyFrame] = None):
        idx_cols = y.columns[:2]
        entity_col = idx_cols[0]
        target_col = y.columns[2]
        sp = self.sp
        # BUG: Cannot run the following in lazy streaming mode?
        # Causes internal error: entered unreachable code
        y_pred = y.sort(idx_cols).group_by(entity_col).agg(pl.col(target_col).tail(sp))
        artifacts = {"y_pred": y_pred}
        return artifacts

    def predict(
        self, fh: int, X: Optional[Union[pl.DataFrame, pl.LazyFrame]] = None
    ) -> pl.DataFrame:
        state = self.state
        entity = state.entity
        target = state.target
        sp = self.sp
        # Cutoffs cannot be lazy
        cutoffs: pl.DataFrame = state.artifacts["__cutoffs"]
        future_ranges = make_future_ranges(
            time_col=state.time,
            cutoffs=cutoffs,
            fh=fh,
            freq=self.freq,
        )
        y_pred_vals = state.artifacts["y_pred"]
        y_pred_vals = (
            y_pred_vals.sort(by=entity)
            .select(
                pl.concat_list(
                    [pl.col(target).list.get(i % sp).alias(f"fh{i}") for i in range(fh)]
                ).alias(target)
            )
            .collect()
        )
        y_pred = pl.concat(
            [future_ranges.sort(by=entity), y_pred_vals], how="horizontal"
        ).explode(pl.all().exclude(entity))
        return y_pred.pipe(self._reset_string_cache)
