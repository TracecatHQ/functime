from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import polars as pl

try:
    POLARS_MAJOR_VERSION = int(pl.__version__.split(".", 1)[0])
except ValueError:
    POLARS_MAJOR_VERSION = 0

if POLARS_MAJOR_VERSION >= 1:
    from polars.plugins import register_plugin_function

    rle_fields = {"value": "value", "len": "len"}

else:

    def register_plugin_function(
        *,
        plugin_path: Path | str,
        function_name: str,
        args: IntoExpr | Iterable[IntoExpr],
        kwargs: dict[str, Any] | None = None,
        is_elementwise: bool = False,
        returns_scalar: bool = False,
        cast_to_supertype: bool = False,
    ):
        expr = args[0]
        args1 = args[1:]
        expr.register_plugin(
            lib=plugin_path,
            args=args1,
            symbol=function_name,
            is_elementwise=is_elementwise,
            returns_scalar=returns_scalar,
            kwargs=kwargs,
            cast_to_supertypes=cast_to_supertype,
        )

    rle_fields = {"value": "values", "len": "lengths"}
